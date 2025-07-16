from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler  # ✅ 新增导入

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可根据需要调整前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

class PredictRequest(BaseModel):
    symbol: str
    days: int  # 预测天数，最多90
    model: Optional[str] = "ensemble"  # 模型选择

def create_sliding_window_features(df: pd.DataFrame, features: list, window_size: int):
    data = df[features].values
    n_samples = len(df) - window_size
    if n_samples <= 0:
        raise ValueError("Not enough data for the given sliding window size")

    X = np.zeros((n_samples, window_size * len(features)))
    for i in range(n_samples):
        window = data[i:i+window_size].flatten()
        X[i, :] = window
    return X

@router.post("/predict")
async def predict_stock(payload: PredictRequest):
    symbol = payload.symbol.upper()
    days = min(payload.days, 90)
    model_name = payload.model or "ensemble"
    window_size = 10

    print(f"[INFO] Predict request: {symbol}, days={days}, model={model_name}")

    try:
        df = yf.download(symbol, period="1460d", interval="1d", progress=False)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found.")

        df.dropna(inplace=True)
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()
        delta = df["Close"].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        avg_gain = up.rolling(window=14).mean()
        avg_loss = down.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df.dropna(inplace=True)

        features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10", "MA20", "RSI"]

        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        total_len = len(df)
        if total_len <= window_size + days:
            raise HTTPException(status_code=400, detail="Not enough data for requested prediction days and window size.")

        train_len = 756
        if train_len > total_len:
            train_len = total_len - days - window_size
        df_train = df.iloc[:train_len].reset_index(drop=True)

        # 训练特征归一化
        scaler = MinMaxScaler()
        scaler.fit(df_train[features])

        df_train_scaled = df_train.copy()
        df_train_scaled[features] = scaler.transform(df_train[features])

        # 训练标签也归一化（只归一化Close列，目标是Close的下一天，所以标签也要归一化）
        target_scaler = MinMaxScaler()
        y_train_close = df_train[["Target"]].values  # shape (n,1)
        target_scaler.fit(y_train_close)
        y_train_scaled = target_scaler.transform(y_train_close).flatten()

        # 构造训练特征和标签（滑动窗口）
        X_train = create_sliding_window_features(df_train_scaled, features, window_size)
        y_train = y_train_scaled[window_size:]  # 对应标签滑动窗口后剔除window_size

        eval_len = 252
        if train_len + eval_len + days > total_len:
            eval_len = total_len - train_len - days
        df_eval = df.iloc[train_len:train_len + eval_len].reset_index(drop=True)

        df_eval_scaled = df_eval.copy()
        df_eval_scaled[features] = scaler.transform(df_eval[features])

        y_eval_close = df_eval[["Target"]].values
        y_eval_scaled = target_scaler.transform(y_eval_close).flatten()

        X_eval = create_sliding_window_features(df_eval_scaled, features, window_size)
        y_eval = y_eval_scaled[window_size:]

        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gb": GradientBoostingRegressor(random_state=42),
            "xgb": XGBRegressor(objective='reg:squarederror', verbosity=0, n_estimators=50, random_state=42),
            "linear": LinearRegression(),
        }

        used_models = models if model_name == "ensemble" else {model_name: models.get(model_name)}
        if None in used_models.values():
            raise HTTPException(status_code=400, detail=f"Unsupported model '{model_name}'.")

        preds_all = []
        metrics_all = []

        for name, model in used_models.items():
            print(f"[INFO] Training model: {name}")
            model.fit(X_train, y_train)

            y_pred_eval_scaled = model.predict(X_eval)
            # 评估指标计算时，将预测和真实标签反归一化再评估更合理
            y_pred_eval = target_scaler.inverse_transform(y_pred_eval_scaled.reshape(-1,1)).flatten()
            y_eval_real = target_scaler.inverse_transform(y_eval.reshape(-1,1)).flatten()
            r2 = r2_score(y_eval_real, y_pred_eval)
            mae = mean_absolute_error(y_eval_real, y_pred_eval)

            preds = []
            # 初始输入特征为训练集最后window_size天（归一化）
            last_days_data = df_train_scaled.iloc[train_len - window_size:train_len][features].values.flatten()
            input_feat = last_days_data.reshape(1, -1)

            for _ in range(days):
                pred_scaled = model.predict(input_feat)[0]  # 预测归一化的Target值
                # 逆归一化成真实股价
                pred_real = target_scaler.inverse_transform([[pred_scaled]])[0,0]
                preds.append(pred_real)

                # 取当前输入特征最后一天的归一化特征（用于保留Volume和计算技术指标）
                last_day_features = input_feat[0, -len(features):]

                # Volume保持不变，直接取最后一天的归一化Volume
                last_volume_scaled = last_day_features[4]

                # 技术指标MA5, MA10, MA20, RSI 用前一天的技术指标值（last_day_features中对应位置）
                # Open, High, Low, Close 用预测的归一化Close值（pred_scaled）
                new_feat_scaled = np.array([
                    pred_scaled,        # Open
                    pred_scaled,        # High
                    pred_scaled,        # Low
                    pred_scaled,        # Close
                    last_volume_scaled,  # Volume
                    last_day_features[5],  # MA5
                    last_day_features[6],  # MA10
                    last_day_features[7],  # MA20
                    last_day_features[8],  # RSI
                ])

                # 滚动窗口左移
                input_feat = np.roll(input_feat, -len(features))
                input_feat[0, -len(features):] = new_feat_scaled

            preds_all.append(preds)
            metrics_all.append((name, r2, mae))

        if model_name == "ensemble":
            final_preds = np.mean(preds_all, axis=0).tolist()
        else:
            final_preds = preds_all[0]

        model_used = "ensemble" if model_name == "ensemble" else list(used_models.keys())[0]
        model_r2 = float(np.mean([m[1] for m in metrics_all]))
        model_mae = float(np.mean([m[2] for m in metrics_all]))

        trend = "upward" if final_preds[-1] > final_preds[0] else "downward"
        risk = "low" if model_mae < 2 else "high"
        suggestion = (
            "Likely to rise, consider buying."
            if trend == "upward" and risk == "low"
            else "Trend uncertain or risk high, be cautious."
        )

        print("[INFO] Prediction completed.")

        return {
            "predictions": final_preds,
            "metrics": {
                "model": model_used,
                "r2": round(model_r2, 4),
                "mae": round(model_mae, 4),
            },
            "advice": {
                "trend": trend,
                "risk": risk,
                "suggestion": suggestion,
            },
        }

    except Exception as e:
        import traceback
        print("[ERROR] Exception occurred:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router, prefix="/api")
