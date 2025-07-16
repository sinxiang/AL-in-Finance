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
    """
    利用过去window_size天的特征构造多维输入矩阵
    输入：DataFrame，特征名列表，窗口大小
    输出：二维np数组，每行是过去window_size天所有特征拼接
    """
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
    window_size = 5  # 滑动窗口大小，可以调整

    print(f"[INFO] Predict request: {symbol}, days={days}, model={model_name}")

    try:
        # 下载最近4年数据（大于训练+评估需求）
        df = yf.download(symbol, period="1460d", interval="1d", progress=False)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found.")

        df.dropna(inplace=True)
        features = ["Open", "High", "Low", "Close", "Volume"]

        # 构造标签，目标是未来1天Close价格
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)  # 去掉最后一天，因为Target为NaN

        # 整体数据长度
        total_len = len(df)
        if total_len <= window_size + days:
            raise HTTPException(status_code=400, detail="Not enough data for requested prediction days and window size.")

        # 训练数据用最近3年（约756个交易日）
        train_len = 756
        if train_len > total_len:
            train_len = total_len - days - window_size  # 保证够预测和窗口
        df_train = df.iloc[:train_len].reset_index(drop=True)

        # 构造训练集特征和标签
        X_train = create_sliding_window_features(df_train, features, window_size)
        y_train = df_train["Target"].values[window_size:]

        # 评估用最近1年数据(约252个交易日)
        eval_len = 252
        if train_len + eval_len + days > total_len:
            eval_len = total_len - train_len - days
        df_eval = df.iloc[train_len:train_len + eval_len].reset_index(drop=True)
        X_eval = create_sliding_window_features(df_eval, features, window_size)
        y_eval = df_eval["Target"].values[window_size:]

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

            # 评估指标计算
            y_pred_eval = model.predict(X_eval)
            r2 = r2_score(y_eval, y_pred_eval)
            mae = mean_absolute_error(y_eval, y_pred_eval)

            # 递归预测未来days天
            preds = []
            # 取训练集最后window_size天数据特征做预测输入
            last_days_data = df.iloc[train_len - window_size:train_len][features].values.flatten()
            input_feat = last_days_data.reshape(1, -1)  # 1行多列

            for _ in range(days):
                pred = model.predict(input_feat)[0]
                preds.append(float(pred))

                # 更新输入特征，剔除最旧一天，添加预测一天的5个特征
                # 这里简单用预测值填充Open/High/Low/Close，Volume用前一天Volume（最后一个Volume）
                new_feat = np.array([pred, pred, pred, pred, input_feat[0, -1]])
                input_feat = np.roll(input_feat, -len(features))
                input_feat[0, -len(features):] = new_feat

            preds_all.append(preds)
            metrics_all.append((name, r2, mae))

        # 集成模型预测取均值
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
