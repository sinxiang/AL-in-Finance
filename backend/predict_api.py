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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

class PredictRequest(BaseModel):
    symbol: str
    days: int
    model: Optional[str] = "ensemble"

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # 填充中性值
    return rsi

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 计算移动均线，允许最小period减少以避免丢失数据
    df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["MA10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    # 前后填充缺失值
    for col in ["MA5", "MA10", "MA20"]:
        df[col] = df[col].bfill().ffill()
    df["RSI14"] = df["RSI14"].fillna(50)
    return df

@router.post("/predict")
async def predict_stock(payload: PredictRequest):
    symbol = payload.symbol.upper()
    days = min(payload.days, 90)  # 最大预测90天
    model_name = payload.model or "ensemble"

    print(f"[INFO] Predict request: {symbol}, days={days}, model={model_name}")

    try:
        # 下载数据720天，关闭auto_adjust防止价格数据调整
        df = yf.download(symbol, period="720d", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found.")

        # 计算技术指标特征
        df = calculate_features(df)

        # 创建Target列为下一个交易日的Close价格
        df["Target"] = df["Close"].shift(-1)
        # 删除Target缺失的行
        df.dropna(subset=["Target"], inplace=True)

        features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10", "MA20", "RSI14"]

        # 训练特征和目标
        X = df[features].values
        y = df["Target"].values

        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gb": GradientBoostingRegressor(random_state=42),
            "xgb": XGBRegressor(objective='reg:squarederror', verbosity=0, n_estimators=50, random_state=42),
            "linear": LinearRegression(),
        }

        preds_all = []
        metrics_all = []

        used_models = models if model_name == "ensemble" else {model_name: models[model_name]}

        window_len = 60  # 滑动窗口长度

        for name, model in used_models.items():
            print(f"[INFO] Training model: {name}")
            model.fit(X, y)

            # 初始化滑动窗口缓存，保留最近window_len条数据
            sliding_window = df.tail(window_len).copy().reset_index(drop=True)

            preds = []

            for _ in range(days):
                input_features = sliding_window[features].iloc[-1].values.reshape(1, -1)

                # 使用滑动窗口所有值均值替代输入中的nan
                flat_vals = sliding_window[features].values.flatten()
                nan_fill_value = np.nanmean(flat_vals)
                input_features = np.nan_to_num(input_features, nan=nan_fill_value)

                pred = model.predict(input_features)[0]
                preds.append(float(pred))

                # 构建新行，Open/High/Low/Close 均用预测值，Volume沿用前一行
                new_row_df = pd.DataFrame([{
                    "Open": float(pred),
                    "High": float(pred),
                    "Low": float(pred),
                    "Close": float(pred),
                    "Volume": sliding_window["Volume"].iloc[-1],
                }])

                # 拼接新行
                sliding_window = pd.concat([sliding_window, new_row_df], ignore_index=True)

                # 重新计算特征，保证滑动窗口长度不变
                sliding_window = calculate_features(sliding_window)
                if len(sliding_window) > window_len:
                    sliding_window = sliding_window.iloc[-window_len:].copy().reset_index(drop=True)

            # 计算模型指标
            r2 = r2_score(y, model.predict(X))
            mae = mean_absolute_error(y, model.predict(X))
            metrics_all.append((name, float(r2), float(mae)))
            preds_all.append(preds)

        # 结果融合
        if model_name == "ensemble":
            final_preds = list(map(float, np.mean(preds_all, axis=0)))
            model_used = "ensemble"
            model_r2 = float(np.mean([m[1] for m in metrics_all]))
            model_mae = float(np.mean([m[2] for m in metrics_all]))
        else:
            final_preds = list(map(float, preds_all[0]))
            model_used = list(used_models.keys())[0]
            model_r2 = metrics_all[0][1]
            model_mae = metrics_all[0][2]

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
