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

# 允许跨域访问
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
    model: Optional[str] = "ensemble"  # 支持"random_forest","gb","xgb","linear","ensemble"

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["MA10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    # 先后向填充均线缺失，保证无NaN
    for col in ["MA5", "MA10", "MA20"]:
        df[col] = df[col].bfill().ffill()
    df["RSI14"] = df["RSI14"].fillna(50)
    return df

@router.post("/predict")
async def predict_stock(payload: PredictRequest):
    symbol = payload.symbol.upper()
    days = max(1, min(payload.days, 90))  # 限制预测天数1~90
    model_name = payload.model or "ensemble"

    print(f"[INFO] Predict request: {symbol}, days={days}, model={model_name}")

    try:
        # 下载720天的历史数据，关闭自动调整以保持原始数据
        df = yf.download(symbol, period="720d", interval="1d", progress=False, auto_adjust=False)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found for symbol.")
        if "Close" not in df.columns:
            raise HTTPException(status_code=400, detail="Close price data missing.")

        df = calculate_features(df)

        # 构造目标列 Target 为下一天收盘价
        df["Target"] = df["Close"].shift(-1)

        # 删除无目标的最后一行（未来无标签）
        df = df.dropna(subset=["Target"])

        # 再确认无NaN特征列，缺失则报错
        features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10", "MA20", "RSI14"]
        if df[features].isnull().values.any():
            raise HTTPException(status_code=500, detail="NaN found in feature columns after processing.")

        X = df[features].values
        y = df["Target"].values

        # 初始化模型
        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gb": GradientBoostingRegressor(random_state=42),
            "xgb": XGBRegressor(objective='reg:squarederror', verbosity=0, n_estimators=50, random_state=42),
            "linear": LinearRegression(),
        }

        if model_name != "ensemble" and model_name not in models:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not supported.")

        used_models = models if model_name == "ensemble" else {model_name: models[model_name]}

        window_len = 60  # 滑动窗口长度
        preds_all = []
        metrics_all = []

        for name, model in used_models.items():
            print(f"[INFO] Training model: {name}")
            model.fit(X, y)

            sliding_window = df.tail(window_len).copy().reset_index(drop=True)
            preds = []

            for _ in range(days):
                input_features = sliding_window[features].iloc[-1].values.reshape(1, -1)

                # 用均值填充NaN，避免模型predict失败
                if np.isnan(input_features).any():
                    nan_fill = np.nanmean(sliding_window[features].values)
                    input_features = np.nan_to_num(input_features, nan=nan_fill)

                pred = model.predict(input_features)[0]
                preds.append(float(pred))

                # 构造下一条预测行，维持数据类型
                new_row_df = pd.DataFrame([{
                    "Open": float(pred),
                    "High": float(pred),
                    "Low": float(pred),
                    "Close": float(pred),
                    "Volume": sliding_window["Volume"].iloc[-1],
                }])

                sliding_window = pd.concat([sliding_window, new_row_df], ignore_index=True)
                sliding_window = calculate_features(sliding_window)

                # 保持滑动窗口大小
                if len(sliding_window) > window_len:
                    sliding_window = sliding_window.iloc[-window_len:].copy().reset_index(drop=True)

            # 计算指标
            r2 = r2_score(y, model.predict(X))
            mae = mean_absolute_error(y, model.predict(X))
            metrics_all.append((name, float(r2), float(mae)))
            preds_all.append(preds)

        # 组合结果
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
