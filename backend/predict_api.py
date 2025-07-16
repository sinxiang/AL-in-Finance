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
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
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
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # 用中性值填充前期NA

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA5"] = df["Close"].rolling(window=5, min_periods=5).mean()
    df["MA10"] = df["Close"].rolling(window=10, min_periods=10).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    # 改用 bfill()，避免 FutureWarning
    df["MA5"] = df["MA5"].bfill()
    df["MA10"] = df["MA10"].bfill()
    df["MA20"] = df["MA20"].bfill()
    df["RSI14"] = df["RSI14"].fillna(50)
    return df

@router.post("/predict")
async def predict_stock(payload: PredictRequest):
    symbol = payload.symbol.upper()
    days = min(payload.days, 90)
    model_name = payload.model or "ensemble"

    print(f"[INFO] Predict request: {symbol}, days={days}, model={model_name}")

    try:
        df = yf.download(symbol, period="720d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found.")

        df = calculate_features(df)
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10", "MA20", "RSI14"]
        X = df[features].values
        y = df["Target"].values

        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gb": GradientBoostingRegressor(random_state=42),
            "xgb": XGBRegressor(objective='reg:squarederror', verbosity=0, n_estimators=100, random_state=42),
            "linear": LinearRegression(),
        }

        if model_name != "ensemble" and model_name not in models:
            raise HTTPException(status_code=400, detail=f"Model {model_name} not supported.")

        used_models = models if model_name == "ensemble" else {model_name: models[model_name]}

        preds_all = []
        metrics_all = []

        window_len = 20

        for name, model in used_models.items():
            print(f"[INFO] Training model: {name}")
            model.fit(X, y)

            if len(df) < window_len:
                raise HTTPException(status_code=400, detail="Not enough historical data for sliding window.")

            sliding_window = df.iloc[-window_len:].copy().reset_index(drop=True)
            preds = []

            for _ in range(days):
                input_features = sliding_window[features].values[-1].reshape(1, -1)
                pred = model.predict(input_features)[0]
                preds.append(float(pred))

                # 用 concat 代替 append
                new_row_df = pd.DataFrame([{
                    "Open": float(pred),
                    "High": float(pred),
                    "Low": float(pred),
                    "Close": float(pred),
                    "Volume": sliding_window["Volume"].iloc[-1],
                }])
                sliding_window = pd.concat([sliding_window, new_row_df], ignore_index=True)

                sliding_window = calculate_features(sliding_window)
                sliding_window = sliding_window.iloc[-window_len:].copy().reset_index(drop=True)

            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            metrics_all.append((name, float(r2), float(mae)))
            preds_all.append(preds)

        final_preds = (
            list(map(float, np.mean(preds_all, axis=0)))
            if model_name == "ensemble"
            else list(map(float, preds_all[0]))
        )

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
        print("[ERROR] Exception occurred:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router, prefix="/api")
