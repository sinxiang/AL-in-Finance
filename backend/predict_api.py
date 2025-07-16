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
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

@router.post("/predict")
async def predict_stock(payload: PredictRequest):
    symbol = payload.symbol.upper()
    days = min(payload.days, 90)
    model_name = payload.model or "ensemble"

    print(f"[INFO] Predict request: {symbol}, days={days}, model={model_name}")

    try:
        df = yf.download(symbol, period="720d", interval="1d", progress=False)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found.")

        # Feature Engineering
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["RSI14"] = calculate_rsi(df["Close"], 14)
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10", "MA20", "RSI14"]
        X = df[features].values
        y = df["Target"].values

        models = {
            "random_forest": RandomForestRegressor(n_estimators=100),
            "gb": GradientBoostingRegressor(),
            "xgb": XGBRegressor(objective='reg:squarederror', verbosity=0, n_estimators=50),
            "linear": LinearRegression(),
        }

        preds_all = []
        metrics_all = []

        used_models = models if model_name == "ensemble" else {model_name: models[model_name]}

        for name, model in used_models.items():
            print(f"[INFO] Training model: {name}")
            model.fit(X, y)

            recent_df = df.iloc[-1:].copy()
            preds = []

            for _ in range(days):
                recent_features = recent_df[features].values
                pred = model.predict(recent_features)[0]
                preds.append(float(pred))  # ✅ 保证输出为 Python float

                # Append new row with predicted Close
                new_row = {
                    "Open": float(pred),
                    "High": float(pred),
                    "Low": float(pred),
                    "Close": float(pred),
                    "Volume": float(recent_df["Volume"].values[0]),
                }
                temp_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                temp_df["MA5"] = temp_df["Close"].rolling(window=5).mean()
                temp_df["MA10"] = temp_df["Close"].rolling(window=10).mean()
                temp_df["MA20"] = temp_df["Close"].rolling(window=20).mean()
                temp_df["RSI14"] = calculate_rsi(temp_df["Close"], 14)
                recent_df = temp_df.iloc[[-1]].copy()

            r2 = r2_score(y, model.predict(X))
            mae = mean_absolute_error(y, model.predict(X))
            metrics_all.append((name, float(r2), float(mae)))
            preds_all.append(preds)

        # Final predictions
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
        import traceback
        print("[ERROR] Exception occurred:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router, prefix="/api")
