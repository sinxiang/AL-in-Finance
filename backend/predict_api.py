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
    return rsi.bfill().ffill().fillna(50)


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df["MA5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["MA10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["RSI14"] = calculate_rsi(df["Close"])
    for col in ["MA5", "MA10", "MA20", "RSI14"]:
        df[col] = df[col].bfill().ffill()
    return df


@router.post("/predict")
async def predict_stock(payload: PredictRequest):
    symbol = payload.symbol.upper()
    days = min(payload.days, 90)
    model_name = payload.model or "ensemble"
    print(f"[INFO] Predict request: {symbol}, days={days}, model={model_name}")

    try:
        df = yf.download(symbol, period="720d", interval="1d", progress=False)
        if df.empty or len(df) < 100:
            raise HTTPException(status_code=400, detail="Not enough historical data.")
        df = df.reset_index(drop=True)
        df = calculate_features(df)
        df["Target"] = df["Close"].shift(-1)
        df.dropna(subset=["Target"], inplace=True)

        features = ["Open", "High", "Low", "Close", "Volume", "MA5", "MA10", "MA20", "RSI14"]
        X = df[features].values
        y = df["Target"].values

        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gb": GradientBoostingRegressor(random_state=42),
            "xgb": XGBRegressor(objective='reg:squarederror', verbosity=0, n_estimators=50, random_state=42),
            "linear": LinearRegression(),
        }

        used_models = models if model_name == "ensemble" else {model_name: models[model_name]}
        preds_all = []
        metrics_all = []

        window_len = 60
        for name, model in used_models.items():
            print(f"[INFO] Training model: {name}")
            model.fit(X, y)

            sliding_window = df.tail(window_len).copy().reset_index(drop=True)
            preds = []
            for _ in range(days):
                input_features = sliding_window[features].iloc[-1:].values
                if np.isnan(input_features).any():
                    input_features = np.nan_to_num(input_features, nan=np.nanmean(sliding_window[features].values.flatten()))
                pred = model.predict(input_features)[0]
                preds.append(float(pred))

                new_row = {
                    "Open": pred,
                    "High": pred,
                    "Low": pred,
                    "Close": pred,
                    "Volume": float(sliding_window["Volume"].iloc[-1]),
                }
                sliding_window = pd.concat([sliding_window, pd.DataFrame([new_row])], ignore_index=True)
                sliding_window = calculate_features(sliding_window)
                if len(sliding_window) > window_len:
                    sliding_window = sliding_window.iloc[-window_len:].reset_index(drop=True)

            r2 = r2_score(y, model.predict(X))
            mae = mean_absolute_error(y, model.predict(X))
            preds_all.append(preds)
            metrics_all.append((name, float(r2), float(mae)))

        if model_name == "ensemble":
            final_preds = list(np.mean(preds_all, axis=0).round(2))
            model_used = "ensemble"
            model_r2 = round(np.mean([m[1] for m in metrics_all]), 4)
            model_mae = round(np.mean([m[2] for m in metrics_all]), 4)
        else:
            final_preds = list(np.round(preds_all[0], 2))
            model_used = list(used_models.keys())[0]
            model_r2 = round(metrics_all[0][1], 4)
            model_mae = round(metrics_all[0][2], 4)

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
            "metrics": {"model": model_used, "r2": model_r2, "mae": model_mae},
            "advice": {"trend": trend, "risk": risk, "suggestion": suggestion},
        }

    except Exception as e:
        import traceback
        print("[ERROR] Exception occurred:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router, prefix="/api")
