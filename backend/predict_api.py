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
from sklearn.preprocessing import MinMaxScaler

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

        scaler = MinMaxScaler()
        scaler.fit(df_train[features])

        df_train_scaled = df_train.copy()
        df_train_scaled[features] = scaler.transform(df_train[features])

        target_scaler = MinMaxScaler()
        y_train_close = df_train[["Target"]].values
        target_scaler.fit(y_train_close)
        y_train_scaled = target_scaler.transform(y_train_close).flatten()

        X_train = create_sliding_window_features(df_train_scaled, features, window_size)
        y_train = y_train_scaled[window_size:]

        eval_len = 100
        if train_len + eval_len + days > total_len:
            eval_len = total_len - train_len - days
        if eval_len <= 0:
            raise HTTPException(status_code=400, detail="Not enough data for evaluation period.")

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
            y_pred_eval = target_scaler.inverse_transform(y_pred_eval_scaled.reshape(-1,1)).flatten()
            y_eval_real = target_scaler.inverse_transform(y_eval.reshape(-1,1)).flatten()
            r2 = r2_score(y_eval_real, y_pred_eval)
            mae = mean_absolute_error(y_eval_real, y_pred_eval)

            preds = []
            for i in range(days):
                # 取训练集最后 window_size 天向后滑动 i 天
                start_idx = train_len - window_size + i
                if start_idx + window_size > total_len:
                    raise HTTPException(status_code=400, detail="Not enough data for independent day prediction.")
                window_data = df.iloc[start_idx : start_idx + window_size].copy()
                window_data[features] = scaler.transform(window_data[features])
                X_input = window_data[features].values.flatten().reshape(1, -1)
                pred_scaled = model.predict(X_input)[0]
                pred_real = target_scaler.inverse_transform([[pred_scaled]])[0,0]
                preds.append(pred_real)

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
