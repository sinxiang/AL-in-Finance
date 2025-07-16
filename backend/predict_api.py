from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import yfinance as yf
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

router = APIRouter()

class PredictRequest(BaseModel):
    symbol: str
    days: int
    model: Optional[str] = "ensemble"

@router.post("/predict")
def predict_stock(req: PredictRequest):
    try:
        print(f"Received prediction request: symbol={req.symbol}, days={req.days}, model={req.model}")
        df = yf.download(req.symbol, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 30:
            return {"message": "Not enough data for prediction."}

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        if req.model == "lstm":
            return lstm_predict(df, req.days)

        return ml_predict(df, req.days, req.model)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {"message": f"Prediction failed: {str(e)}"}

def ml_predict(df, days, model_name):
    X = df[["Open", "High", "Low", "Close", "Volume"]].values[:-1]
    y = df["Close"].values[1:]

    if model_name == "random_forest":
        model = RandomForestRegressor()
    elif model_name == "gb":
        model = GradientBoostingRegressor()
    elif model_name == "xgb":
        model = xgb.XGBRegressor()
    elif model_name == "linear":
        model = LinearRegression()
    else:
        rf = RandomForestRegressor()
        gb = GradientBoostingRegressor()
        lin = LinearRegression()

        rf.fit(X, y)
        gb.fit(X, y)
        lin.fit(X, y)

        preds_rf = rf.predict(X)
        preds_gb = gb.predict(X)
        preds_lin = lin.predict(X)
        preds = (preds_rf + preds_gb + preds_lin) / 3

        return format_output(df, preds, y, days, "Ensemble")

    model.fit(X, y)
    preds = model.predict(X)
    return format_output(df, preds, y, days, model_name)

def lstm_predict(df, days):
    scaler = MinMaxScaler()
    close_data = df[["Close"]].values
    scaled_data = scaler.fit_transform(close_data)

    def create_dataset(data, look_back=30):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back])
        X_np = np.array(X)
        y_np = np.array(y)
        print(f"create_dataset - X shape: {X_np.shape}, y shape: {y_np.shape}")
        return X_np, y_np

    look_back = 30
    X, y = create_dataset(scaled_data, look_back)

    model = Sequential([
        LSTM(50, input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    preds_scaled = model.predict(X, verbose=0).flatten()
    print(f"Predictions shape (preds_scaled): {preds_scaled.shape}")
    print(f"True values shape (y): {y.shape}")

    if preds_scaled.shape[0] != y.shape[0]:
        error_msg = f"Length mismatch: preds_scaled={preds_scaled.shape[0]}, y={y.shape[0]}"
        print(error_msg)
        return {"message": error_msg}

    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    true_vals = close_data[look_back:].flatten()

    r2 = float(r2_score(true_vals, preds))
    mae = float(mean_absolute_error(true_vals, preds))

    last_seq = scaled_data[-look_back:]
    future_preds = []
    for _ in range(days):
        pred = model.predict(last_seq.reshape(1, look_back, 1), verbose=0)
        future_preds.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

    return {
        "predictions": future_prices.tolist(),
        "metrics": {
            "model": "LSTM",
            "r2": r2,
            "mae": mae
        },
        "advice": generate_advice(future_prices.tolist())
    }

def format_output(df, preds, y, days, model_name):
    last_features = df[["Open", "High", "Low", "Close", "Volume"]].values[-1]
    future = []
    model_for_future = LinearRegression()
    model_for_future.fit(df[["Open", "High", "Low", "Close", "Volume"]].values[:-1], df["Close"].values[1:])
    for _ in range(days):
        pred = model_for_future.predict(last_features.reshape(1, -1))[0]
        future.append(float(pred))
        last_features = np.roll(last_features, -1)
        last_features[-1] = pred

    return {
        "predictions": future,
        "metrics": {
            "model": model_name,
            "r2": float(r2_score(y, preds)),
            "mae": float(mean_absolute_error(y, preds))
        },
        "advice": generate_advice(future)
    }

def generate_advice(preds):
    trend = "Uptrend" if preds[-1] > preds[0] else "Downtrend"
    risk = "Low" if abs(preds[-1] - preds[0]) / preds[0] < 0.05 else "High"
    suggestion = "Consider buying." if trend == "Uptrend" and risk == "Low" else "Be cautious."
    return {
        "trend": trend,
        "risk": risk,
        "suggestion": suggestion
    }
