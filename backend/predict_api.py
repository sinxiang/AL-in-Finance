from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
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
        df = yf.download(req.symbol, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 30:
            return {"message": "Not enough data for prediction."}

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        if req.model == "lstm":
            return lstm_predict(df, req.days)

        return ml_predict(df, req.days, req.model)
    except Exception as e:
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
        preds = (rf.predict(X) + gb.predict(X) + lin.predict(X)) / 3
        return format_output(df, preds, y, days)

    model.fit(X, y)
    preds = model.predict(X)
    return format_output(df, preds, y, days)

def lstm_predict(df, days):
    scaler = MinMaxScaler()
    close_data = df[["Close"]].values
    scaled_data = scaler.fit_transform(close_data)

    def create_dataset(data, look_back=30):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    look_back = 30
    X, y = create_dataset(scaled_data, look_back)

    model = Sequential([
        LSTM(50, input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    last_seq = scaled_data[-look_back:]
    future_preds = []
    for _ in range(days):
        pred = model.predict(last_seq.reshape(1, look_back, 1), verbose=0)
        future_preds.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten().tolist()
    preds = scaler.inverse_transform(model.predict(X, verbose=0)).flatten().tolist()
    return {
        "predictions": future_prices,
        "metrics": {
            "model": "LSTM",
            "r2": float(r2_score(close_data[look_back:], preds)),
            "mae": float(mean_absolute_error(close_data[look_back:], preds))
        },
        "advice": generate_advice(future_prices)
    }

def format_output(df, preds, y, days):
    model = LinearRegression()
    model.fit(df[["Open", "High", "Low", "Close", "Volume"]].values, df["Close"].values)
    last_features = df[["Open", "High", "Low", "Close", "Volume"]].values[-1]
    future = []
    for _ in range(days):
        pred = model.predict(last_features.reshape(1, -1))[0]
        future.append(pred)
        last_features = np.roll(last_features, -1)
        last_features[-1] = pred

    return {
        "predictions": future,
        "metrics": {
            "model": "Ensemble" if isinstance(preds, np.ndarray) else model.__class__.__name__,
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
