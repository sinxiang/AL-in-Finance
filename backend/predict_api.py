from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import yfinance as yf

router = APIRouter()  # ⬅️ FastAPI 子路由对象

@router.get("/predict")
def predict_stock(symbol: str = "AAPL"):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)

    df = yf.download(symbol, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

    if df.empty:
        return JSONResponse(status_code=404, content={"message": "No data found"})

    features, labels = [], []
    for i in range(1, len(df)):
        features.append([
            float(df["Open"].iloc[i - 1]),
            float(df["High"].iloc[i - 1]),
            float(df["Low"].iloc[i - 1]),
            float(df["Close"].iloc[i - 1]),
            float(df["Volume"].iloc[i - 1]),
        ])
        labels.append(float(df["Close"].iloc[i]))

    model = RandomForestRegressor(n_estimators=100)
    model.fit(features, labels)

    predictions = []
    last_row = df.iloc[[-1]]
    current_features = [
        float(last_row["Open"].iloc[0]),
        float(last_row["High"].iloc[0]),
        float(last_row["Low"].iloc[0]),
        float(last_row["Close"].iloc[0]),
        float(last_row["Volume"].iloc[0]),
    ]

    for _ in range(7):
        prediction = model.predict([current_features])[0]
        predictions.append(round(prediction, 2))
        current_features = [
            float(last_row["Close"].iloc[0]),
            max(float(last_row["Close"].iloc[0]), prediction),
            min(float(last_row["Close"].iloc[0]), prediction),
            prediction,
            float(last_row["Volume"].iloc[0]) * 1.01,
        ]

    return {"symbol": symbol, "predictions": predictions}
