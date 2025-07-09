from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

router = APIRouter()

class PredictRequest(BaseModel):
    symbol: str
    days: int = 7

@router.post("/api/predict")
async def predict_stock(request: PredictRequest):
    try:
        df = yf.download(request.symbol, period="6mo", interval="1d")
        df.dropna(inplace=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

    if df.empty or len(df) < 60:
        raise HTTPException(status_code=404, detail="Not enough data")

    # 🔧 特征工程：添加更多指标可提高性能
    df["HL_PCT"] = (df["High"] - df["Low"]) / df["Low"]
    df["CHG_PCT"] = (df["Close"] - df["Open"]) / df["Open"]
    df_feat = df[["Open", "High", "Low", "Close", "Volume", "HL_PCT", "CHG_PCT"]].copy()

    # 🔢 归一化
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat)

    X = scaled[:-1]
    y = scaled[1:, 3]  # Close price for next day

    # ⏩ 多模型训练
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    lr = LinearRegression()

    rf.fit(X, y)
    gb.fit(X, y)
    lr.fit(X, y)

    # 🔁 滚动预测
    last_row = scaled[-1:]
    preds = []

    for _ in range(request.days):
        rf_pred = rf.predict(last_row)[0]
        gb_pred = gb.predict(last_row)[0]
        lr_pred = lr.predict(last_row)[0]

        # 🎯 组合预测
        avg_pred = np.mean([rf_pred, gb_pred, lr_pred])
        preds.append(avg_pred)

        # 创建下一时刻的输入（简单滚动方式）
        next_row = last_row[0].copy()
        next_row[3] = avg_pred  # 更新 Close 值
        last_row = np.vstack([last_row[0], next_row])[1:].reshape(1, -1)

    # ⏪ 反归一化 close 值
    dummy = np.zeros((len(preds), df_feat.shape[1]))
    dummy[:, 3] = preds
    inv_preds = scaler.inverse_transform(dummy)[:, 3]

    return {"predictions": [round(x, 2) for x in inv_preds.tolist()]}
