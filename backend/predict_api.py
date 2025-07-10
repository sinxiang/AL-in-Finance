from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False

router = APIRouter()

class PredictRequest(BaseModel):
    symbol: str
    days: int = 30
    model: str = "ensemble"
    start: str = "2015-01-01"

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

@router.post("/predict")
async def predict_stock(request: PredictRequest):
    if request.days > 90:
        raise HTTPException(status_code=400, detail="Max prediction days is 90")

    try:
        df = yf.download(request.symbol, start=request.start, interval="1d")
        df.dropna(inplace=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

    if df.empty or len(df) < 200:
        raise HTTPException(status_code=404, detail="Not enough data")

    # 添加特征
    df["HL_PCT"] = (df["High"] - df["Low"]) / df["Low"]
    df["CHG_PCT"] = (df["Close"] - df["Open"]) / df["Open"]
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["Weekday"] = df.index.weekday / 6
    df.dropna(inplace=True)

    features = [
        "Open", "High", "Low", "Close", "Volume",
        "HL_PCT", "CHG_PCT", "SMA_10", "EMA_10", "RSI", "Weekday"
    ]
    df_feat = df[features]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat)

    # ✅ 滑动窗口构造
    window = 10
    X, y = [], []
    for i in range(len(scaled) - window):
        X.append(scaled[i:i+window])
        y.append(scaled[i + window][3])  # 第 window+1 天的 Close

    X = np.array(X)
    y = np.array(y)

    # 模型选择
    model_name = request.model.lower()
    models = {}

    if model_name == "ensemble":
        models["rf"] = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
        models["gb"] = GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
        models["lr"] = LinearRegression()
        if has_xgb:
            models["xgb"] = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
    else:
        if model_name == "random_forest":
            models["rf"] = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)
        elif model_name == "gb":
            models["gb"] = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
        elif model_name == "linear":
            models["lr"] = LinearRegression()
        elif model_name == "xgb":
            if not has_xgb:
                raise HTTPException(status_code=400, detail="XGBoost not installed.")
            models["xgb"] = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        else:
            raise HTTPException(status_code=400, detail="Invalid model")

    preds_train = {}
    for name, model in models.items():
        model.fit(X, y)
        preds_train[name] = model.predict(X)

    if model_name == "ensemble":
        ensemble_train_pred = np.mean(np.array(list(preds_train.values())), axis=0)
        r2 = r2_score(y, ensemble_train_pred)
        mae = mean_absolute_error(y, ensemble_train_pred)
    else:
        key = list(models.keys())[0]
        r2 = r2_score(y, preds_train[key])
        mae = mean_absolute_error(y, preds_train[key])

    # ✅ 滚动预测（使用滑动窗口）
    last_window = scaled[-window:]
    preds = []

    for _ in range(request.days):
        step_preds = []
        for model in models.values():
            step_preds.append(model.predict(last_window.reshape(1, window, -1))[0])
        avg_pred = np.mean(step_preds)
        preds.append(avg_pred)

        next_day = last_window[-1].copy()
        next_day[3] = avg_pred  # 更新 Close
        last_window = np.vstack([last_window[1:], next_day])

    dummy = np.zeros((len(preds), df_feat.shape[1]))
    dummy[:, 3] = preds
    inv_preds = scaler.inverse_transform(dummy)[:, 3]

    return {
        "predictions": [round(x, 2) for x in inv_preds.tolist()],
        "metrics": {
            "model": model_name,
            "r2": round(r2, 4),
            "mae": round(mae, 4)
        }
    }
