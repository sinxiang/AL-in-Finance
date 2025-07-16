from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import yfinance as yf
import numpy as np
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

@router.post("/predict")
async def predict_stock(payload: PredictRequest):
    symbol = payload.symbol.upper()
    days = min(payload.days, 90)
    model_name = payload.model or "ensemble"

    print(f"[INFO] Predict request: {symbol}, days={days}, model={model_name}")

    try:
        df = yf.download(symbol, period="1080d", interval="1d", progress=False)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found.")

        df.dropna(inplace=True)
        df["Target"] = df["Close"].shift(-1)
        df.dropna(inplace=True)

        features = ["Open", "High", "Low", "Close", "Volume"]
        X = df[features].values
        y = df["Target"].values

        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gb": GradientBoostingRegressor(random_state=42),
            "xgb": XGBRegressor(objective='reg:squarederror', verbosity=0, n_estimators=50, random_state=42),
            "linear": LinearRegression(),
        }

        preds_all = []
        metrics_all = []

        used_models = models if model_name == "ensemble" else {model_name: models.get(model_name)}
        if None in used_models.values():
            raise HTTPException(status_code=400, detail=f"Unsupported model '{model_name}'.")

        # Define evaluation window (1 year = 250 trading days)
        eval_window = 250
        if len(X) < eval_window:
            raise HTTPException(status_code=400, detail="Not enough data for evaluation.")

        for name, model in used_models.items():
            print(f"[INFO] Training model: {name}")
            model.fit(X, y)

            # Calculate metrics on last eval_window samples
            y_pred_eval = model.predict(X[-eval_window:])
            r2 = r2_score(y[-eval_window:], y_pred_eval)
            mae = mean_absolute_error(y[-eval_window:], y_pred_eval)

            preds = []
            recent_X = X[-1:].copy()

            for _ in range(days):
                pred = model.predict(recent_X)[0]
                preds.append(float(pred))
                recent_X = np.array([
                    [pred, pred, pred, pred, recent_X[0][4]]
                ])

            preds_all.append(preds)
            metrics_all.append((name, r2, mae))

        final_preds = (
            np.mean(preds_all, axis=0).tolist()
            if model_name == "ensemble"
            else preds_all[0]
        )

        final_preds = [float(p) for p in final_preds]
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
