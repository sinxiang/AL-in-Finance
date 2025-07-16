from fastapi import APIRouter
from pydantic import BaseModel
import yfinance as yf

router = APIRouter()

class PredictRequest(BaseModel):
    symbol: str

@router.post("/predict")
def predict_stock(req: PredictRequest):
    try:
        df = yf.download(req.symbol, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 30:
            return {"message": "Not enough data."}
        return {
            "symbol": req.symbol,
            "history": df["Close"].tail(30).tolist()
        }
    except Exception as e:
        return {"message": f"Error fetching data: {str(e)}"}
