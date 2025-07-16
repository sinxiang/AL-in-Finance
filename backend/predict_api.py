@router.post("/predict")
def predict_stock(req: PredictRequest):
    try:
        df = yf.download(req.symbol, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 30:
            return {"message": "Not enough data for prediction."}
        
        close_prices = df["Close"].dropna().tolist()  # 取收盘价列表
        return {"history": close_prices}  # 只返回这个，不做预测
    except Exception as e:
        return {"message": f"Error: {str(e)}"}
