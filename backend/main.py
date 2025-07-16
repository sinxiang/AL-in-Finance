from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import predict_api  # ✅ 不需要 from backend

app = FastAPI()

app.include_router(predict_api.router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 可根据需要限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

