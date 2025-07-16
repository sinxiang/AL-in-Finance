# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from predict_api import router  # 这里导入你刚发的那个 router

app = FastAPI()

# 跨域配置，允许所有来源（生产环境建议限制具体域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# 挂载你自己的预测路由
app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
