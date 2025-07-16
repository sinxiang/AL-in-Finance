from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend import predict_api  # 假设你的 predict_api 在 backend 文件夹

app = FastAPI()

app.include_router(predict_api.router, prefix="/api")

# 正确设置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 上线时可改成你 Vercel 域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 本地 & Render 通用启动入口
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
