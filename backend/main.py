from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict_api import router as predict_router

app = FastAPI()

# 添加跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可替换为前端 URL 例如 "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册预测路由
app.include_router(predict_router)
