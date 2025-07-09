from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict_api import router as predict_router

app = FastAPI()

# 🔓 允许跨域访问（Vercel 前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可改为你的 vercel 域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 正确注册路由
app.include_router(predict_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Backend is running."}
