from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict_api import router as predict_router

app = FastAPI()

# ✅ 配置跨域允许（推荐指定你的前端地址）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://al-in-finance.vercel.app", "http://localhost:3000"],  # 或 ["*"] 临时全放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 注册后端接口路由
app.include_router(predict_router, prefix="/api")

# ✅ 健康检查接口
@app.get("/")
def root():
    return {"message": "Backend is running."}
