from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ✅ 必须导入
from predict_api import router as predict_router

app = FastAPI()

# ✅ 正确 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 你也可以改为你的 Vercel 域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 注册路由
app.include_router(predict_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Backend is running."}
