from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 临时屏蔽路由导入
# from predict_api import router as predict_router

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 后期可替换为你的 Vercel 域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is running."}

# app.include_router(predict_router, prefix="/api")
