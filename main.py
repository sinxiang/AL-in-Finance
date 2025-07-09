from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.predict_api import router as predict_router  # ✅ 注意这里加了 backend.

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(predict_router, prefix="/api")
