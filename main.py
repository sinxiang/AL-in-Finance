import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from predict_api import router as predict_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(predict_router, prefix="/api")
