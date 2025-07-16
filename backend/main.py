from fastapi import FastAPI
from predict_api import router

app = FastAPI()

app.include_router(router)
