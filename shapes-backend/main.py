from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_handler import ModelHandler
from io import BytesIO
import base64
from PIL import Image
import uuid


class PredictionRequest(BaseModel):
    image_b64: str

class TrainRequest(BaseModel):
    image_b64: str
    category: int


app = FastAPI()

model_handler = ModelHandler()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(request: PredictionRequest):
    return model_handler.predict(**dict(request))


@app.post("/save")
async def save(request: PredictionRequest):
    return model_handler.save(**dict(request))


@app.post("/train")
async def save(request: TrainRequest):
    return model_handler.train(**dict(request))