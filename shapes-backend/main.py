from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_handler import PerfectModel
from io import BytesIO
import base64
from PIL import Image
import uuid


class PredictionRequest(BaseModel):
    image_b64: str


app = FastAPI()

perfect_model = PerfectModel()

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
    encoded_string = base64.b64decode(request.image_b64)
    image_data = Image.open(BytesIO(encoded_string))
    image_data.save(f"./images/temp/{uuid.uuid1()}.png")
    prediction = perfect_model.classify(image_data)
    print(prediction)
    # return {"message": "Hello World"}'
    return prediction


@app.post("/save")
async def predict(request: PredictionRequest):
    encoded_string = base64.b64decode(request.image_b64)
    image_data = Image.open(BytesIO(encoded_string))
    image_data.save(f"./images/temp/{uuid.uuid1()}.png")
    return {}
