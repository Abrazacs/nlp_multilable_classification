from fastapi import FastAPI
from pydantic import BaseModel
from app.multilable_pipline import TextClassifierPipeline
import os

app = FastAPI()
model_path = "app/multilabel_model"

mlb_path = os.path.join(model_path, "mlb.pkl")
assert os.path.exists(mlb_path), f"File {mlb_path} not found!"

pipeline = TextClassifierPipeline(
    model_path=model_path,
    tokenizer_path=model_path,
    mlb_path=mlb_path
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    labels: list

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    labels = pipeline.predict(request.text, threshold=0.6)
    return PredictResponse(labels=labels)

@app.get("/")
async def root():
    return {"message": "Multilabel Text Classification API is running"}