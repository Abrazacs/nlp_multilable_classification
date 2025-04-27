from fastapi import FastAPI
from pydantic import BaseModel
from multilable_pipline import TextClassifierPipeline
import os

app = FastAPI()

mlb_path = os.path.join("multilabel_model", "mlb.pkl")
assert os.path.exists(mlb_path), f"File {mlb_path} not found!"

pipeline = TextClassifierPipeline(
    model_path="multilabel_model",
    tokenizer_path="multilabel_model",
    mlb_path=mlb_path
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    labels: list

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    labels = pipeline.predict(request.text)
    return PredictResponse(labels=labels)

@app.get("/")
async def root():
    return {"message": "Multilabel Text Classification API is running"}