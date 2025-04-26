from fastapi import FastAPI
from pydantic import BaseModel
from pipeline import TextClassifierPipeline

app = FastAPI()

# Загружаем пайплайн при старте
pipeline = TextClassifierPipeline(
    model_path="multilabel_model",
    tokenizer_path="multilabel_model",
    mlb_path="mlb.pkl"
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