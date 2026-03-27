from fastapi import FastAPI,BackgroundTasks
from pydantic import BaseModel
from typing import Dict
import joblib
from src.pipelines.prediction_pipeline import predict
from deployment.monitoring.utils.logging import log_prediction
from deployment.monitoring_retraining_pipeline import monitoring_retraining_pipeline
from config.config import get_config
from models.model_installer import download_model
import os
from src.utils.model_load_save import load_model

from contextlib import asynccontextmanager



class TransactionInput(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    scaled_amount: float
    scaled_time: float
    actual_label: int = None  

model = None
metadata = None
config = get_config()
MODEL_NAME = config["model_name"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global metadata

    download_model()
    model, metadata = load_model(MODEL_NAME)
    yield

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting fraudulent credit card transactions",
    version="1.0",
    lifespan=lifespan
)



@app.get("/")
def root():
    return {
        "status": "running",
        "service": "fraud-detection-api",
        "model_version": get_config()["model_version"]
    }


@app.post("/predict")
def fraud_prediction(transaction: TransactionInput,background_tasks: BackgroundTasks):

    input_dict = transaction.model_dump()
    global model
    global metadata

    if model is None or metadata is None:
        raise Exception("Model or metadata not loaded")
    

    prediction, probability = predict(input_dict, model, metadata)

    background_tasks.add_task(
        log_prediction,
        input_dict,
        prediction[0],
        probability[0]
    )
    background_tasks.add_task(
        monitoring_retraining_pipeline
    )

    return {
        "fraud_prediction": int(prediction[0]),
        "fraud_probability": float(probability[0])
    }