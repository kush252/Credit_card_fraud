from fastapi import FastAPI,BackgroundTasks
from pydantic import BaseModel
from typing import Dict

from src.pipelines.prediction_pipeline import predict
from deployment.monitoring.utils.logging import log_prediction
from deployment.monitoring_retraining_pipeline import monitoring_retraining_pipeline
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for predicting fraudulent credit card transactions",
    version="1.0"
)


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


@app.get("/")
def health_check():
    return {"status": "API running"}


@app.post("/predict")
def fraud_prediction(transaction: TransactionInput,background_tasks: BackgroundTasks):

    input_dict = transaction.model_dump()

    prediction, probability = predict(input_dict)

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