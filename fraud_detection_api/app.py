from fastapi import FastAPI, Request
from pydantic import BaseModel
import numpy as np
import joblib
import lightgbm as lgb
import json

app = FastAPI()

# Load model and scaler
# Load model, scaler, and feature names
model = lgb.Booster(model_file="C:/Users/DELL/fraud_detection_api/light_gbm.txt")
scaler = joblib.load("C:/Users/DELL/fraud_detection_api/scaler.pkl")
feature_names = joblib.load("C:/Users/DELL/fraud_detection_api/feature_names.pkl")  


# Define expected request schema
class Transaction(BaseModel):
    Time: float
    Amount: float
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

@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    data = json.loads(body)

    # ✅ Corrected feature order (matches training)
    feature_order = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    input_values = np.array([[data[feature] for feature in feature_names]])

    # Scale input
    input_scaled = scaler.transform(input_values)

    # Predict
    prob = model.predict(input_scaled)[0]
    prediction = int(prob >= 0.5)  # your threshold

    return {
        "prediction": "Fraud" if prediction == 1 else "Not Fraud",
        "probability": float(prob)
    }
