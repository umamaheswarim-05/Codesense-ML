"""
CodeSense - ML FastAPI Service
Serves Random Forest predictions to the Node.js backend.
Run with: uvicorn ml_api:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI(title="CodeSense ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "risk_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

model = None
label_encoder = None

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)


@app.get("/")
def root():
    return {"message": "CodeSense ML API is running", "model_loaded": model is not None}


@app.get("/predict-risk")
def predict_risk(total_submissions: int, successful_runs: int, failed_runs: int, distinct_error_types: int):
    if model is None:
        return {"error": "Model not trained yet. Run train_model.py first."}

    error_rate = failed_runs / total_submissions if total_submissions > 0 else 0

    X = pd.DataFrame([{
        "total_submissions": total_submissions,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "distinct_error_types": distinct_error_types,
        "error_rate": error_rate,
    }])

    prediction_encoded = model.predict(X)[0]
    risk_level = label_encoder.inverse_transform([prediction_encoded])[0]

    proba = model.predict_proba(X)[0]
    risk_classes = label_encoder.classes_
    weight_map = {"Low": 20, "Medium": 55, "High": 90}
    risk_score = int(sum(proba[i] * weight_map[risk_classes[i]] for i in range(len(risk_classes))))

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "error_rate": round(error_rate, 2)
    }