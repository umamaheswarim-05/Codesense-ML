# CodeSense ML Service

Python-based Machine Learning service for the CodeSense platform.
Provides student risk prediction and error pattern analysis via a FastAPI REST API.

## Overview
The ML service analyses student code execution history and predicts which students are at risk of falling behind. It uses a Random Forest Classifier trained on submission data including error frequency, attempt count, and success rate. K-Means Clustering groups students by similar error patterns to help educators identify common weak areas across the class. Predictions are served to the Node.js backend through a FastAPI REST endpoint.

## Tech Stack
- Python 3.14 — ML service language
- scikit-learn — Random Forest Classifier, K-Means Clustering
- pandas + numpy — Data processing and feature engineering
- joblib — Model serialisation and loading
- FastAPI + uvicorn — REST API server

## Files
| File | Description |
|------|-------------|
| train_model.py | Trains and saves the Random Forest model |
| ml_api.py | FastAPI server that serves predictions |

## API Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| GET | / | Health check |
| POST | /predict | Predict risk score for a student |
| GET | /docs | Auto-generated Swagger UI documentation |

## How to Run
```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib
python train_model.py
python -m uvicorn ml_api:app --port 8000
```

## Prediction Input
```json
{
  "user_id": 1,
  "error_count": 5,
  "attempt_count": 10,
  "success_rate": 0.4
}
```

## Prediction Output
```json
{
  "risk_level": "High",
  "risk_score": 82,
  "error_rate": 0.5
}
```

## Part of CodeSense Platform
- Frontend: https://github.com/umamaheswarim-05/Codesense-Frontend
- Backend: https://github.com/umamaheswarim-05/Codesense-Backend
