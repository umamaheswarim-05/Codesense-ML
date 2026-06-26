# CodeSense ML Service

Python-based Machine Learning service for the CodeSense platform.
Random Forest risk prediction + FastAPI REST endpoint.

## What This Does
- Trains a Random Forest Classifier on student execution data
- Predicts at-risk students based on error frequency, attempt count, success rate
- Applies K-Means clustering to group students by error patterns
- Serves predictions via FastAPI REST API to the Node.js backend

## Tech Stack
- Python 3.14
- scikit-learn (Random Forest, K-Means)
- FastAPI + uvicorn
- pandas + numpy
- joblib (model save/load)

## Files
- `train_model.py` — trains and saves the Random Forest model
- `ml_api.py` — FastAPI server that serves predictions

## How to Run
```bash
pip install fastapi uvicorn scikit-learn pandas numpy joblib
python train_model.py
uvicorn ml_api:app --reload --port 8000
```

## API Endpoint
