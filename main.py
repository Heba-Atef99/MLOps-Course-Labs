from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Union
import joblib
import pandas as pd
import logging
import os
from prometheus_fastapi_instrumentator import Instrumentator

# -----------------------
# Logging Configuration
# -----------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s"
)

# -----------------------
# FastAPI Setup
# -----------------------
app = FastAPI(
    title="XGBoost Production API",
    description="API for predicting with the Best XGBoost Model",
    version="v1"
)

# -----------------------
# Load Preprocessor & Model
# -----------------------
model = None
preprocessor = None

try:
    preprocessor = joblib.load("model/preprocessor.joblib")
    logging.info("Preprocessor loaded successfully.")
except Exception as e:
    logging.warning(f"Could not load preprocessor: {e}")

try:
    model = joblib.load("model/Best_model_for_production_v1.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.warning(f"Could not load model: {e}")

# -----------------------
# Request Schema
# -----------------------
class ModelInput(BaseModel):
    features: Dict[str, Union[int, float, str]]

    class Config:
        extra = "forbid"  # Strict schema

# -----------------------
# Endpoints
# -----------------------
@app.get("/")
def home():
    return {"message": "Welcome to the XGBoost model API!"}

@app.get("/health")
def health_check():
    status = "Model is ready." if model else "Model not loaded."
    return {"status": status}

@app.post("/predict")
def predict(input_data: ModelInput):
    if model is None or preprocessor is None:
        logging.error("Model or preprocessor not loaded.")
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded.")

    try:
        df = pd.DataFrame([input_data.features])
        transformed = preprocessor.transform(df)
        prediction = model.predict(transformed)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Add Prometheus metrics
# -----------------------
Instrumentator().instrument(app).expose(app)