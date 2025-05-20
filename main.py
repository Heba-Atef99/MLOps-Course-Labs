from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import logging
import os

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
# Load the Trained Model
# -----------------------
model_path = "model/Best_model_for_production_v1.pkl"
try:
    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

# -----------------------
# Request Schema
# -----------------------
class ModelInput(BaseModel):
    features: List[float]  # List of feature values

# -----------------------
# Endpoints
# -----------------------

@app.get("/")
def home():
    logging.info("Home endpoint accessed.")
    return {"message": "Welcome to the XGBoost model API!"}

@app.get("/health")
def health_check():
    status = "Model is ready." if model else "Model not loaded."
    logging.info(f"Health check: {status}")
    return {"status": status}

@app.post("/predict")
def predict(input_data: ModelInput):
    if model is None:
        logging.error("Prediction failed: Model not loaded.")
        return {"error": "Model is not loaded."}
    
    logging.info(f"Received prediction request: {input_data.features}")
    
    try:
        prediction = model.predict([input_data.features])
        result = prediction.tolist()
        logging.info(f"Prediction result: {result}")
        return {"prediction": result}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": str(e)}