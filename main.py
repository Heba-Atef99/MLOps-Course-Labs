from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
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

preprocessor = joblib.load("model/preprocessor.joblib")

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

class Features(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class InputData(BaseModel):
    features: Features

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
def predict(input_data: InputData):
    if model is None:
        logging.error("Prediction failed: Model not loaded.")
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # Convert the Features object to dict then DataFrame
        logging.info(f"Data received for prediction")
        df = pd.DataFrame([input_data.features.dict()])
        logging.info(f"Dataframe created: {df}")

        # Apply preprocessing
        logging.info(f"Applying preprocessing")
        transformed = preprocessor.transform(df)
        logging.info(f"Data transformed: {transformed}")

        # Predict
        logging.info(f"Making prediction")
        prediction = model.predict(transformed)
        result = prediction.tolist()

        logging.info(f"Prediction result: {result}")
        return {"prediction": result}

    except ValueError as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# -----------------------
# Prometheus instrumentation
# -----------------------

from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)