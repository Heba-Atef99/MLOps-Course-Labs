from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import logging

logging.basicConfig(level=logging.INFO)
app = FastAPI()

model = joblib.load("Models/xgb.pkl") 

class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def home():
    logging.info("Home endpoint accessed")
    return {"message": "Welcome to the Bank Churn Prediction API"}

@app.get("/health")
def health():
    logging.info("Health endpoint accessed")
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: CustomerData):
    logging.info(f"Predict endpoint called with data: {data}")

    # one-hot encoding for Geography with 2 categories only to match model features
    X = [[
        data.CreditScore,
        1 if data.Geography == "France" else 0,
        1 if data.Geography == "Germany" else 0,
        # "Spain" تم حذفها عشان يكون عدد الميزات 11 بالضبط
        1 if data.Gender == "Male" else 0,
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumOfProducts,
        data.HasCrCard,
        data.IsActiveMember,
        data.EstimatedSalary,
    ]]
    prediction = model.predict(X)
    logging.info(f"Prediction result: {prediction[0]}")
    return {"churn_prediction": int(prediction[0])}
