import pytest
from fastapi.testclient import TestClient
from main import app
import numpy as np
import pandas as pd

client = TestClient(app)

# Sample valid input
valid_payload = {
    "features": {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
}

# Dummy model and preprocessor
class DummyModel:
    def predict(self, X):
        return np.array([1])

class DummyPreprocessor:
    def __init__(self):
        self.expected_columns = [
            "CreditScore", "Geography", "Gender", "Age", "Tenure",
            "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
        ]
    
    def transform(self, df: pd.DataFrame):
        missing = set(self.expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"columns are missing: {missing}")
        return df[self.expected_columns].values

# Inject mock dependencies on startup
@app.on_event("startup")
def setup_test_dependencies():
    import main
    main.model = DummyModel()
    main.preprocessor = DummyPreprocessor()

# ------------------------
# TEST CASES
# ------------------------

def test_predict_valid_input():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], list)

def test_predict_missing_field():
    payload = valid_payload.copy()
    del payload["features"]["Geography"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
    assert "detail" in response.json()

def test_predict_wrong_type():
    payload = valid_payload.copy()
    payload["features"]["Age"] = "forty"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
    assert "detail" in response.json()