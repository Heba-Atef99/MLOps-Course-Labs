import pytest
from fastapi.testclient import TestClient
from main import app
import numpy as np

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
# Mock model and preprocessor for testing
class DummyModel:
    def predict(self, X):
        return np.array([1])

class DummyPreprocessor:
    def transform(self, df):
        return df.values  # assume a basic transformation

# Inject mock dependencies
@app.on_event("startup")
def setup_test_dependencies():
    import main
    main.model = DummyModel()
    main.preprocessor = DummyPreprocessor()

def test_predict_valid_input():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], list)

def test_predict_missing_field():
    payload = valid_payload.copy()
    del payload["features"]["Geography"]  # Remove required field
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
    assert "detail" in response.json()

def test_predict_wrong_type():
    payload = valid_payload.copy()
    payload["features"]["Age"] = "forty"  # Invalid type
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()