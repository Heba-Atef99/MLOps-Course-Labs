import pytest
from fastapi.testclient import TestClient
from main import app

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

def test_predict_valid_input():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], list)

def test_predict_missing_field():
    payload = valid_payload.copy()
    del payload["features"]["Geography"]  # Remove required field
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()

def test_predict_wrong_type():
    payload = valid_payload.copy()
    payload["features"]["Age"] = "forty"  # Invalid type
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()