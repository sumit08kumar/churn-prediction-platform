"""
tests/test_api.py — FastAPI endpoint tests.

Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# We'll mock the model loading since we won't have the .pkl in CI
with patch("app.model.churn_model.load", return_value=True):
    from app.main import app

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20,
}


def test_root():
    """Root endpoint returns 200."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["version"] == "1.0.0"


def test_health():
    """Health endpoint returns valid response."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_sample_endpoint():
    """Sample endpoint returns a valid customer dict."""
    response = client.get("/sample")
    assert response.status_code == 200
    data = response.json()
    assert "gender" in data
    assert "tenure" in data
    assert "MonthlyCharges" in data


def test_predict_schema_validation():
    """Predict with invalid data returns 422."""
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422


def test_predict_with_mock_model():
    """Predict returns expected shape when model is mocked."""
    mock_result = {
        "churn_probability": 0.72,
        "churn_prediction": True,
        "risk_level": "High",
        "risk_score": 72,
        "confidence": "High",
        "top_factors": [
            {"feature": "Contract", "shap_value": 0.35, "direction": "increases", "impact": 35.0, "raw_name": "contract"}
        ],
        "recommendation": "Intervene now.",
    }

    with patch("app.main.churn_model.is_loaded", True), \
         patch("app.main.churn_model.predict", return_value=mock_result):
        response = client.post("/predict", json=SAMPLE_CUSTOMER)

    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "top_factors" in data
    assert 0 <= data["churn_probability"] <= 1


def test_model_info_endpoint():
    """Model info returns valid structure."""
    mock_info = {
        "model_name": "Churn XGBoost Production",
        "model_type": "XGBoost + sklearn Pipeline",
        "version": "1.0.0",
        "metrics": {"roc_auc": 0.891, "f1": 0.637},
        "training_date": "2024-01-01",
        "dataset": "Telco Customer Churn",
        "status": "production",
    }
    with patch("app.main.churn_model.get_model_info", return_value=mock_info):
        response = client.get("/model/info")

    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "metrics" in data


def test_predict_batch_json_empty_list():
    """Batch predict with empty list returns 422 or 503 if model not loaded."""
    response = client.post("/predict/batch/json", json=[])
    # FastAPI should handle an empty list — might be 200, 422, or 503 (model not loaded)
    assert response.status_code in (200, 422, 503)


def test_predict_batch_too_large():
    """Batch predict with >1000 items returns 400."""
    with patch("app.main.churn_model.is_loaded", True):
        customers = [SAMPLE_CUSTOMER] * 1001
        response = client.post("/predict/batch/json", json=customers)
    assert response.status_code == 400
