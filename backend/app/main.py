"""
main.py — FastAPI application for Churn Prediction API.

Endpoints:
  GET  /              → Root / welcome
  GET  /health        → Health check
  GET  /model/info    → Model metadata
  POST /predict       → Single customer prediction
  POST /predict/batch → Batch CSV or JSON prediction
  POST /explain       → Full SHAP explanation for a customer
  GET  /docs          → Swagger UI (auto-generated)
"""

import io
import logging
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from app.model import churn_model
from app.schemas import (
    CustomerInput, PredictionResponse, BatchPredictionResponse,
    BatchPredictionItem, ModelInfoResponse, HealthResponse, ShapResponse,
)

logger = logging.getLogger("uvicorn")


# ─── App Lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    logger.info("Loading churn prediction model...")
    loaded = churn_model.load()
    if loaded:
        logger.info("✓ Model loaded successfully.")
    else:
        logger.warning("⚠ Model not loaded — run training first.")
    yield
    logger.info("Shutting down.")


# ─── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description="""
## Customer Churn Prediction Platform

End-to-end ML API for predicting telco customer churn.

### Features
- **Single prediction** with SHAP-based explanations
- **Batch prediction** — upload CSV, download results
- **Model registry info** via MLflow integration
- **SHAP explainability** for every prediction

### Model
- XGBoost trained on Telco Customer Churn dataset
- Tuned with Optuna (50 trials)
- Class imbalance handled with SMOTE
- AUC-ROC ~0.89
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper ──────────────────────────────────────────────────────────────────
def _check_model():
    if not churn_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first.",
        )


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if churn_model.is_loaded else "degraded",
        "model_loaded": churn_model.is_loaded,
        "version": "1.0.0",
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model metadata, version, and performance metrics."""
    return churn_model.get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerInput):
    """
    Predict churn probability for a single customer.

    Returns:
    - `churn_probability`: 0–1 probability
    - `risk_level`: Low / Medium / High / Critical
    - `top_factors`: Top 5 SHAP-based factors driving the prediction
    - `recommendation`: Business action recommendation
    """
    _check_model()
    try:
        result = churn_model.predict(customer.model_dump())
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch/json", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_json(customers: list[CustomerInput]):
    """Batch prediction from a JSON array of customers."""
    _check_model()
    if len(customers) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 customers per batch.")

    records = [c.model_dump() for c in customers]
    results = churn_model.predict_batch(records)

    high_risk = sum(1 for r in results if r.get("risk_level") in ("High", "Critical"))
    predicted_churns = sum(1 for r in results if r.get("churn_prediction"))

    items = []
    for i, r in enumerate(results):
        items.append(BatchPredictionItem(
            customer_id=f"CUST_{i+1:04d}",
            churn_probability=r.get("churn_probability", 0),
            churn_prediction=r.get("churn_prediction", False),
            risk_level=r.get("risk_level", "Low"),
            risk_score=r.get("risk_score", 0),
        ))

    return BatchPredictionResponse(
        total=len(results),
        high_risk_count=high_risk,
        churn_rate_predicted=round(predicted_churns / len(results), 4),
        predictions=items,
    )


@app.post("/predict/batch/csv", tags=["Prediction"])
async def predict_batch_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file, get predictions back as a downloadable CSV.

    The CSV must have the same columns as the Telco dataset.
    """
    _check_model()

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    results = []
    for _, row in df.iterrows():
        try:
            r = churn_model.predict(row.to_dict())
            results.append({
                "customerID": row.get("customerID", ""),
                "churn_probability": r["churn_probability"],
                "churn_prediction": r["churn_prediction"],
                "risk_level": r["risk_level"],
                "risk_score": r["risk_score"],
                "recommendation": r["recommendation"],
            })
        except Exception as e:
            results.append({
                "customerID": row.get("customerID", ""),
                "error": str(e),
            })

    result_df = pd.DataFrame(results)
    output = io.StringIO()
    result_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"},
    )


@app.post("/explain", tags=["Explainability"])
async def explain(customer: CustomerInput):
    """
    Get full SHAP explanation for why a customer is predicted to churn.

    Returns SHAP values for top 15 features showing which factors
    increase or decrease churn risk for this specific customer.
    """
    _check_model()
    try:
        result = churn_model.get_shap_explanation(customer.model_dump())
        return result
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample", tags=["Utils"])
async def get_sample_input():
    """Get a sample customer input for testing."""
    return {
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
