"""
model.py — Model loading, prediction, and SHAP explanation logic.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.pipeline import Pipeline

from ml.preprocess import prepare_single_sample, NUMERIC_FEATURES, CATEGORICAL_FEATURES

logger = logging.getLogger("uvicorn")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/churn_pipeline.pkl"))
META_PATH = Path(os.getenv("META_PATH", "artifacts/model_meta.json"))

# ─── Risk Level Mapping ────────────────────────────────────────────────────────
def get_risk_level(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"


def get_confidence(prob: float) -> str:
    dist = abs(prob - 0.5)
    if dist > 0.35:
        return "Very High"
    elif dist > 0.2:
        return "High"
    elif dist > 0.1:
        return "Medium"
    else:
        return "Low"


def get_recommendation(prob: float, risk: str) -> str:
    if risk == "Low":
        return "Customer is stable. Standard engagement recommended."
    elif risk == "Medium":
        return "Monitor closely. Consider a loyalty offer or proactive check-in."
    elif risk == "High":
        return "Intervene now. Offer a contract upgrade or personalized discount."
    else:
        return "URGENT: High churn risk. Escalate to retention team immediately."


# ─── Model Wrapper ─────────────────────────────────────────────────────────────
class ChurnModel:
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.meta: dict = {}
        self.explainer = None
        self._feature_names: list = []
        self.is_loaded = False

    def load(self):
        """Load the trained sklearn pipeline from disk."""
        if not MODEL_PATH.exists():
            logger.warning(f"Model not found at {MODEL_PATH}. Run training first.")
            return False

        try:
            self.pipeline = joblib.load(MODEL_PATH)
            logger.info(f"Model loaded from {MODEL_PATH}")

            if META_PATH.exists():
                with open(META_PATH) as f:
                    self.meta = json.load(f)

            self._build_explainer()
            self.is_loaded = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _build_explainer(self):
        """Build a SHAP TreeExplainer on the XGBoost model."""
        try:
            xgb_model = self.pipeline.named_steps["model"]
            self.explainer = shap.TreeExplainer(xgb_model)
            logger.info("SHAP explainer initialized.")
        except Exception as e:
            logger.warning(f"SHAP explainer not available: {e}")

    def _transform_input(self, df: pd.DataFrame) -> np.ndarray:
        """Run preprocessor only (no SMOTE, no model)."""
        preprocessor = self.pipeline.named_steps["preprocessor"]
        return preprocessor.transform(df)

    def _get_feature_names(self) -> list:
        """Extract feature names from the fitted preprocessor."""
        if self._feature_names:
            return self._feature_names
        try:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            num_names = [
                "tenure", "MonthlyCharges", "TotalCharges",
                "charge_per_tenure", "no_support_mtm",
                "senior_no_security", "total_services",
            ]
            cat_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(
                CATEGORICAL_FEATURES + ["charge_tier"]
            ).tolist()
            self._feature_names = num_names + cat_names
        except Exception:
            self._feature_names = [f"feature_{i}" for i in range(100)]
        return self._feature_names

    def predict(self, customer_data: dict) -> dict:
        """Run full prediction pipeline on a single customer."""
        df = prepare_single_sample(customer_data)
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        prediction = prob >= 0.5
        risk = get_risk_level(prob)
        confidence = get_confidence(prob)
        recommendation = get_recommendation(prob, risk)

        # Get SHAP explanation for top factors
        top_factors = self._explain_factors(df, prob)

        return {
            "churn_probability": round(prob, 4),
            "churn_prediction": prediction,
            "risk_level": risk,
            "risk_score": int(prob * 100),
            "confidence": confidence,
            "top_factors": top_factors,
            "recommendation": recommendation,
        }

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """Batch prediction."""
        results = []
        for rec in records:
            try:
                r = self.predict(rec)
                results.append(r)
            except Exception as e:
                logger.error(f"Error predicting record: {e}")
                results.append({"error": str(e)})
        return results

    def _explain_factors(self, df: pd.DataFrame, prob: float) -> list[dict]:
        """Return top 5 SHAP factors with direction."""
        if self.explainer is None:
            return []

        try:
            X_transformed = self._transform_input(df)
            shap_values = self.explainer.shap_values(X_transformed)[0]
            feature_names = self._get_feature_names()

            n = min(len(shap_values), len(feature_names))
            pairs = sorted(
                zip(feature_names[:n], shap_values[:n]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]

            factors = []
            for name, sv in pairs:
                # Simplify feature name for display
                display_name = name.replace("_", " ").replace("cat__", "").title()
                factors.append({
                    "feature": display_name,
                    "raw_name": name,
                    "shap_value": round(float(sv), 4),
                    "direction": "increases" if sv > 0 else "decreases",
                    "impact": round(abs(float(sv)) * 100, 1),
                })
            return factors
        except Exception as e:
            logger.warning(f"SHAP factor extraction failed: {e}")
            return []

    def get_shap_explanation(self, customer_data: dict) -> dict:
        """Full SHAP explanation for a customer."""
        if self.explainer is None:
            return {"error": "SHAP explainer not available"}

        df = prepare_single_sample(customer_data)
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        X_transformed = self._transform_input(df)
        shap_values = self.explainer.shap_values(X_transformed)[0]
        feature_names = self._get_feature_names()

        n = min(len(shap_values), len(feature_names))
        features = []
        for i in range(n):
            features.append({
                "name": feature_names[i],
                "display_name": feature_names[i].replace("_", " ").title(),
                "value": float(X_transformed[0, i]),
                "shap_value": round(float(shap_values[i]), 4),
                "direction": "risk" if shap_values[i] > 0 else "safe",
            })

        features.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": round(float(self.explainer.expected_value), 4),
            "features": features[:15],
            "churn_probability": round(prob, 4),
        }

    def get_model_info(self) -> dict:
        return {
            "model_name": "Churn XGBoost Production",
            "model_type": "XGBoost + sklearn Pipeline",
            "version": "1.0.0",
            "metrics": {
                k: self.meta.get(k)
                for k in ["accuracy", "f1", "precision", "recall", "roc_auc"]
                if k in self.meta
            },
            "training_date": self.meta.get("training_date", "2024-01-01"),
            "dataset": self.meta.get("dataset", "Telco Customer Churn"),
            "status": "production" if self.is_loaded else "not loaded",
        }


# ─── Singleton ─────────────────────────────────────────────────────────────────
churn_model = ChurnModel()
