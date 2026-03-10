"""
train.py — Full ML training pipeline with MLflow experiment tracking.

Usage:
    python -m ml.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

What this does:
  1. Load & clean data
  2. Feature engineering
  3. Train/test split
  4. Train multiple models (LR, RF, XGBoost)
  5. Tune best model with Optuna
  6. Log everything to MLflow
  7. Register best model in MLflow Model Registry
  8. Save final pipeline as artifact
"""

import argparse
import os
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import optuna
from xgboost import XGBClassifier
import shap

from ml.preprocess import (
    load_data, engineer_features, build_preprocessor,
    get_feature_names, TARGET, CUSTOMER_ID,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── Config ──────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "churn-prediction"
MODEL_NAME = "churn-xgboost-prod"
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ─── Evaluation helpers ───────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Return a dict of all relevant metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }


def plot_confusion_matrix(model, X_test, y_test, path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"], ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, path, top_n=20):
    """Plot SHAP-based feature importance."""
    # Get the XGB model from inside the pipeline
    xgb_model = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    # We need transformed data for SHAP — use a small sample
    # (full SHAP computed in explain endpoint)
    importances = xgb_model.feature_importances_

    # Match length of importances to feature names
    n = min(top_n, len(importances))
    idx = np.argsort(importances)[-n:]

    try:
        names = feature_names[:len(importances)]
        selected_names = [names[i] for i in idx]
    except Exception:
        selected_names = [f"feature_{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(n), importances[idx], color="#3B82F6")
    ax.set_yticks(range(n))
    ax.set_yticklabels(selected_names, fontsize=9)
    ax.set_title("Top Feature Importances (XGBoost)", fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(model, X_test, y_test, path):
    from sklearn.metrics import roc_curve
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#3B82F6", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─── Optuna Objective ─────────────────────────────────────────────────────────
def make_objective(X_train, y_train, preprocessor):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        }

        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )),
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    return objective


# ─── Main Training Function ───────────────────────────────────────────────────
def train(data_path: str):
    print("\n" + "=" * 60)
    print("  CHURN PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load & Engineer Features ──────────────────────────────
    print("\n[1/7] Loading and engineering features...")
    df = load_data(data_path)
    df = engineer_features(df)

    X = df.drop(columns=[TARGET, CUSTOMER_ID])
    y = df[TARGET]

    print(f"  Dataset: {len(df):,} rows | {y.mean():.1%} churn rate")

    # ── 2. Train/Test Split ──────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 3. Setup MLflow ──────────────────────────────────────────
    print("\n[2/7] Setting up MLflow...")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    preprocessor = build_preprocessor()

    # ── 4. Baseline Models ───────────────────────────────────────
    print("\n[3/7] Training baseline models...")
    baseline_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    for name, clf in baseline_models.items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            pipeline = ImbPipeline([
                ("preprocessor", build_preprocessor()),
                ("smote", SMOTE(random_state=42)),
                ("model", clf),
            ])
            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test)

            mlflow.log_params({"model_type": name})
            mlflow.log_metrics(metrics)
            print(f"  {name:30s} | AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")

    # ── 5. Tune XGBoost with Optuna ──────────────────────────────
    print("\n[4/7] Tuning XGBoost with Optuna (50 trials)...")
    study = optuna.create_study(direction="maximize", study_name="xgb-churn")
    study.optimize(
        make_objective(X_train, y_train, build_preprocessor()),
        n_trials=50,
        show_progress_bar=True,
    )

    best_params = study.best_params
    print(f"\n  Best AUC (CV): {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(best_params, indent=4)}")

    # ── 6. Train Final Model & Log to MLflow ─────────────────────
    print("\n[5/7] Training final XGBoost model...")
    with mlflow.start_run(run_name="xgboost_tuned_final") as run:
        final_pipeline = ImbPipeline([
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(
                **best_params,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        final_pipeline.fit(X_train, y_train)
        metrics = evaluate_model(final_pipeline, X_test, y_test)

        # Log params & metrics
        mlflow.log_params({**best_params, "model_type": "xgboost_tuned", "smote": True})
        mlflow.log_metrics(metrics)

        print(f"\n  Final Model Metrics:")
        for k, v in metrics.items():
            print(f"    {k:15s}: {v:.4f}")

        # ── 7. Generate & Log Plots ──────────────────────────────
        print("\n[6/7] Generating plots...")
        cm_path = str(ARTIFACTS_DIR / "confusion_matrix.png")
        roc_path = str(ARTIFACTS_DIR / "roc_curve.png")
        fi_path = str(ARTIFACTS_DIR / "feature_importance.png")

        plot_confusion_matrix(final_pipeline, X_test, y_test, cm_path)
        plot_roc_curve(final_pipeline, X_test, y_test, roc_path)

        try:
            # Feature names after transformation
            fitted_pre = final_pipeline.named_steps["preprocessor"]
            feature_names = get_feature_names(fitted_pre)
            plot_feature_importance(
                Pipeline([
                    ("preprocessor", fitted_pre),
                    ("model", final_pipeline.named_steps["model"]),
                ]),
                feature_names, fi_path,
            )
        except Exception as e:
            print(f"  (Feature importance plot skipped: {e})")

        mlflow.log_artifact(cm_path, "plots")
        mlflow.log_artifact(roc_path, "plots")
        if os.path.exists(fi_path):
            mlflow.log_artifact(fi_path, "plots")

        # Save model metadata
        meta = {
            "model_type": "XGBoost",
            "dataset": "Telco Customer Churn",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "churn_rate": float(y.mean()),
            **metrics,
        }
        meta_path = str(ARTIFACTS_DIR / "model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact(meta_path)

        # ── 8. Log Model + Register ──────────────────────────────
        print("\n[7/7] Registering model in MLflow Registry...")
        signature = infer_signature(X_train, final_pipeline.predict_proba(X_train)[:, 1])

        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        run_id = run.info.run_id

    # Save locally too (for FastAPI to load without MLflow)
    model_path = str(ARTIFACTS_DIR / "churn_pipeline.pkl")
    joblib.dump(final_pipeline, model_path)
    print(f"\n  Model saved to: {model_path}")
    print(f"  MLflow Run ID: {run_id}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE ✓")
    print("=" * 60 + "\n")

    return final_pipeline, metrics, run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to the CSV dataset",
    )
    args = parser.parse_args()
    train(args.data)
