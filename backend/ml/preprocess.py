import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

# ─── Feature Groups ───────────────────────────────────────────────────────────
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"
CUSTOMER_ID = "customerID"


def load_data(filepath: str) -> pd.DataFrame:
    """Load and do initial cleaning on the Telco dataset."""
    df = pd.read_csv(filepath)

    # Fix TotalCharges — it has spaces for new customers (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Encode target
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new business-meaningful features."""
    df = df.copy()

    # Avoid divide-by-zero
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # High-risk combo: month-to-month + no tech support
    df["no_support_mtm"] = (
        (df["Contract"] == "Month-to-month") &
        (df["TechSupport"] == "No")
    ).astype(int)

    # Senior with no online security — high churn risk segment
    df["senior_no_security"] = (
        (df["SeniorCitizen"] == 1) &
        (df["OnlineSecurity"] == "No")
    ).astype(int)

    # Total services subscribed (more = more sticky)
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["total_services"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    # Charge tier
    df["charge_tier"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 65, 95, 200],
        labels=["low", "medium", "high", "premium"],
    ).astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Build the sklearn ColumnTransformer pipeline."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Extended numeric features after feature engineering
    all_numeric = NUMERIC_FEATURES + [
        "charge_per_tenure", "no_support_mtm",
        "senior_no_security", "total_services",
    ]
    all_categorical = CATEGORICAL_FEATURES + ["charge_tier"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, all_numeric),
            ("cat", categorical_pipeline, all_categorical),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    """Extract feature names after transformation."""
    num_features = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "charge_per_tenure", "no_support_mtm",
        "senior_no_security", "total_services",
    ]
    cat_features = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(
        CATEGORICAL_FEATURES + ["charge_tier"]
    ).tolist()
    return num_features + cat_features


def prepare_single_sample(data: dict) -> pd.DataFrame:
    """Convert API input dict to a feature-engineered DataFrame."""
    df = pd.DataFrame([data])

    # Apply same feature engineering
    df = engineer_features(df)

    return df
