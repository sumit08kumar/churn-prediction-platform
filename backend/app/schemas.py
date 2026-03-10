from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────
class ContractType(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class InternetServiceType(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class PaymentMethodType(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class YesNoNoService(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet_service = "No internet service"


class YesNoNoPhone(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone_service = "No phone service"


# ─── Request Schemas ─────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    """Single customer prediction input."""

    model_config = ConfigDict(use_enum_values=True)

    # Demographics
    gender: Literal["Male", "Female"] = Field(..., examples=["Male"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: YesNo = Field(..., examples=["Yes"])
    Dependents: YesNo = Field(..., examples=["No"])

    # Account info
    tenure: int = Field(..., ge=0, le=72, examples=[12], description="Months with company")
    Contract: ContractType = Field(..., examples=["Month-to-month"])
    PaperlessBilling: YesNo = Field(..., examples=["Yes"])
    PaymentMethod: PaymentMethodType = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., gt=0, examples=[65.0])
    TotalCharges: float = Field(..., gt=0, examples=[780.0])

    # Phone
    PhoneService: YesNo = Field(..., examples=["Yes"])
    MultipleLines: YesNoNoPhone = Field(..., examples=["No"])

    # Internet
    InternetService: InternetServiceType = Field(..., examples=["Fiber optic"])
    OnlineSecurity: YesNoNoService = Field(..., examples=["No"])
    OnlineBackup: YesNoNoService = Field(..., examples=["Yes"])
    DeviceProtection: YesNoNoService = Field(..., examples=["No"])
    TechSupport: YesNoNoService = Field(..., examples=["No"])
    StreamingTV: YesNoNoService = Field(..., examples=["Yes"])
    StreamingMovies: YesNoNoService = Field(..., examples=["Yes"])


# ─── Response Schemas ─────────────────────────────────────────────────────────
class RiskLevel(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"
    critical = "Critical"


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="0–1 probability of churning")
    churn_prediction: bool = Field(..., description="True if predicted to churn")
    risk_level: RiskLevel
    risk_score: int = Field(..., description="0–100 risk score")
    confidence: str
    top_factors: list[dict]
    recommendation: str


class BatchPredictionItem(BaseModel):
    customer_id: Optional[str] = None
    churn_probability: float
    churn_prediction: bool
    risk_level: RiskLevel
    risk_score: int


class BatchPredictionResponse(BaseModel):
    total: int
    high_risk_count: int
    churn_rate_predicted: float
    predictions: list[BatchPredictionItem]


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    version: str
    metrics: dict
    training_date: Optional[str]
    dataset: str
    status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ShapResponse(BaseModel):
    base_value: float
    features: list[dict]  # [{name, value, shap_value, direction}]
    churn_probability: float
