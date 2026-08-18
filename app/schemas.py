"""Request/response contracts for the prediction API (validated at the boundary)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app import features


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class CustomerProfile(BaseModel):
    """Raw customer features as they exist at the prediction point."""

    customer_id: Optional[str] = Field(default=None, max_length=64)
    gender: str
    SeniorCitizen: bool
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=72)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0, le=200)
    TotalCharges: Optional[float] = Field(default=None, ge=0, le=20000)

    @field_validator(
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", mode="before"
    )
    @classmethod
    def _check_categories(cls, value: str, info) -> str:
        field_name = info.field_name
        assert field_name is not None
        allowed = features.ALLOWED_VALUES[field_name]
        if value not in allowed:
            raise ValueError(f"{field_name} must be one of {allowed}")
        return value

    @field_validator("customer_id", mode="before")
    @classmethod
    def _strip_id(cls, value):
        if value is None:
            return None
        return str(value).strip()


class PredictionRequest(BaseModel):
    profile: CustomerProfile


class RiskFactor(BaseModel):
    feature: str
    impact: str  # high | medium | low
    direction: str  # positive (raises churn) | negative (lowers churn)
    contribution: float


class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float = Field(ge=0, le=1)
    risk_level: str
    expected_retention_value: float
    model_version: str
    feature_version: str
    prediction_timestamp: str = Field(default_factory=_utcnow)
    top_risk_factors: list[RiskFactor]


class ModelInfoResponse(BaseModel):
    model_version: str
    feature_version: str
    trained_at: str
    algorithm: str
    dataset_version: str
    metrics: dict
    risk_thresholds: dict
    status: str


class HealthResponse(BaseModel):
    status: str
    model_version: Optional[str]
    model_loaded: bool
    uptime_seconds: float


class DriftMetric(BaseModel):
    served_predictions: int
    mean_probability: Optional[float]
    psi_recent_vs_baseline: Optional[float]
    threshold: float
    drift_detected: bool


class RecentPrediction(BaseModel):
    timestamp: str
    customer_id_hash: Optional[str]
    model_version: str
    churn_probability: float
    risk_level: str
    latency_ms: float
