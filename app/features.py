"""Shared feature contract.

Single source of truth for the feature schema so training, the API, the batch
job and the Streamlit demo all agree on the exact columns, allowed categories
and derived features. Prevents the schema drift that plagues script-based
pipelines.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

TARGET = "Churn"

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "num_addons"]

CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC + CATEGORICAL

ADDON_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

ALLOWED_VALUES: dict[str, list[str]] = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 72),
    "MonthlyCharges": (0, 200),
    "TotalCharges": (0, 20000),
    "num_addons": (0, 6),
}

FEATURE_VERSION = "features-2.0"


def derive_num_addons(values: dict[str, Any]) -> int:
    """Count of internet add-ons the customer has (0-6)."""
    return sum(1 for col in ADDON_COLUMNS if values.get(col) == "Yes")


def derive_num_addons_series(df: pd.DataFrame) -> pd.Series:
    return (df[ADDON_COLUMNS] == "Yes").sum(axis=1).astype(int)


def make_dataframe(row: dict[str, Any]) -> pd.DataFrame:
    """Build a single-row DataFrame in the exact model column order."""
    base = {col: row.get(col) for col in FEATURE_COLUMNS if col != "num_addons"}
    base["num_addons"] = derive_num_addons(row)
    df = pd.DataFrame([base])
    return df[FEATURE_COLUMNS]


def validate_profile(row: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty = valid)."""
    errors: list[str] = []
    for col, allowed in ALLOWED_VALUES.items():
        value = row.get(col)
        if value not in allowed:
            errors.append(f"{col} must be one of {allowed}, got {value!r}")
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col == "num_addons":
            continue  # derived, always valid
        value = row.get(col)
        if value is None:
            errors.append(f"{col} is required")
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"{col} must be numeric, got {value!r}")
                continue
            if not (lo <= number <= hi):
                errors.append(f"{col} must be in [{lo}, {hi}], got {value!r}")
    return errors
