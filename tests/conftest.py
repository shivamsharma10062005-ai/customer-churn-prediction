"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app import features as feat
from app.model_runtime import ModelBundle

VALID_PROFILE = {
    "customer_id": "CUST-1",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0,
    "TotalCharges": 960.0,
}


def _tiny_logistic_bundle() -> ModelBundle:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(240):
        row = dict(VALID_PROFILE)
        row["tenure"] = int(rng.integers(0, 72))
        row["Contract"] = rng.choice(["Month-to-month", "One year", "Two year"])
        row["PaymentMethod"] = rng.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        rows.append(row)
    df = pd.DataFrame([feat.make_dataframe(r).iloc[0].to_dict() for r in rows])

    logit = (
        (df["Contract"] == "Month-to-month").astype(int) * 1.5
        - df["tenure"] * 0.03
        + (df["PaymentMethod"] == "Electronic check").astype(int) * 0.8
        + rng.normal(0, 0.2, len(df))
    )
    y = (1.0 / (1.0 + np.exp(-logit)) > 0.3).astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feat.NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feat.CATEGORICAL),
        ]
    )
    clf = LogisticRegression(max_iter=1000)
    pipeline = Pipeline([("pre", pre), ("clf", clf)])
    pipeline.fit(df[feat.FEATURE_COLUMNS], y)

    card = {
        "model_version": "v-test",
        "feature_version": feat.FEATURE_VERSION,
        "algorithm": "logistic",
        "status": "test",
        "reference_profile": {
            "Contract": "One year", "tenure": 24.0, "MonthlyCharges": 60.0,
            "TotalCharges": 1440.0, "SeniorCitizen": 0.0, "PaymentMethod": "Mailed check",
            "gender": "Male", "Partner": "Yes", "Dependents": "No",
            "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "PaperlessBilling": "Yes", "num_addons": 0,
        },
        "risk_thresholds": {"low": 0.30, "medium": 0.60, "high": 0.80},
        "baseline_prediction_distribution": [0.1] * 100,
    }
    return ModelBundle(pipeline=pipeline, model_card=card)


@pytest.fixture
def tiny_bundle() -> ModelBundle:
    return _tiny_logistic_bundle()


@pytest.fixture
def valid_profile() -> dict:
    return dict(VALID_PROFILE)
