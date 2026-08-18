"""Unit tests for the shared feature contract."""
import pandas as pd

from app import features as feat


def test_make_dataframe_column_order():
    row = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0, "TotalCharges": 960.0,
    }
    df = feat.make_dataframe(row)
    assert list(df.columns) == feat.FEATURE_COLUMNS
    assert df.shape == (1, len(feat.FEATURE_COLUMNS))


def test_derive_num_addons_counts_yes():
    row = {"OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "Yes",
           "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No"}
    assert feat.derive_num_addons(row) == 3


def test_validate_profile_accepts_valid():
    row = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0, "TotalCharges": 960.0,
    }
    assert feat.validate_profile(row) == []


def test_validate_profile_rejects_bad_category_and_range():
    row = {
        "gender": "Alien", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 500, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0, "TotalCharges": 960.0,
    }
    errors = feat.validate_profile(row)
    assert any("gender" in e for e in errors)
    assert any("tenure" in e for e in errors)


def test_derive_num_addons_series():
    df = pd.DataFrame({
        "OnlineSecurity": ["Yes", "No"], "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "No"], "TechSupport": ["No", "No"],
        "StreamingTV": ["Yes", "No"], "StreamingMovies": ["No", "No"],
    })
    assert feat.derive_num_addons_series(df).tolist() == [3, 0]
