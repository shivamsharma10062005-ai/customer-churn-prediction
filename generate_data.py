"""Generate a realistic synthetic telecom customer-churn dataset.

Mirrors the patterns in real post-paid telecom churn (e.g. the classic
Telco Customer Churn dataset): a logit-shaped churn probability driven by
contract type, tenure, payment method, internet service, and add-on services.

Output: data/telco_churn.csv (one row per customer)
"""
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_CUSTOMERS = 15000
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "telco_churn.csv")

INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def churn_logit(row: dict) -> float:
    """Linear log-odds of churning; tuned so overall churn ~26%."""
    w = 0.0
    w += -1.15                                       # intercept
    if row["Contract"] == "Month-to-month":
        w += 1.75
    elif row["Contract"] == "One year":
        w += 0.25
    w += -0.055 * row["tenure"]                       # loyalty
    if row["PaymentMethod"] == "Electronic check":
        w += 0.85
    if row["InternetService"] == "Fiber optic":
        w += 0.45
    w += -0.55 if row["OnlineSecurity"] == "Yes" else 0
    w += -0.50 if row["TechSupport"] == "Yes" else 0
    w += -0.35 if row["OnlineBackup"] == "Yes" else 0
    w += -0.35 if row["DeviceProtection"] == "Yes" else 0
    w += 0.35 if row["SeniorCitizen"] == 1 else 0
    w += -0.30 if row["Partner"] == "Yes" else 0
    w += -0.35 if row["Dependents"] == "Yes" else 0
    w += 0.30 if row["PaperlessBilling"] == "Yes" else 0
    w += 0.10 if row["StreamingTV"] == "Yes" else 0
    w += 0.10 if row["MultipleLines"] == "Yes" else 0
    return w


def monthly_charges(row: dict) -> float:
    c = RNG.normal(25, 5)                             # base line
    if row["InternetService"] == "DSL":
        c += 20
    elif row["InternetService"] == "Fiber optic":
        c += 50
    if row["MultipleLines"] == "Yes":
        c += 10
    for col in INTERNET_ADDONS:
        if row[col] == "Yes":
            c += 10
    return float(np.clip(c, 15, 120))


def build_dataset() -> pd.DataFrame:
    contracts = ["Month-to-month", "One year", "Two year"]
    pay_methods = ["Electronic check", "Mailed check",
                   "Bank transfer (automatic)", "Credit card (automatic)"]
    internet = ["DSL", "Fiber optic", "No"]
    yes_no = ["Yes", "No"]

    rows = []
    for _ in range(N_CUSTOMERS):
        contract = RNG.choice(contracts, p=[0.55, 0.24, 0.21])
        # tenure distribution conditioned on contract
        if contract == "Month-to-month":
            tenure = int(np.clip(RNG.gamma(2.2, 6.0), 0, 71))
        else:
            tenure = int(np.clip(RNG.gamma(4.5, 10.0), 1, 72))

        internet_service = RNG.choice(internet, p=[0.44, 0.44, 0.12])
        has_internet = internet_service != "No"

        phone_service = RNG.choice(yes_no, p=[0.90, 0.10])
        row = {
            "customerID": f"{RNG.integers(1000, 9999)}-{RNG.integers(10000, 99999)}",
            "gender": RNG.choice(["Male", "Female"]),
            "SeniorCitizen": int(RNG.random() < 0.16),
            "Partner": RNG.choice(yes_no, p=[0.48, 0.52]),
            "Dependents": RNG.choice(yes_no, p=[0.70, 0.30]),
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": (RNG.choice(["Yes", "No", "No phone service"],
                                         p=[0.45, 0.42, 0.13]) if phone_service == "Yes"
                              else "No phone service"),
            "InternetService": internet_service,
            "OnlineSecurity": RNG.choice(["Yes", "No", "No internet service"],
                                         p=[0.42, 0.45, 0.13]) if has_internet else "No internet service",
            "OnlineBackup": RNG.choice(["Yes", "No", "No internet service"],
                                       p=[0.40, 0.47, 0.13]) if has_internet else "No internet service",
            "DeviceProtection": RNG.choice(["Yes", "No", "No internet service"],
                                           p=[0.38, 0.49, 0.13]) if has_internet else "No internet service",
            "TechSupport": RNG.choice(["Yes", "No", "No internet service"],
                                      p=[0.30, 0.57, 0.13]) if has_internet else "No internet service",
            "StreamingTV": RNG.choice(["Yes", "No", "No internet service"],
                                      p=[0.38, 0.49, 0.13]) if has_internet else "No internet service",
            "StreamingMovies": RNG.choice(["Yes", "No", "No internet service"],
                                          p=[0.38, 0.49, 0.13]) if has_internet else "No internet service",
            "Contract": contract,
            "PaperlessBilling": RNG.choice(yes_no, p=[0.60, 0.40]),
            "PaymentMethod": RNG.choice(pay_methods, p=[0.33, 0.22, 0.25, 0.20]),
        }

        monthly = monthly_charges(row)
        row["MonthlyCharges"] = round(monthly, 2)
        row["TotalCharges"] = round(monthly * tenure * RNG.uniform(0.95, 1.05), 2)

        p_churn = logistic(churn_logit(row))
        row["Churn"] = "Yes" if RNG.random() < p_churn else "No"
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = build_dataset()
    df.to_csv(OUT_PATH, index=False)
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    print(f"Wrote {len(df):,} rows -> {OUT_PATH}")
    print(f"Churn rate: {churn_rate:.1f}%")
