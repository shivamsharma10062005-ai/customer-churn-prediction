"""Live demo: Customer Churn Prediction.

Run:  streamlit run demo.py
"""
import os
import sys

import pandas as pd
import streamlit as st
from joblib import load as joblib_load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from demo_theme import apply_black_theme  # noqa: E402

ART = os.path.join(HERE, "artifacts")
MODEL_PATH = os.path.join(ART, "churn_model.joblib")

CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

YES_NO = ["Yes", "No"]
INTENT_NO = ["Yes", "No", "No internet service"]
PHONE_NO = ["Yes", "No", "No phone service"]


@st.cache_resource
def load_model():
    return joblib_load(MODEL_PATH)


st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
apply_black_theme()
st.title("Customer Churn Prediction")
st.caption("Predicts whether a post-paid customer will churn within the next period.")

c1, c2 = st.columns(2)
with c1:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly charges ($)", 15.0, 120.0, 60.0, 0.1)
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    has_internet = internet != "No"
    online_security = st.selectbox("Online security", INTENT_NO, index=1)
    tech_support = st.selectbox("Tech support", INTENT_NO, index=1)
    online_backup = st.selectbox("Online backup", INTENT_NO, index=1)
    device_protection = st.selectbox("Device protection", INTENT_NO, index=1)
with c2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.toggle("Senior citizen", False)
    partner = st.selectbox("Has partner", YES_NO)
    dependents = st.selectbox("Has dependents", YES_NO)
    paperless = st.selectbox("Paperless billing", YES_NO)
    multiple_lines = st.selectbox("Multiple lines", PHONE_NO, index=1)
    streaming_tv = st.selectbox("Streaming TV", INTENT_NO, index=1)
    streaming_movies = st.selectbox("Streaming Movies", INTENT_NO, index=1)

# demo assumes a sample customer with full tenure matched to total charges
total = monthly * tenure

row = {
    "gender": gender,
    "SeniorCitizen": int(senior),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": "No" if multiple_lines == "No phone service" else "Yes",
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_security if has_internet else "No internet service",
    "OnlineBackup": online_backup if has_internet else "No internet service",
    "DeviceProtection": device_protection if has_internet else "No internet service",
    "TechSupport": tech_support if has_internet else "No internet service",
    "StreamingTV": streaming_tv if has_internet else "No internet service",
    "StreamingMovies": streaming_movies if has_internet else "No internet service",
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": round(total, 2),
}

prob = float(load_model().predict_proba(pd.DataFrame([row])[NUMERIC + CATEGORICAL])[0][1])
risk = "High" if prob >= 0.5 else "Low"
st.metric("Churn probability", f"{prob:.1%}",
          delta=fr"{risk} risk  ·  {contract} contract", delta_color="inverse")

if prob >= 0.5:
    st.warning("High churn risk — recommend a retention offer (e.g. loyalty discount "
               "or contract renewal incentive) before this customer leaves.")
else:
    st.success("Low churn risk — customer is likely to stay.")

with st.expander("What drives this score"):
    st.write(f"""
    - **Contract:** {contract} ({'higher churn driver — month-to-month renews at will' if contract == 'Month-to-month' else 'stable contract, lower churn'})
    - **Tenure:** {tenure} months (longer tenure = more loyalty)
    - **Payment method:** {payment} (electronic check is historically the churniest)
    - **Internet:** {internet} {'— fiber users churn more' if internet == 'Fiber optic' else ''}
    - **Add-ons:** {'OnlineSecurity/TechSupport reduce churn' if online_security == 'Yes' or tech_support == 'Yes' else 'no add-ons — engagement risk'}
    """)

with st.expander("Model details"):
    st.write("""
    - **Algorithms compared:** Logistic Regression, Random Forest, XGBoost
    - **Imbalance handling:** SMOTE applied on the train set only (never the test set)
    - **Metrics:** ROC-AUC (correctly scores ranking quality on imbalanced data),
      precision/recall/F1 at 0.5 threshold + confusion matrix
    - **Deployment:** model saved as a single sklearn pipeline
      (`artifacts/churn_model.joblib`) — preprocessor + classifier in one file
    - **Results:** see `artifacts/roc_pr_curves.png`, `artifacts/confusion_matrix.png`
    """)