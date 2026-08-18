"""Live demo: Customer Churn Prediction.

Run:  streamlit run demo.py

Uses the same production services as the API (feature contract, versioned
model runtime, risk engine, explainer) so the demo and the backend can never
drift apart. Gracefully shows an error if no model artifact is present.
"""
import os
import sys

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from churn_theme import apply_theme  # noqa: E402

from app.explanations import explain  # noqa: E402
from app.features import make_dataframe  # noqa: E402
from app.model_runtime import ModelLoadError, runtime  # noqa: E402
from app.risk import classify, expected_retention_value  # noqa: E402

YES_NO = ["Yes", "No"]
INTENT_NO = ["Yes", "No", "No internet service"]
PHONE_NO = ["Yes", "No", "No phone service"]

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🛰️", layout="centered")
apply_theme()
st.markdown('<div class="nx-eyebrow">● Telecom Retention · ML Interactive</div>',
            unsafe_allow_html=True)
st.title("Customer Churn Prediction")
st.caption("Predicts whether a post-paid customer will churn within the next period.")


@st.cache_resource
def get_bundle_info():
    return runtime.info()


try:
    model_info = get_bundle_info()
except ModelLoadError as exc:
    st.error(f"Model artifact not found: {exc}")
    st.stop()


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
    "TotalCharges": round(monthly * tenure, 2),
}

prob = runtime.predict(make_dataframe(row))
bundle = runtime.bundle
risk = classify(prob, bundle)
erv = expected_retention_value(prob)
factors = explain(bundle, row, top_n=5)

st.metric("Churn probability", f"{prob:.1%}",
          delta=f"{risk} risk  ·  {contract} contract", delta_color="inverse")

risk_help = {
    "low": "Customer is likely to stay. No intervention needed.",
    "medium": "At-risk — a light touch (e.g. a check-in) can help.",
    "high": "High churn risk — offer a retention incentive.",
    "critical": "Critical — prioritize a retention offer now.",
}
st.success(risk_help[risk]) if risk in ("low", "medium") else st.warning(risk_help[risk])

with st.expander("Why this score (top risk factors)"):
    if factors:
        for factor in factors:
            direction = "raises churn risk" if factor["direction"] == "positive" else "lowers churn risk"
            st.markdown(f"- **{factor['feature']}** — {factor['impact']} impact, "
                        f"{direction} (contribution {factor['contribution']:+.2f})")
    else:
        st.write("No explanation available for this model version.")

with st.expander("Model details"):
    metrics = model_info.get("metrics", {})
    st.write(f"""
    - **Model:** `{model_info.get('model_version')}` · **Features:** `{model_info.get('feature_version')}`
    - **Algorithm:** {model_info.get('algorithm')} (chosen by validation PR-AUC)
    - **Calibrated probabilities:** isotonic calibration on held-out validation —
      Brier {metrics.get('brier')}, ECE {metrics.get('ece')}
    - **Deployed threshold:** {metrics.get('threshold')} (cost-optimized)
    - **Test metrics:** ROC-AUC {metrics.get('roc_auc')}, PR-AUC {metrics.get('pr_auc')},
      recall {metrics.get('recall')}, top-decile recall {metrics.get('top_decile_recall')}
    - **Risk bands:** low < {model_info.get('risk_thresholds', {}).get('low')} ·
      medium < {model_info.get('risk_thresholds', {}).get('medium')} ·
      high < {model_info.get('risk_thresholds', {}).get('high')} · critical above
    """)

st.markdown(
    '<div class="nx-footer"><b>NEXUS · CUSTOMER CHURN PREDICTION</b> — '
    'versioned pipeline · calibrated probabilities · Streamlit</div>',
    unsafe_allow_html=True,
)
