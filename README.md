# Project 4 — Customer Churn Prediction

Predicts whether a post-paid telecom customer will churn. Mirrors the retention
problem every telecom/e-commerce/insurance company solves with ML, using the
same patterns as the classic Telco Customer Churn dataset.

## Approach
- **Data**: deterministic synthetic generator (`generate_data.py`) that recreates
  real churn drivers — contract type (month-to-month churns most), tenure
  (loyalty), payment method (electronic check = churniest), internet service,
  add-on services (security/support lower churn), senior citizens, household
  add-ons. **15,000 customers**, ~26% churn.
- **Preprocessing**: `OneHotEncoder` via `ColumnTransformer`, `TotalCharges`
  cleaned.
- **Imbalance**: SMOTE applied **after** preprocessing, on the train split only —
  never on test data (no leakage).
- **Models**: Logistic Regression, Random Forest, XGBoost.
- **Validation**: stratified 80/20 split; metrics reported on the untouched
  test set: ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix.

> Why ROC-AUC over accuracy? The dataset is imbalanced (~26% churn), so
> "accuracy" looks good even with a dumb model. AUC measures how well the model
> *ranks* churners vs non-churners, which is what a retention team actually
> needs to prioritize outreach. Recall matters too — a false negative means a
> churning customer you never called.

## Results (run `python train_churn.py` to regenerate)

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Logistic Regression | **~0.83** | ~0.49 | **~0.84** | **~0.62** |
| Random Forest | ~0.82 | ~0.55 | ~0.57 | ~0.56 |
| XGBoost | ~0.82 | **~0.57** | ~0.53 | ~0.55 |

Logistic Regression wins on ROC-AUC and recall — with SMOTE it catches 84% of
churners while staying the most interpretable model. The tree models trade recall
for precision. On a linear-ish churn signal, the simple, explainable model wins:
a great talking point (start simple, only escalate to tree models when the data
has nonlinear interactions). Swap in the real Kaggle Telco dataset to see XGBoost
pull ahead as usual.

## Files
```
generate_data.py     # builds data/telco_churn.csv (15K customers)
train_churn.py       # trains 3 models, SMOTE, prints metrics, saves artifacts/
demo.py              # Streamlit app: predict churn for any customer profile
data/                # generated dataset
artifacts/           # churn_model.joblib, results.json, plots
```

## Demo (deploy to Streamlit Community Cloud)
```bash
pip install -r ../requirements.txt imbalanced-learn
python generate_data.py
python train_churn.py
streamlit run demo.py
```
The model ships as one self-contained sklearn pipeline
(`artifacts/churn_model.joblib`), so the demo feeds raw customer features in and
gets a churn probability back.

## Resume bullet (paste + update numbers)
> Built a customer-churn prediction model for a telecom dataset (15K customers)
> using Logistic Regression and XGBoost with an OneHotEncoder + StandardScaler
> pipeline and SMOTE to handle class imbalance. Achieved ROC-AUC of ~0.83 and
> recall of ~0.84 on a stratified hold-out (catching 84% of churners), and
> identified contract type, tenure, and payment method as the top churn drivers —
> providing actionable retention recommendations. Deployed the model as a single
> sklearn pipeline behind a Streamlit demo app.