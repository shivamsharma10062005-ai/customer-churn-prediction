# Project 4 — Customer Churn Prediction

Predicts whether a post-paid telecom customer will churn. Mirrors the retention
problem every telecom/e-commerce/insurance company solves with ML.

## What's here

```
app/                   FastAPI backend package (production services)
  config.py            env-driven configuration (pydantic-settings)
  features.py          single feature contract (train + inference share it)
  schemas.py           Pydantic request/response contracts
  model_runtime.py     versioned model loading, checksum verify, legacy fallback
  risk.py              risk engine (bands + expected retention value)
  explanations.py      coefficient-based top risk factors (exact for logistic)
  drift.py             PSI drift monitor vs training-time baseline
  store.py             privacy-safe append-only prediction store (hashed ids)
  api.py               FastAPI app (/api/v1/*)
scripts/
  batch_predict.py     vectorized batch risk segmentation (idempotent)
tests/                 22 pytest tests (features, risk, explanations, runtime, API)
generate_data.py       deterministic synthetic generator -> data/telco_churn.csv
train_churn.py         trains a versioned, calibrated bundle -> artifacts/model/v*/
demo.py                Streamlit app (same services as the API)
churn_theme.py         NEXUS theme for the demo
```

## ML approach (and what changed)

- **Data**: 15,000 synthetic customers, ~26% churn, deterministic seed.
- **Split**: stratified train/val/test (70/15/15). Validation is used *only* for
  calibration and threshold selection — never for model fitting, never for the
  reported metrics.
- **Imbalance**: SMOTE applied to the **train split only**. Because SMOTE
  rebalances the class prior, predicted probabilities are systematically biased
  upward — so the base model is then **isotonic-calibrated** on the
  unresampled validation set. Result: mean predicted probability (0.279)
  matches the true churn rate (0.26), Brier 0.146, ECE 0.022.
- **Threshold**: not assumed to be 0.5. Chosen to maximize expected profit
  (`TP × value × effect − FP × cost`) on validation. Champion threshold is
  configurable via `--value/--cost/--effect`.
- **Champion selection**: best validation PR-AUC (ranking quality) among
  Logistic / Random Forest / XGBoost → **Logistic** wins here (linear-ish
  signal, best calibration, and fully explainable).
- **Leakage**: no target/future/prediction-point leakage; `customerID` dropped;
  preprocessors fit on train only. Known limitation: synthetic data is a
  snapshot with no time axis, so a *temporal* split (predict future periods)
  should be used when real data lands. `TotalCharges` is deterministic of
  `tenure × MonthlyCharges` in the generator — redundant, kept because it is
  available before the prediction point.

## Current results (v2.0.0 — `python train_churn.py`)

| Model | ROC-AUC | PR-AUC | Brier | ECE | Recall | Top-decile recall |
|-------|---------|--------|-------|-----|--------|-------------------|
| Logistic | 0.822 | **0.607** | **0.146** | **0.022** | 0.895 | 0.253 |
| Random Forest | 0.809 | 0.558 | 0.150 | 0.019 | — | 0.239 |
| XGBoost | 0.813 | 0.570 | 0.150 | 0.028 | — | 0.247 |

Champion threshold: **0.17** (cost-optimized: value $120, cost $10, effect 0.35).
Risk bands derived from it: low < 0.07 · medium < 0.17 · high < 0.37 · critical above.

## Backend API

```bash
pip install -r requirements.txt
python train_churn.py          # build artifacts/model/v2.0.0/ + model_card.json
uvicorn app.api:app --port 8000
```

| Endpoint | Description |
|---|---|
| `GET /api/v1/health` | liveness + model readiness |
| `GET /api/v1/model/info` | model card (version, metrics, thresholds) |
| `POST /api/v1/predictions` | predict + risk + explanation for one customer |
| `GET /api/v1/predictions/recent` | recent predictions (hashed ids) |
| `GET /api/v1/metrics` | served stats + drift PSI |

```bash
curl -X POST localhost:8000/api/v1/predictions -H 'Content-Type: application/json' \
  -d '{"profile":{"gender":"Male","SeniorCitizen":0,"Partner":"No","Dependents":"No",
       "tenure":3,"PhoneService":"Yes","MultipleLines":"Yes","InternetService":"Fiber optic",
       "OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No",
       "StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month",
       "PaperlessBilling":"Yes","PaymentMethod":"Electronic check",
       "MonthlyCharges":95.5,"TotalCharges":286.5}}'
```

Response includes `churn_probability`, `risk_level`, `expected_retention_value`,
`model_version`, `feature_version`, and `top_risk_factors`.

- **Auth**: set `API_KEY` to require `X-API-Key` on every call.
- **Rate limiting**: in-memory sliding window (configurable).
- **Observability**: structured JSON logs (request_id, latency, model_version),
  per-prediction store (PII hashed), PSI drift monitor.
- **Docs**: `http://localhost:8000/api/v1/docs`.

## Batch risk segmentation

```bash
python scripts/batch_predict.py --input data/telco_churn.csv \
  --output artifacts/batch_scores.csv --limit 2000
```
Writes `churn_probability`, `risk_level`, `expected_retention_value` and model
version per customer. Vectorized; deterministic output path.

## Tests

```bash
python -m pytest tests -q
```

## Deploy

- **API**: `Dockerfile` + `render.yaml` (Render web service, health check at
  `/api/v1/health`).
- **Demo**: `streamlit run demo.py` or deploy to Streamlit Community Cloud.

## Model versioning

Each training run writes `artifacts/model/v<version>/` containing
`pipeline.joblib`, `isotonic.joblib`, plots and `model_card.json`
(model_version, feature_version, dataset_version, trained_at, checksum,
metrics, thresholds, reference profile, baseline prediction distribution).
`artifacts/model/latest.json` points at the active version. Every prediction is
traceable to a model version.

## Resume bullet (paste + update numbers)

> Productionized a customer-churn prediction system: versioned, checksummed model
> bundles with isotonic-calibrated probabilities (ECE 0.02, Brier 0.15), a
> cost-optimized decision threshold, a FastAPI service with validated contracts,
> API-key auth, rate limiting, structured JSON logging, PSI drift monitoring and a
> privacy-safe prediction store — plus a 22-test suite and batch risk segmentation.
> Logistic regression champion: ROC-AUC 0.82, catching 89% of churners while the
> mean predicted probability matches the true churn rate.
