"""Batch risk segmentation.

Scores a CSV of customers (one row per customer, same columns as the training
data) and writes a scored CSV with churn_probability, risk_level,
expected_retention_value and the model versions used. Scoring is vectorized;
per-customer explanations are computed only for the highest-risk customers to
bound cost.

Idempotent and restartable: the output path is deterministic given input +
model version, so re-running overwrites a stable artifact safely.

Usage:
    python scripts/batch_predict.py --input data/telco_churn.csv \
        --output artifacts/batch_scores.csv --limit 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.config import settings  # noqa: E402
from app.explanations import explain  # noqa: E402
from app.features import (FEATURE_COLUMNS, FEATURE_VERSION,  # noqa: E402
                          derive_num_addons_series)
from app.model_runtime import ModelLoadError, runtime  # noqa: E402
from app.risk import risk_thresholds  # noqa: E402

EXPLAIN_TOP_K = 50


def score(input_path: Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["num_addons"] = derive_num_addons_series(df)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    if limit:
        df = df.head(limit)

    bundle = runtime.bundle
    probs = bundle.predict_proba_batch(df[FEATURE_COLUMNS])

    t = risk_thresholds(bundle)
    risk_levels = np.select(
        [probs >= t["high"], probs >= t["medium"], probs >= t["low"]],
        ["critical", "high", "medium"],
        default="low",
    )
    erv = probs * settings.retained_value * settings.intervention_effect - settings.intervention_cost

    out = pd.DataFrame({
        "customer_id": df.get("customerID", df.index).astype(str) if "customerID" in df else df.index,
        "churn_probability": np.round(probs, 6),
        "risk_level": risk_levels,
        "expected_retention_value": np.round(erv, 2),
        "model_version": bundle.model_version,
        "feature_version": bundle.feature_version,
    })

    top_k = int(min(EXPLAIN_TOP_K, len(out)))
    top_idx = np.argsort(probs)[::-1][:top_k]
    factor_texts = [""] * len(out)
    for position in top_idx:
        row_dict = df.iloc[position].to_dict()
        factors = [f["feature"] for f in explain(bundle, row_dict, top_n=3)]
        factor_texts[position] = " | ".join(factors)
    out["top_risk_factors"] = factor_texts
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch churn risk segmentation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        scored = score(Path(args.input), args.limit)
    except ModelLoadError as exc:
        raise SystemExit(f"Model unavailable: {exc}")
    except ValueError as exc:
        raise SystemExit(f"Input error: {exc}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output, index=False)
    summary = scored["risk_level"].value_counts().to_dict()
    print(f"Scored {len(scored):,} customers -> {output}")
    print(f"Risk segments: {summary}")
    print(f"Mean probability: {scored['churn_probability'].mean():.4f}")


if __name__ == "__main__":
    main()
