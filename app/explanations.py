"""Prediction explanations.

For linear models we compute an exact, additive decomposition of the logit
deviation from a reference customer (mode categories / median numerics) and map
it back to original feature names. For tree models we fall back to local
perturbation importance. Both are stable, model-faithful and fast enough for
synchronous production inference.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.features import CATEGORICAL, NUMERIC, make_dataframe
from app.model_runtime import ModelBundle


def _transformer(pre, name: str):
    """Return the fitted transformer, unwrapping an optional single-step Pipeline."""
    transformer = pre.named_transformers_[name]
    if isinstance(transformer, Pipeline):
        return list(transformer.steps)[-1][1]
    return transformer


def _onehot_mapping(bundle: ModelBundle) -> dict[str, tuple[str, str]]:
    """Map encoded column name -> (original feature, category)."""
    pre = bundle.pipeline.named_steps["pre"]
    ohe = _transformer(pre, "cat")
    mapping: dict[str, tuple[str, str]] = {}
    for feature, categories in zip(CATEGORICAL, ohe.categories_):
        for category in categories:
            mapping[f"{feature}_{category}"] = (feature, category)
    return mapping


def _reference_row(bundle: ModelBundle) -> dict[str, Any]:
    card = bundle.model_card
    ref = card.get("reference_profile")
    if ref:
        return dict(ref)
    return {}


def explain_linear(bundle: ModelBundle, row: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
    """Exact additive contribution of each original feature vs the reference."""
    clf = bundle.pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    pre = bundle.pipeline.named_steps["pre"]
    scaler = _transformer(pre, "num")
    ohe = _transformer(pre, "cat")

    coef = np.asarray(clf.coef_).ravel()
    mapping = _onehot_mapping(bundle)
    ref = _reference_row(bundle)

    x_df = make_dataframe(row)
    ref_df = make_dataframe(ref) if ref else x_df.copy()
    x_enc = pre.transform(x_df)
    ref_enc = pre.transform(ref_df)

    contributions: dict[str, float] = {}
    # numeric features (scaled)
    for i, feature in enumerate(NUMERIC):
        contribution = float(coef[i] * (x_enc[0, i] - ref_enc[0, i]))
        if abs(contribution) > 1e-12:
            contributions[feature] = contribution

    # categorical features (one-hot block)
    start = len(NUMERIC)
    for feature in CATEGORICAL:
        cats = ohe.categories_[CATEGORICAL.index(feature)]
        block_sum = 0.0
        for category in cats:
            col = f"{feature}_{category}"
            if col not in mapping:
                continue
            col_idx = list(mapping).index(col)
            block_sum += float(coef[start + col_idx] * (x_enc[0, start + col_idx] - ref_enc[0, start + col_idx]))
        if abs(block_sum) > 1e-12:
            contributions[feature] = block_sum

    return _rank(contributions, top_n)


def explain_perturbation(bundle: ModelBundle, row: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
    """Local perturbation importance for non-linear models (bounded cost)."""
    ref = _reference_row(bundle)
    if not ref:
        return []
    base_prob = bundle.predict_proba(make_dataframe(row))
    contributions: dict[str, float] = {}
    for feature in list(NUMERIC) + list(CATEGORICAL):
        if feature == "num_addons":
            continue
        variant = dict(row)
        if feature in NUMERIC:
            variant[feature] = ref.get(feature, variant[feature])
        else:
            variant[feature] = ref.get(feature, variant[feature])
        prob = bundle.predict_proba(make_dataframe(variant))
        contributions[feature] = base_prob - prob
    return _rank(contributions, top_n)


def _rank(contributions: dict[str, float], top_n: int) -> list[dict[str, Any]]:
    if not contributions:
        return []
    max_abs = max(abs(v) for v in contributions.values()) or 1.0
    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))
    factors = []
    for feature, contribution in ranked[:top_n]:
        ratio = abs(contribution) / max_abs
        impact = "high" if ratio >= 0.5 else ("medium" if ratio >= 0.25 else "low")
        factors.append({
            "feature": feature,
            "impact": impact,
            "direction": "positive" if contribution > 0 else "negative",
            "contribution": round(float(contribution), 5),
        })
    return factors


def explain(bundle: ModelBundle, row: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
    clf = bundle.pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        factors = explain_linear(bundle, row, top_n)
        if factors:
            return factors
    return explain_perturbation(bundle, row, top_n)
