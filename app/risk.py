"""Business-oriented risk engine.

Risk bands are derived from a cost-optimized threshold and the model card, not
hard-coded. Expected Retention Value (ERV) prioritizes customers by value and
intervention economics, not probability alone.
"""
from __future__ import annotations

from app.config import settings
from app.model_runtime import ModelBundle


def risk_thresholds(bundle: ModelBundle) -> dict[str, float]:
    card = bundle.model_card
    thresholds = card.get("risk_thresholds") or {"low": 0.30, "medium": 0.60, "high": 0.80}
    return {
        "low": float(thresholds.get("low", 0.30)),
        "medium": float(thresholds.get("medium", 0.60)),
        "high": float(thresholds.get("high", 0.80)),
    }


def classify(probability: float, bundle: ModelBundle) -> str:
    """Map a calibrated probability to a risk level."""
    t = risk_thresholds(bundle)
    if probability >= t["high"]:
        return "critical"
    if probability >= t["medium"]:
        return "high"
    if probability >= t["low"]:
        return "medium"
    return "low"


def expected_retention_value(
    probability: float,
    customer_value: float | None = None,
    effect: float | None = None,
    cost: float | None = None,
) -> float:
    """ERV = P(churn) * value * effectiveness - intervention cost."""
    value = customer_value if customer_value is not None else settings.retained_value
    effectiveness = effect if effect is not None else settings.intervention_effect
    intervention_cost = cost if cost is not None else settings.intervention_cost
    return probability * value * effectiveness - intervention_cost
