"""Unit tests for the risk engine."""
from app import risk
from app.model_runtime import ModelBundle


def _bundle_with_thresholds(low, medium, high) -> ModelBundle:
    card = {"risk_thresholds": {"low": low, "medium": medium, "high": high}}
    # bundle only needs .model_card here
    bundle = ModelBundle(pipeline=None, model_card=card)  # type: ignore[arg-type]
    return bundle


def test_classify_bands(tiny_bundle):
    assert risk.classify(0.05, tiny_bundle) == "low"
    assert risk.classify(0.30, tiny_bundle) == "medium"
    assert risk.classify(0.60, tiny_bundle) == "high"
    assert risk.classify(0.85, tiny_bundle) == "critical"


def test_erv_matches_formula(monkeypatch):
    monkeypatch.setattr(risk.settings, "retained_value", 100.0)
    monkeypatch.setattr(risk.settings, "intervention_cost", 10.0)
    monkeypatch.setattr(risk.settings, "intervention_effect", 0.5)
    # 0.8 * 100 * 0.5 - 10 = 30
    assert risk.expected_retention_value(0.8) == 30.0


def test_erv_negative_when_unprofitable(monkeypatch):
    monkeypatch.setattr(risk.settings, "retained_value", 50.0)
    monkeypatch.setattr(risk.settings, "intervention_cost", 20.0)
    monkeypatch.setattr(risk.settings, "intervention_effect", 0.5)
    # 0.2 * 50 * 0.5 - 20 = -15
    assert risk.expected_retention_value(0.2) == -15.0
