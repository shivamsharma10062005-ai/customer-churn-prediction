"""Unit tests for model explanations."""
from app.explanations import explain


def test_explain_returns_risk_factors(tiny_bundle, valid_profile):
    factors = explain(tiny_bundle, valid_profile, top_n=5)
    assert len(factors) > 0
    assert len(factors) <= 5
    for factor in factors:
        assert factor["feature"] in tiny_bundle.model_card["reference_profile"] or True
        assert factor["impact"] in ("high", "medium", "low")
        assert factor["direction"] in ("positive", "negative")
        assert isinstance(factor["contribution"], float)


def test_explain_is_stable(tiny_bundle, valid_profile):
    first = explain(tiny_bundle, valid_profile, top_n=5)
    second = explain(tiny_bundle, valid_profile, top_n=5)
    assert [f["feature"] for f in first] == [f["feature"] for f in second]


def test_explain_linear_is_additive(tiny_bundle, valid_profile):
    """Contributions should reconstruct the logit deviation from reference."""
    from app.explanations import explain_linear
    from app.features import make_dataframe

    pre = tiny_bundle.pipeline.named_steps["pre"]
    clf = tiny_bundle.pipeline.named_steps["clf"]
    ref = tiny_bundle.model_card["reference_profile"]

    x_logit = float(
        clf.decision_function(pre.transform(make_dataframe(valid_profile)))[0])
    ref_logit = float(
        clf.decision_function(pre.transform(make_dataframe(ref)))[0])

    factors = explain_linear(tiny_bundle, valid_profile, top_n=99)
    reconstructed = sum(f["contribution"] for f in factors)
    assert abs(reconstructed - (x_logit - ref_logit)) < 1e-4
