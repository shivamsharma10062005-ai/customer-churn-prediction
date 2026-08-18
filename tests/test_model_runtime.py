"""Tests for the model runtime (versioned bundle loading + legacy fallback)."""
import pytest

from app.config import settings
from app.model_runtime import ModelLoadError, ModelRuntime, load_bundle


def test_runtime_predict_range(tiny_bundle):
    runtime = ModelRuntime()
    runtime._bundle = tiny_bundle
    from app.features import make_dataframe

    prob = runtime.predict(make_dataframe(tiny_bundle.model_card["reference_profile"]))
    assert 0.0 <= prob <= 1.0


def test_load_bundle_when_absent_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        load_bundle(model_dir=tmp_path, version="v9.9.9")


@pytest.mark.skipif(not (settings.model_dir_path / "latest.json").exists(),
                    reason="no trained bundle on disk")
def test_load_latest_bundle_from_disk():
    bundle = load_bundle()
    assert bundle.model_version.startswith("v")
    assert "model_version" in bundle.model_card
    assert "risk_thresholds" in bundle.model_card
