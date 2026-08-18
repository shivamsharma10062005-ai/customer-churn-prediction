"""Versioned model runtime.

Loads the trained bundle (pipeline + calibration + model card) lazily and
thread-safely, verifies its checksum, and exposes `predict`. Falls back to the
legacy single-file artifact if no versioned bundle exists, so old deployments
keep working.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.config import settings


class ModelLoadError(RuntimeError):
    pass


@dataclass
class ModelBundle:
    pipeline: Pipeline
    model_card: dict[str, Any]
    isotonic: Optional[Any] = None
    bundle_dir: Optional[Path] = None
    loaded_from: str = "versioned"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def model_version(self) -> str:
        return str(self.model_card.get("model_version", "legacy-unknown"))

    @property
    def feature_version(self) -> str:
        return str(self.model_card.get("feature_version", "features-1"))

    def predict_proba(self, df: pd.DataFrame) -> float:
        raw = float(self.pipeline.predict_proba(df)[:, 1][0])
        if self.isotonic is not None:
            return float(self.isotonic.predict([[raw]])[0])
        return raw

    def predict_proba_batch(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(df)[:, 1]
        if self.isotonic is not None:
            return self.isotonic.predict(raw.reshape(-1, 1)).ravel()
        return raw


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_version(model_dir: Path) -> Optional[str]:
    pointer = model_dir / "latest.json"
    if pointer.exists():
        try:
            return str(json.loads(pointer.read_text(encoding="utf-8")).get("version"))
        except Exception:
            pass
    versions = sorted(
        (d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("v")),
        key=lambda s: [int(p) if p.isdigit() else 0 for p in s.lstrip("v").split(".")],
    )
    return versions[-1] if versions else None


def load_bundle(model_dir: Optional[Path] = None, version: Optional[str] = None) -> ModelBundle:
    """Load a model bundle. Raises ModelLoadError if none is available."""
    model_dir = model_dir or settings.model_dir_path
    version = version or settings.model_version or _latest_version(model_dir)

    if version:
        bundle_dir = model_dir / version
        if not bundle_dir.is_dir():
            raise ModelLoadError(f"Model bundle not found: {bundle_dir}")

        card_path = bundle_dir / "model_card.json"
        if not card_path.exists():
            raise ModelLoadError(f"Missing model_card.json in {bundle_dir}")
        card = json.loads(card_path.read_text(encoding="utf-8"))

        pipeline_path = bundle_dir / "pipeline.joblib"
        if not pipeline_path.exists():
            raise ModelLoadError(f"Missing pipeline.joblib in {bundle_dir}")

        checksum = card.get("pipeline_sha256")
        if checksum and _sha256(pipeline_path) != checksum:
            raise ModelLoadError(f"Checksum mismatch for {pipeline_path} — refusing to load")

        isotonic_path = bundle_dir / "isotonic.joblib"
        isotonic = joblib.load(isotonic_path) if isotonic_path.exists() else None

        return ModelBundle(
            pipeline=joblib.load(pipeline_path),
            model_card=card,
            isotonic=isotonic,
            bundle_dir=bundle_dir,
        )

    # --- legacy fallback (no versioned bundle yet) ---
    legacy = settings.model_dir_path.parent / "churn_model.joblib"
    if not legacy.exists():
        raise ModelLoadError("No model artifact found (checked versioned bundle and legacy file).")
    card = {
        "model_version": "legacy",
        "feature_version": "features-1",
        "algorithm": "unknown",
        "trained_at": "",
        "status": "legacy",
        "risk_thresholds": {"low": 0.30, "medium": 0.60, "high": 0.80},
    }
    return ModelBundle(
        pipeline=joblib.load(legacy),
        model_card=card,
        loaded_from="legacy",
    )


class ModelRuntime:
    """Thread-safe lazy holder for the active model bundle."""

    def __init__(self) -> None:
        self._bundle: Optional[ModelBundle] = None
        self._lock = threading.Lock()
        self._failed: Optional[str] = None

    @property
    def bundle(self) -> ModelBundle:
        with self._lock:
            if self._bundle is None:
                if self._failed:
                    raise ModelLoadError(self._failed)
                try:
                    self._bundle = load_bundle()
                except ModelLoadError as exc:
                    self._failed = str(exc)
                    raise
            return self._bundle

    def predict(self, df: pd.DataFrame) -> float:
        return self.bundle.predict_proba(df)

    def reload(self) -> None:
        with self._lock:
            self._bundle = load_bundle()
            self._failed = None

    def info(self) -> dict[str, Any]:
        card = self.bundle.model_card
        return {
            "model_version": self.bundle.model_version,
            "feature_version": self.bundle.feature_version,
            "trained_at": card.get("trained_at", ""),
            "algorithm": card.get("algorithm", ""),
            "dataset_version": card.get("dataset_version", ""),
            "metrics": card.get("metrics", {}),
            "risk_thresholds": card.get("risk_thresholds", {}),
            "status": card.get("status", ""),
            "baseline_mean_probability": card.get("baseline_mean_probability"),
        }


runtime = ModelRuntime()
