"""Lightweight drift monitoring.

Tracks the served prediction distribution against the training-time baseline
(binned in the model card) using Population Stability Index, plus a rolling
mean probability. In-memory and process-scoped — durable drift telemetry can be
added by persisting the store records (see app/store.py).
"""
from __future__ import annotations

import math
import threading
from collections import deque
from typing import Optional

from app.config import settings
from app.model_runtime import ModelBundle

_BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _bin_counts(values: list[float]) -> list[int]:
    counts = [0] * (len(_BIN_EDGES) - 1)
    for value in values:
        index = min(int(value * 10), len(counts) - 1)
        counts[index] += 1
    return counts


def psi(expected: list[float], actual: list[float]) -> Optional[float]:
    """Population Stability Index between two probability distributions."""
    if not expected or not actual:
        return None
    exp_counts = _bin_counts(expected)
    act_counts = _bin_counts(actual)
    score = 0.0
    for e, a in zip(exp_counts, act_counts):
        e_rate = (e + 1e-9) / (sum(exp_counts) + 1e-9)
        a_rate = (a + 1e-9) / (sum(act_counts) + 1e-9)
        score += (a_rate - e_rate) * math.log(a_rate / e_rate)
    return score


class DriftMonitor:
    def __init__(self, window: int = 2000) -> None:
        self._window = window
        self._recent: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, probability: float) -> None:
        with self._lock:
            self._recent.append(probability)

    def summary(self, bundle: ModelBundle) -> dict:
        baseline = bundle.model_card.get("baseline_prediction_distribution")
        with self._lock:
            served = list(self._recent)
        stats = {
            "served_predictions": len(served),
            "mean_probability": (sum(served) / len(served)) if served else None,
            "psi_recent_vs_baseline": None,
            "threshold": settings.drift_psi_threshold,
            "drift_detected": False,
        }
        if baseline and len(served) >= 100:
            value = psi(baseline, served) or 0.0
            stats["psi_recent_vs_baseline"] = round(value, 4)
            stats["drift_detected"] = value > settings.drift_psi_threshold
        return stats


monitor = DriftMonitor()
