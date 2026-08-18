"""Append-only prediction store (privacy-safe).

Persists a compact JSONL record per prediction. Customer identifiers are
sha256-hashed before writing so raw PII never lands on disk or in logs.
Designed as a simple interface so a Postgres-backed implementation can be
swapped in later without touching callers.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Optional

from app.config import settings


def hash_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


class PredictionStore:
    def __init__(self, path=None) -> None:
        self._path = path or settings.store_path_obj
        self._lock = threading.Lock()

    def append(
        self,
        *,
        customer_id: Optional[str],
        model_version: str,
        feature_version: str,
        probability: float,
        risk_level: str,
        latency_ms: float,
        request_id: str,
    ) -> None:
        record = {
            "request_id": request_id,
            "timestamp": time.time(),
            "customer_id_hash": hash_id(customer_id),
            "model_version": model_version,
            "feature_version": feature_version,
            "churn_probability": round(float(probability), 6),
            "risk_level": risk_level,
            "latency_ms": round(latency_ms, 3),
        }
        line = json.dumps(record, separators=(",", ":"))
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + os.linesep)

    def recent(self, limit: int = 20) -> list[dict]:
        if not self._path.exists():
            return []
        records: list[dict] = []
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return records[-limit:]


store = PredictionStore()
