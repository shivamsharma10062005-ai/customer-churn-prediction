"""Structured JSON logging with request correlation.

All logs are single-line JSON (timestamp, level, request_id, message, extras).
PII is never logged; customer ids are hashed before they enter log fields.
"""
from __future__ import annotations

import contextvars
import json
import logging
import time
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def setup_logging(level: str = "INFO") -> logging.Logger:
    root = logging.getLogger("churn")
    if root.handlers:
        return root
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())
    root.propagate = False
    return root


def get_logger(name: str = "churn") -> logging.Logger:
    logger = logging.getLogger(f"churn.{name}")
    logger.propagate = True
    return logger
