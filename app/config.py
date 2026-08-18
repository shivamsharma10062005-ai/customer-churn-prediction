"""Application configuration.

Everything is driven by environment variables (optionally via a .env file
loaded by pydantic-settings). No secrets or environment-specific values are
hard-coded here.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

HERE = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = HERE.parent


class Settings(BaseSettings):
    """Runtime configuration for the churn prediction backend."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- service ---
    app_name: str = "customer-churn-api"
    api_version: str = "v1"
    log_level: str = "INFO"

    # --- security ---
    api_key: str = ""  # if empty, the API key gate is DISABLED (dev convenience)
    allowed_origins: str = "http://localhost:8501,http://localhost:8000"

    # --- model artifacts ---
    model_dir: str = str(PROJECT_ROOT / "artifacts" / "model")
    model_version: str = ""  # pin a version; empty = use latest on disk

    # --- risk engine business parameters ---
    intervention_cost: float = 10.0      # $ cost to reach one customer
    retained_value: float = 120.0        # $ expected value of a retained customer
    intervention_effect: float = 0.35    # fraction of would-be churners saved

    # --- rate limiting ---
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # --- drift ---
    drift_psi_threshold: float = 0.20

    # --- prediction store ---
    store_path: str = str(PROJECT_ROOT / "data" / "store" / "predictions.jsonl")

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def model_dir_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def store_path_obj(self) -> Path:
        return Path(self.store_path)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
