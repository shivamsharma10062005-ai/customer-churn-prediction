"""FastAPI application: churn prediction service.

Endpoints (all under /api/v1):
  GET  /api/v1/health            liveness + model readiness
  GET  /api/v1/model/info        model card summary (version, metrics, thresholds)
  POST /api/v1/predictions       predict + risk + explanation for one customer
  GET  /api/v1/predictions/recent  recent predictions (hashed ids only)
  GET  /api/v1/metrics           served stats + drift PSI
"""
from __future__ import annotations

import secrets
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import drift, schemas
from app.config import settings
from app.explanations import explain
from app.features import make_dataframe, validate_profile
from app.logging_setup import get_logger, request_id_var, setup_logging
from app.model_runtime import ModelLoadError, runtime
from app.risk import classify, expected_retention_value
from app.store import store

logger = setup_logging(settings.log_level)
log = get_logger("api")

START_TIME = time.time()

# --- in-memory rate limiter ---
_rate_lock = threading.Lock()
_rate_hits: dict[str, deque] = {}


def _rate_allowed(client_ip: str) -> bool:
    now = time.time()
    with _rate_lock:
        times = _rate_hits.setdefault(client_ip, deque())
        while times and now - times[0] > settings.rate_limit_window_seconds:
            times.popleft()
        if len(times) >= settings.rate_limit_requests:
            return False
        times.append(now)
        return True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # warm the model so the first request isn't slow
    try:
        runtime.info()
    except ModelLoadError as exc:
        log.error("model unavailable at startup", extra={"extra_fields": {"error": str(exc)}})
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        log.info(
            "request",
            extra={"extra_fields": {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code if response is not None else "n/a",
                "latency_ms": round(latency_ms, 3),
            }},
        )
        request_id_var.reset(token)


def _verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.auth_enabled:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _check_rate(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")


@app.exception_handler(ModelLoadError)
async def model_load_error_handler(_request: Request, exc: ModelLoadError):
    log.error("model load failed", extra={"extra_fields": {"error": str(exc)}})
    return JSONResponse(status_code=503, content={"detail": "Model unavailable. Try again later."})


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError):
    log.warning("bad request", extra={"extra_fields": {"error": str(exc)}})
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/v1/health", response_model=schemas.HealthResponse, dependencies=[Depends(_verify_api_key)])
def health():
    try:
        info = runtime.info()
        model_loaded = True
    except ModelLoadError:
        info = {}
        model_loaded = False
    return schemas.HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_version=info.get("model_version"),
        model_loaded=model_loaded,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.get("/api/v1/model/info", response_model=schemas.ModelInfoResponse, dependencies=[Depends(_verify_api_key)])
def model_info():
    return schemas.ModelInfoResponse(**runtime.info())


@app.post("/api/v1/predictions", response_model=schemas.PredictionResponse, dependencies=[Depends(_verify_api_key)])
async def predict(payload: schemas.PredictionRequest, request: Request):
    _check_rate(request)
    profile = payload.profile
    row = profile.model_dump()
    row["SeniorCitizen"] = int(row["SeniorCitizen"])
    if row.get("TotalCharges") is None:
        row["TotalCharges"] = round(float(row["MonthlyCharges"]) * float(row["tenure"]), 2)

    validation_errors = validate_profile(row)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))

    df = make_dataframe(row)
    start = time.perf_counter()
    probability = runtime.predict(df)
    latency_ms = (time.perf_counter() - start) * 1000

    bundle = runtime.bundle
    risk_level = classify(probability, bundle)
    erv = expected_retention_value(probability)
    factors = explain(bundle, row)
    drift.monitor.record(probability)
    store.append(
        customer_id=profile.customer_id,
        model_version=bundle.model_version,
        feature_version=bundle.feature_version,
        probability=probability,
        risk_level=risk_level,
        latency_ms=latency_ms,
        request_id=request_id_var.get(),
    )

    return schemas.PredictionResponse(
        customer_id=profile.customer_id,
        churn_probability=round(probability, 6),
        risk_level=risk_level,
        expected_retention_value=round(erv, 2),
        model_version=bundle.model_version,
        feature_version=bundle.feature_version,
        top_risk_factors=[schemas.RiskFactor(**factor) for factor in factors],
    )


@app.get("/api/v1/predictions/recent", response_model=list[schemas.RecentPrediction], dependencies=[Depends(_verify_api_key)])
def recent_predictions(limit: int = 20):
    limit = max(1, min(limit, 100))
    records = store.recent(limit)
    return [
        schemas.RecentPrediction(
            timestamp=rec["timestamp"],
            customer_id_hash=rec.get("customer_id_hash"),
            model_version=rec.get("model_version", ""),
            churn_probability=rec.get("churn_probability", 0.0),
            risk_level=rec.get("risk_level", ""),
            latency_ms=rec.get("latency_ms", 0.0),
        )
        for rec in records
    ]


@app.get("/api/v1/metrics", response_model=schemas.DriftMetric, dependencies=[Depends(_verify_api_key)])
def metrics():
    return schemas.DriftMetric(**drift.monitor.summary(runtime.bundle))
