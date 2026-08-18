"""API integration tests (FastAPI TestClient with a tiny in-memory model)."""
import pytest
from fastapi.testclient import TestClient

import app.api as api
from app import schemas
from app.config import settings


@pytest.fixture
def client(tiny_bundle):
    # swap in a tiny bundle so tests never depend on the trained artifact
    api.runtime._bundle = tiny_bundle
    api.runtime._failed = None
    with TestClient(api.app) as c:
        yield c


def _payload():
    return {
        "profile": {
            "customer_id": "CUST-1",
            "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
            "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
            "StreamingMovies": "Yes", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
            "MonthlyCharges": 80.0, "TotalCharges": 960.0,
        }
    }


def test_health_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_success(client):
    r = client.post("/api/v1/predictions", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in ("low", "medium", "high", "critical")
    assert body["model_version"]
    assert body["feature_version"]
    assert isinstance(body["top_risk_factors"], list)


def test_predict_rejects_bad_category(client):
    payload = _payload()
    payload["profile"]["gender"] = "Alien"
    r = client.post("/api/v1/predictions", json=payload)
    assert r.status_code == 422


def test_predict_rejects_out_of_range(client):
    payload = _payload()
    payload["profile"]["tenure"] = 999
    r = client.post("/api/v1/predictions", json=payload)
    assert r.status_code == 422


def test_predict_derives_total_charges_when_absent(client):
    payload = _payload()
    payload["profile"].pop("TotalCharges")
    r = client.post("/api/v1/predictions", json=payload)
    assert r.status_code == 200


def test_api_key_required_when_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret-key")
    r = client.get("/api/v1/health")
    assert r.status_code == 401
    r = client.get("/api/v1/health", headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200


def test_model_info(client):
    r = client.get("/api/v1/model/info")
    assert r.status_code == 200
    body = r.json()
    assert "model_version" in body
    assert "metrics" in body


def test_metrics_endpoint(client):
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200
    assert "served_predictions" in r.json()
