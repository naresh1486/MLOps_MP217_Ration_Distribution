"""Automated tests for FastAPI prediction service (Part 10).

Covers:
- GET /health endpoint (status: ok)
- GET / root endpoint
- POST /predict successful prediction with valid payloads
- Client Rule 1: biometric_failures > 150 rejected with HTTP 422
- Boundary check: biometric_failures == 150 accepted with HTTP 200
- Client Rule 2: working_hours < 1.5 rejected with HTTP 422
- Boundary check: working_hours == 1.5 accepted with HTTP 200
- Missing required fields validation (HTTP 422)
- Invalid data types validation (HTTP 422)
- Categorical casing and whitespace tolerance
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Create a reusable FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture
def valid_payload():
    """Standard valid payload containing all 8 operational features."""
    return {
        "cards_attached": 1150,
        "working_hours": 7.0,
        "biometric_failures": 35,
        "temp_c": 29.5,
        "stock_kg_morning": 1600.0,
        "days_to_month_end": 12,
        "shop_area": "tribal",
        "block": "Block North",
    }


def test_health_endpoint(client):
    """GET /health must return status 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert data.get("model_name") == "mp217_ration_model"
    assert data.get("model_alias") == "champion"


def test_root_endpoint(client):
    """GET / must return status 200 with service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "health" in data
    assert "predict" in data


def test_predict_endpoint_success(client, valid_payload):
    """POST /predict must return status 200 and predicted grain_kg."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "prediction" in data
    assert "predicted_grain_kg" in data
    assert "grain_kg" in data

    # Verify predictions are within a sensible physical distribution range for Kochi ration shops
    pred_val = float(data["prediction"])
    assert 200.0 < pred_val < 3000.0
    assert data["predicted_grain_kg"] == pred_val
    assert data["grain_kg"] == pred_val


def test_client_rule_1_biometric_failures_exceeding_150_rejected(client, valid_payload):
    """Client Rule 1: biometric_failures > 150 must be rejected with HTTP 422."""
    payload = valid_payload.copy()
    payload["biometric_failures"] = 151

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("biometric_failures" in str(err) for err in errors)


def test_client_rule_1_biometric_failures_boundary_accepted(client, valid_payload):
    """Client Rule 1 Boundary: biometric_failures == 150 must be accepted with HTTP 200."""
    payload = valid_payload.copy()
    payload["biometric_failures"] = 150

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_client_rule_2_working_hours_below_1_5_rejected(client, valid_payload):
    """Client Rule 2: working_hours < 1.5 must be rejected with HTTP 422."""
    payload = valid_payload.copy()
    payload["working_hours"] = 1.4

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    errors = response.json().get("detail", [])
    assert any("working_hours" in str(err) for err in errors)


def test_client_rule_2_working_hours_boundary_accepted(client, valid_payload):
    """Client Rule 2 Boundary: working_hours == 1.5 must be accepted with HTTP 200."""
    payload = valid_payload.copy()
    payload["working_hours"] = 1.5

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


@pytest.mark.parametrize(
    "missing_field",
    [
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
        "shop_area",
        "block",
    ],
)
def test_missing_required_field_rejected(client, valid_payload, missing_field):
    """Omitting any of the 8 required features must trigger HTTP 422."""
    payload = valid_payload.copy()
    del payload[missing_field]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,invalid_value",
    [
        ("cards_attached", "many_cards"),
        ("working_hours", "full_day"),
        ("biometric_failures", "none"),
        ("temp_c", "very_hot"),
        ("stock_kg_morning", "ten_bags"),
        ("days_to_month_end", "two_weeks"),
    ],
)
def test_invalid_data_types_rejected(client, valid_payload, field, invalid_value):
    """Passing string/invalid types for numeric features must trigger HTTP 422."""
    payload = valid_payload.copy()
    payload[field] = invalid_value

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_prediction_casing_and_whitespace_tolerance(client, valid_payload):
    """Area strings with different casing and whitespace should be properly handled."""
    payload = valid_payload.copy()
    payload["shop_area"] = "  Tribal  "

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
