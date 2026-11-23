# PART 10 REPORT — FastAPI Prediction API

**Project:** MP217 — Ration Shop Distribution (Kochi)  
**Group:** G230  
**Date:** November 24, 2026  
**Status:** PASS  

---

## 1. Executive Summary

In Part 10, we built and verified the production FastAPI prediction service for MP217 fair price (ration) shop grain distribution forecasting in Kochi. The service loads the registered `@champion` model (`mp217_ration_model`) from MLflow (`sqlite:///mlflow.db`) and exposes standard HTTP endpoints for health monitoring and batch-ready single-record predictions.

Strict Pydantic input validation enforces the client's mandatory operational constraints (rejecting `biometric_failures > 150` and `working_hours < 1.5` with HTTP 422 Unprocessable Entity). A custom structured JSON logger streams single-line operational records to stdout (screen) without recording sensitive personal customer data.

The service was thoroughly tested using `fastapi.testclient.TestClient`. The test suite expanded from 40 to **62 total passing tests**, with all 22 new API tests passing in under 4 seconds.

---

## 2. API Endpoints

The FastAPI application is implemented in `app/main.py` and provides the following endpoints:

| Endpoint | Method | Purpose | Success Status | Expected Response Structure |
|---|---|---|---|---|
| `/health` | `GET` | Readiness & Liveness probe | `200 OK` | `{"status": "ok", "model_name": "mp217_ration_model", "model_alias": "champion"}` |
| `/` | `GET` | Service index & navigation | `200 OK` | `{"service": "...", "health": "/health", "predict": "/predict", "docs": "/docs"}` |
| `/predict` | `POST` | Grain distribution inference | `200 OK` | `{"prediction": 1177.92, "predicted_grain_kg": 1177.92, "grain_kg": 1177.92, "status": "ok"}` |
| `/docs` | `GET` | Interactive Swagger UI | `200 OK` | OpenAPI 3.1 documentation interface |
| `/openapi.json` | `GET` | OpenAPI JSON schema | `200 OK` | Machine-readable API schema |

---

## 3. Model Loading Architecture

In accordance with Section 8 and Section 9 of `STUDENT_HANDBOOK.md`:
1. **Primary Model Source:** MLflow Registry:
   - **Tracking Backend:** `sqlite:///mlflow.db`
   - **Registered Model Name:** `mp217_ration_model`
   - **Registered Alias:** `champion` (points to Version 1 GradientBoostingRegressor pipeline)
   - **Loading Method:** `mlflow.sklearn.load_model("models:/mp217_ration_model@champion")`
2. **Resilient Local Fallback:**
   - If SQLite locking or tracking backend connectivity errors occur during isolated CI/CD or container startup, the service gracefully falls back to `joblib.load("models/model.joblib")`.
   - Both artifacts represent the exact identical trained scikit-learn pipeline (Preprocessor + GradientBoostingRegressor, MAE = 60.18 kg).
3. **Application Lifespan:**
   - The model is eagerly loaded during FastAPI startup via an `@asynccontextmanager lifespan(app: FastAPI)` hook, ensuring zero cold-start latency on initial prediction requests.

---

## 4. Input Validation & Client Rules

The request payload is governed by Pydantic V2 schema `RationPredictionRequest`. The schema requires all 8 operational features identified in `PROBLEM.md` Section 5:

```python
class RationPredictionRequest(BaseModel):
    cards_attached: int = Field(..., description="Number of ration cards attached to shop")
    working_hours: float = Field(..., ge=1.5, description="Daily working hours (must be >= 1.5; client rule 2)")
    biometric_failures: int = Field(..., le=150, description="Daily biometric failure count (must be <= 150; client rule 1)")
    temp_c: float = Field(..., description="Daily temperature in Celsius")
    stock_kg_morning: float = Field(..., description="Morning stock in kg at shop opening")
    days_to_month_end: int = Field(..., description="Days remaining until end of the month")
    shop_area: str = Field(..., description="Shop area category (tribal, rural, urban)")
    block: str = Field(..., description="Administrative block (e.g. Block North, Block South)")
```

### Client Validation Rules Enforced:
1. **Rule 1 (Biometric Failures Cap):**
   - `biometric_failures > 150` is impossible for Kochi fair price shops.
   - Configured via `Field(..., le=150)`.
   - Payloads with `biometric_failures = 151` immediately return `HTTP 422 Unprocessable Entity` without invoking the machine learning model.
   - Payloads with `biometric_failures = 150` are accepted (`HTTP 200`).
2. **Rule 2 (Minimum Working Hours):**
   - `working_hours < 1.5` represents clerical typing mistakes.
   - Configured via `Field(..., ge=1.5)`.
   - Payloads with `working_hours = 1.4` immediately return `HTTP 422 Unprocessable Entity`.
   - Payloads with `working_hours = 1.5` are accepted (`HTTP 200`).
3. **Completeness & Type Safety:**
   - Omitting any required feature returns `HTTP 422`.
   - Passing incompatible data types (e.g., strings for numeric fields) returns `HTTP 422`.
   - Area strings are normalized (`strip().lower()`) to safely accommodate whitespace and casing differences (`" Tribal "` $\rightarrow$ `"tribal"`).

---

## 5. Structured JSON Logging

In compliance with Section 9 and Section 11 (Grading Criteria J) of `STUDENT_HANDBOOK.md`:
- Logging is implemented via Python's standard `logging` library using a custom `JSONFormatter`.
- Log records are streamed directly to `sys.stdout` as single-line JSON strings (never written to file).
- Predictions emit an `INFO` event containing operational feature inputs and the resulting prediction.
- Customer privacy is preserved: no individual cardholder IDs or personal identifying details are accepted or logged.

### Example JSON Log Line:
```json
{"timestamp": "2026-11-24 03:28:40,729", "level": "INFO", "logger": "mp217.api", "message": "Prediction generated successfully", "event": "prediction", "predicted_grain_kg": 1177.92, "cards_attached": 1150, "working_hours": 7.0, "biometric_failures": 35, "temp_c": 29.5, "stock_kg_morning": 1600.0, "days_to_month_end": 12, "shop_area": "tribal", "block": "Block North"}
```

---

## 6. Example Requests and Responses

### A. Health Check (`GET /health`)

#### Request:
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

#### Response (`HTTP 200 OK`):
```json
{
  "status": "ok",
  "model_name": "mp217_ration_model",
  "model_alias": "champion"
}
```

---

### B. Valid Prediction Request (`POST /predict`)

#### Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cards_attached": 1150,
    "working_hours": 7.0,
    "biometric_failures": 35,
    "temp_c": 29.5,
    "stock_kg_morning": 1600.0,
    "days_to_month_end": 12,
    "shop_area": "tribal",
    "block": "Block North"
  }'
```

#### Response (`HTTP 200 OK`):
```json
{
  "prediction": 1177.92,
  "predicted_grain_kg": 1177.92,
  "grain_kg": 1177.92,
  "status": "ok"
}
```

---

### C. Client Rule 1 Rejection (`biometric_failures > 150`)

#### Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cards_attached": 1150,
    "working_hours": 7.0,
    "biometric_failures": 180,
    "temp_c": 29.5,
    "stock_kg_morning": 1600.0,
    "days_to_month_end": 12,
    "shop_area": "tribal",
    "block": "Block North"
  }'
```

#### Response (`HTTP 422 Unprocessable Entity`):
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "biometric_failures"],
      "msg": "Input should be less than or equal to 150",
      "input": 180,
      "ctx": {"le": 150}
    }
  ]
}
```

---

### D. Client Rule 2 Rejection (`working_hours < 1.5`)

#### Request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cards_attached": 1150,
    "working_hours": 1.0,
    "biometric_failures": 35,
    "temp_c": 29.5,
    "stock_kg_morning": 1600.0,
    "days_to_month_end": 12,
    "shop_area": "tribal",
    "block": "Block North"
  }'
```

#### Response (`HTTP 422 Unprocessable Entity`):
```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "working_hours"],
      "msg": "Input should be greater than or equal to 1.5",
      "input": 1.0,
      "ctx": {"ge": 1.5}
    }
  ]
}
```

---

## 7. Automated Test Results

A dedicated automated test suite was created in `tests/test_api.py` utilizing `fastapi.testclient.TestClient`.

### Test Summary:
- **API Tests Added:** 22 tests
- **Total Project Tests:** 62 tests across 8 test modules
- **Total Passing:** 62 / 62 (100%)
- **Execution Duration:** 3.97 seconds

```
============================= test session starts ==============================
platform darwin -- Python 3.11.9, pytest-8.3.4, pluggy-1.7.0
rootdir: /Users/nareshyadav/Desktop/mlops_project
configfile: pytest.ini
collected 62 items

tests/test_api.py ......................                                 [ 35%]
tests/test_cleaning.py ........                                          [ 48%]
tests/test_cleaning_validation.py .....                                  [ 56%]
tests/test_config_and_utils.py ....                                      [ 62%]
tests/test_evaluate.py .....                                             [ 70%]
tests/test_model_inference.py .....                                      [ 79%]
tests/test_quality_checks.py ........                                    [ 91%]
tests/test_train.py .....                                                [100%]

======================= 62 passed, 18 warnings in 3.97s ========================
```

### Breakdown of API Test Coverage (`tests/test_api.py`):
1. `test_health_endpoint`: Verifies HTTP 200, status "ok", and model metadata.
2. `test_root_endpoint`: Verifies service index endpoint and route discovery.
3. `test_predict_endpoint_success`: Verifies valid prediction returns HTTP 200 and physically realistic prediction range (200 kg – 3,000 kg).
4. `test_client_rule_1_biometric_failures_exceeding_150_rejected`: Validates HTTP 422 for `biometric_failures = 151`.
5. `test_client_rule_1_biometric_failures_boundary_accepted`: Validates HTTP 200 for exact boundary `biometric_failures = 150`.
6. `test_client_rule_2_working_hours_below_1_5_rejected`: Validates HTTP 422 for `working_hours = 1.4`.
7. `test_client_rule_2_working_hours_boundary_accepted`: Validates HTTP 200 for exact boundary `working_hours = 1.5`.
8. `test_missing_required_field_rejected` (8 parametrized tests): Validates HTTP 422 when each of the 8 required fields is omitted.
9. `test_invalid_data_types_rejected` (6 parametrized tests): Validates HTTP 422 when strings are passed for numeric attributes.
10. `test_prediction_casing_and_whitespace_tolerance`: Validates that area names with irregular casing or padding (e.g., `" Tribal "`) predict successfully.

---

## 8. Files Created or Modified

| File | Status | Description |
|---|---|---|
| `app/__init__.py` | Created | Package initializer for FastAPI application module |
| `app/main.py` | Created | FastAPI prediction service, Pydantic schemas, lifespan model loader, and stdout JSON logger |
| `pytest.ini` | Created | Root pytest configuration defining `pythonpath = src .` and `testpaths = tests` |
| `tests/test_api.py` | Created | 22 comprehensive unit and integration tests using `TestClient` |
| `reports/PART_10_REPORT.md` | Created | Part 10 technical report |
| `G230_MP217/reports/PART_10_REPORT.md` | Created | Synchronized report copy in group distribution folder |

---

## 9. Verification & Quality Assurance

- **Handbook & Problem Rules:** 100% adherence to `PROBLEM.md` Section 5 and `STUDENT_HANDBOOK.md` Sections 5, 9, 11 (Criteria B, F, J).
- **Model Integrity:** The registered champion model in MLflow (`mp217_ration_model@champion`) was loaded and evaluated without retraining or mutating registry aliases.
- **Dataset Integrity:** Original raw datasets and processed files remain completely unmodified with SHA-256 hashes preserved.
- **Next Steps:** The service is fully prepared for containerization (`Dockerfile`, `.dockerignore`), CI workflow configuration, and Kubernetes manifests in subsequent phases.
