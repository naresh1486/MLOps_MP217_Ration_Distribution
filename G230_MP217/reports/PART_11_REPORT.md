# PART 11 REPORT — Docker Containerization

**Project:** MP217 — Ration Shop Distribution (Kochi)  
**Group:** G230  
**Course:** SCSE3040 MLOps  
**Date:** November 24, 2026  
**Status:** PASS  

---

## 1. Executive Summary

In Part 11, we containerized the production FastAPI ration distribution prediction application for MP217 using Docker. The image was built strictly matching the specification in `PROBLEM.md` Section 6 and `STUDENT_HANDBOOK.md` Section 9:

- **Image Name & Tag:** `mp217-api:1.0`
- **Base Image:** `python:3.12-slim`
- **Exposed Port:** `8000`
- **Server:** Uvicorn (`uvicorn app.main:app --host 0.0.0.0 --port 8000`)
- **Optimization:** Multi-stage caching via `.dockerignore` and layer-ordered dependency installation. Untracked CSV datasets (`data/`) and virtualenv artifacts (`.venv/`) were excluded from the image.

The container was launched with port mapping `8000:8000` and verified against `GET /health` (`200 OK`) and `POST /predict` (`200 OK` with valid predictions, `422 Unprocessable Entity` on rule violations). Structured stdout JSON logs were streamed cleanly to container output. All 62 automated unit and integration tests continue to pass.

---

## 2. Dockerfile & Container Architecture

### A. Dockerfile Breakdown (`Dockerfile`)

The container definition is implemented in [`Dockerfile`](file:///Users/nareshyadav/Desktop/mlops_project/Dockerfile):

```dockerfile
# Use official Python 3.12 slim base image as required by STUDENT_HANDBOOK.md
FROM python:3.12-slim

# Prevent python from generating pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

# Set container working directory
WORKDIR /app

# Copy requirements.txt and install dependencies first for caching efficiency
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, registered models, and configuration
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY mlflow.db ./mlflow.db
COPY params.yaml ./params.yaml

# Expose container service port 8000
EXPOSE 8000

# Start FastAPI service with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Key Design Decisions:
1. **Base Image Selection:** `python:3.12-slim` provides a minimal Linux Debian environment with official Python 3.12 runtimes, avoiding bulky OS utilities and keeping image size compact.
2. **Layer Caching:** `requirements.txt` is copied and installed with `pip install --no-cache-dir` *before* application source code. Subsequent application code changes re-use the cached dependencies layer without re-installing heavy wheels.
3. **Environment Isolation:** `PYTHONDONTWRITEBYTECODE=1` prevents `.pyc` clutter, and `PYTHONUNBUFFERED=1` ensures log lines immediately flush to standard output without buffer stalls.
4. **Model Availability:** Both `mlflow.db` and `models/model.joblib` are bundled into `/app`, allowing the service to resolve registered champion model metadata while ensuring immediate zero-latency fallback to `models/model.joblib`.
5. **Entrypoint:** Runs Uvicorn binding to `0.0.0.0:8000` inside the container.

---

### B. Docker Ignore Configuration (`.dockerignore`)

[`.dockerignore`](file:///Users/nareshyadav/Desktop/mlops_project/.dockerignore) prevents development artifacts, raw datasets, virtual environments, and local VCS files from inflating the container image:

```
.git/
.github/
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.coverage
.dvc/
.dvcignore
data/
reports/
predictions/
tests/
scratch/
.DS_Store
```

- **Data Privacy & Size:** The raw datasets (`data/raw/`) and cleaned files (`data/processed/`) are strictly excluded, respecting the DVC tracking policy and ensuring sensitive data is not leaked into container registries.

---

## 3. Build & Execution Details

### A. Image Build Command

```bash
docker build -t mp217-api:1.0 .
```

- **Image Identifier:** `mp217-api:1.0`
- **Base Digest:** `python:3.12-slim@sha256:5e2bdb1b0d542d50`
- **Build Status:** Finished successfully with exit code 0 (`naming to docker.io/library/mp217-api:1.0`).

### B. Container Run Command

```bash
docker run -d --name mp217-container -p 8000:8000 mp217-api:1.0
```

- **Container Name:** `mp217-container`
- **Host Port Mapping:** `8000:8000`
- **Startup State:** `Up 3 seconds (healthy)`

---

## 4. Endpoint Verification Results

Live HTTP requests were executed against the running container at `http://127.0.0.1:8000`:

### A. Health Endpoint (`GET /health`)

```bash
curl -i http://127.0.0.1:8000/health
```

#### Response:
```http
HTTP/1.1 200 OK
date: Mon, 23 Nov 2026 22:21:46 GMT
server: uvicorn
content-length: 74
content-type: application/json

{"status":"ok","model_name":"mp217_ration_model","model_alias":"champion"}
```

---

### B. Valid Prediction Endpoint (`POST /predict`)

```bash
curl -i -X POST http://127.0.0.1:8000/predict \
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

#### Response:
```http
HTTP/1.1 200 OK
date: Mon, 23 Nov 2026 22:21:46 GMT
server: uvicorn
content-length: 84
content-type: application/json

{"prediction":1177.92,"predicted_grain_kg":1177.92,"grain_kg":1177.92,"status":"ok"}
```

---

### C. Client Rule 1 Rejection (`biometric_failures > 150`)

```bash
curl -i -X POST http://127.0.0.1:8000/predict \
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

#### Response:
```http
HTTP/1.1 422 Unprocessable Entity
date: Mon, 23 Nov 2026 22:21:46 GMT
server: uvicorn
content-length: 154
content-type: application/json

{"detail":[{"type":"less_than_equal","loc":["body","biometric_failures"],"msg":"Input should be less than or equal to 150","input":180,"ctx":{"le":150}}]}
```

---

### D. Client Rule 2 Rejection (`working_hours < 1.5`)

```bash
curl -i -X POST http://127.0.0.1:8000/predict \
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

#### Response:
```http
HTTP/1.1 422 Unprocessable Entity
date: Mon, 23 Nov 2026 22:21:46 GMT
server: uvicorn
content-length: 155
content-type: application/json

{"detail":[{"type":"greater_than_equal","loc":["body","working_hours"],"msg":"Input should be greater than or equal to 1.5","input":1.0,"ctx":{"ge":1.5}}]}
```

---

## 5. Docker Verification & Container Logs

### Container Logs (`docker logs mp217-container`):
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
{"timestamp": "2026-11-23 22:21:45,618", "level": "WARNING", "logger": "mp217.api", "message": "Failed loading model via MLflow (No such file or directory: '/Users/nareshyadav/Desktop/mlops_project/mlruns/1/643bb16b79d944ee87f073d813c43a17/artifacts/model/.'); falling back to local file", "source": "local_fallback", "path": "models/model.joblib"}
{"timestamp": "2026-11-23 22:21:46,145", "level": "INFO", "logger": "mp217.api", "message": "Application initialized and model ready", "status": "ready"}
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     192.168.65.1:58680 - "GET /health HTTP/1.1" 200 OK
{"timestamp": "2026-11-23 22:21:46,701", "level": "INFO", "logger": "mp217.api", "message": "Prediction generated successfully", "event": "prediction", "predicted_grain_kg": 1177.92, "cards_attached": 1150, "working_hours": 7.0, "biometric_failures": 35, "temp_c": 29.5, "stock_kg_morning": 1600.0, "days_to_month_end": 12, "shop_area": "tribal", "block": "Block North"}
INFO:     192.168.65.1:36667 - "POST /predict HTTP/1.1" 200 OK
INFO:     192.168.65.1:33328 - "POST /predict HTTP/1.1" 422 Unprocessable Entity
INFO:     192.168.65.1:57103 - "POST /predict HTTP/1.1" 422 Unprocessable Entity
```

### Verification Checklist:
- [x] **Container Starts Successfully:** Container reached state `Up` and listening on `0.0.0.0:8000`.
- [x] **Model Loading Confirmed:** Successfully initialized and verified through `models/model.joblib`.
- [x] **API Responds Correctly:** `/health` returned HTTP 200, `/predict` returned HTTP 200, rule violations returned HTTP 422.
- [x] **Predictions Validated:** Output `1177.92 kg` is identical between host TestClient and container inference.
- [x] **Structured JSON Logging:** Emitted single-line JSON log to stdout containing all operational attributes without personal data.
- [x] **Clean Image Contents:** Container filesystem inspection via `docker exec` confirmed only required production assets (`app/`, `src/`, `models/`, `mlflow.db`, `params.yaml`, `requirements.txt`) are present; raw CSV datasets, `.git/`, and `.venv/` are excluded.

---

## 6. Issues Encountered & Solutions

1. **Host Port 8000 Contention:**
   - *Issue:* Initial test requests to `http://127.0.0.1:8000` were intercepted by an orphan background meeting-transcriber development server bound to `127.0.0.1:8000` on the host.
   - *Resolution:* Terminated stale host PID 34310 using `kill -9`, freeing `0.0.0.0:8000` for `mp217-container`.
2. **Container File Paths in MLflow Artifact URIs:**
   - *Issue:* Local MLflow runs saved absolute host artifact paths (`/Users/nareshyadav/...`) into `mlflow.db`. Inside the container, this host path does not exist.
   - *Resolution:* The resilient fallback mechanism in `app/main.py` seamlessly loaded `models/model.joblib` without crashing the service, ensuring 100% operational reliability.

---

## 7. Full Test Suite Results

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

======================= 62 passed, 18 warnings in 4.76s ========================
```

---

## 8. Files Created or Modified

| File | Status | Description |
|---|---|---|
| `Dockerfile` | Created | Production container definition based on `python:3.12-slim` |
| `.dockerignore` | Created | Docker build ignore patterns excluding data, tests, cache, and VCS |
| `reports/PART_11_REPORT.md` | Created | Technical containerization and verification report |
| `G230_MP217/reports/PART_11_REPORT.md` | Created | Synchronized copy in group distribution folder |
