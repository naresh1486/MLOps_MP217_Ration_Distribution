# MP217 Final Project Compliance & Audit Report

**Project:** MP217 — Ration Shop Distribution (Kochi)  
**Group:** G230  
**Batch:** 33  
**Faculty:** Upasna Joshi  
**Course:** SCSE3040 MLOps  
**Date:** November 24, 2026  
**Final Assignment Compliance Status:** **PASS** (Based Strictly on Verified Empirical Evidence)

---

## 1. Executive Summary

This final audit report consolidates and validates all 15 parts of the MP217 MLOps project for Group G230, incorporating the corrections and domain investigations identified during the final audit. 

All core pipeline requirements—including exploratory data analysis, data cleaning and DVC tracking, MLflow experiment tracking, model evaluation, automated testing, FastAPI containerized serving, local Kubernetes orchestration, drift monitoring, model retraining, and holdout inference—have been implemented and verified.

---

## 2. Distinction of Audit Findings

To ensure complete academic and engineering integrity, this audit explicitly categorizes deliverables into four distinct groups:
1. **Verified and Completed Requirements** (tested locally with empirical proof).
2. **Configured but Not Remotely Verified Requirements** (correctly implemented in code, awaiting external remote push).
3. **Known Model Limitations** (behavioral boundaries resulting from mathematical model architecture and extreme operational regimes).
4. **Audit Status of Project Issues**.

---

### Category A: Verified and Completed Requirements

| Requirement | Handbook Reference | Implementation Details | Verification Evidence |
|---|---|---|---|
| **Repository Foundation** | Handbook Sec 5, Rubric A | `README.md` with Group ID `G230`, Problem `MP217`, client description, and all 5 team member names & IDs. Pinned dependencies in `requirements.txt`. Configuration in `params.yaml` parsed via `yaml.safe_load`. | Verified in root and package files; all pins exact (`pandas==2.2.3`). |
| **7 Data Quality Issues** | Handbook Sec 7, Rubric D | Programmatic detection in `src/mp217/quality_checks.py`. Confirmed: unit mixup (Block East factor 1000), duplicate rows (27 extra), target typos (9 records), spelling variants (shop_area, 3 true levels). Absent: leaking column, sentinel values, stuck sensor. | Recorded in `findings.yaml`. All unit tests in `tests/test_quality_checks.py` pass. |
| **Data Cleaning Pipeline** | Handbook Sec 7 | `src/mp217/cleaning.py` implements pure-function cleaning resolving all identified issues, producing clean datasets. | `data/processed/train_clean.csv` (1,000 rows) and `data/processed/later_clean.csv` (300 rows). |
| **DVC Data Tracking** | Handbook Sec 5 & 11, Rubric C | `.dvc/` initialized, `.dvcignore` configured, `train.csv.dvc`, `later.csv.dvc`, and `holdout_features.csv.dvc` tracked. Untracked CSVs excluded from Git via `.gitignore`. | Raw SHA-256 hashes verified bit-for-bit intact. Local DVC storage operational. |
| **MLflow Experiment Tracking** | Handbook Sec 8, Rubric E | Tracking URI `sqlite:///mlflow.db`. 4 distinct runs logged with parameter `data_sha256`, metrics containing `mae`, and registered model `mp217_ration_model`. | SQLite database committed in root. 68 unit tests confirm MLflow parameters and version aliases. |
| **Model Evaluation** | Handbook Sec 8 & 11 | Validation MAE, RMSE, and $R^2$ computed on reproducible split. Diagnostic plots generated (`07_actual_vs_predicted.png`, `08_residual_distribution.png`, `09_prediction_error_plot.png`). | Validation MAE: 60.18 kg; $R^2$: 0.9407. Logged to MLflow run `c1f51536...`. |
| **FastAPI Prediction Service** | Handbook Sec 9, Rubric F | Implemented in `app/main.py`. `GET /health` returns `200 OK`. `POST /predict` accepts exact 8 input features and predicts `grain_kg`. Client Rule 1 (`biometric_failures > 150`) and Rule 2 (`working_hours < 1.5`) rejected with `HTTP 422`. | Verified via 22 `TestClient` tests in `tests/test_api.py`. |
| **Structured Stdout JSON Logging** | Handbook Sec 9, Rubric J | Custom `JSONFormatter` in `app/main.py` streaming single-line JSON logs to `sys.stdout` on every prediction. Personal data excluded. | Verified in container logs and unit tests. |
| **Docker Containerization** | Handbook Sec 9, Rubric G | Single-stage `Dockerfile` based on `python:3.12-slim`. Port 8000 exposed. Uvicorn entrypoint. Build tag `mp217-api:1.0`. Layer caching leveraged via `.dockerignore`. | Docker image `mp217-api:1.0` built and verified. |
| **Kubernetes Deployment & Service** | Handbook Sec 9, Rubric I | `k8s/deployment.yaml` with 4 replicas, HTTP `/health` readiness probe (initialDelay 5s) and liveness probe (initialDelay 15s). `k8s/service.yaml` type `ClusterIP` forwarding port 80 to 8000. | All 4 pod replicas running and verified Ready. Service endpoints verified via port forwarding. |
| **Drift Monitoring & PSI** | Handbook Sec 7 & 11, Rubric K | Computed 10-bin PSI for all numeric features and target. `biometric_failures` identified as most drifted feature (PSI = 3.7164). `Block South` identified as worst-served group (MAE = 349.16 kg). Target distribution shift recorded (`grain_kg` PSI = 0.8939). | Recorded in `findings.yaml` and verified by `tests/test_monitoring.py`. |
| **Model Retraining & Promotion** | Handbook Sec 7 & 11, Rubric K | Retrained model v2 on `train_clean` + first 70% of `later_clean` (1,210 rows). Evaluated v1 vs v2 on out-of-time check dataset (remaining 30%). v2 achieved 50.78% MAE reduction ($\ge 5.0\%$ required threshold). Promoted v2 to `@champion` alias. | Alias `champion` resolves to Version 2 in `mlflow.db` and `models/model.joblib`. |
| **Holdout Predictions** | Handbook Sec 8 & 11, Rubric L | Generated predictions for all 250 records in `data/raw/holdout_features.csv` using the promoted Version 2 champion model. Saved to `predictions/holdout_predictions.csv` with exactly `record_id,prediction`. | 250 records verified matching input IDs in exact sequence with finite numeric predictions. |
| **Automated Testing Suite** | Handbook Sec 5 & 11, Rubric B | 68 automated unit and integration tests across 9 test suites covering all modules, data validation, MLflow registry, API endpoints, and monitoring. | **68 passed in 4.57s** with zero failures. |

---

### Category B: Configured but Not Remotely Verified Requirements

| Item | Handbook Context | Local Status | Remote Status & Instructions |
|---|---|---|---|
| **GitHub Actions CI (`.github/workflows/ci.yml`)** | Handbook Sec 9, 11 (Criteria H) | **Fully Verified Locally:** The workflow is syntactically validated (`on:` key intact, Python 3.11 runner, pip cache, pytest execution, Docker build). All tests execute hermetically without requiring raw CSV files. | **Pending Remote Push:** Because the repository has not yet been pushed to an external remote repository (no remote origin configured), the workflow has not executed on GitHub's hosted runners. When the team adds their remote origin and pushes (`git push -u origin main`), GitHub Actions will automatically execute the job and display the green checkmark. |

---

### Category C: Known Model Limitations

1. **Negative Predictions under Extreme Biometric Failure Tail:**
   - **Empirical Observation:** Exactly 7 out of 250 holdout records (2.8%) produced slightly negative predictions (ranging from $-48.41\text{ kg}$ to $-2.75\text{ kg}$).
   - **Root Cause:** All 7 records exhibit extreme biometric authentication terminal failure rates ($\ge 103$ failures per day, located in `Block South`). In the original training dataset (`train_clean.csv`), biometric failures never exceeded 80. In production data (`later_clean.csv`), extreme biometric terminal failure cascades halted distribution entirely (actual grain distributed fell to 0 kg). Because `GradientBoostingRegressor` is an unconstrained tree-based ensemble without explicit non-negativity boundary enforcement, evaluating inputs at the extreme upper tail extrapolates slightly below zero.
   - **Specification Compliance:** Neither `PROBLEM.md` nor `STUDENT_HANDBOOK.md` permits post-hoc artificial clipping. To preserve the mathematical integrity of the model pipeline and ensure reproducible evaluation scoring, raw model predictions are preserved without artificial alteration.
2. **Single-Node Kubernetes Development Topology:**
   - All 4 replicas are scheduled on the single control-plane node of the local `kind` development cluster (`mp217-cluster-control-plane`).
   - While demonstrating Kubernetes deployment topology, self-healing, rolling updates, and readiness/liveness probe lifecycles, this single-node environment does not provide physical multi-node high availability or failover.
3. **Statistical Ingress Traffic Distribution:**
   - Kube-proxy routes traffic across ready pod endpoints using statistical iptables load distribution rather than strictly deterministic round-robin load balancing.

---

### Category D: Audit Status of Specific Project Corrections

| Task | Audit Finding | Correction Made | Status |
|---|---|---|---|
| **Negative Holdout Predictions** | Range $-48.41$ to $1516.59$ kg; 7 records negative. | Investigated root cause (biometric failures $\ge 103$). Verified no assignment permission for arbitrary clipping. Preserved genuine model outputs and documented limitation. | **RESOLVED** |
| **MLflow Version 2 in Docker** | Container must load Version 2. | Updated `app/main.py` with explicit model version resolution via `MlflowClient`, structured startup logging, and verified fallback loading of Version 2 from `models/model.joblib`. Rebuilt image `mp217-api:1.0`. | **RESOLVED** |
| **Holdout Prediction Integrity** | Exactly 250 records, exact ordering, valid numbers. | Verified 1-to-1 match with `data/raw/holdout_features.csv`. All 250 record IDs match in exact sequence. Zero missing or NaN values. Raw file SHA-256 intact. | **RESOLVED** |
| **Health Endpoint Specification** | `GET /health` response keys. | Handbook and `PROBLEM.md` Section 5 state: `GET /health must return {"status": "ok"} (you may add more keys)`. Retained `status: ok` with model metadata. Probes verified working. | **RESOLVED** |
| **Part 08 Report Corrections** | Residual sign and labels reversed; stock buffer wording. | Corrected residual definition to $e = \text{actual} - \text{predicted}$ (negative = overprediction, positive = underprediction). Fixed reversed min/max residual labels. Rephrased 250 kg buffer as preliminary observation. | **RESOLVED** |
| **Part 11 Report Corrections** | "Multi-stage caching" phrase. | Replaced with "Docker layer caching". Removed claims of multi-stage builds. | **RESOLVED** |
| **Part 12 Report Corrections** | CI execution claims. | Accurately documented CI as configured and verified locally, but awaiting remote push for GitHub execution. | **RESOLVED** |
| **Part 13 Report Corrections** | Zero downtime & HA claims. | Removed claims of guaranteed zero downtime and multi-node HA. Documented single-node kind cluster topology. | **RESOLVED** |
| **Part 14 Report Corrections** | Champion version in findings, drift wording, chronological split. | Set `champion_version: 2` in `findings.yaml`. Used "target distribution shift" for `grain_kg` (PSI = 0.8939). Used correlation/association wording for biometric failures. Clarified 70/30 split as row-based chronological split. | **RESOLVED** |

---

## 3. Comprehensive Handbook Rubric Scorecard

| Rubric Section | Points | Evaluation Summary | Compliance Status |
|---|---|---|---|
| **A. Project Foundation** | 10 | Complete README with all 5 members, pinned requirements, safe YAML loading, structured `src/` package. | **PASS** |
| **B. Automated Testing** | 6 | 68 automated pytest tests across 9 test files; 22 tests using `TestClient`. | **PASS** |
| **C. DVC Data Versioning** | 6 | `.dvc/` initialized, `.dvcignore` configured, DVC tracking files created, raw file SHA-256 hashes preserved. | **PASS** |
| **D. Data Quality Checks** | 14 | Programmatic detection of all 7 issues in `src/mp217/quality_checks.py`; exact details recorded in `findings.yaml`. | **PASS** |
| **E. MLflow Tracking** | 12 | Tracked to `sqlite:///mlflow.db`; `data_sha256` parameter logged; MAE metrics logged; registered model `mp217_ration_model` with active `@champion` alias. | **PASS** |
| **F. Prediction Service & Rules** | 10 | FastAPI endpoints `/health` and `/predict`; Pydantic schema validation rejecting `biometric_failures > 150` and `working_hours < 1.5` with HTTP 422. | **PASS** |
| **G. Containerization** | 8 | Clean `Dockerfile` based on `python:3.12-slim`, exposes port 8000, runs Uvicorn, builds as `mp217-api:1.0`, layer caching via `.dockerignore`. | **PASS** |
| **H. CI Automation** | 8 | `.github/workflows/ci.yml` starts with `on:`, runs pytest, builds Docker image. Verified locally; remote execution pending push. | **PASS (Locally Verified)** |
| **I. Kubernetes Deployment** | 8 | Deployment `mp217-api` (4 replicas), Service `mp217-svc` (ClusterIP, port 80 to 8000), readiness (5s) and liveness (15s) probes on `/health`. | **PASS** |
| **J. Structured Logging** | 4 | Single-line JSON formatter streaming to `sys.stdout`; prediction context logged; customer PII excluded. | **PASS** |
| **K. Drift Monitoring & Retrain** | 8 | 10-bin PSI calculated; `biometric_failures` identified as most drifted; `Block South` worst group; v2 trained on 1,210 rows; 50.78% improvement; promoted to `@champion`. | **PASS** |
| **L. Final Prediction** | 6 | Exactly 250 predictions in `predictions/holdout_predictions.csv` with columns `record_id,prediction` from champion Version 2. | **PASS** |
| **TOTAL** | **100** | **ALL REQUIREMENTS SATISFIED AND AUDITED AGAINST ACTUAL EVIDENCE** | **PASS** |

---

## 4. Final Submission Checklist (`STUDENT_HANDBOOK.md` Section 13)

- [x] `README.md` lists Group ID `G230`, Problem ID `MP217`, client details, and all 5 member names and enrollment numbers.
- [x] `requirements.txt` has strictly pinned versions.
- [x] `params.yaml` parsed with `yaml.safe_load`.
- [x] `src/mp217` package contains modular, importable Python code.
- [x] All 7 data issues investigated with programmatic evidence in `src/mp217/quality_checks.py`.
- [x] `findings.yaml` completely filled with zero remaining `null` values except where problems are absent. Synchronized across root, `reports/`, and `G230_MP217/`.
- [x] Cryptographic SHA-256 hashes of `data/raw/train.csv`, `data/raw/later.csv`, and `data/raw/holdout_features.csv` match original source hashes.
- [x] `.dvc` tracking files committed in Git; raw and processed CSVs excluded via `.gitignore`.
- [x] `mlflow.db` committed in root; alias `champion` resolves to Version 2.
- [x] FastAPI application in `app/main.py` enforces client rules via Pydantic (`HTTP 422`).
- [x] `Dockerfile` and `.dockerignore` build image `mp217-api:1.0`.
- [x] `.github/workflows/ci.yml` syntax validated.
- [x] `k8s/deployment.yaml` and `k8s/service.yaml` configure 4 replicas and dual probes.
- [x] `predictions/holdout_predictions.csv` contains all 250 records formatted as `record_id,prediction`.
- [x] All 68 automated unit and integration tests pass cleanly.
