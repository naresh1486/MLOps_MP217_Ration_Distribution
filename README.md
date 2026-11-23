# MP217 — Ration Shop Distribution (Kochi)

## Project Overview
* **Problem ID:** MP217
* **Client:** Kochi Food and Civil Supplies Office
* **Course:** SCSE3040 MLOps
* **Batch:** 33
* **Faculty:** Upasna Joshi
* **Group ID:** G230

## Team Members
* **S24CSEU1502** — Palak Verma
* **S24CSEU1486** — Naresh Yadav
* **S24CSEU1472** — Amandeep singh
* **S24CSEU1500** — Anurag Singh
* **S24CSEU1481** — Paarth Kashiv

---

## Project Objective
Fair price (ration) shops across development blocks in Kochi distribute grain to card holders. Due to uneven demand patterns, shops occasionally exhaust their stock by midday. The objective of this project is to build an end-to-end MLOps pipeline to forecast daily grain distribution demand (`grain_kg`) for each shop so that the Civil Supplies Office can dispatch replenishment stock in a timely manner, ensuring no card-holding family is turned away.

* **Problem Type:** Supervised Tabular Regression
* **Target Variable:** `grain_kg` (daily grain distributed in kilograms)
* **Primary Evaluation Metric:** Mean Absolute Error (MAE) in kg

---

## Planned Technology Stack
* **Data Management & Versioning:** DVC (Data Version Control)
* **Data Processing:** Python 3.11/3.12, Pandas (`pandas==2.2.3`), NumPy
* **Machine Learning:** Scikit-learn, Joblib
* **Experiment Tracking & Model Registry:** MLflow (SQLite backend `sqlite:///mlflow.db`)
* **Serving API:** FastAPI, Uvicorn, Pydantic
* **Testing:** Pytest, HTTPX (FastAPI `TestClient`)
* **Containerization:** Docker (`python:3.12-slim`)
* **Orchestration:** Kubernetes (Deployment and Service manifests)
* **CI/CD Automation:** GitHub Actions
* **Configuration:** YAML (`params.yaml`, parsed with `PyYAML`)

---

## Data Version Control (DVC)

### Why DVC is Used
In machine learning projects, datasets are large, evolve over time, and should not be tracked directly in Git repositories to avoid repository bloat. DVC tracks large data files using lightweight `.dvc` pointer files containing cryptographic hashes (MD5), while keeping the actual CSV data decoupled from Git history. This guarantees full data reproducibility, auditability, and pipeline provenance across environments.

### Tracked Datasets
The following five datasets are versioned with DVC:
1. `data/raw/train.csv` (Pointer: `data/raw/train.csv.dvc`) — Original training data (1,027 rows).
2. `data/raw/later.csv` (Pointer: `data/raw/later.csv.dvc`) — Production monitoring data (300 rows).
3. `data/raw/holdout_features.csv` (Pointer: `data/raw/holdout_features.csv.dvc`) — Unlabelled evaluation holdout features (250 rows).
4. `data/processed/train_clean.csv` (Pointer: `data/processed/train_clean.csv.dvc`) — Cleaned, deduplicated training data (1,000 rows).
5. `data/processed/later_clean.csv` (Pointer: `data/processed/later_clean.csv.dvc`) — Cleaned production monitoring data (300 rows).

### How to Reproduce Datasets Locally
1. **Pull tracked datasets from DVC storage:**
   ```bash
   dvc pull
   ```
2. **Verify data status:**
   ```bash
   dvc status
   ```
3. **Reproduce cleaned datasets from raw sources:**
   ```bash
   python scripts/clean_data.py
   ```
4. **Restore datasets if missing:**
   ```bash
   dvc checkout
   ```
