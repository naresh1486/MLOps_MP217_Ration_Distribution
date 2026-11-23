# PART 02 — Project Structure and Environment Setup Report

**Project:** MP217 — Ration Shop Distribution (Kochi)  
**Group:** G230  
**Batch:** 33  
**Faculty:** Upasna Joshi  
**Course:** SCSE3040 MLOps  
**Timestamp:** 2026-11-23T23:53:30+05:30  
**Status:** PASS — All Part 02 requirements completed  

---

## 1. Files and Folders Created

In accordance with the repository structure specified in `STUDENT_HANDBOOK.md` and Part 02 instructions, the following directories and files were created while preserving all original project documentation files:

```
/Users/nareshyadav/Desktop/mlops_project/
├── README.md                      # Project title, G230 group members, problem overview & tech stack
├── requirements.txt               # Minimal pinned dependencies (pandas==2.2.3, etc.)
├── params.yaml                    # Simple central configuration file
├── src/
│   └── mp217/
│       ├── __init__.py            # Package entry point and version metadata
│       ├── config.py              # Configuration loader with safe YAML parsing and error handling
│       └── data_utils.py          # Data ingestion utility with error handling
├── tests/
│   └── .gitkeep                   # Prepared test directory
├── reports/
│   ├── PART_01_REPORT.md          # Part 01 inspection report
│   └── PART_02_REPORT.md          # Part 02 environment & setup report (this file)
├── models/
│   └── .gitkeep                   # Prepared directory for serialized models
├── predictions/
│   └── .gitkeep                   # Prepared directory for predictions
├── app/
│   └── .gitkeep                   # Prepared directory for FastAPI service
├── k8s/
│   └── .gitkeep                   # Prepared directory for Kubernetes manifests
├── .github/
│   └── workflows/
│       └── .gitkeep               # Prepared directory for GitHub Actions CI
├── data/
│   └── raw/                       # Staging area for raw data files for future DVC tracking
│       ├── train.csv              # Exact replica of original training dataset
│       ├── later.csv              # Exact replica of production monitoring dataset
│       └── holdout_features.csv   # Exact replica of holdout features dataset
└── .venv/                         # Isolated Python virtual environment
```

*Note: All original project documentation files (`G230_MP217/GROUP.md`, `G230_MP217/PROBLEM.md`, `G230_MP217/STUDENT_HANDBOOK.md`, and `G230_MP217/findings.yaml`) have been strictly preserved in place.*

---

## 2. Dependencies and Versions Installed

A minimal, compatible set of dependencies was pinned in `requirements.txt` and installed into the dedicated `.venv` virtual environment:

| Library Category | Package Name | Target Pinned Version | Installed Version | Purpose |
|---|---|---|---|---|
| **Data Processing** | `pandas` | `2.2.3` | **2.2.3** | Core tabular manipulation (matches Student Handbook specification) |
| | `numpy` | `2.0.2` | **2.0.2** | Numerical and array computation |
| **Machine Learning** | `scikit-learn` | `1.6.1` | **1.6.1** | Regression algorithms and preprocessing pipelines |
| | `joblib` | `1.4.2` | **1.4.2** | Model serialization and disk persistence |
| **Tracking** | `mlflow` | `2.20.1` | **2.20.1** | Experiment tracking, SQLite backend logging, model registry |
| **Configuration** | `PyYAML` | `6.0.2` | **6.0.2** | Safe YAML configuration parsing (`yaml.safe_load`) |
| **Data Versioning** | `dvc` | `3.59.0` | **3.59.0** | Data version control and `.dvc` tracking |
| **Serving API** | `fastapi` | `0.115.6` | **0.115.6** | High-performance REST API for inference |
| | `uvicorn` | `0.34.0` | **0.34.0** | ASGI web server for FastAPI |
| | `pydantic` | `2.10.4` | **2.10.4** | Request schema validation and business rule enforcement |
| **Testing** | `pytest` | `8.3.4` | **8.3.4** | Automated test suite execution |
| | `httpx` | `0.28.1` | **0.28.1** | HTTP test client for FastAPI integration testing |

No unnecessary or unpinned dependencies were installed.

---

## 3. Virtual Environment Status

* **Virtual Environment Path:** `/Users/nareshyadav/Desktop/mlops_project/.venv`
* **Python Executable:** `/Users/nareshyadav/Desktop/mlops_project/.venv/bin/python`
* **Python Version:** `Python 3.11.9`
* **Pip Version:** `24.0`
* **Environment Status:** Active, fully isolated from global system site-packages, with all 12 core libraries and their required sub-dependencies successfully installed.

---

## 4. Configuration Loading Test Results

The configuration module `src/mp217/config.py` was executed and tested using `load_config()`:

* **File Tested:** `params.yaml`
* **Parsing Method:** `yaml.safe_load` via standard library `open` context manager.
* **Keys Loaded:** `['paths', 'target_column', 'split', 'model', 'mlflow']`
  * `paths.train_data`: `"data/raw/train.csv"`
  * `paths.later_data`: `"data/raw/later.csv"`
  * `paths.holdout_data`: `"data/raw/holdout_features.csv"`
  * `paths.model_path`: `"models/model.joblib"`
  * `paths.holdout_predictions`: `"predictions/holdout_predictions.csv"`
  * `target_column`: `"grain_kg"`
  * `split.random_seed`: `42`
  * `split.test_size`: `0.2`
  * `model.algorithm`: `"RandomForestRegressor"`
  * `model.n_estimators`: `100`
  * `model.max_depth`: `10`
  * `mlflow.tracking_uri`: `"sqlite:///mlflow.db"`
  * `mlflow.experiment_name`: `"mp217_ration_distribution"`
  * `mlflow.registered_model_name`: `"mp217_ration_model"`
* **Error Handling Test:** Attempting to load an invalid or missing path (`non_existent_params.yaml`) properly raised `FileNotFoundError`:
  * *Result:* `PASS: Configuration file not found at: non_existent_params.yaml. Please ensure params.yaml exists in the project root.`

---

## 5. Data Utility Test Results

The data utility module `src/mp217/data_utils.py` was tested using `load_data()`:

* **Test Sample Ingestion:** `data_utils.load_data("data/raw/train.csv", nrows=5)`
  * **Resulting DataFrame Shape:** `(5, 11)`
  * **Columns Returned:** `['record_id', 'date', 'cards_attached', 'working_hours', 'biometric_failures', 'temp_c', 'stock_kg_morning', 'days_to_month_end', 'shop_area', 'block', 'grain_kg']`
  * **Data Integrity Sample:**
    ```
      record_id        date  cards_attached  working_hours  biometric_failures  temp_c  stock_kg_morning  days_to_month_end shop_area        block  grain_kg
    0    R00001  2026-01-01            1811            6.0                  42    27.8            2721.0                 30    tribal  Block North    1302.1
    1    R00002  2026-01-01            1142            8.1                  15    37.1             579.0                 30    tribal  Block North     938.5
    ```
* **Error Handling Test:** Calling `load_data("data/raw/non_existent.csv")` properly raised `FileNotFoundError`:
  * *Result:* `PASS: Dataset file not found at: /Users/nareshyadav/Desktop/mlops_project/data/raw/non_existent.csv. Please verify the dataset path.`

---

## 6. Package Import Test Results

The Python package `src/mp217` was verified for importability:

* `import mp217` $\rightarrow$ Version `0.1.0` verified.
* `from mp217 import config` $\rightarrow$ Module path `/Users/nareshyadav/Desktop/mlops_project/src/mp217/config.py` verified.
* `from mp217 import data_utils` $\rightarrow$ Module path `/Users/nareshyadav/Desktop/mlops_project/src/mp217/data_utils.py` verified.
* All top-level dependencies (`pandas`, `numpy`, `sklearn`, `joblib`, `mlflow`, `yaml`, `dvc`, `fastapi`, `uvicorn`, `pydantic`, `pytest`, `httpx`) successfully imported with zero deprecation warnings or binary incompatibility issues.

---

## 7. Errors and Unresolved Issues

* **Errors Encountered:** None. All packages installed cleanly, and all utility functions performed as expected.
* **Unresolved Issues:** None.

---

## 8. Confirmation of Dataset Immutability

The original CSV dataset files residing in `G230_MP217/data/` were audited and confirmed to be completely unmodified. Their SHA-256 cryptographic hashes match the exact hashes verified in Part 01:

| Original File Path | Expected SHA-256 Hash | Current SHA-256 Hash | Status |
|---|---|---|---|
| `G230_MP217/data/train.csv` | `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` | `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` | **UNCHANGED & VERIFIED** |
| `G230_MP217/data/later.csv` | `966d0c8e759b263250febeaa45b365b28470d2047bb7e6417a21921f40603d59` | `966d0c8e759b263250febeaa45b365b28470d2047bb7e6417a21921f40603d59` | **UNCHANGED & VERIFIED** |
| `G230_MP217/data/holdout_features.csv` | `aa6d44b46bb337d52b6cfc3102f2d7dc3f94fddabad4171c2b3af36e964d1310` | `aa6d44b46bb337d52b6cfc3102f2d7dc3f94fddabad4171c2b3af36e964d1310` | **UNCHANGED & VERIFIED** |

---

## 9. Distinction: Tested vs Untested Functionality

* **Successfully Tested:**
  * Virtual environment instantiation and pip resolution of pinned libraries.
  * Importing all core dependencies (`pandas==2.2.3`, `scikit-learn`, `mlflow`, `dvc`, `fastapi`, `pydantic`, `pytest`, etc.).
  * Package structure and import of `src.mp217`.
  * Safe YAML parsing of `params.yaml` and missing-file error handling.
  * Ingestion of raw CSV data via `load_data()` with row sampling and missing-file error handling.
  * Cryptographic integrity verification of all three original datasets.
* **Untested (Intentionally Deferred per Part 02 Restrictions):**
  * Data cleaning logic (duplicate removal, unit conversion, typo correction, spelling normalization).
  * Feature transformations and encoding.
  * Model training, MLflow experiment logging, and model registry.
  * FastAPI endpoints (`POST /predict`, `GET /health`) and Pydantic validation rules.
  * Docker image building and Kubernetes manifest application.
  * DVC tracking initialization (`dvc init`).
  * GitHub repository initialization and remote pushes.

---

## Final Status

**PASS — All Part 02 requirements completed.**
