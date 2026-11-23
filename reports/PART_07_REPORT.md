# PART 07 — MLflow Experiment Tracking and Model Training Report

**Project:** MP217 — Ration Shop Distribution (Kochi)  
**Group:** G230  
**Batch:** 33  
**Faculty:** Upasna Joshi  
**Course:** SCSE3040 MLOps  
**Timestamp:** 2026-11-24T03:11:00+05:30  
**Status:** PASS — MLflow tracking and v1 model registration completed  

---

## 1. Training Dataset Used

* **Source File:** `data/processed/train_clean.csv`
* **Dataset Characteristics:**
  * Total Rows: **1,000** (clean, deduplicated)
  * Total Columns: **11**
  * Missing Values: **0**
  * Duplicate Rows: **0**
  * Target Units: Kilograms ($\text{kg}$), normalized distribution ($\text{min} = 145.0\text{ kg}$, $\text{max} = 1,864.0\text{ kg}$, $\text{mean} = 1,060.52\text{ kg}$)

---

## 2. Features and Target

* **Target Variable:** `grain_kg` (continuous numerical target in kilograms).
* **Model Features (Exactly 8 features used):**
  * **Numerical Features (6):** `cards_attached`, `working_hours`, `biometric_failures`, `temp_c`, `stock_kg_morning`, `days_to_month_end`.
  * **Categorical Features (2):** `shop_area`, `block`.
* **Explicitly Excluded Columns:**
  * `record_id` (unique transaction identifier; omitted to prevent overfitting).
  * `date` (temporal timestamp; omitted from `/predict` operational schema).

---

## 3. Train-Validation Split

* **Configuration Source:** Loaded dynamically from `params.yaml` via `src/mp217/config.py`.
* **Random Seed (`random_seed`):** `42`
* **Validation Split Ratio (`test_size`):** `0.20` (20%)
* **Subsample Shapes:**
  * Training Subset: **800 rows** × 8 feature columns
  * Validation Subset: **200 rows** × 8 feature columns
* **Determinism:** Verified through unit testing; splitting produces identical train/validation indices across repeated runs.

---

## 4. Model Configurations Tested

Three distinct regression architectures were evaluated using scikit-learn pipelines with standardized preprocessing:
* **Preprocessing Pipeline:**
  * Numerical: `StandardScaler()` applied to all 6 numeric features.
  * Categorical: `OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)` applied to `shop_area` and `block`.
* **Candidate Regressors:**
  1. `RandomForestRegressor`: `n_estimators=100`, `max_depth=10`, `random_state=42`.
  2. `ExtraTreesRegressor`: `n_estimators=100`, `max_depth=12`, `random_state=42`.
  3. `GradientBoostingRegressor`: `n_estimators=100`, `learning_rate=0.1`, `max_depth=4`, `random_state=42`.

---

## 5. Actual Validation MAE for Each Run

All models were evaluated on the held-out validation set ($N=200$). The actual validation Mean Absolute Errors are:

| Model Architecture | MLflow Run Name | Validation MAE (kg) | Primary Metric (`mae`) | Rank |
|---|---|---|---|---|
| **GradientBoostingRegressor** | `v1_candidate_GradientBoostingRegressor` | **60.1811 kg** | **60.1811** | **1 (Best)** |
| **ExtraTreesRegressor** | `v1_candidate_ExtraTreesRegressor` | **70.0752 kg** | **70.0752** | 2 |
| **RandomForestRegressor** | `v1_candidate_RandomForestRegressor` | **76.8753 kg** | **76.8753** | 3 |

*Metric Naming Compliance:* Every logged metric key contains the required substring `"mae"` (`val_mae` and `mae`).

---

## 6. MLflow Configuration & Tracking URI

In compliance with `STUDENT_HANDBOOK.md` (Section 8):
* **Tracking Backend:** SQLite database located in repository root: `sqlite:///mlflow.db`.
* **Experiment Name:** `mp217_ration_distribution`.
* **Experiment ID:** `1`.
* **Database File Status:** `mlflow.db` is tracked and committed in Git.

---

## 7. MLflow Run IDs

Direct query from the MLflow SQLite store:

| Run ID | Model Name | Status | Artifact URI |
|---|---|---|---|
| `643bb16b79d944ee87f073d813c43a17` | `GradientBoostingRegressor` | `FINISHED` | `mlflow-artifacts:/1/643bb16b79d944ee87f073d813c43a17/artifacts` |
| `2a539532be754b238e3fbb03eadb0a39` | `ExtraTreesRegressor` | `FINISHED` | `mlflow-artifacts:/1/2a539532be754b238e3fbb03eadb0a39/artifacts` |
| `0cac2c9aff434631b14191d82e995d23` | `RandomForestRegressor` | `FINISHED` | `mlflow-artifacts:/1/0cac2c9aff434631b14191d82e995d23/artifacts` |

---

## 8. Confirmation of `data_sha256` Parameter in Every Run

As mandated by rubric Section E, the SHA-256 hash of the **original uncleaned** `data/raw/train.csv` was computed directly from binary bytes:

* **Expected Original Hash:** `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a`
* **Computed Hash:** `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` (Exact match)
* **MLflow Run Verification:**
  * Run `643bb16b79d944ee87f073d813c43a17` $\rightarrow$ `params.data_sha256 = "d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a"` (Verified)
  * Run `2a539532be754b238e3fbb03eadb0a39` $\rightarrow$ `params.data_sha256 = "d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a"` (Verified)
  * Run `0cac2c9aff434631b14191d82e995d23` $\rightarrow$ `params.data_sha256 = "d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a"` (Verified)

---

## 9. Best Model Selection & Justification

* **Selected Model:** `GradientBoostingRegressor` (Run ID: `643bb16b79d944ee87f073d813c43a17`)
* **Selection Metric:** Lowest validation MAE of **60.1811 kg**, outperforming `ExtraTreesRegressor` (70.08 kg) by **14.1%** and `RandomForestRegressor` (76.88 kg) by **21.7%**.
* **Model Characteristics:** Boosting sequentially corrects additive residual errors on the non-linear interaction between `biometric_failures` and ration shop demand.

---

## 10. Registered Model Name and Version

* **Registered Model Name:** `mp217_ration_model`
* **Registered Version:** **Version 1** (`v1`)
* **Source Run ID:** `643bb16b79d944ee87f073d813c43a17`
* **Status:** `READY`

---

## 11. Champion Alias Verification

The model alias `@champion` was registered through `MlflowClient.set_registered_model_alias()`:

* **Registered Model:** `mp217_ration_model`
* **Alias Name:** `champion`
* **Target Version:** `1`
* **Verification Query:**
  ```python
  client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
  mv = client.get_model_version_by_alias("mp217_ration_model", "champion")
  assert mv.version == "1"
  assert mv.run_id == "643bb16b79d944ee87f073d813c43a17"
  ```
  *Result: Successfully verified that `@champion` points to version 1.*

---

## 12. Saved Model Path and Persistence Verification

* **Destination Path:** `models/model.joblib` (File size: 330 KB).
* **Serialization Method:** `joblib.dump(pipeline, "models/model.joblib")`.
* **Deserialization Test:** `joblib.load("models/model.joblib")`.
* **Inference Test on Validation Sample:**
  ```python
  sample_input = X_val.iloc[:2]
  preds = loaded_model.predict(sample_input)
  # Returns: [1291.28 kg, 878.43 kg]
  ```
  *Result: Deserialization and prediction executed cleanly with 0 errors.*

---

## 13. Unit Test Results

The test suite in `tests/test_train.py` was executed alongside all previous tests via `pytest`:

```
============================= test session starts ==============================
platform darwin -- Python 3.11.9, pytest-8.3.4, pluggy-1.7.0
rootdir: /Users/nareshyadav/Desktop/mlops_project
plugins: platformdirs-4.14.8, hydra-core-1.3.8, anyio-4.16.0
collected 21 items

tests/test_cleaning.py::test_remove_duplicates PASSED                    [  4%]
tests/test_cleaning.py::test_fix_unit_mixup PASSED                       [  9%]
tests/test_cleaning.py::test_fix_target_typos PASSED                     [ 14%]
tests/test_cleaning.py::test_normalize_spelling PASSED                   [ 19%]
tests/test_cleaning.py::test_preservation_of_unaffected_values PASSED    [ 23%]
tests/test_cleaning.py::test_clean_train_data_shape_and_validation PASSED [ 28%]
tests/test_cleaning.py::test_clean_later_data PASSED                     [ 33%]
tests/test_cleaning.py::test_immutability_of_input_dataframe PASSED      [ 38%]
tests/test_quality_checks.py::test_detect_leaking_column PASSED          [ 42%]
tests/test_quality_checks.py::test_detect_unit_mixup PASSED              [ 47%]
tests/test_quality_checks.py::test_detect_sentinel_values PASSED         [ 52%]
tests/test_quality_checks.py::test_detect_duplicate_rows PASSED          [ 57%]
tests/test_quality_checks.py::test_detect_stuck_sensor PASSED            [ 61%]
tests/test_quality_checks.py::test_detect_target_typos PASSED            [ 66%]
tests/test_quality_checks.py::test_detect_spelling_variants PASSED       [ 71%]
tests/test_quality_checks.py::test_quality_checks_do_not_modify_inputs PASSED [ 76%]
tests/test_train.py::test_compute_file_sha256 PASSED                     [ 80%]
tests/test_train.py::test_prepare_features_and_target PASSED             [ 85%]
tests/test_train.py::test_reproducible_split PASSED                      [ 90%]
tests/test_train.py::test_build_pipeline_and_prediction PASSED           [ 95%]
tests/test_train.py::test_saved_model_loading_and_prediction PASSED      [100%]

======================= 21 passed in 2.27s ========================
```

---

## 14. Confirmation of Dataset Immutability

Cryptographic SHA-256 hashes of all raw datasets were re-verified:

| Dataset File Path | Expected SHA-256 Hash | Current SHA-256 Hash | Immutability Status |
|---|---|---|---|
| `G230_MP217/data/train.csv` | `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` | `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` | **VERIFIED UNCHANGED** |
| `G230_MP217/data/later.csv` | `966d0c8e759b263250febeaa45b365b28470d2047bb7e6417a21921f40603d59` | `966d0c8e759b263250febeaa45b365b28470d2047bb7e6417a21921f40603d59` | **VERIFIED UNCHANGED** |
| `G230_MP217/data/holdout_features.csv` | `aa6d44b46bb337d52b6cfc3102f2d7dc3f94fddabad4171c2b3af36e964d1310` | `aa6d44b46bb337d52b6cfc3102f2d7dc3f94fddabad4171c2b3af36e964d1310` | **VERIFIED UNCHANGED** |
| `data/raw/train.csv` | `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` | `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` | **VERIFIED UNCHANGED** |
| `data/raw/later.csv` | `966d0c8e759b263250febeaa45b365b28470d2047bb7e6417a21921f40603d59` | `966d0c8e759b263250febeaa45b365b28470d2047bb7e6417a21921f40603d59` | **VERIFIED UNCHANGED** |
| `data/raw/holdout_features.csv` | `aa6d44b46bb337d52b6cfc3102f2d7dc3f94fddabad4171c2b3af36e964d1310` | `aa6d44b46bb337d52b6cfc3102f2d7dc3f94fddabad4171c2b3af36e964d1310` | **VERIFIED UNCHANGED** |

---

## Final Status

**PASS — MLflow tracking and v1 model registration completed.**
