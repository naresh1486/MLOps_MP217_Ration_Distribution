# PART 09 — Automated Testing Improvements Report

**Project:** MP217 — Ration Shop Distribution (Kochi)  
**Group:** G230  
**Batch:** 33  
**Faculty:** Upasna Joshi  
**Course:** SCSE3040 MLOps  
**Timestamp:** 2026-11-24T03:22:15+05:30  
**Status:** PASS — Automated testing improvements completed successfully  

---

## 1. What Was Reviewed

The entire existing test suite and codebase across Parts 01–08 were systematically reviewed:
* **Existing Passing Tests (26 total):**
  * `tests/test_cleaning.py` (8 tests): Covered basic duplicate removal, Block East unit scaling, target typo correction, spelling normalization, preservation of unaffected values, clean dataset shapes, later dataset cleaning, and DataFrame immutability.
  * `tests/test_quality_checks.py` (8 tests): Covered detection functions for all 7 data quality issues and input immutability.
  * `tests/test_train.py` (5 tests): Covered original file SHA-256 calculation, feature matrix extraction, reproducible splitting, pipeline training, and `models/model.joblib` persistence.
  * `tests/test_evaluate.py` (5 tests): Covered champion model loading from registry, metric computation formulas, validation prediction dimensions, diagnostic plotting, and evaluation workflows.

### Gaps and Opportunities Identified:
1. **Data Cleaning Validation Guardrails:** While positive cleaning transformations were tested, `validate_cleaned_data()` had zero negative/error-handling tests verifying that it properly raises `ValueError` when confronted with corrupted data (e.g. unhandled duplicates, injected NaNs, unmapped categories, unscaled stock, uncorrected typos).
2. **Model Inference Robustness:** The pipeline had not been tested against edge cases such as:
   - Direct loading through the MLflow model registry URI (`models:/mp217_ration_model@champion`).
   - Invariance between single-record and batch predictions.
   - Robustness to unseen or unknown categorical levels (verifying OneHotEncoder `handle_unknown="ignore"` behavior).
   - Error handling when required feature columns are missing from prediction inputs.
   - Realistic physical bounds checking on predictions.
3. **Configuration and Data Utilities:** No unit tests previously existed for `src/mp217/config.py` and `src/mp217/data_utils.py` verifying accurate parsing, sample row loading, and non-file error handling.

---

## 2. Additional Tests Added

A total of **14 new tests** were constructed across three dedicated test modules:

### 2.1 `tests/test_cleaning_validation.py` (5 New Tests)
* `test_validation_catches_duplicate_rows`: Verifies that `validate_cleaned_data` raises `ValueError` matching `"no_duplicates"` when duplicate rows are present.
* `test_validation_catches_missing_values`: Verifies that `validate_cleaned_data` raises `ValueError` matching `"no_missing_values"` when NaN entries are introduced.
* `test_validation_catches_invalid_shop_area`: Verifies that `validate_cleaned_data` raises `ValueError` matching `"canonical_shop_areas"` when unmapped categories (e.g. `'coastal'`) are present.
* `test_validation_catches_unconverted_block_east`: Verifies that `validate_cleaned_data` raises `ValueError` matching `"block_east_in_kg"` when Block East stock is unscaled ($> 5,000\text{ kg}$).
* `test_validation_catches_unresolved_target_typos`: Verifies that `validate_cleaned_data` raises `ValueError` matching `"target_typos_resolved"` when target typos ($> 2,000\text{ kg}$) remain.

### 2.2 `tests/test_model_inference.py` (5 New Tests)
* `test_mlflow_model_loading_via_champion_alias`: Verifies loading the champion model directly through the MLflow registry URI (`models:/mp217_ration_model@champion`) and confirms accurate inference.
* `test_model_prediction_batch_invariance`: Asserts that predicting a single row yields identical numerical results to the corresponding row evaluated in a batch.
* `test_model_handles_unknown_categorical_values`: Asserts that the pipeline handles unseen categorical values (e.g. `shop_area="unknown_category"`, `block="Unknown_Block"`) gracefully without raising exceptions.
* `test_model_missing_feature_raises_error`: Asserts that passing an incomplete payload missing required features (e.g. omitted `cards_attached`) properly raises `(KeyError, ValueError)`.
* `test_model_output_range_and_datatype`: Asserts that predictions are continuous positive floats within realistic physical bounds ($0 < \hat{y} < 3,000\text{ kg}$).

### 2.3 `tests/test_config_and_utils.py` (4 New Tests)
* `test_config_reads_expected_structure`: Verifies accurate parsing of `params.yaml` sections (`paths`, `target_column`, `split`, `model`, `mlflow`).
* `test_config_raises_on_missing_file`: Verifies `load_config` raises `FileNotFoundError` when pointed to a non-existent configuration file.
* `test_data_utils_load_data_sample`: Verifies `data_utils.load_data` accurately respects the `nrows` sampling argument.
* `test_data_utils_raises_on_directory_path`: Verifies `data_utils.load_data` raises `ValueError` when passed a directory path instead of a file.

---

## 3. Total Number of Tests & Breakdown

The automated test suite expanded from **26 to 40 tests**:

| Test Module | Tests Before Part 09 | Tests Added | Total Tests Now | Focus Area |
|---|---|---|---|---|
| `tests/test_cleaning.py` | 8 | 0 | **8** | Core cleaning transformations & immutability |
| `tests/test_cleaning_validation.py` | 0 | 5 | **5** | Negative cases & validation error detection |
| `tests/test_model_inference.py` | 0 | 5 | **5** | Inference robustness, MLflow URI loading, edge cases |
| `tests/test_config_and_utils.py` | 0 | 4 | **4** | Configuration parsing & generic CSV loader |
| `tests/test_quality_checks.py` | 8 | 0 | **8** | Detection routines for all 7 data quality issues |
| `tests/test_train.py` | 5 | 0 | **5** | Data preparation, splitting, and model persistence |
| `tests/test_evaluate.py` | 5 | 0 | **5** | Metric formulas, dimensions, and plot generation |
| **Total** | **26** | **14** | **40** | **Complete pipeline test coverage** |

---

## 4. Final Test Execution Results

Executing the complete test suite via `pytest`:

```
============================= test session starts ==============================
platform darwin -- Python 3.11.9, pytest-8.3.4, pluggy-1.7.0
rootdir: /Users/nareshyadav/Desktop/mlops_project
collected 40 items

tests/test_cleaning.py ........                                          [ 20%]
tests/test_cleaning_validation.py .....                                  [ 32%]
tests/test_config_and_utils.py ....                                      [ 42%]
tests/test_evaluate.py .....                                             [ 55%]
tests/test_model_inference.py .....                                      [ 67%]
tests/test_quality_checks.py ........                                    [ 87%]
tests/test_train.py .....                                                [100%]

======================= 40 passed, 17 warnings in 3.92s ========================
```

* **Passed:** **40**
* **Failed:** **0**
* **Execution Time:** **3.92 seconds**

---

## 5. Limitations Encountered

* None. All tests execute deterministically in offline mode without requiring network access, cloud infrastructure, or external APIs.
* Deprecation warnings originating from transitive dependencies within MLflow/Pydantic are captured cleanly without impacting test outcomes.

---

## 6. Confirmation of Dataset Immutability

Cryptographic SHA-256 hashes of all original files were verified against their original baselines:

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

**PASS — Automated testing improvements completed successfully.**
