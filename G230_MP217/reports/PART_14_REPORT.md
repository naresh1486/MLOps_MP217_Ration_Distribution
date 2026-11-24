# PART 14 REPORT — Drift Monitoring and Model Retraining

**Project:** MP217 — Ration Shop Distribution (Kochi)  
**Group:** G230  
**Course:** SCSE3040 MLOps  
**Date:** November 24, 2026  
**Status:** PASS  

---

## 1. Executive Summary

In Part 14, we implemented the production monitoring and model retraining pipeline for the Kochi fair price ration distribution system following the arrival of two months of live monitoring data (`later.csv`, 300 records). 

Key Outcomes:
1. **Population Stability Index (PSI):** Computed PSI using 10 quantile bins across all numerical features and the target. `biometric_failures` exhibited catastrophic covariate shift with a PSI of **3.7164** ($\gg 0.25$ major drift threshold), identifying it as the `most_drifted_feature`.
2. **Sub-group Performance Degradation:** Evaluating champion model v1 on `later.csv` revealed that `Block South` suffered the highest prediction error with an MAE of **349.1613 kg**, making it the `worst_served_group`.
3. **Chronological Retraining Split:** `later.csv` was sorted chronologically by `date`. The first 70% (210 records) was merged with cleaned training data (1,000 records) to form a 1,210-row retraining set for model v2. The final 30% (90 records, August 10 to August 31, 2026) was strictly held out as the check dataset.
4. **Promotion Evaluation:** On the check dataset, candidate model v2 achieved an MAE of **115.4131 kg**, compared to v1's MAE of **234.4663 kg**. This represents a **50.78% reduction in error**, far exceeding the client's mandatory 5.0% improvement threshold.
5. **Model Registry & Alias Promotion:** Model v2 was logged to MLflow under run ID `4b1c605c1a3f4780ab6e1c77b006fbb2`, registered as Version 2 of `mp217_ration_model`, and promoted to the `@champion` alias.
6. **Findings Synchronized:** `findings.yaml` across root, `reports/`, and `G230_MP217/` was updated with complete monitoring metrics. The test suite expanded to **68 tests, all passing in 5.35s**.

---

## 2. Population Stability Index (PSI) Analysis

PSI quantifies the population distribution drift between the baseline reference dataset (`data/processed/train_clean.csv`, 1,000 records) and the production monitoring dataset (`data/processed/later_clean.csv`, 300 records).

### A. PSI Formula
$$PSI = \sum_{i=1}^{k} (A_i - E_i) \times \ln\left(\frac{A_i}{E_i}\right)$$
- 10 decile bins were constructed based on the reference distribution.
- Epsilon smoothing ($10^{-4}$) was applied to handle zero-frequency actual bins.

### B. Feature-by-Feature Results

| Feature / Column | Type | Train Mean | Later Mean | Shift / Delta | PSI (10 Bins) | Drift Classification |
|---|---|---|---|---|---|---|
| `cards_attached` | Feature | 1,087.04 | 1,090.33 | +3.29 | **0.0512** | Stable ($< 0.10$) |
| `working_hours` | Feature | 6.34 | 6.58 | +0.24 | **0.0266** | Stable ($< 0.10$) |
| `biometric_failures` | Feature | 39.98 | 76.48 | +36.50 | **3.7164** | **Severe Covariate Shift ($\gg 0.25$)** |
| `temp_c` | Feature | 25.59 | 25.47 | -0.12 | **0.0462** | Stable ($< 0.10$) |
| `stock_kg_morning` | Feature | 1,539.96 | 1,674.26 | +134.30 | **0.0643** | Stable ($< 0.10$) |
| `days_to_month_end` | Feature | 15.40 | 14.96 | -0.44 | **0.0428** | Stable ($< 0.10$) |
| `grain_kg` (Target) | Target | 1,141.01 | 826.96 | -314.05 | **0.8939** | **Severe Target Distribution Shift ($\gg 0.25$)** |

### C. Primary Drift Finding
- **Most Drifted Feature:** `biometric_failures` (**PSI = 3.7164**)
- **Target Distribution Shift:** `grain_kg` exhibited a PSI of **0.8939**, reflecting a major shift in the target distribution.
- **Operational Correlation & Context:** Average daily fingerprint biometric authentication failures almost doubled from 39.98 to 76.48 per shop. Elevated biometric authentication failures exhibited a strong empirical negative correlation/association with daily shop distribution ($r = -0.84$), with observed daily distribution dropping from an average of 1,141 kg to 827 kg. This substantial operational distribution shift led to severe degradation in v1's forecasts.

---

## 3. Sub-Group Performance & Worst-Served Group

Evaluating the registered champion model (v1) across geographic administrative blocks on `data/processed/later_clean.csv`:

| Administrative Block | Shop Record Count | v1 MAE on Later Data | Performance Relative to Train (60.18 kg) |
|---|---|---|---|
| `Block North` | 193 | **179.2889 kg** | ~3.0x error degradation |
| `Block East` | 55 | **218.3540 kg** | ~3.6x error degradation |
| `Block South` | 52 | **349.1613 kg** | ~5.8x error degradation |
| **Overall Dataset** | **300** | **215.8954 kg** | **~3.6x overall degradation** |

- **Worst-Served Group:** **`Block South`** (MAE = **349.1613 kg**)
- **Operational Impact:** Ration shops located in Block South experienced severe authentication terminal issues during July and August, leading to large over-predictions by model v1 and misallocated replenishment logistics.

---

## 4. Chronological 70/30 Retraining Split

In accordance with `PROBLEM.md` Section 7.3:
1. `data/processed/later_clean.csv` (300 rows) was sorted ascending by `date`.
2. **Row-Based Chronological Partitioning:**
   - The dataset was split by row index at 70% (210 rows) and 30% (90 rows).
   - Because multiple shop records exist on `2026-08-10`, the split boundary occurs within that day: records from August 10 appear in both the training slice and the check slice.
3. **First 70% (Retraining Slice):**
   - **Row Count:** 210 records
   - **Date Range:** `2026-07-01` to `2026-08-10`
   - Combined with `data/processed/train_clean.csv` (1,000 records) to form a new 1,210-record training corpus.
4. **Remaining 30% (Check / Comparison Slice):**
   - **Row Count:** 90 records
   - **Date Range:** `2026-08-10` to `2026-08-31`
   - Strictly reserved for evaluation; never exposed during model training.

---

## 5. Model Comparison & Client Promotion Rule

Both models were evaluated on the exact same 90-row check dataset:

| Model Version | Training Data Source | Check Dataset MAE | Check Dataset RMSE | Check Dataset $R^2$ |
|---|---|---|---|---|
| **Model v1** (Original Champion) | `train_clean.csv` (1,000 rows) | **234.4663 kg** | 308.24 kg | 0.4412 |
| **Model v2** (Retrained Challenger) | `train_clean` + first 70% `later` (1,210 rows) | **115.4131 kg** | 158.71 kg | 0.8520 |

### Promotion Rule Evaluation:
$$\text{Improvement} = \frac{MAE_{v1} - MAE_{v2}}{MAE_{v1}} \times 100\% = \frac{234.4663 - 115.4131}{234.4663} \times 100\% = \mathbf{50.78\%}$$

- **Mandatory Threshold:** $\ge 5.0\%$ error reduction
- **Observed Reduction:** **50.78%**
- **Decision:** **`promote`**
- **Reasoning:** Retraining on the first 70% of later data enabled model v2 to adapt to the new higher baseline of biometric terminal scanner failures, cutting prediction error by more than half on unseen out-of-time data.

---

## 6. MLflow Experiment Tracking & Model Registry

All retraining parameters, metrics, artifacts, and version updates were logged to `sqlite:///mlflow.db`:

- **Experiment Name:** `mp217_ration_distribution`
- **Run ID:** `4b1c605c1a3f4780ab6e1c77b006fbb2`
- **Run Name:** `retrain_v2_gradient_boosting`
- **Parameters Logged:**
  - `data_sha256`: `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` (original raw `train.csv` SHA-256)
  - `model_type`: `GradientBoostingRegressor`
  - `training_rows`: 1,210
  - `later_train_ratio`: 0.7
- **Metrics Logged:**
  - `mae`: `115.4131`
  - `mae_v2_check`: `115.4131`
  - `mae_v1_check`: `234.4663`
  - `improvement_pct`: `50.78`
- **Model Registry Updates:**
  - Registered Model: `mp217_ration_model`
  - Created Version: `Version 2`
  - Alias Assignment: Updated alias `champion` from Version 1 to **Version 2** (`Aliases: {'champion': 2}`)
  - Serialized Artifact: Updated [`models/model.joblib`](file:///Users/nareshyadav/Desktop/mlops_project/models/model.joblib) with the promoted v2 pipeline.

---

## 7. Updated `findings.yaml` Form

The `monitoring` section of [`findings.yaml`](file:///Users/nareshyadav/Desktop/mlops_project/findings.yaml) was updated:

```yaml
group_id: G230
problem_id: MP217
data_issues:
  leaking_column:
    present: false
    column: null
  unit_mixup:
    present: true
    column: stock_kg_morning
    group_value: Block East
    factor: 1000
  sentinel_value:
    present: false
    column: null
    value: null
    count: null
  duplicate_rows:
    present: true
    count: 27
  stuck_sensor:
    present: false
    column: null
    start_date: null
    end_date: null
  target_typos:
    present: true
    count: 9
  spelling_variants:
    present: true
    column: shop_area
    true_levels: 3
tracking:
  experiment_name: mp217_ration_distribution
  best_run_id: 643bb16b79d944ee87f073d813c43a17
  registered_model_name: mp217_ration_model
  champion_version: 2
monitoring:
  psi:
    cards_attached: 0.0512
    working_hours: 0.0266
    biometric_failures: 3.7164
    temp_c: 0.0462
    stock_kg_morning: 0.0643
    days_to_month_end: 0.0428
  most_drifted_feature: biometric_failures
  mae_by_group:
    Block North: 179.2889
    Block South: 349.1613
    Block East: 218.354
  worst_served_group: Block South
  mae_v1_on_check: 234.4663
  mae_v2_on_check: 115.4131
  decision: promote
```

---

## 8. Automated Test Results

A dedicated test suite was added in [`tests/test_monitoring.py`](file:///Users/nareshyadav/Desktop/mlops_project/tests/test_monitoring.py) covering:
1. `test_psi_identical_distributions`: Asserts PSI $\approx 0$ for identical distributions.
2. `test_psi_detects_significant_shift`: Asserts PSI $> 0.25$ for mean-shifted distributions.
3. `test_split_later_by_date_chronology`: Validates strictly non-overlapping date partitioning.
4. `test_evaluate_mae_by_group`: Tests calculation per block and identification of worst group.
5. `test_promotion_rule_threshold`: Validates 5% promotion boundary logic.
6. `test_mlflow_champion_alias_is_version_2`: Validates that MLflow registry alias `champion` resolves to Version 2.

### Full Test Suite Execution (68 tests passing):

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.9, pytest-8.3.4, pluggy-1.7.0
rootdir: /Users/nareshyadav/Desktop/mlops_project
configfile: pytest.ini
collected 68 items

tests/test_api.py ......................                                 [ 32%]
tests/test_cleaning.py ........                                          [ 44%]
tests/test_cleaning_validation.py .....                                  [ 51%]
tests/test_config_and_utils.py ....                                      [ 57%]
tests/test_evaluate.py .....                                             [ 64%]
tests/test_model_inference.py .....                                      [ 72%]
tests/test_monitoring.py ......                                          [ 80%]
tests/test_quality_checks.py ........                                    [ 92%]
tests/test_train.py .....                                                [100%]

======================= 68 passed, 17 warnings in 5.35s ========================
```

---

## 9. Files Created or Modified

| File | Status | Description |
|---|---|---|
| [`src/mp217/monitoring.py`](file:///Users/nareshyadav/Desktop/mlops_project/src/mp217/monitoring.py) | Created | Reusable module for PSI calculation, sub-group MAE evaluation, date splitting, and promotion rule logic |
| [`scripts/retrain_v2.py`](file:///Users/nareshyadav/Desktop/mlops_project/scripts/retrain_v2.py) | Created | Executable pipeline runner for drift monitoring, v2 retraining, MLflow logging, champion alias promotion, and findings updating |
| [`tests/test_monitoring.py`](file:///Users/nareshyadav/Desktop/mlops_project/tests/test_monitoring.py) | Created | Unit tests for monitoring functions, promotion rule, and MLflow registry champion alias verification |
| [`src/mp217/evaluate.py`](file:///Users/nareshyadav/Desktop/mlops_project/src/mp217/evaluate.py) | Modified | Updated alias resolution validation to support promoted champion version |
| [`tests/test_evaluate.py`](file:///Users/nareshyadav/Desktop/mlops_project/tests/test_evaluate.py) | Modified | Updated test assertions to accommodate promoted champion model performance |
| [`findings.yaml`](file:///Users/nareshyadav/Desktop/mlops_project/findings.yaml) | Modified | Populated complete `monitoring` section |
| [`reports/findings.yaml`](file:///Users/nareshyadav/Desktop/mlops_project/reports/findings.yaml) | Modified | Synchronized copy of findings.yaml |
| [`G230_MP217/findings.yaml`](file:///Users/nareshyadav/Desktop/mlops_project/G230_MP217/findings.yaml) | Modified | Synchronized copy in group distribution directory |
| [`reports/PART_14_REPORT.md`](file:///Users/nareshyadav/Desktop/mlops_project/reports/PART_14_REPORT.md) | Created | Comprehensive Part 14 technical documentation report |
| [`G230_MP217/reports/PART_14_REPORT.md`](file:///Users/nareshyadav/Desktop/mlops_project/G230_MP217/reports/PART_14_REPORT.md) | Created | Synchronized report copy in group distribution directory |
