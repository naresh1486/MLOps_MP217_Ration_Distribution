# PART 08 — Model Evaluation and Performance Analysis Report

**Project:** MP217 — Ration Shop Distribution (Kochi)  
**Group:** G230  
**Batch:** 33  
**Faculty:** Upasna Joshi  
**Course:** SCSE3040 MLOps  
**Timestamp:** 2026-11-24T03:17:15+05:30  
**Status:** PASS — Model evaluation completed successfully  

---

## 1. Registered Model and Version

The champion regression pipeline trained in Part 07 was loaded and verified:
* **Registered Model Name:** `mp217_ration_model`
* **Target Version:** **Version 1** (`v1`)
* **Underlying Algorithm:** `GradientBoostingRegressor` (`n_estimators=100`, `learning_rate=0.1`, `max_depth=4`, `random_state=42`)
* **Pipeline Structure:** `ColumnTransformer(StandardScaler, OneHotEncoder) -> GradientBoostingRegressor`
* **Training Run ID:** `643bb16b79d944ee87f073d813c43a17`
* **Persisted Model Artifact:** `models/model.joblib` (330 KB)

---

## 2. Champion Alias Verification

The model alias was audited via `MlflowClient` against the SQLite tracking database (`sqlite:///mlflow.db`):
* **Registered Model:** `mp217_ration_model`
* **Alias Queried:** `champion`
* **Resolved Model Version:** `1`
* **Source Run ID:** `643bb16b79d944ee87f073d813c43a17`
* **Alias Invariance:** The `@champion` alias remained unchanged and points strictly to Version 1.

---

## 3. Evaluation Dataset and Split

* **Evaluation Source:** Held-out validation split generated from `data/processed/train_clean.csv`.
* **Dataset Size:** 200 validation samples out of 1,000 total clean records.
* **Partitioning Parameters:** `test_size = 0.20`, `random_seed = 42` (configured via `params.yaml`).
* **Feature Schema (8 Operational Features):** `cards_attached`, `working_hours`, `biometric_failures`, `temp_c`, `stock_kg_morning`, `days_to_month_end`, `shop_area`, `block`.
* **Target:** `grain_kg` ($\text{min} = 201.6\text{ kg}$, $\text{max} = 1,823.1\text{ kg}$, $\text{mean} = 1,048.56\text{ kg}$ on validation split).

---

## 4. Evaluation Metrics (MAE, RMSE, R²)

Regression performance on the 200 validation instances:

| Metric | Computed Value | Interpretation |
|---|---|---|
| **Mean Absolute Error (MAE)** | **60.1811 kg** | On average, daily shop grain distribution forecasts deviate from true ground truth by approximately 60.2 kg. |
| **Root Mean Squared Error (RMSE)** | **77.4806 kg** | Penalizes larger deviations; closely aligned with MAE, indicating few large residual outliers. |
| **Coefficient of Determination ($R^2$)** | **0.9407** | The champion model explains **94.07%** of the total variance in daily grain distribution. |

### Practical Domain Meaning of MAE in this Project
In the context of the Kochi Food and Civil Supplies Office, ration shops distribute approximately $1,050\text{ kg}$ of grain per day on average. An MAE of **$60.2\text{ kg}$** corresponds to an average forecasting error of only **$5.7\%$** of daily shop turnover. This degree of accuracy enables the civil supplies supply chain to dispatch truck replenishments with high confidence, preventing midday stockouts while minimizing unnecessary depot overhead.

---

## 5. Actual versus Predicted Analysis

* **Range of Actual Values ($y$):** $[201.60\text{ kg}, 1,823.10\text{ kg}]$
* **Range of Predicted Values ($\hat{y}$):** $[242.84\text{ kg}, 1,801.32\text{ kg}]$
* **Prediction Correlation:** Pearson $r = 0.970$ between actual and predicted values.
* **Linear Alignment:** In the actual vs predicted scatter plot, validation observations closely cluster along the ideal $45^\circ$ reference line ($y = x$) across rural, urban, and tribal shops.

---

## 6. Residual Analysis ($e = y - \hat{y}$)

Residuals are defined as actual minus predicted ($e = y - \hat{y}$):
* A **negative residual** ($e < 0$) indicates **overprediction / overestimation** ($\hat{y} > y$).
* A **positive residual** ($e > 0$) indicates **underprediction / underestimation** ($\hat{y} < y$).

Summary of validation residuals:
* **Mean Residual:** **$-10.2992\text{ kg}$**
  * *Interpretation:* A slightly negative mean residual indicates a very modest tendency to overestimate daily demand by $\approx 10.3\text{ kg}$ on average. In a food relief and ration context, slight conservative overestimation is operationally desirable over underestimation, as it prevents families from arriving at empty ration counters.
* **Standard Deviation of Residuals:** **$76.7930\text{ kg}$**
* **Minimum Residual (Largest Overprediction):** **$-227.70\text{ kg}$** (actual distribution was $227.70\text{ kg}$ lower than predicted).
* **Maximum Residual (Largest Underprediction):** **$+214.50\text{ kg}$** (actual distribution was $214.50\text{ kg}$ higher than predicted).
* **Maximum Absolute Error:** **$227.7041\text{ kg}$**
* **Symmetry & Homoscedasticity:**
  * The residual distribution is symmetric and approximately normal around zero.
  * Residual variance remains relatively constant across predicted grain volumes (no severe heteroscedastic funneling).

---

## 7. Diagnostic Visualizations

Three figures were generated using `src/mp217/evaluate.py` and saved to `reports/figures/`:

| Figure Description | File Path | Visual Finding |
|---|---|---|
| **Actual vs Predicted Scatter Plot** | [07_actual_vs_predicted.png](file:///Users/nareshyadav/Desktop/mlops_project/reports/figures/07_actual_vs_predicted.png) | Strong linear alignment along the identity line ($y = x$) with high fidelity ($R^2 = 0.9407$). |
| **Residual Distribution Histogram** | [08_residual_distribution.png](file:///Users/nareshyadav/Desktop/mlops_project/reports/figures/08_residual_distribution.png) | Bell-shaped normal residual density centered near zero ($\mu = -10.3\text{ kg}$). |
| **Prediction Error / Homoscedasticity Plot** | [09_prediction_error_plot.png](file:///Users/nareshyadav/Desktop/mlops_project/reports/figures/09_prediction_error_plot.png) | Even spread of residuals across low, medium, and high prediction intervals. |

---

## 8. MLflow Logging Details

The evaluation metrics and diagnostic plots were logged into the active MLflow SQLite experiment:
* **Tracking URI:** `sqlite:///mlflow.db`
* **Experiment Name:** `mp217_ration_distribution` (ID: `1`)
* **Evaluation Run Name:** `v1_champion_validation_evaluation`
* **Logged Run ID:** `c1f51536b1314abf911355e3df42f331`
* **Logged Parameters:**
  * `evaluated_model`: `mp217_ration_model`
  * `evaluated_version`: `1`
  * `model_alias`: `champion`
  * `source_training_run_id`: `643bb16b79d944ee87f073d813c43a17`
  * `data_sha256`: `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a`
* **Logged Metrics:** `val_mae = 60.1811`, `mae = 60.1811`, `val_rmse = 77.4806`, `val_r2 = 0.9407`, `residuals_mean = -10.2992`, `max_absolute_error = 227.7041`.
* **Artifacts Logged:** `07_actual_vs_predicted.png`, `08_residual_distribution.png`, `09_prediction_error_plot.png`.

---

## 9. Unit Test Results

The test suite in `tests/test_evaluate.py` was executed alongside all existing tests:

```
============================= test session starts ==============================
platform darwin -- Python 3.11.9, pytest-8.3.4, pluggy-1.7.0
rootdir: /Users/nareshyadav/Desktop/mlops_project
collected 26 items

tests/test_cleaning.py ........                                          [ 30%]
tests/test_evaluate.py .....                                             [ 50%]
tests/test_quality_checks.py ........                                    [ 80%]
tests/test_train.py .....                                                [100%]

======================= 26 passed in 3.90s ========================
```

All 26 unit tests passed with zero failures.

---

## 10. Confirmation of Dataset Immutability

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

## 11. Model Limitations and Operational Observations

1. **Biometric Failure Regime Limitation:**
   * The training distribution (`train_clean.csv`) only contains `biometric_failures` between 0 and 80.
   * As discovered during EDA, production monitoring data (`later.csv`) features a surge in biometric failures up to 118.
   * Model v1 has never seen operational conditions with $\ge 80$ biometric failures, which may cause performance degradation during production deployment. This reinforces the need for drift monitoring and retraining in Stage 4.
2. **Peak Validation Residual Observation:**
   * Maximum absolute error on the validation set is $227.70\text{ kg}$, occurring on an unusually high-volume urban shop on the penultimate day of the month. Maintaining a precautionary buffer stock of $\approx 250\text{ kg}$ is noted as a preliminary operational observation from empirical error bounds, not an established or validated supply chain policy.

---

## Final Status

**PASS — Model evaluation completed successfully.**
