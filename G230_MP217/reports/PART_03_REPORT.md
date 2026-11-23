# PART 03 — Exploratory Data Analysis (EDA) Report

**Project:** MP217 — Ration Shop Distribution (Kochi)  
**Group:** G230  
**Batch:** 33  
**Faculty:** Upasna Joshi  
**Course:** SCSE3040 MLOps  
**Timestamp:** 2026-11-23T23:59:30+05:30  
**Status:** PASS — All Part 03 requirements completed  

---

## 1. Dataset Overview

An exploratory data analysis was conducted on the three raw datasets stored under `data/raw/` (`train.csv`, `later.csv`, `holdout_features.csv`). All observations and metrics below were computed directly on the original unmodified data.

### 1.1 Structural Overview Table

| Dataset | File Path | Total Rows | Total Columns | Temporal Range | Target Column Present | Inferred Target Type |
|---|---|---|---|---|---|---|
| **Training Dataset** | `data/raw/train.csv` | 1,027 | 11 | 2026-01-01 to 2026-06-30 | Yes (`grain_kg`) | Continuous float ($\text{kg}$) |
| **Production Monitor** | `data/raw/later.csv` | 300 | 11 | 2026-07-01 to 2026-08-31 | Yes (`grain_kg`) | Continuous float ($\text{kg}$) |
| **Unlabelled Holdout** | `data/raw/holdout_features.csv` | 250 | 10 | 2026-09-01 to 2026-09-30 | No (Features only) | N/A |

### 1.2 Column Inventory and Data Types

| Column Name | Inferred Dtype | Storage Type | Role in Pipeline | Description |
|---|---|---|---|---|
| `record_id` | `object` | string | Identifier | Unique transaction alphanumeric ID (e.g. `R00001` - `R50250`) |
| `date` | `object` | string (`YYYY-MM-DD`) | Temporal Metadata | Date of the shop operation |
| `cards_attached` | `int64` | integer | Numeric Feature | Number of ration cards attached to the fair price shop |
| `working_hours` | `float64` | float | Numeric Feature | Operating hours of the shop for the day |
| `biometric_failures` | `int64` | integer | Numeric Feature | Number of failed fingerprint biometric scans |
| `temp_c` | `float64` | float | Numeric Feature | Ambient maximum temperature in Celsius |
| `stock_kg_morning` | `float64` / `int64` | numeric | Numeric Feature | Opening stock in kilograms (in `train.csv`, affected by unit mix-up) |
| `days_to_month_end` | `int64` | integer | Numeric Feature | Calendar days remaining until the month concludes |
| `shop_area` | `object` | string | Categorical Feature | Geographic categorization (`tribal`, `rural`, `urban`) |
| `block` | `object` | string | Categorical Feature | Administrative block (`Block North`, `Block South`, `Block East`) |
| `grain_kg` | `float64` | float | **Regression Target** | Total quantity of grain distributed in kilograms |

---

## 2. Descriptive Statistics

All figures below are exact calculations directly computed from the raw data files.

### 2.1 Training Dataset (`data/raw/train.csv`, $N=1,027$)

| Column | Count | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
|---|---|---|---|---|---|---|---|---|
| `cards_attached` | 1027 | 1087.17 | 518.79 | 202.0 | 660.5 | 1066.0 | 1544.5 | 1997.0 |
| `working_hours` | 1027 | 6.35 | 2.05 | 3.0 | 4.6 | 6.4 | 8.2 | 10.0 |
| `biometric_failures` | 1027 | 39.87 | 23.27 | 0.0 | 20.0 | 39.0 | 60.0 | 80.0 |
| `temp_c` | 1027 | 25.51 | 12.05 | 5.0 | 15.0 | 25.3 | 35.8 | 46.0 |
| `stock_kg_morning` | 1027 | 247,555.98 | 622,969.83 | 207.0 | 973.5 | 1861.0 | 2664.0 | 2,997,000.0 |
| `days_to_month_end` | 1027 | 15.42 | 8.95 | 0.0 | 8.0 | 15.0 | 23.0 | 30.0 |
| `grain_kg` (Target) | 1027 | 1157.86 | 1087.76 | 145.0 | 839.15 | 1063.5 | 1301.35 | 14,858.0 |

*Note on Raw Anomalies:*
* `stock_kg_morning` has a mean of 247,555.98 kg and a maximum of 2,997,000.0 kg due to grams vs kilograms unit mixup in `Block East`.
* `grain_kg` has a standard deviation of 1087.76 kg and max of 14,858.0 kg due to 9 entries typed with an extra zero (10x too large).

### 2.2 Production Monitoring Dataset (`data/raw/later.csv`, $N=300$)

| Column | Count | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
|---|---|---|---|---|---|---|---|---|
| `cards_attached` | 300 | 1090.33 | 524.23 | 203.0 | 639.75 | 1082.0 | 1546.75 | 1999.0 |
| `working_hours` | 300 | 6.58 | 2.00 | 3.0 | 4.8 | 6.6 | 8.4 | 10.0 |
| `biometric_failures` | 300 | **76.48** | 23.98 | **38.0** | 56.75 | **76.0** | 98.0 | **118.0** |
| `temp_c` | 300 | 25.47 | 11.77 | 5.2 | 15.08 | 25.25 | 35.85 | 45.5 |
| `stock_kg_morning` | 300 | 1674.26 | 826.05 | 208.0 | 973.5 | 1688.5 | 2408.0 | 2993.0 |
| `days_to_month_end` | 300 | 14.96 | 8.56 | 0.0 | 7.75 | 15.0 | 22.0 | 30.0 |
| `grain_kg` (Target) | 300 | **690.20** | 451.75 | **0.0** | 346.22 | **691.25** | 994.30 | 1981.9 |

### 2.3 Unlabelled Holdout Dataset (`data/raw/holdout_features.csv`, $N=250$)

| Column | Count | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
|---|---|---|---|---|---|---|---|---|
| `cards_attached` | 250 | 1066.94 | 513.38 | 201.0 | 635.5 | 1032.5 | 1500.25 | 1995.0 |
| `working_hours` | 250 | 6.34 | 2.07 | 3.1 | 4.4 | 6.15 | 8.18 | 10.0 |
| `biometric_failures` | 250 | **79.13** | 24.31 | **38.0** | 57.25 | **81.5** | 100.75 | **118.0** |
| `temp_c` | 250 | 25.19 | 12.25 | 5.2 | 14.5 | 24.7 | 36.38 | 45.9 |
| `stock_kg_morning` | 250 | 1540.98 | 799.42 | 212.0 | 858.75 | 1465.5 | 2248.5 | 2981.0 |
| `days_to_month_end` | 250 | 15.34 | 8.31 | 0.0 | 8.0 | 15.0 | 22.75 | 30.0 |

---

## 3. Missing Values and Duplicate Analysis

### 3.1 Missing Values Analysis
Across all three datasets, an explicit `.isnull().sum()` check confirms that there are **0 missing (null or NaN) values** in any column:
* `data/raw/train.csv`: 0 missing values across all 11 columns.
* `data/raw/later.csv`: 0 missing values across all 11 columns.
* `data/raw/holdout_features.csv`: 0 missing values across all 10 columns.

### 3.2 Duplicate Record Analysis
* **Training Dataset (`train.csv`):**
  * Exact full-row duplicate count: **27 duplicate rows** (`df.duplicated().sum() == 27`).
  * Unique record IDs: 1,000 distinct IDs out of 1,027 total rows.
  * All 27 duplicated rows have identical values across all 11 columns (including `record_id` and `date`).
* **Production Monitor (`later.csv`):**
  * Exact duplicate count: **0 duplicate rows** (all 300 rows are unique).
* **Holdout Features (`holdout_features.csv`):**
  * Exact duplicate count: **0 duplicate rows** (all 250 rows are unique).

---

## 4. Numerical Feature Observations

1. **`cards_attached`:**
   * Uniformly spread across ~200 to 2000 ration cards across all datasets (mean ~1087, median ~1066).
   * Stable behavior across training, later, and holdout periods.
2. **`working_hours`:**
   * Ranges from 3.0 to 10.0 hours (mean ~6.35h to 6.58h).
   * No records violate the client rule (`working_hours < 1.5`), confirming no invalid data-entry errors exist in the raw files.
3. **`biometric_failures` (Significant Phenomenon):**
   * In `train.csv`, failures range from **0 to 80** with a mean of **39.87** and median of **39.0**.
   * In `later.csv` and `holdout_features.csv`, biometric failures jump to a range of **38 to 118** with a mean of **76.48** and **79.13** respectively.
   * This represents a severe operational drift (nearly doubling of POS biometric authentication failures in July–September compared to January–June).
   * No records exceed the physical hardware ceiling of 150 (`Field(..., le=150)`).
4. **`temp_c`:**
   * Spans 5.0 °C to 46.0 °C with an average of ~25.5 °C. Normal seasonal and diurnal variation.
5. **`stock_kg_morning` (Unit Mixup Phenomenon):**
   * In `train.csv`, values in `Block East` ($N=179$) are scaled in grams (ranging from 206,000 to 2,997,000 g), while `Block North` and `Block South` are in kilograms (207 to 3000 kg).
   * In `later.csv` and `holdout_features.csv`, all blocks (including `Block East`) are uniformly recorded in kilograms (208 to 2993 kg).
6. **`days_to_month_end`:**
   * Uniform distribution between 0 and 30 days across all three datasets (mean ~15 days), representing day of the ration distribution cycle.
7. **`grain_kg` (Target Variable Analysis):**
   * In `train.csv`, 1,018 rows sit comfortably between 145.0 kg and 1,864.0 kg.
   * Exactly 9 rows have values between 6,891.0 kg and 14,858.0 kg (separated by a huge void from 1864 kg to 6891 kg). These correspond to manual data-entry typos where an extra zero was appended (10x inflation).
   * In `later.csv`, target values range from 0.0 kg to 1,981.9 kg (mean 690.20 kg). The overall average is lower than in `train.csv` due to the surge in biometric failures restricting distribution.

---

## 5. Categorical Feature Observations

### 5.1 Development Block (`block`) Distribution

| Block Name | `train.csv` Count (%) | `later.csv` Count (%) | `holdout_features.csv` Count (%) | Distribution Assessment |
|---|---|---|---|---|
| **Block North** | 673 (65.53%) | 193 (64.33%) | 154 (61.60%) | Highly stable primary block (~62–65%) |
| **Block East** | 179 (17.43%) | 55 (18.33%) | 56 (22.40%) | Stable secondary block (~17–22%) |
| **Block South** | 175 (17.04%) | 52 (17.33%) | 40 (16.00%) | Stable secondary block (~16–17%) |
| **Total** | **1,027 (100%)** | **300 (100%)** | **250 (100%)** | Uniform administrative allocation |

### 5.2 Shop Area (`shop_area`) Distribution

* **`train.csv` (Messy Strings - 17 Variants):**
  * Contains leading/trailing whitespace, capitalization differences, and abbreviations:
    * `tribal` variants: `'tribal'` (568), `' tribal'` (28), `'tri.'` (27), `'Tribal'` (24), `'TRIBAL'` (20), `'tribal '` (16) $\rightarrow$ **Total: 683 (66.50%)**
    * `rural` variants: `'rural'` (209), `'rural '` (7), `' rural'` (6), `'Rural'` (6), `'rur.'` (4), `'RURAL'` (4) $\rightarrow$ **Total: 236 (22.98%)**
    * `urban` variants: `'urban'` (94), `' urban'` (6), `'URBAN'` (4), `'urb.'` (3), `'Urban'` (1) $\rightarrow$ **Total: 108 (10.52%)**
* **`later.csv` (Clean - 3 Levels):**
  * `tribal`: 189 (63.00%)
  * `rural`: 65 (21.67%)
  * `urban`: 46 (15.33%)
* **`holdout_features.csv` (Clean - 3 Levels):**
  * `tribal`: 154 (61.60%)
  * `rural`: 64 (25.60%)
  * `urban`: 32 (12.80%)

*Key Takeaway:* While `train.csv` has significant surface noise in string formatting, the underlying geographical proportions are stable across all three evaluation periods (~62–66% tribal, ~21–25% rural, ~10–15% urban).

---

## 6. Important Visualizations and Figures

All figures were generated programmatically using `src/mp217/eda.py` and saved to `reports/figures/`:

| Figure Description | Generated File Path | Visual Insight Captured |
|---|---|---|
| **Numerical Feature Histograms** | [01_numerical_histograms.png](file:///Users/nareshyadav/Desktop/mlops_project/reports/figures/01_numerical_histograms.png) | Displays raw feature distributions in `train.csv`; illustrates the massive right-skew in `stock_kg_morning` due to the unit mixup. |
| **Numerical Feature Boxplots** | [02_numerical_boxplots.png](file:///Users/nareshyadav/Desktop/mlops_project/reports/figures/02_numerical_boxplots.png) | Highlights extreme values and scale distortions across all 6 features. |
| **Target Distribution (grain_kg)** | [03_target_distribution.png](file:///Users/nareshyadav/Desktop/mlops_project/reports/figures/03_target_distribution.png) | Contrast between normal grain distribution ($\le 2000\text{ kg}$) and the 9 isolated typo entries ($> 6800\text{ kg}$). |
| **Correlation Heatmap** | [04_correlation_heatmap.png](file:///Users/nareshyadav/Desktop/mlops_project/reports/figures/04_correlation_heatmap.png) | Heatmap of Pearson correlation coefficients across numerical columns in `train.csv`. |
| **Categorical Distributions** | [05_categorical_distribution.png](file:///Users/nareshyadav/Desktop/mlops_project/reports/figures/05_categorical_distribution.png) | Frequency bar charts of `block` and top raw levels of `shop_area`. |
| **Train vs Later Distribution Shifts** | [06_train_vs_later_comparison.png](file:///Users/nareshyadav/Desktop/mlops_project/reports/figures/06_train_vs_later_comparison.png) | Side-by-side density overlay proving the major distribution shift in `biometric_failures` and correction of `stock_kg_morning`. |

---

## 7. Feature-Target Relationships

### 7.1 Quantitative Correlations with `grain_kg`

| Feature | Raw `train.csv` Correlation | Clean `later.csv` Correlation | Domain Explanation |
|---|---|---|---|
| `biometric_failures` | **-0.238** | **-0.840** | **Dominant negative driver.** When biometric fingerprint scanners fail at the shop, cardholders cannot authenticate, directly blocking grain dispersal. In clean data (`later.csv`), this correlation is overwhelmingly strong ($-0.84$). |
| `stock_kg_morning` | **+0.056** | **+0.328** | **Positive driver.** Shops that begin the morning with higher physical stock are equipped to satisfy larger distribution volumes without stocking out early. In raw train, this relationship was obscured by the Block East gram scale error. |
| `cards_attached` | **+0.123** | **+0.160** | **Positive driver.** More ration cards assigned to a shop creates higher aggregate daily footfall and demand. |
| `days_to_month_end` | **-0.046** | **-0.233** | **Negative driver.** As the end of the month nears (fewer days remaining), beneficiaries rush to collect unavailed monthly grain rations before quotas lapse. |
| `temp_c` | **-0.130** | **-0.152** | **Mild negative driver.** Extreme heat discourages rural/tribal cardholders from traveling to shops during peak afternoon hours. |
| `working_hours` | **+0.025** | **-0.057** | **Weak driver.** Minor impact once biometric and stock bottlenecks are accounted for. |

### 7.2 Group-Level Target Comparisons

* **Distribution by Development Block (`block` in `train.csv`):**
  * `Block South`: Mean = 1,289.70 kg (Std = 1,555.44 kg, Median = 1,080.80 kg)
  * `Block East`: Mean = 1,172.11 kg (Std = 1,285.50 kg, Median = 1,030.10 kg)
  * `Block North`: Mean = 1,119.79 kg (Std = 858.10 kg, Median = 1,063.30 kg)
  * *Observation:* Medians across blocks are very consistent (~1,030 to 1,080 kg). Discrepancies in means and standard deviations are heavily driven by where the 9 target typo rows landed.
* **Distribution by Shop Area (`shop_area`):**
  * `urban`: Median ~1,200 kg (higher volume per shop due to density).
  * `rural`: Median ~1,037 kg.
  * `tribal`: Median ~1,006 kg.

---

## 8. Train vs Later Dataset Comparison

A comparative evaluation between `train.csv` and `later.csv` revealed three critical differences that will shape the downstream MLOps pipeline:

1. **Biometric Failures Surge (Severe Operational Covariate Shift):**
   * `train.csv` mean was 39.87 (max 80).
   * `later.csv` mean increased to 76.48 (max 118), representing an increase of ~92%.
   * In `holdout_features.csv`, this shift persists with a mean of 79.13.
   * *Impact:* A model trained solely on `train.csv` has seen very few examples with `biometric_failures > 70`. This will likely degrade model performance on `later.csv`, motivating retraining in Stage 4.
2. **Target Distribution Compression (Concept/Target Shift):**
   * In `train.csv`, average grain distribution was ~1,063 kg (median).
   * In `later.csv`, average grain distribution dropped significantly to ~691 kg (median) and mean 690.20 kg.
   * *Impact:* Directly caused by the heightened biometric failure rate, which systematically throttled shop distribution capacity.
3. **Unit Mixup Disappearance:**
   * In `later.csv`, `stock_kg_morning` for `Block East` is in standard kilograms (mean: 1,546.47 kg; max: 2,861 kg). The grams-scale mixup observed in `train.csv` was an isolated historical data entry issue.

---

## 9. Important Findings for the Next Stage (Data Cleaning & Preparation)

The findings from this EDA establish clear specifications for the upcoming Data Cleaning stage:
1. **Deduplication:** Exactly 27 duplicate rows must be removed from `train.csv` to reduce the dataset to 1,000 unique records.
2. **Unit Conversion:** For all rows where `block == 'Block East'` in `train.csv`, `stock_kg_morning` must be divided by `1000.0`.
3. **Target Typo Correction:** For the 9 rows where `grain_kg > 2000.0`, `grain_kg` must be divided by `10.0`.
4. **Categorical Normalization:** `shop_area` strings must be trimmed, lowercased, and mapped to the 3 canonical categories (`tribal`, `rural`, `urban`).
5. **Validation Rule Compliance:** All records in both `later.csv` and `holdout_features.csv` respect the client's serving rules (`biometric_failures <= 150` and `working_hours >= 1.5`), confirming these rules will not reject legitimate production records.

---

## 10. Confirmation of Dataset Immutability

All original CSV files in both `G230_MP217/data/` and `data/raw/` were verified against their original cryptographic SHA-256 hashes:

| Dataset File Path | Expected SHA-256 Hash | Current SHA-256 Hash | Status |
|---|---|---|---|
| `G230_MP217/data/train.csv` | `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` | `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` | **UNCHANGED** |
| `G230_MP217/data/later.csv` | `966d0c8e759b263250febeaa45b365b28470d2047bb7e6417a21921f40603d59` | `966d0c8e759b263250febeaa45b365b28470d2047bb7e6417a21921f40603d59` | **UNCHANGED** |
| `G230_MP217/data/holdout_features.csv` | `aa6d44b46bb337d52b6cfc3102f2d7dc3f94fddabad4171c2b3af36e964d1310` | `aa6d44b46bb337d52b6cfc3102f2d7dc3f94fddabad4171c2b3af36e964d1310` | **UNCHANGED** |
| `data/raw/train.csv` | `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` | `d65889d074586614eaddcb806298ccd08f60c4555c3ad8709842f67dffec0d6a` | **UNCHANGED** |
| `data/raw/later.csv` | `966d0c8e759b263250febeaa45b365b28470d2047bb7e6417a21921f40603d59` | `966d0c8e759b263250febeaa45b365b28470d2047bb7e6417a21921f40603d59` | **UNCHANGED** |
| `data/raw/holdout_features.csv` | `aa6d44b46bb337d52b6cfc3102f2d7dc3f94fddabad4171c2b3af36e964d1310` | `aa6d44b46bb337d52b6cfc3102f2d7dc3f94fddabad4171c2b3af36e964d1310` | **UNCHANGED** |

---

## 11. Distinction: Actual Findings vs Assumptions

* **Actual Findings (Empirically Confirmed via Code):**
  * Exactly 0 missing values across all files.
  * Exactly 27 duplicate rows in `train.csv`.
  * `stock_kg_morning` for `Block East` in `train.csv` is inflated by a factor of 1000.
  * Exactly 9 rows in `train.csv` have `grain_kg > 2000` (typed 10x too big).
  * `shop_area` in `train.csv` has 17 spelling variants of 3 true levels (`tribal`, `rural`, `urban`).
  * `biometric_failures` doubled in `later.csv` and `holdout_features.csv` compared to `train.csv`.
  * Strong negative correlation between `biometric_failures` and `grain_kg` ($-0.84$ in clean data).
* **Assumptions:**
  * Downstream regressors trained on cleaned data will benefit significantly from retraining once the first 70% of `later.csv` is incorporated, due to exposure to the higher `biometric_failures` regime.

---

## Final Status

**PASS — All Part 03 requirements completed.**
