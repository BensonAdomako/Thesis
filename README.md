# Crop Yields and Conflict in Sub-Saharan Africa

Economics thesis investigating whether predicted crop yields causally affect conflict in six Sub-Saharan African countries — Ethiopia, Malawi, Mali, Nigeria, Tanzania, and Uganda — using an admin-2 district × year panel from 2010–2024.

Predicted yields from satellite/weather machine learning models serve as the key regressor to avoid reverse causality. Conflict data comes from ACLED.

---

## Research Design

- **Unit of analysis:** Admin-2 district × year
- **Outcome:** Conflict event count (ACLED), with 3-month, 6-month, and 12-month forward windows
- **Key regressor:** ML-predicted maize yield (kg/ha) — predicted from satellite imagery using XGBoost, instrumented to avoid reverse causality
- **Controls:** Population (gridded, Admin-2 × year)
- **Fixed effects:** Admin-2 + year (two-way)
- **Estimators:** TWFE OLS, TWFE Poisson, log-log OLS, linear probability model

**Niger is excluded** — both country-specific and global XGBoost models produce negative out-of-sample R² for Niger (only 16 LSMS training observations). This is flagged as a thesis limitation.

---

## Pipeline

Steps must be run in order from the project root.

### Step 1 — Satellite Panel (Google Colab only)

**File:** `Satellite_Year_Panel.ipynb`
**Cannot be run locally** — depends on large files in Google Drive and `google.colab` APIs.

Reads `EE_combined_long.parquet` (pre-extracted Earth Engine satellite data) from Google Drive. For each LSMS survey point, computes the mode harvest month across waves, then constructs 12-month lag features (lags 0–11) relative to each synthetic harvest date for every year 2010–2024.

**Output:** `EE_harvest_ml_full_panel.csv` (~98,280 rows × 90 cols) — download this to `Data/` before Step 2b.

---

### Step 2 — Yield Predictions (local)

**File:** `Code/generate_predictions.py`

```bash
python Code/generate_predictions.py
```

Reads `Data/EE_harvest_ml_full_panel.csv` and generates ML yield predictions using the hybrid model selection approach. Applies the full feature engineering pipeline (seasonal aggregates, interactions), predicts with the appropriate XGBoost model per country, merges LSMS observed yields for validation, and saves the output.

**Requires:** `Data/EE_harvest_ml_full_panel.csv` (download from Google Drive after Step 1)

**Output:** `Data/all_data_with_predictions.csv`

| Column | Description |
|---|---|
| `country` | Country name |
| `year` | Year (2010–2024) |
| `lat_modified`, `lon_modified` | Survey point coordinates |
| `harvest_end_month` | Harvest month (first of month) |
| `observed_yield_kg` | LSMS observed yield (survey points only, ~9% match) |
| `predicted_yield_xgb` | ML-predicted yield in kg/ha |
| `model_used` | Which model was used (e.g. `XGB_Ethiopia`) |
| `precip_missing` | `TRUE` for all rows — precipitation is 100% null in source data |

---

### Step 3 — Conflict Processing & Yield Merge

**File:** `Code /Conflict_Yields.r`

```bash
Rscript "Code /Conflict_Yields.r"
```

Downloads FAO/GAUL Admin-2 boundaries via `rgee` (requires authenticated Earth Engine session — see [rgee auth](#rgee-authentication) below). Spatially joins ACLED conflict events and yield predictions to Admin-2 polygons, builds monthly conflict panels with 3-, 6-, and 12-month forward windows, aggregates to Admin-2 × year, and merges population controls.

**Output:** `Data/conflict_yields_panel.csv` (13,890 rows × 22 cols)

---

### Step 4 — Regression Analysis

**File:** `Code /Reg_analysis.r`

```bash
Rscript "Code /Reg_analysis.r"
```

Runs the full specification suite and saves LaTeX tables.

**Output:**
- `Data/regression_results.tex` — preferred specifications (3-month window)
- `Data/regression_results_robustness.tex` — robustness checks

---

## Models

Pre-trained XGBoost and Random Forest models are in `Data/models_v3_agg/`. **Do not retrain** unless explicitly needed.

Hybrid model selection by best spatial out-of-fold R² (from `r2_matrix.csv`):

| Country | Model | Spatial R² |
|---|---|---|
| Ethiopia | `model_XGB_Ethiopia.joblib` | 0.066 |
| Malawi | `model_XGB_Malawi.joblib` | 0.178 |
| Mali | `model_XGB_GLOBAL.joblib` | 0.192 |
| Nigeria | `model_XGB_GLOBAL.joblib` | 0.018 |
| Tanzania | `model_XGB_Tanzania.joblib` | 0.092 |
| Uganda | `model_XGB_Uganda.joblib` | 0.016 |

**Known limitations:**
- Nigeria and Uganda: near-zero R² — high within-country heterogeneity and complex multi-season cropping systems limit satellite-based prediction quality
- Precipitation and soil moisture are 100% null in the source satellite extract — after imputation these features are constant (`precip_missing = TRUE` for all rows)
- Temporal R² (~0.10) is lower than spatial R² (~0.17)

---

## Data Files

| File | Description | In repo |
|---|---|---|
| `Data/Conflict.csv` | Raw ACLED conflict events export | No (gitignored) |
| `Data/adm2_pop_area.csv` | Admin-2 gridded population by year (2010–2020) | No |
| `Data/Plot_dataset.dta` | LSMS plot-level yield data | No |
| `Data/Plotcrop_dataset.dta` | LSMS plot-crop data (harvest months) | No |
| `Data/EE_harvest_ml_full_panel.csv` | Step 1 output — download from Google Drive | No |
| `Data/all_data_with_predictions.csv` | Step 2 output — ML yield predictions | No |
| `Data/conflict_yields_panel.csv` | Step 3 output — regression panel | No |
| `Data/models_v3_agg/` | Pre-trained XGB and RF models + imputers | No |
| `lsms_crop_yields/train_models.py` | Original model training pipeline (reference only) | Yes |

Data files are gitignored. The Google Drive source files (`EE_combined_long.parquet`, `EE_harvest_ml_full_panel.csv`) are large and not stored in this repo.

---

## rgee Authentication

Step 3 requires a live Earth Engine connection via `rgee`. If `ee_Initialize()` fails:

```r
library(rgee)
ee_install()          # installs the Python EE environment (first time only)
ee_Initialize()       # opens browser for Google account auth
```

Once authenticated, credentials are cached and `ee_Initialize()` will work silently on subsequent runs. The deprecation warnings printed at startup are from unrelated datasets in your EE environment and do not affect the `FAO/GAUL/2015/level2` boundary fetch used in this pipeline.

---

## Dependencies

**Python (Step 2):**
```
pandas, numpy, joblib, scikit-learn, xgboost
```

**R (Steps 3–4):**
```
rgee, sf, dplyr, readr, slider, lubridate, tidyr, fixest
```

---

## Key Results

Across most specifications, predicted crop yield has no statistically significant effect on conflict at the Admin-2 × year level. The log-log TWFE OLS specification (M4) finds a positive and significant coefficient (β = 0.175, p = 0.022), suggesting a 1% increase in predicted yield is associated with a 0.175% increase in log-conflict — consistent with a resource or opportunity-cost channel rather than a grievance channel. Results should be interpreted cautiously given the low predictive R² of the yield models for several countries (especially Nigeria and Uganda).
