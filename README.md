# Crop Yields and Conflict in Sub-Saharan Africa

Economics thesis investigating whether predicted crop yields causally affect conflict in six Sub-Saharan African countries — Ethiopia, Malawi, Mali, Nigeria, Tanzania, and Uganda — using an admin-2 district × year panel from 2010–2024.

Predicted yields from satellite/weather machine learning models serve as the key regressor to avoid reverse causality. Conflict data comes from ACLED.

---

## Research Design

- **Unit of analysis:** Admin-2 district × year
- **Outcome:** Conflict event count (ACLED), with 3-month, 6-month, and 12-month forward windows
- **Key regressor:** ML-predicted maize yield (kg/ha) — v2 XGBoost predictions from GROW-Africa training data (spatial R² ≈ 0.35)
- **Controls:** Population (gridded, Admin-2 × year)
- **Fixed effects:** Admin-2 + year (two-way)
- **Estimators:** TWFE OLS, TWFE Poisson, log-log OLS; per-country specifications

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

### Step 2b — v1 Yield Predictions (local, reference only)

**File:** `Code/generate_predictions.py`

```bash
python Code/generate_predictions.py
```

Generates v1 ML yield predictions using the hybrid model selection approach (country-specific XGBoost where available, global model fallback). Output used in the base conflict panel but superseded by v2 predictions for regressions.

**Output:** `Data/all_data_with_predictions.csv`

---

### Step 2c — v2 Yield Predictions (armazi server)

**File:** `Code/generate_predictions_v2.py`
**Must be run on armazi** — depends on `admin_features_v2.parquet` and v2 models.

```bash
# On armazi:
python generate_predictions_v2.py
# Then copy back:
scp biadomako@armazi:~/lsms_crop_yields/predictions_v2.csv ~/Desktop/Thesis/Data/
```

Uses a single global XGBoost model trained on GROW-Africa survey data with richer satellite features (SAR, LSWI, GCVI). Produces both absolute yield predictions (kg/ha) and anomaly (z-score) predictions for all 6 countries × 2010–2024.

**Output:** `Data/predictions_v2.csv`

| Column | Description |
|---|---|
| `GID_0` | ISO3 country code |
| `GID_2` | GADM Admin-2 code |
| `NAME_2` | GADM Admin-2 name |
| `year` | Year (2010–2024) |
| `predicted_yield_abs_kgha` | Predicted maize yield in kg/ha |
| `log_predicted_yield_abs` | Log-scale prediction (used in regressions) |
| `predicted_yield_anomaly` | Yield z-score (anomaly model) |

---

### Step 2d — Merge v2 Predictions into Panel

**File:** `Code/update_panel_v2.R`

```bash
Rscript Code/update_panel_v2.R
```

Merges `Data/predictions_v2.csv` (GADM naming) into `Data/conflict_yields_panel.csv` (FAO/GAUL naming) using a three-pass district name matching strategy:
- **Pass 1:** Normalised exact match (diacritics stripped, lowercase, punctuation removed)
- **Pass 2:** Suffix-stripped match (removes administrative suffixes like "boma", "city", "urban")
- **Pass 3:** Manual crosswalk (Ethiopia Amharic directional translations, Nigeria spelling corrections)

**Coverage:** 93.3% of districts matched. Unmatched districts receive NA yield and are dropped from regressions.

**Output:** `Data/conflict_yields_panel_v2.csv` (13,890 rows × 25 cols)

---

### Step 3 — Conflict Processing & Yield Merge

**File:** `Code /Conflict_Yields.r`

```bash
Rscript "Code /Conflict_Yields.r"
```

Downloads FAO/GAUL Admin-2 boundaries via `rgee`. Spatially joins ACLED conflict events and yield predictions to Admin-2 polygons, builds monthly conflict panels with 3-, 6-, and 12-month forward windows, aggregates to Admin-2 × year, and merges population controls.

**Output:** `Data/conflict_yields_panel.csv` (13,890 rows × 22 cols)

> **Note:** Binary conflict indicators (`any_conflict_3/6/12mo`) are identical across all three windows by construction — any district with conflict in the 3-month window trivially has conflict in the 6- and 12-month windows. These have been dropped from the robustness analysis as uninformative.

---

### Step 4 — Regression Analysis

**File:** `Code /Reg_analysis.r`

```bash
Rscript "Code /Reg_analysis.r"
```

Uses `Data/conflict_yields_panel_v2.csv` (v2 predictions). Runs the full specification suite and saves four LaTeX tables.

**Output:**

| File | Contents |
|---|---|
| `Data/regression_results.tex` | Preferred specs: pooled OLS, TWFE OLS, TWFE Poisson, log-log OLS |
| `Data/regression_results_robustness.tex` | Alternative windows (6mo, 12mo) + fatality Poisson |
| `Data/regression_results_bycountry_ols.tex` | Per-country TWFE log-log OLS (one column per country) |
| `Data/regression_results_bycountry_poisson.tex` | Per-country TWFE Poisson (one column per country) |

---

## Models

### v2 Models (current — used in all regressions)

Trained on GROW-Africa survey data with richer satellite features including SAR (Sentinel-1 VV/VH), LSWI, and GCVI alongside standard EVI/NDVI and climate variables. Single global model applied to all 6 countries.

- **Spatial R² ≈ 0.35** (vs v1 range of 0.016–0.192)
- Stored on armazi: `/home/ahobbs/lsms_crop_yields/models_v2/`
- 210 features post-variance-threshold

### v1 Models (reference only)

Pre-trained XGBoost and Random Forest models in `Data/models_v3_agg/`. Hybrid selection by best spatial out-of-fold R²:

| Country | Model | Spatial R² |
|---|---|---|
| Ethiopia | `model_XGB_Ethiopia.joblib` | 0.066 |
| Malawi | `model_XGB_Malawi.joblib` | 0.178 |
| Mali | `model_XGB_GLOBAL.joblib` | 0.192 |
| Nigeria | `model_XGB_GLOBAL.joblib` | 0.018 |
| Tanzania | `model_XGB_Tanzania.joblib` | 0.092 |
| Uganda | `model_XGB_Uganda.joblib` | 0.016 |

**Known limitations:**
- Nigeria and Uganda: near-zero R² — high within-country heterogeneity and complex multi-season cropping systems
- Precipitation and soil moisture are 100% null in the v1 source satellite extract

---

## Data Files

| File | Description | In repo |
|---|---|---|
| `Data/Conflict.csv` | Raw ACLED conflict events export | No (gitignored) |
| `Data/adm2_pop_area.csv` | Admin-2 gridded population by year (2010–2020) | No |
| `Data/Plot_dataset.dta` | LSMS plot-level yield data | No |
| `Data/Plotcrop_dataset.dta` | LSMS plot-crop data (harvest months) | No |
| `Data/EE_harvest_ml_full_panel.csv` | Step 1 output — download from Google Drive | No |
| `Data/all_data_with_predictions.csv` | Step 2b output — v1 ML predictions | No |
| `Data/predictions_v2.csv` | Step 2c output — v2 predictions (GADM) | No |
| `Data/conflict_yields_panel.csv` | Step 3 output — panel with v1 yields | No |
| `Data/conflict_yields_panel_v2.csv` | Step 2d output — panel with v2 yields | No |
| `Data/models_v3_agg/` | Pre-trained v1 XGB and RF models + imputers | No |
| `Code/generate_predictions_v2.py` | v2 prediction script (run on armazi) | Yes |
| `Code/update_panel_v2.R` | Merges v2 predictions into panel | Yes |
| `Code/match_districts.R` | Diagnostic: GADM–GAUL district name matching | Yes |
| `lsms_crop_yields/train_models.py` | Original model training pipeline (reference only) | Yes |

Data files are gitignored. The Google Drive source files and armazi data are not stored in this repo.

---

## rgee Authentication

Step 3 requires a live Earth Engine connection via `rgee`. If `ee_Initialize()` fails:

```r
library(rgee)
ee_install()          # installs the Python EE environment (first time only)
ee_Initialize()       # opens browser for Google account auth
```

---

## Dependencies

**Python (Steps 2b, 2c):**
```
pandas, numpy, joblib, scikit-learn, xgboost
```

**R (Steps 2d, 3, 4):**
```
rgee, sf, dplyr, readr, slider, lubridate, tidyr, fixest, stringi
```

---

## Key Results

Using v2 XGBoost predictions (spatial R² ≈ 0.35), the preferred log-log TWFE OLS specification finds β = 1.18 (p = 0.003), suggesting a 1% increase in predicted yield is associated with a 1.18% increase in log-conflict events in the 3-month forward window. The TWFE Poisson specification is not significant (β = -0.94, p = 0.59). Results are heterogeneous across countries: Nigeria shows a positive significant effect under Poisson (β = 4.28, p = 0.008) while Ethiopia, Mali, and Uganda show no significant effect. Results should be interpreted cautiously given the limited within-district year-to-year variation captured by the satellite features.
