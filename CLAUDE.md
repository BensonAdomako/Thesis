# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Economics thesis investigating whether crop yields causally affect conflict in 6
Sub-Saharan African countries (Ethiopia, Malawi, Mali, Nigeria, Tanzania, Uganda)
using an admin-2 district × year panel from 2010–2024.

Niger is excluded from predictions — both XGB_Niger (R²=-0.265) and XGB_GLOBAL
applied to Niger (R²=-0.168) produce negative out-of-sample R². Niger had only 16
LSMS training observations. Flag this as a thesis limitation.

Predicted yields from satellite/weather ML models serve as the key regressor to avoid
reverse causality. Conflict data comes from ACLED (Data/Conflict.csv).

## Key Variables

- Unit of analysis: Admin-2 district × year
- Outcome variable: Conflict event count per district-year (from ACLED); also run
  as fatality count as robustness check
- Key regressor: ML-predicted crop yield (maize primary); v2 predictions used
  (log_pred_yield_v2) from Data/conflict_yields_panel_v2.csv
- Controls: Population data each year per Admin-2 found in Data/adm2_pop_area.csv
- Forward conflict windows: 3-month, 6-month, and 12-month; preferred specification
  is 3-month, others are robustness checks
- Binary conflict indicators (any_conflict_3/6/12mo) are identical across windows
  by construction and have been dropped from robustness checks as uninformative

## Running Code

All scripts must be run from the project root (benson_thesis/).

**Important:** Steps 1 and 2 are Google Colab notebooks and cannot be run locally.
They depend on large data files stored in Google Drive and use google.colab APIs.
The local pipeline begins at Step 2b (generate_predictions.py), once the Step 1
output has been downloaded from Google Drive into Data/.

Full pipeline:

  # Step 1 — generate satellite panel [COLAB ONLY]
  # Run Satellite_Year_Panel.ipynb in Google Colab.
  # Source data: EE_combined_long.parquet on Google Drive (pre-extracted satellite data).
  # Output: EE_harvest_ml_full_panel.csv — download this to Data/ before Step 2b.

  # Step 2 — generate yield predictions [COLAB ONLY — for reference]
  # Yield_Predictions.ipynb also runs in Colab. See discrepancy note in Models section.
  # The authoritative local prediction step is Step 2b below.

  # Step 2b — generate yield predictions locally (run after downloading Step 1 output)
  python Code/generate_predictions.py

  # Step 2c — generate v2 predictions on armazi server [SERVER ONLY]
  # SSH into armazi, then:
  python generate_predictions_v2.py
  # Then scp back:
  scp biadomako@armazi:~/lsms_crop_yields/predictions_v2.csv ~/Desktop/Thesis/Data/

  # Step 2d — merge v2 predictions into panel
  Rscript Code/update_panel_v2.R

  # Step 3 — process conflict data and merge with yields
  Rscript "Code /Conflict_Yields.r"

  # Step 4 — run regression analysis (uses v2 panel)
  Rscript "Code /Reg_analysis.r"

Note: generate_predictions.py depends on Data/EE_harvest_ml_full_panel.csv (Step 1
output from Colab). Do not run it until that file is downloaded to Data/.
Reg_analysis.r reads Data/conflict_yields_panel_v2.csv (v2 panel) and depends
on all prior steps.

## Pipeline & Data Flow

### Step 1 — Satellite Variable Generation (Satellite_Year_Panel.ipynb) [COLAB ONLY]

**Must be run in Google Colab. Cannot be run locally.**

Source data on Google Drive:
- EE_combined_long.parquet — pre-extracted Earth Engine satellite data (all bands,
  all months 2010–2024) for all LSMS survey point locations. This file is large and
  lives only on Google Drive; it is not in this repo.
- LSMS Data/Plotcrop_dataset.dta and Plot_dataset.dta — also read from Drive paths.

No live Google Earth Engine API connection is required — all satellite data is
already extracted and stored in EE_combined_long.parquet.

What it does: For each LSMS survey point, computes the mode harvest month across
waves, then constructs 12-month lag features (lags 0–11) relative to each
synthetic harvest date for every year 2010–2024. Matches satellite data to survey
points via nearest-neighbour spatial join (tolerance 1e-4°).

Output saved to Google Drive: EE_harvest_ml_full_panel.csv
  - Shape: ~98,280 rows × 90 columns
  - Columns: country, year, lat_modified, lon_modified, harvest_end_month,
    harvest_end_month_dt, plus lag features (e.g. EVI_max_0 … EVI_max_11,
    NDVI_max_0 … KDD_mean_11, temperature_2m_max_mean_0 …)

After Colab run: download EE_harvest_ml_full_panel.csv to Data/ in this repo.

### Step 2b — v1 Yield Predictions (Code/generate_predictions.py) [LOCAL]

Reads Data/EE_harvest_ml_full_panel.csv and generates v1 predictions using the
hybrid model selection approach (see Models section — v1).

Output: Data/all_data_with_predictions.csv

### Step 2c — v2 Yield Predictions (Code/generate_predictions_v2.py) [armazi SERVER]

**Must be run on armazi server** — depends on admin_features_v2.parquet and
v2 models in /home/ahobbs/lsms_crop_yields/.

What it does: loads admin_features_v2.parquet (35,230 × 1,463 columns), filters to
6 thesis countries and 2010–2024, runs two XGBoost pipelines:
  - Absolute: imputer_v2 → vt_v2 → model_XGB_v2 → log scale predictions
  - Anomaly: imputer_v2_anomaly → vt_v2_anomaly → model_XGB_v2_anomaly → z-score

Back-transform: expm1(log_pred) × 1000 = kg/ha

Output: predictions_v2.csv with columns:
  GID_0, GID_1, GID_2, NAME_1, NAME_2, year,
  predicted_yield_abs_kgha, log_predicted_yield_abs, predicted_yield_anomaly

Copy back to thesis: scp biadomako@armazi:~/lsms_crop_yields/predictions_v2.csv Data/

### Step 2d — Merge v2 Predictions (Code/update_panel_v2.R) [LOCAL]

Merges Data/predictions_v2.csv (GADM GID_2 × year) into Data/conflict_yields_panel.csv
(GAUL admin-2 × year). Handles GADM→GAUL name mismatch with 3-pass matching:
  Pass 1: normalised exact match (diacritics stripped, lowercase)
  Pass 2: suffix-stripped match (boma, city, town, urban, municipal etc.)
  Pass 3: manual crosswalk (Ethiopia directional translations, Nigeria typos)

Coverage: 93.3% of districts matched (861/926). Unmatched districts get NA yield
and are dropped from regressions.

Output: Data/conflict_yields_panel_v2.csv
  - 13,890 rows × 25 cols (original panel + pred_yield_v2_kgha, log_pred_yield_v2,
    pred_yield_anomaly)
  - 17% NA on log_pred_yield_v2 (unmatched districts — dropped in Step 4)

### Step 3 — Conflict Processing & Yield Merge (Code /Conflict_Yields.r)

1. Loads Data/Conflict.csv (ACLED raw export)
2. Spatially joins conflict events to FAO/GAUL admin-2 boundaries via Earth Engine
3. Aggregates conflict counts by admin-2 × year × conflict type
4. Loads Data/all_data_with_predictions.csv, spatially joins to admin-2 polygons
5. Merges conflict panel with yield predictions
6. Loads Data/adm2_pop_area.csv and merges population controls
7. Computes 3-month, 6-month, and 12-month forward conflict windows from conflict
   end months

Output: Data/conflict_yields_panel.csv — merged admin-2 × year panel (v1 yields)

**Known issue — binary conflict windows:** any_conflict_3/6/12mo are all identical
because any district with conflict in the 3mo window trivially has conflict in the
6mo and 12mo windows (supersets). Binary window robustness has been dropped.

### Step 4 — Regression Analysis (Code /Reg_analysis.r)

Loads Data/conflict_yields_panel_v2.csv (v2 predictions) and runs:
- Preferred: 3-month forward window, log-log OLS + TWFE OLS + TWFE Poisson
- Robustness: 6-month and 12-month count windows; fatality Poisson
- Per-country: log-log OLS and Poisson separately for each of the 6 countries
- Key regressor: log_pred_yield_v2 (v2 XGBoost predictions)
- All TWFE models: Admin-2 + year FEs, SEs clustered at Admin-2

Outputs:
  Data/regression_results.tex              — preferred specs
  Data/regression_results_robustness.tex   — robustness checks
  Data/regression_results_bycountry_ols.tex    — per-country log-log OLS
  Data/regression_results_bycountry_poisson.tex — per-country Poisson

## Models

### v2 Models (current — used in regressions)

Trained on GROW-Africa survey data with richer satellite features including SAR
(Sentinel-1 VV/VH), LSWI, GCVI, and longer temporal coverage. Spatial R² ≈ 0.35
(vs v1 range of 0.016–0.192).

Models stored on armazi: /home/ahobbs/lsms_crop_yields/models_v2/
  - imputer_v2.joblib, vt_v2.joblib, model_XGB_v2.joblib  (absolute mode)
  - imputer_v2_anomaly.joblib, vt_v2_anomaly.joblib, model_XGB_v2_anomaly.joblib

Both models use 210 features (post-VT). The 20 admin embedding PCs (emb_PC_0..19)
were set to NaN at prediction time (no saved PCA transformer); imputer fills with
training medians.

Single global model applied to all 6 countries.

### v1 Models (reference only — not used in main regressions)

Pre-trained models in Data/models_v3_agg/.
Do not retrain unless explicitly instructed.

Hybrid model selection (best spatial OOF R² per country, from r2_matrix.csv):

  Country     | Model                        | Spatial R²
  ------------|------------------------------|------------
  Ethiopia    | model_XGB_Ethiopia.joblib    | 0.066
  Malawi      | model_XGB_Malawi.joblib      | 0.178
  Mali        | model_XGB_GLOBAL.joblib      | 0.192
  Nigeria     | model_XGB_GLOBAL.joblib      | 0.018
  Tanzania    | model_XGB_Tanzania.joblib    | 0.092
  Uganda      | model_XGB_Uganda.joblib      | 0.016

  Niger is EXCLUDED (XGB_Niger R²=-0.265; XGB_GLOBAL on Niger R²=-0.168).

Known v1 limitations (flag in thesis):
- Nigeria and Uganda: near-zero R² — high within-country heterogeneity
- Precipitation and soil moisture: 100% null in source extract (precip_missing=TRUE)
- Temporal R² (~0.10) lower than spatial R² (~0.17)

## Data Files

  File                                      | Description
  ------------------------------------------|------------------------------------------
  Data/Conflict.csv                         | Raw ACLED conflict events export
  Data/adm2_pop_area.csv                    | Admin-2 population by year (control)
  Data/Plot_dataset.dta                     | LSMS plot-level yield data
  Data/Plotcrop_dataset.dta                 | LSMS plot-crop data (harvest months etc.)
  Data/EE_harvest_ml_full_panel.csv         | Output of Step 1 (download from Drive)
  Data/all_data_with_predictions.csv        | Output of Step 2b — v1 ML predictions
  Data/predictions_v2.csv                   | Output of Step 2c — v2 predictions (GADM)
  Data/conflict_yields_panel.csv            | Output of Step 3 — panel with v1 yields
  Data/conflict_yields_panel_v2.csv         | Output of Step 2d — panel with v2 yields
  Data/models_v3_agg/                        | Pre-trained v1 XGB + RF models + imputers
  Code/generate_predictions_v2.py           | v2 prediction script (runs on armazi)
  Code/update_panel_v2.R                    | Merges v2 predictions into panel
  Code/match_districts.R                    | Diagnostic: GADM-GAUL name matching
  lsms_crop_yields/train_models.py          | Original model training pipeline (reference)

  Google Drive (not in repo):
  EE_combined_long.parquet                  | Pre-extracted satellite data — Step 1 input
  EE_harvest_ml_full_panel.csv              | Step 1 output — download to Data/ before Step 2b

  armazi server:
  /home/ahobbs/lsms_crop_yields/models_v2/  | v2 XGBoost models and preprocessors
  /home/ahobbs/lsms_crop_yields/data/admin_features_v2.parquet | v2 feature matrix

## Countries in Scope

Ethiopia, Malawi, Mali, Nigeria, Tanzania, Uganda (6 countries).
Niger is excluded — negative R² for all models due to only 16 LSMS training
observations. Note this as a thesis limitation.
All scripts should validate that outputs contain exactly these 6 countries — flag
any missing or unexpected country codes.
