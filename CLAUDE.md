# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Economics thesis investigating whether crop yields causally affect conflict in 7
Sub-Saharan African countries (Ethiopia, Malawi, Mali, Niger, Nigeria, Tanzania, Uganda)
using an admin-2 district × year panel from 2010–2024.

Predicted yields from satellite/weather ML models serve as the key regressor to avoid
reverse causality. Conflict data comes from ACLED (Data/Conflict.csv).

## Key Variables

- Unit of analysis: Admin-2 district × year
- Outcome variable: Conflict event count per district-year (from ACLED); also run
  as binary (any conflict) and fatality count as robustness checks
- Key regressor: ML-predicted crop yield (maize primary) from Data/all_data_with_predictions.csv
- Controls: Population data each year per Admin-2 found in Data/adm2_pop_area.csv
- Forward conflict windows: 3-month, 6-month, and 12-month; preferred specification
  is 3-month, others are robustness checks

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

  # Step 3 — process conflict data and merge with yields
  Rscript Code /Conflict_Yields.r

  # Step 4 — run regression analysis
  Rscript Code /Reg_analysis.r

Note: generate_predictions.py depends on Data/EE_harvest_ml_full_panel.csv (Step 1
output from Colab). Do not run it until that file is downloaded to Data/.
Reg_analysis.r depends on outputs from all prior steps.

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

### Step 2b — Yield Predictions (Code/generate_predictions.py) [LOCAL]

Reads Data/EE_harvest_ml_full_panel.csv and generates predictions using the
hybrid model selection approach (see Models section).

Feature engineering: the lag features (EVI_max_0 … temperature_2m_min_mean_11)
are already computed in Step 1. generate_predictions.py applies the imputer and
variance threshold from models_v3/, then predicts with the appropriate model
per country.

Output: Data/all_data_with_predictions.csv with columns:
country, year, lat_modified, lon_modified, harvest_end_month,
predicted_yield_xgb, model_used

### Step 3 — Conflict Processing & Yield Merge (Code /Conflict_Yields.r)

1. Loads Data/Conflict.csv (ACLED raw export)
2. Spatially joins conflict events to FAO/GAUL admin-2 boundaries via Earth Engine
3. Aggregates conflict counts by admin-2 × year × conflict type
4. Loads Data/all_data_with_predictions.csv, spatially joins to admin-2 polygons
5. Merges conflict panel with yield predictions
6. Loads Data/adm2_pop_area.csv and merges population controls
7. Computes 3-month, 6-month, and 12-month forward conflict windows from conflict
   end months

Output: merged admin-2 × year panel ready for regression

### Step 4 — Regression Analysis (Code /Reg_analysis.r)

Loads the final merged panel and runs the following model specifications:
- Preferred: 3-month forward conflict window, maize predicted yield, population control
- Robustness: 6-month and 12-month windows; RF predictions as alternative regressor
- All specs include admin-2 and year fixed effects

## Models

Pre-trained models are in Data/models_v3/ (not models_v3/ at root).
Do not retrain unless explicitly instructed.

Model selection (hybrid — best spatial R² per country):

  Country     | Model                        | Spatial R²
  ------------|------------------------------|------------
  Ethiopia    | model_XGB_GLOBAL.joblib      | 0.174
  Malawi      | model_XGB_Malawi.joblib      | 0.193
  Mali        | model_XGB_GLOBAL.joblib      | 0.174
  Niger       | model_XGB_GLOBAL.joblib      | 0.174
  Nigeria     | model_XGB_GLOBAL.joblib      | 0.174
  Tanzania    | model_XGB_GLOBAL.joblib      | 0.174
  Uganda      | model_XGB_GLOBAL.joblib      | 0.174

For the global model use imputer_GLOBAL.joblib and vt_GLOBAL.joblib.
For Malawi use imputer_Malawi.joblib and refit VarianceThreshold(threshold=1e-5).

Predictions are in log scale — always back-transform with np.expm1().

**Known discrepancy — Yield_Predictions.ipynb vs hybrid approach:**
Yield_Predictions.ipynb (the Colab notebook) used country-specific models for
ALL 7 countries (e.g. model_XGB_Ethiopia.joblib for Ethiopia), with no GLOBAL
model fallback. This differs from the hybrid approach specified above.
generate_predictions.py implements the hybrid approach (GLOBAL for 6 countries,
Malawi-specific for Malawi), which is the more methodologically defensible choice
as it is grounded in the spatial R² comparison in Data/models_v3/r2_matrix.csv.
If re-running for robustness, be aware the Colab notebook used a different model
selection rule.

Known limitations (flag in thesis):
- Nigeria and Uganda: near-zero R² under both spatial and temporal CV.
  High within-country heterogeneity and complex multi-season cropping
  systems limit satellite-based prediction quality for these countries.
- Temporal R² (~0.10) is lower than spatial R² (~0.17), reflecting
  year-to-year shifts in the satellite-yield relationship not captured
  by features.

## Data Files

  File                                    | Description
  ----------------------------------------|------------------------------------------
  Data/Conflict.csv                       | Raw ACLED conflict events export
  Data/adm2_pop_area.csv                  | Admin-2 population by year (control)
  Data/Plot_dataset.dta                   | LSMS plot-level yield data
  Data/Plotcrop_dataset.dta               | LSMS plot-crop data (harvest months etc.)
  Data/EE_harvest_ml_full_panel.csv       | Output of Step 1 (download from Drive)
  Data/all_data_with_predictions.csv      | Output of Step 2b — ML yield predictions
  Data/[final merged panel]               | Output of Step 3 — regression panel
  Data/models_v3/                         | Pre-trained XGB and RF models + imputers
  lsms_crop_yields/train_models.py        | Original model training pipeline (reference)

  Google Drive (not in repo):
  EE_combined_long.parquet                | Pre-extracted satellite data — Step 1 input
  EE_harvest_ml_full_panel.csv            | Step 1 output — download to Data/ before Step 2b

## Countries in Scope

Ethiopia, Malawi, Mali, Niger, Nigeria, Tanzania, Uganda.
All scripts should validate that outputs contain exactly these 7 countries — flag
any missing or unexpected country codes.
