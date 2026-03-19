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
- Key regressor: ML-predicted crop yield (maize primary; also sorghum, millet as
  robustness) from Data/all_data_with_predictions.csv
- Controls: Population data each year per Admin-2 found in Data/adm2_pop_area.csv
- Forward conflict windows: 3-month, 6-month, and 12-month; preferred specification
  is 3-month, others are robustness checks

## Running Code

All scripts must be run from the project root (benson_thesis/).

Full pipeline order (must be run in sequence):

  # Step 1 — generate satellite variables
  jupyter nbconvert --to notebook --execute Code/Satellite_Year_Panel.ipynb

  # Step 2 — generate yield predictions
  jupyter nbconvert --to notebook --execute Code/Yield_Predictions.ipynb

  # Step 3 — process conflict data and merge with yields
  Rscript Code/Conflict_Yields.R

  # Step 4 — run regression analysis
  Rscript Code/Reg_analysis.R

Note: Reg_analysis.R depends on outputs from all prior steps. Do not run it
standalone unless Data/all_data_with_predictions.csv and the conflict panel
already exist.

## Pipeline & Data Flow

### Step 1 — Satellite Variable Generation (Code/Satellite_Year_Panel.ipynb)

For each LSMS survey point, generates satellite variables for all years (2010–2024)
using the most frequently occurring harvest end month per point. Satellite variables
are constructed over the 12 months preceding the harvest end month.

Output: Data/Satellite_Year_Panel.csv — one row per point × year, with all
satellite and weather variables needed as model inputs.

### Step 2 — Yield Predictions (Code/Yield_Predictions.ipynb)

Pre-trained models are already available in models_v3/ (no retraining needed).

Model files present:
- models_v3/model_XGB_GLOBAL.joblib — global model (preferred for most countries)
- models_v3/model_XGB_Malawi.joblib — country model (preferred for Malawi)
- models_v3/model_RF_GLOBAL.joblib
- models_v3/imputer_GLOBAL.joblib
- models_v3/vt_GLOBAL.joblib
- Per-country XGB and RF models + imputers for all 7 countries

Note: per-country vt_ files were not saved. For Malawi's country model, refit
VarianceThreshold(threshold=1e-5) on the Malawi subset at prediction time.

Workflow:
1. Load Data/Satellite_Year_Panel.csv (output of Step 1)
2. Read train_models.py from lsms_crop_yields/ to understand the exact feature
   engineering pipeline and apply that same pipeline to the satellite panel
3. Generate predictions using hybrid model selection (best spatial R² per country):

   FOR Malawi:
     - Filter rows to Malawi
     - Load models_v3/imputer_Malawi.joblib
     - Apply imputer
     - Refit VarianceThreshold(threshold=1e-5) on the Malawi subset
     - Load models_v3/model_XGB_Malawi.joblib
     - Generate predictions

   FOR Ethiopia, Mali, Niger, Nigeria, Tanzania, Uganda:
     - Filter rows to each country
     - Load models_v3/imputer_GLOBAL.joblib and models_v3/vt_GLOBAL.joblib
     - Load models_v3/model_XGB_GLOBAL.joblib
     - Generate predictions

4. Back-transform all predictions with np.expm1()
5. Stack all 7 country prediction sets into one dataframe
6. Add a column model_used indicating which model was applied per row
   (e.g. 'XGB_Malawi' or 'XGB_GLOBAL')

Output: Data/all_data_with_predictions.csv with columns:
country, admin2_id, year, predicted_yield_xgb, model_used

Sanity checks to run after saving:
- Row count per country and year — flag any gaps in 2010-2024 coverage
- Null count in predicted_yield_xgb — should be zero
- Min, mean, max predicted yield per country after back-transform
  — flag anything outside 200-8000 kg range as suspicious
- Confirm all 7 countries present
- Print a warning if Nigeria or Uganda predictions look unreliable
  (these countries have near-zero model R² — flag for thesis)

### Step 3 — Conflict Processing & Yield Merge (Code/Conflict_Yields.R)

1. Loads Data/Conflict.csv (ACLED raw export)
2. Spatially joins conflict events to FAO/GAUL admin-2 boundaries via Earth Engine
3. Aggregates conflict counts by admin-2 × year × conflict type
4. Loads Data/all_data_with_predictions.csv, spatially joins to admin-2 polygons
5. Merges conflict panel with yield predictions
6. Loads Data/adm2_pop_area.csv and merges population controls
7. Computes 3-month, 6-month, and 12-month forward conflict windows from conflict
   end months

Output: merged admin-2 × year panel ready for regression

### Step 4 — Regression Analysis (Code/Reg_analysis.R)

Loads the final merged panel and runs the following model specifications:
- Preferred: 3-month forward conflict window, maize predicted yield, population control
- Robustness: 6-month and 12-month windows; RF predictions as alternative regressor
- All specs include admin-2 and year fixed effects

## Models

Pre-trained models are in models_v3/. Do not retrain unless explicitly instructed.

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

Known limitations (flag in thesis):
- Nigeria and Uganda: near-zero R² under both spatial and temporal CV.
  High within-country heterogeneity and complex multi-season cropping
  systems limit satellite-based prediction quality for these countries.
- Temporal R² (~0.10) is lower than spatial R² (~0.17), reflecting
  year-to-year shifts in the satellite-yield relationship not captured
  by features.

## Data Files

  File                                  | Description
  --------------------------------------|------------------------------------------
  Data/Conflict.csv                     | Raw ACLED conflict events export
  Data/adm2_pop_area.csv                | Admin-2 population by year (control)
  Data/Satellite_Year_Panel.csv         | Output of Step 1
  Data/all_data_with_predictions.csv    | Output of Step 2 — ML yield predictions
  Data/[final merged panel]             | Output of Step 3 — regression panel

## Countries in Scope

Ethiopia, Malawi, Mali, Niger, Nigeria, Tanzania, Uganda.
All scripts should validate that outputs contain exactly these 7 countries — flag
any missing or unexpected country codes.
