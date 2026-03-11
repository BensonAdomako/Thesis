# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Economics thesis investigating whether crop yields causally affect conflict in 7 Sub-Saharan African countries (Ethiopia, Malawi, Mali, Niger, Nigeria, Tanzania, Uganda) using an admin-2 district × year panel from 2010–2024. Predicted yields from satellite/weather ML models serve as the key regressor to avoid reverse causality.

## Running Code

All scripts must be run from the project root (`benson_thesis/`):

```bash
Rscript Code/thesis_regressions.R    # Main regression analysis (standalone)
```

The earlier pipeline scripts (`Conflict_Counts.r`, `Conflict_Counts2.r`, `Yields_Conf_Merged.r`) require Google Earth Engine authentication via `rgee::ee_Initialize()` and are not typically re-run.

`Reg analysis.r` is a legacy script superseded by `thesis_regressions.R`.

## Pipeline & Data Flow

Three stages, run in order:

1. **Conflict processing** (`Conflict_Counts.r` / `Conflict_Counts2.r`): Loads `Data/Conflict.csv` (ACLED), spatially joins to FAO/GAUL admin-2 boundaries via Earth Engine, aggregates conflict counts by admin-year-type. Outputs `Data/conflicts_admin_year_type.csv`.

2. **Yields-conflict merge** (`Yields_Conf_Merged.r`): Loads yield predictions (`Data/all_data_with_predictions.csv`), spatially joins to admin-2 polygons, merges with conflict panel, computes 3/6/12-month forward conflict windows. Outputs `Data/Conflict_Yields.csv`.

3. **Regression analysis** (`thesis_regressions.R`): Loads the final dataset `Data/reg_final.csv` and runs 6 model specifications (OLS, Negative Binomial, Hurdle NB, PPML with admin FE, PPML + lagged conflict, PPML + yield×distance interaction). This is the main deliverable script.

## Key R Packages

- **fixest**: `feols`, `fepois`, `fenegbin` for fixed-effects estimation (Models 1-2, 4-6)
- **pscl**: `hurdle` for hurdle negative binomial (Model 3)
- **rgee** + **sf**: Earth Engine API access and spatial joins (Stages 1-2 only)
- **slider**: Rolling window aggregation for forward conflict counts

## Data

The `Data/` directory is gitignored. Key files:
- `reg_final.csv` — Final regression-ready panel (used by `thesis_regressions.R`)
- `Conflict.csv` — Raw ACLED conflict events
- `all_data_with_predictions.csv` — ML yield predictions by country
- `conflicts_admin_year_type.csv` — Intermediate: conflict counts by admin-year-type
