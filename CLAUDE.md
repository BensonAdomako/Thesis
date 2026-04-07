# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Economics thesis investigating whether crop yields causally affect conflict in 7
Sub-Saharan African countries (Burkina Faso, Ethiopia, Mali, Malawi, Niger,
Nigeria, Tanzania) using an Admin-2 district × year panel from 2010–2024.

**Uganda is excluded** — not in GROW-Africa training data; LOCO R² < 0.

Predicted yields from the v2 XGBoost model (trained on GROW-Africa administrative
yield records, spatial R² = 0.35) serve as the key regressor. Conflict data comes
from ACLED (Data/acled_africa.csv).

## Key Variables

- Unit of analysis: GADM Admin-2 district × year (18,096 obs, 1,338 districts)
- Panel file: Data/conflict_yields_panel_v3.csv
- Outcome: conflict event count per district-year (ACLED); 3-month forward window
  preferred; 6-month and 12-month used as robustness checks; fatality count as
  additional robustness
- Key regressor: log_pred_yield_v2 (log of v2 ML-predicted maize yield, kg/ha)
- Controls: log_pop (log of WorldPop gridded population per Admin-2; source:
  Data/admn2_pop.csv; 2021–2024 carried forward from 2020)
- Binary conflict indicators dropped — identical across windows by construction
- IV/2SLS dropped — first-stage F < 2 across all specs

## AEZ Classification

- Sahel zone: BFA, MLI, NER (semi-arid, rain-fed, agro-pastoral)
- Non-Sahel zone: ETH, MWI, NGA, TZA (higher rainfall, highland, Guinea Savanna)
- KEY RESULT: Non-Sahel = opportunity-cost channel (β = −0.47, p < 0.001)
  Sahel = resource-predation channel (net β = +2.38, interaction p < 0.001)
  The Sahel net effect is POSITIVE — better harvests attract armed groups.

## Key Results (v3 panel)

Preferred spec: TWFE OLS log-log, 3-month window
- β = −0.666 (SE = 0.153, p < 0.001): 10% yield increase → 6.7% fewer conflict events
- Survives spatial lag control: β = −0.377 (p < 0.001)
- Strengthens at longer horizons: 6mo β = −0.949, 12mo β = −1.290 (both p < 0.001)
- Conflict type: strongest for violence vs civilians (β = −0.524), null for riots
- AEZ interaction: yield × Sahel = +3.513 (p < 0.001); net Sahel = +2.383
- Ethiopia anomaly: within-ETH β = +1.719 (p = 0.009) — Tigray/Amhara confound

## Running Code

All scripts must be run from the project root.

Full pipeline:

  # Step 1 — Yield predictions (armazi server)
  python generate_predictions_v2.py          # on armazi
  scp biadomako@armazi:~/lsms_crop_yields/predictions_v2.csv Data/

  # Step 2 — Rebuild conflict-yields panel (local)
  Rscript Code/build_panel_v3.R

  # Step 3 — Run all regressions and save 7 LaTeX tables (local)
  Rscript "Code /Reg_analysis.r"

  # Exploratory only — NOT part of main pipeline
  Rscript Code/test_extensions.R

## Pipeline & Data Flow

### build_panel_v3.R

Builds Data/conflict_yields_panel_v3.csv from scratch. Steps:
1. Extract GROW-Africa harvest months (area-weighted mode per district)
2. Download GADM Level-2 shapefiles via geodata::gadm() → cached in Data/gadm/
3. Spatial join ACLED events to GADM polygons (st_within, 99.8% match)
4. Build full GID_2 × month grid (zeros filled), compute 3/6/12-month forward
   windows using slider::slide_dbl(.before=0, .after=2/5/11)
5. Build panel spine from predictions_v2.csv; drop water-body units
6. Join conflict windows at harvest_date
7. Merge population from Data/admn2_pop.csv with 3-pass name matching:
   - MWI: GADM NAME_1 (parent district) ↔ GAUL ADM2_NAME
   - ETH: normalised match + manual Amharic crosswalk
   - NGA/NER: normalised match + suffix-stripped + manual crosswalk
   - Others: normalised exact + suffix-stripped

Population file: Data/admn2_pop.csv (WorldPop via GEE; NOT adm2_pop_area.csv
which has all-zero sum column due to wrong system:index filter in GEE script)

### Reg_analysis.r

Generates 7 LaTeX tables in Data/:
  regression_results.tex              — 4 main specs (preferred: col 4 TWFE OLS log-log)
  regression_results_robustness.tex   — alt windows (3/6/12mo log-log) + count Poisson + fatality Poisson
  regression_results_spatial_lag.tex  — baseline vs spatial-lag-augmented (OLS + Poisson)
  regression_results_conflict_types.tex — by ACLED event type (battles/riots/VAC/explosions)
  regression_results_aez.tex          — Sahel vs Non-Sahel + interaction
  regression_results_bycountry_ols.tex    — per-country TWFE OLS log-log
  regression_results_bycountry_poisson.tex — per-country TWFE Poisson

Tables are wrapped with \begin{table}...\label{tab:...} in Write up/06_results.tex.

### Harvest months (GROW-Africa weighted mode)

ETH=January, BFA/MLI/NER=December, MWI=May, NGA=October, TZA=July

## Models (v2 — current)

Trained on GROW-Africa administrative yield records:
- 3,363 obs, 204 admin units, 2000–2023
- 7 countries: BFA, ETH, MWI, MLI, NER, NGA, TZA
- 210 features: Landsat/Sentinel-1/2 composites (EVI, NDVI, GCVI, LSWI, SAR),
  CHIRPS precipitation, ERA5 temperature (GDD, KDD), iSDA soil, SRTM terrain,
  AlphaEarth embedding PCs
- XGBoost Spatial R² = 0.35, Stacked Ensemble R² = 0.39
- LOCO mean R² = −0.34 (poor cross-country generalisation — Uganda excluded)
- Stored on armazi: /home/ahobbs/lsms_crop_yields/models_v2/

## Data Files

  File                               | Description
  -----------------------------------|------------------------------------------
  Data/acled_africa.csv              | Raw ACLED full-Africa export 2010–2024
  Data/admn2_pop.csv                 | WorldPop population by Admin-2 × year (GEE export)
  Data/predictions_v2.csv            | v2 XGBoost predictions (7 countries, 2010–2024)
  Data/conflict_yields_panel_v3.csv  | Main analysis panel (v3 rebuild)
  Data/grow_maize_yields.csv         | GROW-Africa training data (harvest months source)
  Data/gadm/                         | Cached GADM shapefiles (geodata::gadm output)
  Code/build_panel_v3.R              | Panel build script
  Code /Reg_analysis.r               | Regression + table generation script
  Code/test_extensions.R             | Exploratory: spatial lag, types, AEZ (not in pipeline)
  Code/generate_predictions_v2.py    | v2 prediction script (run on armazi)

  NOT in repo (gitignored):
  Data/acled_africa.csv, Data/admn2_pop.csv, Data/predictions_v2.csv,
  Data/conflict_yields_panel_v3.csv, Data/grow_maize_yields.csv, Data/gadm/

## Write-up Files

  Write up/04_data.tex     — Data section (v3 panel, GROW-Africa model, AEZ classification)
  Write up/06_results.tex  — Results (main, robustness 1-2, heterogeneity 1-2, per-country)
  Write up/07_discussion.tex — Discussion (opp-cost channel, predation channel, limitations)

## Countries in Scope

BFA, ETH, MLI, MWI, NER, NGA, TZA (7 countries).
Uganda excluded — not in GROW-Africa training data; LOCO R² < 0.
Do not add Uganda back without re-training the model with Ugandan observations.
Do not re-introduce IV/2SLS specs — first-stage F < 2, instrument is too weak.
Do not re-introduce binary conflict window robustness — all three windows identical.
