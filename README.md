# Crop Yields and Conflict in Sub-Saharan Africa

Economics thesis investigating whether predicted crop yields causally affect conflict
in seven Sub-Saharan African countries using an Admin-2 district × year panel from
2010–2024.

**Countries:** Burkina Faso, Ethiopia, Mali, Malawi, Niger, Nigeria, Tanzania
(Uganda excluded — not in model training data, LOCO R² < 0)

---

## Research Design

- **Unit of analysis:** GADM Admin-2 district × year (18,096 obs, 1,338 districts)
- **Outcome:** ACLED conflict event count; 3-month post-harvest forward window (preferred)
- **Key regressor:** v2 XGBoost-predicted maize yield (kg/ha), trained on GROW-Africa
  administrative yield records; spatial R² = 0.35
- **Controls:** Log WorldPop gridded population per Admin-2
- **Fixed effects:** Admin-2 + year (two-way)
- **Preferred estimator:** TWFE OLS log-log

**Main result:** β = −0.666 (p < 0.001): a 10% increase in predicted yield is
associated with a 6.7% reduction in log conflict events in the 3-month post-harvest
window. The result survives spatial lag control (β = −0.377, p < 0.001) and
strengthens at longer horizons (12-month: β = −1.290, p < 0.001).

**Novel finding — AEZ heterogeneity:** The yield–conflict relationship is negative
in Non-Sahel countries (ETH, MWI, NGA, TZA: β = −0.467, p < 0.001, opportunity-cost
channel) but positive in Sahel countries (BFA, MLI, NER: net β = +2.383, p < 0.001,
resource-predation channel). The interaction term (yield × Sahel) = +3.513 (p < 0.001).

---

## Pipeline

Run all scripts from the project root.

### Step 1 — v2 Yield Predictions (armazi server)

```bash
# On armazi:
python generate_predictions_v2.py
# Copy back:
scp biadomako@armazi:~/lsms_crop_yields/predictions_v2.csv Data/
```

Generates district-year predicted maize yield (kg/ha) and anomaly z-score for all
7 countries, 2010–2024, using the GROW-Africa XGBoost model (210 satellite/climate/
soil features, spatial R² = 0.35).

**Output:** `Data/predictions_v2.csv`

---

### Step 2 — Build Conflict-Yields Panel

```bash
Rscript Code/build_panel_v3.R
```

Builds the full analysis panel from scratch:
- GROW-Africa harvest months (area-weighted mode per Admin-2)
- GADM Level-2 shapefiles downloaded via `geodata::gadm()`
- ACLED events spatially joined to GADM polygons (99.8% match)
- Full district × month grid with zeros, then 3/6/12-month sliding windows
- WorldPop population matched via 3-pass name crosswalk

**Output:** `Data/conflict_yields_panel_v3.csv` (18,096 rows × 19 cols)

| Column | Description |
|---|---|
| `GID_0` | ISO3 country code |
| `GID_2` | GADM Admin-2 code (unique identifier) |
| `year` | Year (2010–2024) |
| `log_pred_yield_v2` | Log predicted maize yield (kg/ha) |
| `conflict_3mo` | ACLED events in 3-month post-harvest window |
| `conflict_6mo` / `conflict_12mo` | 6- and 12-month window counts |
| `fatalities_3mo` | Fatalities in 3-month window |
| `log_pop` | Log WorldPop population |
| `harvest_month` | District harvest month (from GROW-Africa) |

---

### Step 3 — Regression Analysis

```bash
Rscript "Code /Reg_analysis.r"
```

Runs all specifications and saves 7 LaTeX tables to `Data/`:

| Output file | Contents |
|---|---|
| `regression_results.tex` | Main results: pooled OLS, TWFE OLS, TWFE Poisson, TWFE log-log |
| `regression_results_robustness.tex` | Alt windows (3/6/12mo log-log) + Poisson + fatalities |
| `regression_results_spatial_lag.tex` | Baseline vs spatial-lag-augmented (OLS + Poisson) |
| `regression_results_conflict_types.tex` | By ACLED event type: battles/riots/VAC/explosions |
| `regression_results_aez.tex` | Sahel vs Non-Sahel split + interaction model |
| `regression_results_bycountry_ols.tex` | Per-country TWFE OLS log-log |
| `regression_results_bycountry_poisson.tex` | Per-country TWFE Poisson |

---

## Key Results

| Specification | β (log yield) | SE | p |
|---|---|---|---|
| Pooled OLS | −0.650 | 0.211 | 0.002 |
| TWFE OLS | −3.750 | 1.155 | 0.001 |
| TWFE Poisson | −2.629 | 0.855 | 0.002 |
| **TWFE OLS log-log (preferred)** | **−0.666** | **0.153** | **<0.001** |
| + Spatial lag control | −0.377 | 0.110 | <0.001 |
| 6-month window | −0.949 | 0.193 | <0.001 |
| 12-month window | −1.290 | 0.236 | <0.001 |

**Conflict type heterogeneity (TWFE OLS log-log):**

| Type | β | p |
|---|---|---|
| Battles | −0.309 | 0.002 |
| Riots | −0.005 | 0.887 |
| Violence vs civilians | −0.524 | <0.001 |
| Explosions/remote | −0.155 | 0.020 |

**AEZ heterogeneity:**

| Sample | β (OLS) | p |
|---|---|---|
| Non-Sahel (ETH/MWI/NGA/TZA) | −0.467 | <0.001 |
| Sahel (BFA/MLI/NER) | −0.877 | 0.212 |
| Interaction: yield × Sahel | +3.513 | <0.001 |
| **Net Sahel effect** | **+2.383** | — |

---

## Model

| Property | Value |
|---|---|
| Architecture | Global XGBoost |
| Training data | GROW-Africa admin yield records (3,363 obs, 204 units, 2000–2023) |
| Countries | BFA, ETH, MLI, MWI, NER, NGA, TZA |
| Features | 210 (satellite composites, CHIRPS, ERA5, iSDA soil, SRTM terrain) |
| Satellite inputs | Landsat, Sentinel-1 SAR, Sentinel-2, MODIS, CHIRPS, ERA5 |
| Spatial R² | 0.35 |
| Stacked ensemble R² | 0.39 |
| Location | armazi: `/home/ahobbs/lsms_crop_yields/models_v2/` |

---

## Data Files

Data files are gitignored. Key files needed to run the pipeline:

| File | Source |
|---|---|
| `Data/acled_africa.csv` | ACLED full-Africa download (2010–2024) |
| `Data/admn2_pop.csv` | WorldPop via Google Earth Engine (see GEE script in repo) |
| `Data/predictions_v2.csv` | Step 1 output — copy from armazi |
| `Data/grow_maize_yields.csv` | GROW-Africa training data — copy from armazi |

**Note on population export:** The GEE script `Code/gee_population_export.js` uses
`ee.Filter.eq('year', useYear)` with a client-side loop to correctly filter the
WorldPop collection. The `system:index` in `WorldPop/GP/100m/pop` uses
`COUNTRY_YEAR` format (e.g. `NGA_2015`), not plain year strings — filtering by
`system:index` returns an empty collection and produces all-zero sums.

---

## Write-up

LaTeX source files in `Write up/`:

| File | Contents |
|---|---|
| `04_data.tex` | Panel structure, ACLED, GROW-Africa model, AEZ classification, summary stats |
| `05_empirical.tex` | Identification strategy, specifications |
| `06_results.tex` | All results: main, robustness 1–2, heterogeneity 1–2, per-country |
| `07_discussion.tex` | Opportunity-cost vs predation channels, limitations |

---

## Dependencies

**R:** `fixest`, `dplyr`, `readr`, `sf`, `spdep`, `geodata`, `lubridate`,
`slider`, `tidyr`, `purrr`

**Python (armazi):** `pandas`, `numpy`, `joblib`, `scikit-learn`, `xgboost`
