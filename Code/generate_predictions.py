"""
generate_predictions.py
=======================
Step 2b of the thesis pipeline: apply the hybrid XGBoost model to the
satellite panel and produce predicted crop yields for 6 countries.

Niger is EXCLUDED — both XGB_Niger (R²=-0.265) and XGB_GLOBAL applied to
Niger (R²=-0.168) produce negative out-of-sample R². Niger had only 16 LSMS
training observations — too few for reliable prediction. Note this limitation
in the thesis.

Hybrid model selection (best spatial OOF R² per country, from r2_matrix.csv):
  Ethiopia  → model_XGB_Ethiopia.joblib   Spatial R²=0.066
  Malawi    → model_XGB_Malawi.joblib     Spatial R²=0.178
  Mali      → model_XGB_GLOBAL.joblib     Spatial R²=0.192
  Nigeria   → model_XGB_GLOBAL.joblib     Spatial R²=0.018
  Tanzania  → model_XGB_Tanzania.joblib   Spatial R²=0.092
  Uganda    → model_XGB_Uganda.joblib     Spatial R²=0.016

Known limitation: precipitation and soil moisture are 100% missing from
EE_combined_long.parquet (the source satellite data). After imputation, these
features are constant (training medians). All output rows are flagged
precip_missing=True. Flag this in the thesis.

Feature engineering pipeline (must match train_models.py exactly):
  - Rename raw lag columns: EVI_max_N → EVI_N, temperature_2m_max_mean_N → tmax_N, etc.
  - Create precip_N and soil_moist_N as NaN (100% missing from source)
  - Compute seasonal aggregates (gs=lags 3-6, peak=lags 4-5, near=lags 0-2, pre=lags 7-11)
  - Compute interactions: water_balance_gs, aridity_gs
  - Use imputer.feature_names_in_ as ground truth for feature selection and order

Usage:
    python Code/generate_predictions.py

Prerequisites:
    Data/EE_harvest_ml_full_panel.csv   (download from Google Drive after Colab step 1)
    Data/models_v3_agg/                 (pre-trained models)
    Data/Plotcrop_dataset.dta           (LSMS plot-crop data)
    Data/Plot_dataset.dta               (LSMS plot yield data)
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "Data")
MODEL_DIR    = os.path.join(DATA_DIR, "models_v3_agg")

PANEL_CSV    = os.path.join(DATA_DIR, "EE_harvest_ml_full_panel.csv")
PLOTCROP_DTA = os.path.join(DATA_DIR, "Plotcrop_dataset.dta")
PLOT_DTA     = os.path.join(DATA_DIR, "Plot_dataset.dta")
OUTPUT_CSV   = os.path.join(DATA_DIR, "all_data_with_predictions.csv")

# ── Constants ─────────────────────────────────────────────────────────────────

COUNTRIES_6 = ["Ethiopia", "Malawi", "Mali", "Nigeria", "Tanzania", "Uganda"]

# Hybrid model selection: country -> (model_key, uses_global_model)
MODEL_SELECTION = {
    "Ethiopia": ("Ethiopia", False),
    "Malawi":   ("Malawi",   False),
    "Mali":     ("GLOBAL",   True),
    "Nigeria":  ("GLOBAL",   True),
    "Tanzania": ("Tanzania", False),
    "Uganda":   ("Uganda",   False),
}

# Features dropped by VarianceThreshold during training for each country model.
# VT is NOT refit at prediction time — the dropped features are hard-coded here
# based on imputer.statistics_ evidence (median=0 → near-constant in training data).
#
# Tanzania: imputer statistics_ shows KDD_peak/near/pre/trend all have median=0.000,
# meaning KDD was near-constant zero in Tanzania LSMS training data (Tanzania rarely
# exceeds the extreme heat threshold). VT(1e-5) dropped exactly these 4 features.
# Verified: model.n_features_in_=26 = 30 imputer features minus 4 KDD features.
#
# Ethiopia, Malawi, Uganda: model.n_features_in_=30 = imputer output — no VT drop.
VT_DROPPED_FEATURES = {
    "Tanzania": ["KDD_peak_mean", "KDD_near_mean", "KDD_pre_mean", "KDD_trend"],
}


# ── Column rename map ─────────────────────────────────────────────────────────

def build_rename_map():
    """Map EE_harvest_ml_full_panel.csv verbose column names to train_models.py short names."""
    rmap = {}
    for i in range(12):
        rmap[f"EVI_max_{i}"]                 = f"EVI_{i}"
        rmap[f"NDVI_max_{i}"]                = f"NDVI_{i}"
        rmap[f"GCVI_max_{i}"]                = f"GCVI_{i}"
        rmap[f"KDD_mean_{i}"]                = f"KDD_{i}"
        rmap[f"GDD_mean_{i}"]                = f"GDD_{i}"
        rmap[f"temperature_2m_max_mean_{i}"] = f"tmax_{i}"
        rmap[f"temperature_2m_min_mean_{i}"] = f"tmin_{i}"
    return rmap


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df):
    """Compute seasonal aggregate features from raw lag columns.

    Replicates the feature engineering in train_models.py exactly.
    precip_0-11 and soil_moist_0-11 will all be NaN — the imputer will
    fill them with training medians, producing constant columns.

    Parameters
    ----------
    df : DataFrame with renamed lag columns (EVI_0…11, tmax_0…11, etc.)

    Returns
    -------
    X : DataFrame of all engineered features (superset — imputer will select).
    """
    weather_vars = ["precip", "tmax", "tmin", "soil_moist", "KDD"]
    veg_vars     = ["NDVI", "EVI", "GCVI"]
    all_vars     = weather_vars + veg_vars

    X = pd.DataFrame(index=df.index)

    for var in all_vars:
        gs_lags   = list(range(3, 7))   # growing season: lags 3-6
        peak_lags = list(range(4, 6))   # peak season: lags 4-5
        near_lags = list(range(0, 3))   # near-harvest: lags 0-2
        pre_lags  = list(range(7, 12))  # pre-season: lags 7-11

        def lc(lags):
            return [f"{var}_{i}" for i in lags]

        gs   = df[lc(gs_lags)]
        peak = df[lc(peak_lags)]
        near = df[lc(near_lags)]
        pre  = df[lc(pre_lags)]

        if var == "precip":
            X[f"{var}_gs_sum"]   = gs.sum(axis=1)
            X[f"{var}_peak_sum"] = peak.sum(axis=1)
            X[f"{var}_near_sum"] = near.sum(axis=1)
            X[f"{var}_pre_sum"]  = pre.sum(axis=1)
        else:
            X[f"{var}_gs_mean"]   = gs.mean(axis=1)
            X[f"{var}_peak_mean"] = peak.mean(axis=1)
            X[f"{var}_near_mean"] = near.mean(axis=1)
            X[f"{var}_pre_mean"]  = pre.mean(axis=1)

        # Trend: growing-season to harvest (lag 3 minus lag 0)
        X[f"{var}_trend"] = df[f"{var}_3"] - df[f"{var}_0"]

    # Interaction terms
    X["water_balance_gs"]   = X["precip_gs_sum"]  / (X["tmax_gs_mean"]   + 1)
    X["water_balance_peak"] = X["precip_peak_sum"] / (X["tmax_peak_mean"] + 1)
    X["aridity_gs"]         = X["tmax_gs_mean"]   / (X["precip_gs_sum"]  + 1)

    # KDD fraction of GDD (heat stress intensity)
    gdd_gs = df[[f"GDD_{i}" for i in range(3, 7)]].sum(axis=1)
    X["kdd_fraction_gs"] = X["KDD_gs_mean"] * 4 / (gdd_gs + 1)

    # Coordinates (included during training; correlation drop may have removed them)
    X["latitude"]  = df["lat_modified"].values
    X["longitude"] = df["lon_modified"].values

    return X


# ── LSMS observed yield merge ─────────────────────────────────────────────────

def load_observed_yields():
    """Merge Plotcrop + Plot .dta files and return mean observed yield per location-year.

    Returns a DataFrame with columns:
        country, year, lat_modified, lon_modified, observed_yield_kg
    or None if files cannot be loaded.
    """
    print("\n--- Loading LSMS observed yields ---")

    merge_keys = ["wave", "season", "ea_id_merge", "parcel_id_merge", "plot_id_merge"]

    try:
        plotcrop = pd.read_stata(PLOTCROP_DTA)
        print(f"  Plotcrop: {plotcrop.shape[0]:,} rows, {plotcrop.shape[1]} cols")
    except Exception as e:
        print(f"  WARNING: Could not load Plotcrop_dataset.dta: {e}")
        return None

    try:
        plot = pd.read_stata(PLOT_DTA)
        print(f"  Plot:     {plot.shape[0]:,} rows, {plot.shape[1]} cols")
    except Exception as e:
        print(f"  WARNING: Could not load Plot_dataset.dta: {e}")
        return None

    # Merge on shared keys
    common_keys = [k for k in merge_keys if k in plotcrop.columns and k in plot.columns]
    if not common_keys:
        print("  WARNING: No common merge keys found between Plotcrop and Plot datasets")
        return None
    print(f"  Merging on: {common_keys}")

    merged = pd.merge(plotcrop, plot, on=common_keys, how="inner", suffixes=("_crop", "_plot"))
    print(f"  After merge: {merged.shape[0]:,} rows")

    # Identify yield column
    yield_col = None
    for candidate in ["yield_kg", "yield_kg_plot", "yield_kg_crop"]:
        if candidate in merged.columns:
            yield_col = candidate
            break
    if yield_col is None:
        print("  WARNING: No yield_kg column found after merge")
        return None
    print(f"  Using yield column: {yield_col}")

    # Parse harvest date and extract year
    hm_col = None
    for candidate in ["harvest_end_month", "harvest_end_month_crop", "harvest_end_month_plot"]:
        if candidate in merged.columns:
            hm_col = candidate
            break
    if hm_col is None:
        print("  WARNING: harvest_end_month not found")
        return None

    merged["harvest_year"] = pd.to_datetime(merged[hm_col], errors="coerce").dt.year

    # Filter valid years (Tanzania has corrupt dates 1920-2009)
    before = len(merged)
    merged = merged[(merged["harvest_year"] >= 2010) & (merged["harvest_year"] <= 2024)]
    print(f"  After year filter 2010-2024: {len(merged):,} rows "
          f"(dropped {before - len(merged):,} corrupt dates)")

    # Filter valid yields (matches training filter in train_models.py)
    merged = merged[(merged[yield_col] > 0) & (merged[yield_col] < 15_000)]
    print(f"  After yield filter (0 < yield < 15000): {len(merged):,} rows")

    # Resolve country / lat / lon column names after merge suffix
    def resolve_col(base, candidates):
        for c in candidates:
            if c in merged.columns:
                return c
        return None

    ctry_col = resolve_col("country",      ["country", "country_plot", "country_crop"])
    lat_col  = resolve_col("lat_modified", ["lat_modified", "lat_modified_plot", "lat_modified_crop"])
    lon_col  = resolve_col("lon_modified", ["lon_modified", "lon_modified_plot", "lon_modified_crop"])

    group_cols = []
    if ctry_col:
        group_cols.append(ctry_col)
    group_cols.append("harvest_year")
    if lat_col:
        group_cols.append(lat_col)
    if lon_col:
        group_cols.append(lon_col)

    if len(group_cols) < 2:
        print("  WARNING: Cannot identify grouping columns — skipping observed yield merge")
        return None

    agg = merged.groupby(group_cols, as_index=False)[yield_col].mean()

    # Standardise column names
    rename = {yield_col: "observed_yield_kg", "harvest_year": "year"}
    if ctry_col and ctry_col != "country":
        rename[ctry_col] = "country"
    if lat_col and lat_col != "lat_modified":
        rename[lat_col] = "lat_modified"
    if lon_col and lon_col != "lon_modified":
        rename[lon_col] = "lon_modified"
    agg = agg.rename(columns=rename)

    print(f"  Aggregated to {len(agg):,} location-year rows")

    # Diagnostics for problem countries
    for ctry in ["Uganda", "Tanzania"]:
        if "country" in agg.columns:
            n = (agg["country"].astype(str) == ctry).sum()
            if n == 0:
                print(f"  WARNING: {ctry} has 0 observed rows after filtering — "
                      f"check harvest_end_month encoding")
            else:
                print(f"  {ctry} observed yields: {n} location-year rows")

    return agg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", category=UserWarning)   # sklearn version mismatch

    print("=" * 70)
    print("STEP 2b: GENERATE YIELD PREDICTIONS")
    print("=" * 70)
    print(f"Niger excluded — negative R² for all models (only 16 training obs)")
    print(f"Countries in scope: {COUNTRIES_6}")

    # ── 1. Check prerequisites ───────────────────────────────────────────────
    missing_files = [p for p in [PANEL_CSV, MODEL_DIR, PLOTCROP_DTA, PLOT_DTA]
                     if not os.path.exists(p)]
    if missing_files:
        print("\nERROR: Missing required files/directories:")
        for m in missing_files:
            print(f"  {m}")
        sys.exit(1)

    # ── 2. Load satellite panel ──────────────────────────────────────────────
    print(f"\n--- Loading satellite panel ---")
    df = pd.read_csv(PANEL_CSV)
    print(f"  Raw: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Countries in panel: {sorted(df['country'].unique())}")

    # Filter to 6-country scope
    df = df[df["country"].isin(COUNTRIES_6)].copy().reset_index(drop=True)
    print(f"  After Niger exclusion: {df.shape[0]:,} rows, "
          f"countries: {sorted(df['country'].unique())}")

    # ── 3. Rename lag columns to short names ─────────────────────────────────
    print("\n--- Renaming raw lag columns ---")
    rename_map = {k: v for k, v in build_rename_map().items() if k in df.columns}
    df = df.rename(columns=rename_map)
    print(f"  Renamed {len(rename_map)} columns "
          f"(e.g. EVI_max_0 → EVI_0, temperature_2m_max_mean_0 → tmax_0)")

    # ── 4. Add missing precip and soil_moist columns as NaN ─────────────────
    print("\n--- Adding missing precip and soil_moist lag columns (all NaN) ---")
    for band in ["precip", "soil_moist"]:
        added = 0
        for i in range(12):
            col = f"{band}_{i}"
            if col not in df.columns:
                df[col] = np.nan
                added += 1
        if added:
            print(f"  Added {added} {band} columns as NaN "
                  f"(100% missing in EE_combined_long.parquet)")

    # ── 5. Clip vegetation indices to physical bounds ────────────────────────
    for band in ["EVI", "NDVI", "GCVI"]:
        cols = [f"{band}_{i}" for i in range(12) if f"{band}_{i}" in df.columns]
        if cols:
            n_clipped = (df[cols].abs() > 2).sum().sum()
            df[cols] = df[cols].clip(-1, 2)
            if n_clipped:
                print(f"  Clipped {n_clipped} out-of-range {band} values to [-1, 2]")

    # ── 6. Engineer seasonal aggregate features ──────────────────────────────
    print("\n--- Engineering seasonal aggregate features ---")
    X_sat = engineer_features(df)
    print(f"  Satellite feature matrix: {X_sat.shape[1]} cols")

    # Country dummies for global model (drop_first=True → Ethiopia is reference)
    # We'll add any missing dummies (e.g. country_Niger) as 0 when needed.
    country_dummies = pd.get_dummies(df["country"], prefix="country", drop_first=True,
                                     dtype=float)
    X_global = pd.concat([X_sat, country_dummies], axis=1)
    print(f"  Global feature matrix: {X_global.shape[1]} cols (incl. country dummies)")

    # ── 7. Predict for each country ──────────────────────────────────────────
    print("\n--- Loading models and generating predictions ---")

    pred_log       = np.full(len(df), np.nan, dtype=float)
    model_used_arr = np.full(len(df), "", dtype=object)

    for country in COUNTRIES_6:
        mask      = df["country"] == country
        n         = int(mask.sum())
        model_key, is_global = MODEL_SELECTION[country]

        if is_global:
            # ── GLOBAL model ─────────────────────────────────────────────────
            imp   = joblib.load(os.path.join(MODEL_DIR, "imputer_GLOBAL.joblib"))
            vt    = joblib.load(os.path.join(MODEL_DIR, "vt_GLOBAL.joblib"))
            model = joblib.load(os.path.join(MODEL_DIR, "model_XGB_GLOBAL.joblib"))

            feat_names = list(imp.feature_names_in_)

            # Add any dummy columns expected by imputer but absent from our data
            # (e.g. country_Niger=0 since Niger is excluded from predictions)
            for fn in feat_names:
                if fn not in X_global.columns:
                    X_global[fn] = 0.0

            X_sub = X_global.loc[mask, feat_names].copy()

            X_imp = imp.transform(X_sub)
            X_vt  = vt.transform(X_imp)
            preds = model.predict(X_vt)

        else:
            # ── Country-specific model ────────────────────────────────────────
            imp   = joblib.load(os.path.join(MODEL_DIR, f"imputer_{country}.joblib"))
            model = joblib.load(os.path.join(MODEL_DIR, f"model_XGB_{country}.joblib"))

            feat_names = list(imp.feature_names_in_)

            # Add any features expected by imputer but missing from our data
            for fn in feat_names:
                if fn not in X_sat.columns:
                    print(f"  WARNING: {country} — imputer expects '{fn}' not found; adding NaN")
                    X_sat[fn] = np.nan

            X_sub = X_sat.loc[mask, feat_names].copy()
            X_imp = imp.transform(X_sub)

            n_imp_out   = X_imp.shape[1]
            n_model_exp = model.n_features_in_

            if n_model_exp == n_imp_out:
                # No VT applied during training for this country (VT dropped 0 features)
                X_final = X_imp
            else:
                # VT dropped features during training. Use hard-coded feature list
                # (NOT a prediction-time refit — see VT_DROPPED_FEATURES note above).
                dropped = VT_DROPPED_FEATURES.get(country)
                if dropped is None:
                    raise ValueError(
                        f"{country}: model expects {n_model_exp} features but imputer "
                        f"outputs {n_imp_out}, and no VT_DROPPED_FEATURES entry exists. "
                        f"Check imputer.statistics_ to identify zero-median features."
                    )
                keep_feats = [f for f in feat_names if f not in dropped]
                if len(keep_feats) != n_model_exp:
                    raise ValueError(
                        f"{country}: after removing hard-coded drops, have {len(keep_feats)} "
                        f"features but model expects {n_model_exp}. "
                        f"Hard-coded VT_DROPPED_FEATURES may be wrong."
                    )
                keep_idx = [feat_names.index(f) for f in keep_feats]
                X_final  = X_imp[:, keep_idx]
                print(f"  {country}: dropped {len(dropped)} hard-coded VT features: {dropped}")

            preds = model.predict(X_final)

        pred_log[mask]       = preds
        model_used_arr[mask] = f"XGB_{model_key}"
        print(f"  {country:<12}: {n:>6,} rows predicted  (model: XGB_{model_key})")

    # Back-transform from log scale (log1p → expm1)
    pred_yield = np.expm1(pred_log)

    # ── 8. Assemble output DataFrame ─────────────────────────────────────────
    print("\n--- Assembling output ---")
    df_out = df[["country", "year", "lat_modified", "lon_modified",
                 "harvest_end_month"]].copy()
    df_out["predicted_yield_xgb"] = pred_yield
    df_out["model_used"]          = model_used_arr
    df_out["precip_missing"]      = True   # 100% null in source parquet

    # ── 9. Merge LSMS observed yields ────────────────────────────────────────
    obs = load_observed_yields()

    if obs is not None and "country" in obs.columns:
        df_out = pd.merge(
            df_out, obs,
            on=["country", "year", "lat_modified", "lon_modified"],
            how="left",
        )
        n_matched = df_out["observed_yield_kg"].notna().sum()
        pct       = 100 * n_matched / len(df_out)
        print(f"\n  Matched {n_matched:,} rows with observed yields ({pct:.1f}%)")
    else:
        df_out["observed_yield_kg"] = np.nan
        print("\n  observed_yield_kg = NaN for all rows (LSMS data unavailable)")

    # ── 10. Sanity checks ────────────────────────────────────────────────────
    print("\n--- Sanity checks ---")

    out_countries = sorted(df_out["country"].unique())
    print(f"  Countries: {out_countries}")

    for c in COUNTRIES_6:
        if c not in out_countries:
            print(f"  ERROR: {c} missing from output!")
    if "Niger" in out_countries:
        print("  ERROR: Niger should not appear in output!")

    yr_min = int(df_out["year"].min())
    yr_max = int(df_out["year"].max())
    print(f"  Year range: {yr_min}–{yr_max}")
    if yr_min < 2010 or yr_max > 2024:
        print(f"  WARNING: Unexpected years outside 2010-2024")

    assert df_out["precip_missing"].all(), "precip_missing should be True for all rows"
    print("  precip_missing=True for all rows ✓")

    print(f"\n  Per-country prediction summary:")
    print(f"  {'Country':<12} {'Rows':>6}  {'Median pred (kg/ha)':>20}  {'NA pred':>8}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*20}  {'-'*8}")
    for c in COUNTRIES_6:
        sub     = df_out[df_out["country"] == c]
        na_pct  = 100 * sub["predicted_yield_xgb"].isna().mean()
        med_p   = sub["predicted_yield_xgb"].median()
        print(f"  {c:<12} {len(sub):>6,}  {med_p:>20.1f}  {na_pct:>7.1f}%")

    # ── 11. Save ─────────────────────────────────────────────────────────────
    col_order = [
        "country", "year", "lat_modified", "lon_modified",
        "harvest_end_month", "observed_yield_kg",
        "predicted_yield_xgb", "model_used", "precip_missing",
    ]
    df_out = df_out[[c for c in col_order if c in df_out.columns]]
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'=' * 70}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Shape: {df_out.shape[0]:,} rows × {df_out.shape[1]} cols")
    print(f"Columns: {list(df_out.columns)}")


if __name__ == "__main__":
    main()
