"""
LSMS Crop Yield Model Training (v3)

Trains Ridge baseline, Random Forest, and XGBoost models for maize yield
prediction using satellite-derived features with lat/lon coordinates.

Uses expanding-window temporal CV (train on years <= t, test on years > t)
so the model can learn spatial patterns from coordinates without temporal
leakage.  Runs global and country-specific models, then builds a cross-model
evaluation matrix.

Optimized for multi-core CPU (uses all available cores).

If the data file is not found locally, it is automatically downloaded from
Google Drive (requires the same Google auth used for Earth Engine).

Usage:
    python train_models.py --data EE_harvest_ml_v3.csv --out models_v3
    python train_models.py --data EE_harvest_ml_v3.csv --out models_v3 --countries Uganda Nigeria
"""

import argparse
import os
import sys
from collections import Counter
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

N_JOBS = -1  # use all cores

# Google Drive folder and filename used by the Colab merge notebook
DRIVE_FOLDER  = 'Research/Crop Yield Prediction/Nov25Data/EE_exports'
DRIVE_DEFAULT = 'EE_harvest_ml_v3.csv'


# ─── Google Drive download ──────────────────────────────────────────────────

def download_from_drive(local_path, drive_folder=DRIVE_FOLDER, filename=None):
    """Download a file from Google Drive using Application Default Credentials.

    Searches for ``filename`` inside ``drive_folder`` and saves it to
    ``local_path``.  Uses the same Google auth that Earth Engine relies on
    (application-default credentials).
    """
    if filename is None:
        filename = os.path.basename(local_path)

    try:
        import google.auth
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError:
        sys.exit(
            "ERROR: google-api-python-client is required for Drive download.\n"
            "Install with:  pip install google-api-python-client google-auth"
        )

    print(f"File not found at {local_path}")
    print(f"Searching Google Drive for '{filename}' in folder '{drive_folder}'...")

    creds, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    creds = creds.with_quota_project(project or 'ee-ahobbs')
    service = build('drive', 'v3', credentials=creds, cache_discovery=False)

    # Walk nested folder path (e.g. "Research/Crop Yield Prediction/EE_exports")
    folder_parts = drive_folder.split('/')
    parent_id = None

    for part in folder_parts:
        folder_q = (
            f"name = '{part}' and mimeType = 'application/vnd.google-apps.folder' "
            f"and trashed = false"
        )
        if parent_id:
            folder_q += f" and '{parent_id}' in parents"
        folders = service.files().list(q=folder_q, fields='files(id,name)').execute()
        folder_hits = folders.get('files', [])

        if not folder_hits:
            sys.exit(f"ERROR: Could not find Drive folder '{part}' in path '{drive_folder}'")

        parent_id = folder_hits[0]['id']

    folder_id = parent_id

    # Find the file inside that folder
    file_q = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
    files = service.files().list(q=file_q, fields='files(id,name,size)').execute()
    file_hits = files.get('files', [])

    if not file_hits:
        sys.exit(
            f"ERROR: Could not find '{filename}' in Drive folder '{drive_folder}'.\n"
            f"Make sure the Colab merge notebook has been run and the file exists."
        )

    file_id = file_hits[0]['id']
    file_size = int(file_hits[0].get('size', 0))
    print(f"Found: {file_hits[0]['name']} ({file_size / 1e6:.1f} MB)")

    # Download
    import io
    request = service.files().get_media(fileId=file_id)
    os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)

    with open(local_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"  Downloaded {status.progress() * 100:.0f}%", end='\r')

    print(f"  Saved to {local_path} ({os.path.getsize(local_path) / 1e6:.1f} MB)")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}m"


def make_temporal_splits(harvest_years, fold_boundaries):
    """Create expanding-window temporal CV splits.

    Parameters
    ----------
    harvest_years : array-like of ints (same length as X)
    fold_boundaries : list of (train_end_year, test_end_year) pairs

    Returns list of (train_indices, test_indices) tuples.
    """
    hy = np.asarray(harvest_years)
    splits = []
    for train_end, test_end in fold_boundaries:
        tr = np.where(hy <= train_end)[0]
        te = np.where((hy > train_end) & (hy <= test_end))[0]
        if len(tr) > 0 and len(te) > 0:
            splits.append((tr, te))
    return splits


# Default temporal fold boundaries (expanding window) for global model
TEMPORAL_FOLDS = [
    (2013, 2015),   # Train 2010-2013, Test 2014-2015
    (2015, 2017),   # Train 2010-2015, Test 2016-2017
    (2017, 2019),   # Train 2010-2017, Test 2018-2019
    (2019, 2025),   # Train 2010-2019, Test 2020-2024
]

# Year groups for LOYO (Leave-One-Year-Group-Out) in country models
YEAR_GROUPS = [
    (2010, 2012),
    (2013, 2014),
    (2015, 2016),
    (2017, 2018),
    (2019, 2025),
]


def make_loyo_splits(harvest_years, year_groups=YEAR_GROUPS, min_test=10):
    """Leave-One-Year-Group-Out: hold out each year group, train on all others."""
    hy = np.asarray(harvest_years)
    splits = []
    for yg_start, yg_end in year_groups:
        te = np.where((hy >= yg_start) & (hy <= yg_end))[0]
        tr = np.where((hy < yg_start) | (hy > yg_end))[0]
        if len(tr) >= 10 and len(te) >= min_test:
            splits.append((tr, te))
    return splits


def _eval_on_splits(X, y, splits, ModelClass, p):
    """Evaluate a single param combo on a set of CV splits."""
    train_r2s, test_r2s = [], []
    for tr_idx, va_idx in splits:
        imp = SimpleImputer(strategy='median')
        vt = VarianceThreshold(threshold=1e-5)
        X_tr = vt.fit_transform(imp.fit_transform(X.iloc[tr_idx]))
        X_va = vt.transform(imp.transform(X.iloc[va_idx]))

        model = ModelClass(**p)
        model.fit(X_tr, y.iloc[tr_idx])

        train_r2s.append(r2_score(y.iloc[tr_idx], model.predict(X_tr)))
        test_r2s.append(r2_score(y.iloc[va_idx], model.predict(X_va)))

    return float(np.mean(train_r2s)), float(np.mean(test_r2s))


def tune_with_cv(
    X, y, splits, ModelClass, param_grid,
    patience=15, min_delta=1e-3, label='', random_state=42,
    gap_penalty=0.3,
):
    """Grid search scored on a single set of CV splits.

    score = test_r2 - gap_penalty * max(0, train_r2 - test_r2)
    """
    grid_list = list(ParameterGrid(param_grid))
    rng = np.random.default_rng(random_state)
    grid_list = [grid_list[i] for i in rng.permutation(len(grid_list))]

    best_score = -np.inf
    best_params = None
    best_train, best_test = 0.0, 0.0
    no_improve = 0
    t0 = perf_counter()
    total = len(grid_list)

    for idx, p in enumerate(grid_list, 1):
        tr_r2, te_r2 = _eval_on_splits(X, y, splits, ModelClass, p)

        gap = max(0, tr_r2 - te_r2)
        score = te_r2 - gap_penalty * gap
        marker = ''

        if score > best_score + min_delta:
            best_score = score
            best_params = p
            best_train, best_test = tr_r2, te_r2
            no_improve = 0
            marker = ' *'
        else:
            no_improve += 1

        elapsed = perf_counter() - t0
        print(
            f"  [{label}] {idx:3d}/{total}  "
            f"test={te_r2:+.3f} gap={tr_r2-te_r2:.3f}  "
            f"best={best_test:+.3f}  "
            f"({fmt_time(elapsed)}){marker}"
        )

        if no_improve >= patience:
            print(f"  [{label}] Early stop after {no_improve} non-improving combos")
            break

    total_time = perf_counter() - t0
    print(
        f"  [{label}] DONE  "
        f"train={best_train:.4f} test={best_test:.4f} "
        f"gap={best_train-best_test:.4f}  "
        f"({fmt_time(total_time)}, {idx} combos)\n"
    )
    return best_params, {'s_train': best_train, 's_test': best_test}


def collect_oof_predictions(X, y, splits, ModelClass, params):
    """Run CV with best params and return out-of-fold predictions."""
    oof = pd.Series(np.nan, index=X.index)

    for tr_idx, va_idx in splits:
        imp = SimpleImputer(strategy='median')
        vt = VarianceThreshold(threshold=1e-5)
        X_tr = vt.fit_transform(imp.fit_transform(X.iloc[tr_idx]))
        X_va = vt.transform(imp.transform(X.iloc[va_idx]))
        model = ModelClass(**params)
        model.fit(X_tr, y.iloc[tr_idx])
        oof.iloc[va_idx] = model.predict(X_va)

    return oof


def drop_correlated_features(X, threshold=0.95):
    """Drop features with pairwise |correlation| above threshold.

    For each highly-correlated pair, drops the feature with the higher mean
    absolute correlation to all other features (removes the more redundant one).
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            if upper.at[row, col] > threshold and row not in to_drop and col not in to_drop:
                # Drop whichever has higher average correlation overall
                if corr[col].mean() >= corr[row].mean():
                    to_drop.add(col)
                else:
                    to_drop.add(row)
    to_drop = sorted(to_drop)
    return X.drop(columns=to_drop), to_drop


# ─── Param grids ─────────────────────────────────────────────────────────────

# Grids constrained to reduce overfitting:
#  - RF: shallower trees, higher min_samples, bootstrap subsampling
#  - XGB: shallower, stronger L1/L2 regularization, min_child_weight
RF_GRID = {
    'n_estimators':      [300, 600],
    'max_depth':         [3, 5, 8],
    'max_features':      [0.3, 0.5, 0.7],
    'min_samples_split': [10, 20, 50],
    'min_samples_leaf':  [10, 20, 40],
    'max_samples':       [0.5, 0.7],
    'bootstrap':         [True],
    'n_jobs':            [N_JOBS],
    'random_state':      [42],
}

XGB_GRID = {
    'n_estimators':      [300, 600],
    'max_depth':         [2, 3, 4],
    'learning_rate':     [0.03, 0.05, 0.1],
    'gamma':             [0, 1, 5],
    'subsample':         [0.5, 0.7],
    'colsample_bytree':  [0.3, 0.5],
    'reg_lambda':        [10.0, 20.0, 50.0],
    'reg_alpha':         [0.5, 1.0, 5.0],
    'min_child_weight':  [10, 20, 40],
    'tree_method':       ['hist'],
    'nthread':           [N_JOBS],
    'random_state':      [42],
}

# Tighter grids for country models (fewer groups → more regularization)
RF_GRID_COUNTRY = {
    'n_estimators':      [300, 600],
    'max_depth':         [3, 4, 5],
    'max_features':      [0.3, 0.5],
    'min_samples_split': [15, 30, 50],
    'min_samples_leaf':  [15, 25, 40],
    'max_samples':       [0.5],
    'bootstrap':         [True],
    'n_jobs':            [N_JOBS],
    'random_state':      [42],
}

XGB_GRID_COUNTRY = {
    'n_estimators':      [300, 600],
    'max_depth':         [2, 3],
    'learning_rate':     [0.03, 0.05],
    'gamma':             [0, 1, 5],
    'subsample':         [0.5, 0.7],
    'colsample_bytree':  [0.3, 0.5],
    'reg_lambda':        [20.0, 50.0],
    'reg_alpha':         [1.0, 5.0],
    'min_child_weight':  [20, 40],
    'tree_method':       ['hist'],
    'nthread':           [N_JOBS],
    'random_state':      [42],
}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train LSMS crop yield models')
    parser.add_argument('--data', default=DRIVE_DEFAULT,
                        help='Path to EE_harvest_ml_v3.csv (downloaded from Drive if missing)')
    parser.add_argument('--out', default='models_v3',
                        help='Output directory for models and results')
    parser.add_argument('--countries', nargs='*', default=None,
                        help='Train country-specific models only for these (default: all)')
    parser.add_argument('--max-lag', type=int, default=11,
                        help='Maximum lag to include (0-11, default: 11 = all lags)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total_t0 = perf_counter()

    # ── 1. Load data (download from Drive if needed) ─────────────────────────
    print('=' * 70)
    print('LOADING DATA')
    print('=' * 70)

    if not os.path.exists(args.data):
        download_from_drive(args.data)

    df = pd.read_csv(args.data, dtype={'ea_id_obs': 'Int64'})
    print(f"Raw: {df.shape[0]:,} rows, {df.shape[1]} cols")

    df = df.dropna(subset=['yield_kg'])
    df = df[(df['yield_kg'] > 0) & (df['yield_kg'] < 15_000)]
    df['log_yield'] = np.log1p(df['yield_kg'])
    print(f"After yield filter (0 < yield < 15000): {len(df):,} rows")

    # ── Clip vegetation indices to physical bounds ───────────────────────────
    veg_cols = [c for c in df.columns
                if any(c.startswith(v + '_') for v in ['NDVI', 'EVI', 'GCVI'])]
    n_clipped = (df[veg_cols].abs() > 2).sum().sum()
    df[veg_cols] = df[veg_cols].clip(-1, 2)
    if n_clipped > 0:
        print(f"Clipped {n_clipped} out-of-range vegetation index values")

    # ── Feature engineering: growing-season aggregates ────────────────────────
    weather_vars = ['precip', 'tmax', 'tmin', 'soil_moist', 'KDD']
    veg_vars = ['NDVI', 'EVI', 'GCVI']
    all_vars = weather_vars + veg_vars

    def lag_cols(var, lags):
        return [f'{var}_{i}' for i in lags]

    X = pd.DataFrame(index=df.index)

    for var in all_vars:
        # Growing season aggregate (lags 3-6)
        gs = df[lag_cols(var, range(3, 7))]
        if var == 'precip':
            X[f'{var}_gs_sum'] = gs.sum(axis=1)
        else:
            X[f'{var}_gs_mean'] = gs.mean(axis=1)

        # Peak season (lags 4-5)
        peak = df[lag_cols(var, range(4, 6))]
        if var == 'precip':
            X[f'{var}_peak_sum'] = peak.sum(axis=1)
        else:
            X[f'{var}_peak_mean'] = peak.mean(axis=1)

        # Near-harvest (lags 0-2)
        near = df[lag_cols(var, range(0, 3))]
        if var == 'precip':
            X[f'{var}_near_sum'] = near.sum(axis=1)
        else:
            X[f'{var}_near_mean'] = near.mean(axis=1)

        # Pre-season (lags 7-11)
        pre = df[lag_cols(var, range(7, 12))]
        if var == 'precip':
            X[f'{var}_pre_sum'] = pre.sum(axis=1)
        else:
            X[f'{var}_pre_mean'] = pre.mean(axis=1)

        # Trend: growing season to harvest (lag 3 minus lag 0)
        X[f'{var}_trend'] = df[f'{var}_3'] - df[f'{var}_0']

    # Interaction: water balance = precip / (tmax + 1)
    X['water_balance_gs'] = X['precip_gs_sum'] / (X['tmax_gs_mean'] + 1)
    X['water_balance_peak'] = X['precip_peak_sum'] / (X['tmax_peak_mean'] + 1)

    # Aridity index
    X['aridity_gs'] = X['tmax_gs_mean'] / (X['precip_gs_sum'] + 1)

    # KDD fraction of GDD (heat stress intensity)
    gdd_gs = df[lag_cols('GDD', range(3, 7))].sum(axis=1)
    X['kdd_fraction_gs'] = X['KDD_gs_mean'] * 4 / (gdd_gs + 1)

    # ── Add coordinates as features ───────────────────────────────────────────
    X['latitude'] = df['lat_modified'].values
    X['longitude'] = df['lon_modified'].values

    feature_cols = sorted(X.columns.tolist())
    y = df['log_yield'].copy()
    harvest_years = df['harvest_year'].values

    print(f"Engineered features: {len(feature_cols)} (incl. lat/lon)")
    print(f"Locations (ea_ids): {df['ea_id_obs'].nunique():,}")
    print(f"\nCountry distribution:")
    for c, n in df['country'].value_counts().items():
        print(f"  {c:<15} {n:>6,}")

    countries = sorted(df['country'].unique())
    if args.countries:
        countries = [c for c in countries if c in args.countries]
        print(f"\nWill train country models for: {', '.join(countries)}")

    # Feature selection: drop highly correlated features (but never drop lat/lon)
    n_before = len(feature_cols)
    X, dropped_feats = drop_correlated_features(X, threshold=0.90)
    # Ensure lat/lon survive
    for coord in ['latitude', 'longitude']:
        if coord in dropped_feats:
            dropped_feats.remove(coord)
            X[coord] = df['lat_modified'].values if coord == 'latitude' \
                else df['lon_modified'].values
    feature_cols = list(X.columns)
    print(f"\nFeature selection: {n_before} -> {len(feature_cols)} "
          f"(dropped {len(dropped_feats)} correlated features)")
    if dropped_feats:
        print(f"  Dropped: {', '.join(dropped_feats)}")

    # Country one-hot encoding for global models
    country_dummies = pd.get_dummies(df['country'], prefix='country', drop_first=True, dtype=float)
    X_global = pd.concat([X, country_dummies], axis=1)
    global_feature_cols = list(X_global.columns)
    print(f"Global model features: {len(global_feature_cols)} "
          f"({len(feature_cols)} satellite+coords + {len(country_dummies.columns)} country)")

    # ── Build CV splits (dual: spatial + temporal) ──────────────────────────
    groups = df['ea_id_obs']
    spatial_cv = GroupKFold(n_splits=5)
    global_spatial_splits = list(spatial_cv.split(X_global, y, groups))
    global_temporal_splits = make_temporal_splits(harvest_years, TEMPORAL_FOLDS)

    print(f"\nSpatial CV: {len(global_spatial_splits)} folds (GroupKFold by ea_id_obs)")
    print(f"Temporal CV: {len(global_temporal_splits)} folds (expanding window)")
    for i, (tr, te) in enumerate(global_temporal_splits):
        tr_years = sorted(set(harvest_years[tr]))
        te_years = sorted(set(harvest_years[te]))
        shared = len(set(df.iloc[tr]['ea_id_obs']) & set(df.iloc[te]['ea_id_obs']))
        print(f"  Temporal fold {i}: train {tr_years[0]}-{tr_years[-1]} ({len(tr)}), "
              f"test {te_years[0]}-{te_years[-1]} ({len(te)}), "
              f"shared locs={shared}")

    # ── 2. Ridge baseline (evaluated on both CV schemes) ─────────────────────
    print('\n' + '=' * 70)
    print('RIDGE BASELINE')
    print('=' * 70)

    def _run_ridge(splits, label):
        trains, tests = [], []
        for tr_idx, va_idx in splits:
            imp = SimpleImputer(strategy='median')
            vt = VarianceThreshold(threshold=1e-5)
            scl = RobustScaler()
            X_tr = scl.fit_transform(vt.fit_transform(imp.fit_transform(X_global.iloc[tr_idx])))
            X_va = scl.transform(vt.transform(imp.transform(X_global.iloc[va_idx])))
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr, y.iloc[tr_idx])
            trains.append(r2_score(y.iloc[tr_idx], ridge.predict(X_tr)))
            tests.append(r2_score(y.iloc[va_idx], ridge.predict(X_va)))
        tr, te = np.mean(trains), np.mean(tests)
        print(f"  {label}: Train={tr:.4f}  Test={te:.4f}  Gap={tr - te:.4f}")
        return tr, te

    ridge_s_train, ridge_s_test = _run_ridge(global_spatial_splits, 'Spatial')
    ridge_t_train, ridge_t_test = _run_ridge(global_temporal_splits, 'Temporal')
    ridge_info = {
        's_train': ridge_s_train, 's_test': ridge_s_test,
        't_train': ridge_t_train, 't_test': ridge_t_test,
    }

    # ── 3. Global Random Forest ──────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('GLOBAL RANDOM FOREST')
    print('=' * 70)

    rf_params, rf_info = tune_with_cv(
        X_global, y, global_spatial_splits,
        RandomForestRegressor, RF_GRID, patience=15, label='RF-Global',
    )
    # Evaluate best params on temporal CV for reporting
    t_train, t_test = _eval_on_splits(
        X_global, y, global_temporal_splits, RandomForestRegressor, rf_params,
    )
    rf_info['t_train'] = t_train
    rf_info['t_test'] = t_test
    print(f"  Temporal eval: train={t_train:.4f} test={t_test:.4f} "
          f"gap={t_train-t_test:.4f}")
    print(f"  Best params: {rf_params}")

    # Collect OOF from spatial splits (full coverage for cross-model matrix)
    print("  Collecting OOF predictions (spatial CV)...")
    oof_global_rf = collect_oof_predictions(
        X_global, y, global_spatial_splits, RandomForestRegressor, rf_params,
    )

    imp_global = SimpleImputer(strategy='median')
    vt_global = VarianceThreshold(threshold=1e-5)
    X_imp = vt_global.fit_transform(imp_global.fit_transform(X_global))
    global_rf = RandomForestRegressor(**rf_params)
    global_rf.fit(X_imp, y)

    joblib.dump(global_rf, os.path.join(args.out, 'model_RF_GLOBAL.joblib'))
    joblib.dump(imp_global, os.path.join(args.out, 'imputer_GLOBAL.joblib'))
    joblib.dump(vt_global, os.path.join(args.out, 'vt_GLOBAL.joblib'))

    # ── 4. Global XGBoost ────────────────────────────────────────────────────
    print('=' * 70)
    print('GLOBAL XGBOOST')
    print('=' * 70)

    xgb_params, xgb_info = tune_with_cv(
        X_global, y, global_spatial_splits,
        XGBRegressor, XGB_GRID, patience=15, label='XGB-Global',
    )
    t_train, t_test = _eval_on_splits(
        X_global, y, global_temporal_splits, XGBRegressor, xgb_params,
    )
    xgb_info['t_train'] = t_train
    xgb_info['t_test'] = t_test
    print(f"  Temporal eval: train={t_train:.4f} test={t_test:.4f} "
          f"gap={t_train-t_test:.4f}")
    print(f"  Best params: {xgb_params}")

    print("  Collecting OOF predictions (spatial CV)...")
    oof_global_xgb = collect_oof_predictions(
        X_global, y, global_spatial_splits, XGBRegressor, xgb_params,
    )

    global_xgb = XGBRegressor(**xgb_params)
    global_xgb.fit(X_imp, y)  # X_imp already has VT applied from RF step

    joblib.dump(global_xgb, os.path.join(args.out, 'model_XGB_GLOBAL.joblib'))

    # ── 5. Country-specific models ───────────────────────────────────────────
    print('=' * 70)
    print('COUNTRY-SPECIFIC MODELS')
    print('=' * 70)

    country_models = {}
    country_top_features = {}

    for c in countries:
        mask = df['country'] == c
        Xc = X.loc[mask].copy()
        yc = df.loc[mask, 'log_yield'].copy()
        gc = df.loc[mask, 'ea_id_obs']
        hy_c = harvest_years[mask.values]

        n_locs = gc.nunique()
        n_years = len(set(hy_c))
        if n_locs < 20:
            print(f"\n  Skipping {c}: only {n_locs} locations\n")
            continue

        # Spatial splits: GroupKFold by location
        n_spatial = min(5, n_locs)
        c_spatial_cv = GroupKFold(n_splits=n_spatial)
        c_spatial_splits = list(c_spatial_cv.split(Xc, yc, gc))

        # Temporal splits: LOYO year groups (for evaluation only)
        c_temporal_splits = make_loyo_splits(hy_c, min_test=10)

        print(f"\n--- {c}: {len(Xc):,} rows, {n_locs} locations, "
              f"{n_years} years, {n_spatial} spatial + "
              f"{len(c_temporal_splits)} temporal folds ---")

        rf_p, rf_ci = tune_with_cv(
            Xc, yc, c_spatial_splits,
            RandomForestRegressor, RF_GRID_COUNTRY,
            patience=10, label=f'RF-{c}',
        )
        # Temporal evaluation for reporting (if enough folds)
        if len(c_temporal_splits) >= 2:
            rf_t_train, rf_t_test = _eval_on_splits(
                Xc, yc, c_temporal_splits, RandomForestRegressor, rf_p,
            )
            rf_ci['t_train'] = rf_t_train
            rf_ci['t_test'] = rf_t_test
            print(f"  RF temporal: train={rf_t_train:.4f} test={rf_t_test:.4f} "
                  f"gap={rf_t_train-rf_t_test:.4f}")
        else:
            print(f"  RF temporal: skipped ({len(c_temporal_splits)} folds)")

        xgb_p, xgb_ci = tune_with_cv(
            Xc, yc, c_spatial_splits,
            XGBRegressor, XGB_GRID_COUNTRY,
            patience=10, label=f'XGB-{c}',
        )
        if len(c_temporal_splits) >= 2:
            xgb_t_train, xgb_t_test = _eval_on_splits(
                Xc, yc, c_temporal_splits, XGBRegressor, xgb_p,
            )
            xgb_ci['t_train'] = xgb_t_train
            xgb_ci['t_test'] = xgb_t_test
            print(f"  XGB temporal: train={xgb_t_train:.4f} test={xgb_t_test:.4f} "
                  f"gap={xgb_t_train-xgb_t_test:.4f}")
        else:
            print(f"  XGB temporal: skipped ({len(c_temporal_splits)} folds)")

        # Collect OOF from spatial splits (full coverage for cross-model matrix)
        print(f"  Collecting OOF predictions for {c}...")
        oof_rf = collect_oof_predictions(
            Xc, yc, c_spatial_splits, RandomForestRegressor, rf_p,
        )
        oof_xgb = collect_oof_predictions(
            Xc, yc, c_spatial_splits, XGBRegressor, xgb_p,
        )
        # Expand to full df index for consistent cross-eval indexing
        oof_rf_full = pd.Series(np.nan, index=df.index, dtype=float)
        oof_rf_full[Xc.index] = oof_rf.values
        oof_xgb_full = pd.Series(np.nan, index=df.index, dtype=float)
        oof_xgb_full[Xc.index] = oof_xgb.values

        imp_c = SimpleImputer(strategy='median')
        vt_c = VarianceThreshold(threshold=1e-5)
        Xc_imp = vt_c.fit_transform(imp_c.fit_transform(Xc))

        rf_c = RandomForestRegressor(**rf_p)
        rf_c.fit(Xc_imp, yc)

        xgb_c = XGBRegressor(**xgb_p)
        xgb_c.fit(Xc_imp, yc)

        # Track which features survived VT for feature importance reporting
        vt_feature_mask = vt_c.get_support()
        vt_feature_cols = [f for f, keep in zip(feature_cols, vt_feature_mask) if keep]

        country_models[c] = {
            'rf': rf_c, 'xgb': xgb_c, 'imputer': imp_c, 'vt': vt_c,
            'rf_params': rf_p, 'xgb_params': xgb_p,
            'rf_info': rf_ci, 'xgb_info': xgb_ci,
            'oof_rf': oof_rf_full, 'oof_xgb': oof_xgb_full,
        }

        fi = pd.Series(xgb_c.feature_importances_, index=vt_feature_cols)
        country_top_features[c] = fi.nlargest(5).index.tolist()

        joblib.dump(rf_c, os.path.join(args.out, f'model_RF_{c}.joblib'))
        joblib.dump(xgb_c, os.path.join(args.out, f'model_XGB_{c}.joblib'))
        joblib.dump(imp_c, os.path.join(args.out, f'imputer_{c}.joblib'))

    # ── 6. Cross-model evaluation matrix (using OOF predictions) ─────────────
    print('=' * 70)
    print('CROSS-MODEL EVALUATION (out-of-fold)')
    print('=' * 70)

    # OOF lookup: model_name -> pd.Series aligned with df.index
    # Note: temporal CV means only test-fold observations have OOF predictions
    oof_lookup = {
        'GLOBAL_RF':  oof_global_rf,
        'GLOBAL_XGB': oof_global_xgb,
    }
    for c, m in country_models.items():
        oof_lookup[f'{c}_RF']  = m['oof_rf']
        oof_lookup[f'{c}_XGB'] = m['oof_xgb']

    # Final model lookup for out-of-distribution (cross-country) predictions
    all_models = {
        'GLOBAL_RF':  (global_rf,  imp_global, vt_global),
        'GLOBAL_XGB': (global_xgb, imp_global, vt_global),
    }
    for c, m in country_models.items():
        all_models[f'{c}_RF']  = (m['rf'],  m['imputer'], m['vt'])
        all_models[f'{c}_XGB'] = (m['xgb'], m['imputer'], m['vt'])

    all_countries = sorted(df['country'].unique())
    rows = []

    for model_name in all_models:
        if model_name.startswith('GLOBAL'):
            train_countries = set(all_countries)
        else:
            train_country = model_name.rsplit('_', 1)[0]
            train_countries = {train_country}

        for tc in all_countries:
            mask = df['country'] == tc

            if tc in train_countries:
                # Use spatial OOF predictions (full coverage)
                oof = oof_lookup[model_name][mask]
                has_pred = oof.notna()
                if has_pred.sum() < 5:
                    continue
                yt = df.loc[mask, 'log_yield'][has_pred].values
                yp = oof[has_pred].values
                eval_type = 'oof'
            else:
                # Out-of-distribution: use final model predictions
                model, model_imp, model_vt = all_models[model_name]
                Xt = X.loc[mask]
                Xt_imp = model_vt.transform(model_imp.transform(Xt))
                yp = model.predict(Xt_imp)
                yt = df.loc[mask, 'log_yield']
                eval_type = 'ood'

            rows.append({
                'train_model':  model_name,
                'test_country': tc,
                'r2':           r2_score(yt, yp),
                'rmse':         np.sqrt(mean_squared_error(yt, yp)),
                'mae':          mean_absolute_error(yt, yp),
                'n_test':       len(yt),
                'eval_type':    eval_type,
            })

    results_df = pd.DataFrame(rows)

    r2_matrix = results_df.pivot(index='train_model', columns='test_country', values='r2')
    rmse_matrix = results_df.pivot(index='train_model', columns='test_country', values='rmse')

    print('\nR² Matrix (rows=model, cols=test country):')
    print(r2_matrix.round(3).to_string())
    print('\nRMSE Matrix:')
    print(rmse_matrix.round(3).to_string())
    print('\n(In-distribution entries use temporal OOF predictions)')

    results_df.to_csv(os.path.join(args.out, 'cross_model_results.csv'), index=False)
    r2_matrix.to_csv(os.path.join(args.out, 'r2_matrix.csv'))
    rmse_matrix.to_csv(os.path.join(args.out, 'rmse_matrix.csv'))

    # ── 7. Summary ───────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('RESULTS SUMMARY')
    print('=' * 70)

    def _fmt_row(label, info):
        s_gap = info['s_train'] - info['s_test']
        t_gap = info['t_train'] - info['t_test']
        return (f"{label:<20} {info['s_test']:>8.4f} {s_gap:>8.4f} "
                f"{info['t_test']:>8.4f} {t_gap:>8.4f}")

    print(f"\n{'Model':<20} {'Spat R²':>8} {'S Gap':>8} "
          f"{'Temp R²':>8} {'T Gap':>8}")
    print('-' * 56)
    print(_fmt_row('Ridge (baseline)', ridge_info))
    print(_fmt_row('RF (global)', rf_info))
    print(_fmt_row('XGBoost (global)', xgb_info))

    if country_models:
        print(f"\n{'Country':<12} {'Rows':>5} "
              f"{'RF Spat':>8} {'RF Temp':>8} "
              f"{'XGB Spat':>9} {'XGB Temp':>9}")
        print('-' * 62)
        for c in countries:
            if c not in country_models:
                continue
            m = country_models[c]
            n = (df['country'] == c).sum()
            rf_t = m['rf_info'].get('t_test')
            xgb_t = m['xgb_info'].get('t_test')
            rf_t_str = f"{rf_t:>8.4f}" if rf_t is not None else "     N/A"
            xgb_t_str = f"{xgb_t:>9.4f}" if xgb_t is not None else "      N/A"
            print(f"{c:<12} {n:>5} "
                  f"{m['rf_info']['s_test']:>8.4f} {rf_t_str} "
                  f"{m['xgb_info']['s_test']:>9.4f} {xgb_t_str}")

    # ── 8. Feature importance ────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('FEATURE IMPORTANCE')
    print('=' * 70)

    vt_global_mask = vt_global.get_support()
    global_vt_feature_cols = [f for f, keep in zip(global_feature_cols, vt_global_mask) if keep]
    gi = pd.Series(global_xgb.feature_importances_, index=global_vt_feature_cols)
    print('\nGlobal XGBoost — Top 15:')
    for feat, val in gi.nlargest(15).items():
        print(f"  {feat:<20} {val:.4f}")

    if country_top_features:
        print(f"\nTop 5 by country:")
        for c in countries:
            if c in country_top_features:
                print(f"  {c:<15}: {', '.join(country_top_features[c])}")

        all_top = [f for feats in country_top_features.values() for f in feats]
        counts = Counter(all_top)
        n_countries = len(country_top_features)
        print(f"\nMost common top-5 features across {n_countries} countries:")
        for feat, cnt in counts.most_common(10):
            print(f"  {feat:<20}: {cnt}/{n_countries} ({100*cnt/n_countries:.0f}%)")

    total_time = perf_counter() - total_t0
    print(f"\nTotal runtime: {fmt_time(total_time)}")
    print(f"Output saved to: {os.path.abspath(args.out)}/")


if __name__ == '__main__':
    main()
