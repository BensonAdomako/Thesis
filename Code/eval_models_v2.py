"""
eval_models_v2.py
=================
Evaluates v2 XGBoost models (absolute and anomaly) by country.
Computes:
  - In-sample R² and RMSE per country
  - Spatial leave-one-admin1-out cross-validated R² per country
    (holds out each GID_1 region in turn; approximates out-of-sample performance)

Run on armazi from the lsms_crop_yields/ directory:
    python eval_models_v2.py

Output: eval_v2.csv  (copy back to thesis Data/)
    scp biadomako@armazi:~/lsms_crop_yields/eval_v2.csv ~/Desktop/Thesis/Data/
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

DATA_DIR  = Path("/home/ahobbs/lsms_crop_yields/data")
MODEL_DIR = Path("/home/ahobbs/lsms_crop_yields/models_v2")

THESIS_ISO3 = {"ETH", "MWI", "MLI", "NGA", "TZA", "UGA"}

# ── 1. Load models ─────────────────────────────────────────────────────────────
print("Loading v2 models...")
imputer_abs = joblib.load(MODEL_DIR / "imputer_v2.joblib")
vt_abs      = joblib.load(MODEL_DIR / "vt_v2.joblib")
model_abs   = joblib.load(MODEL_DIR / "model_XGB_v2.joblib")

imputer_anom = joblib.load(MODEL_DIR / "imputer_v2_anomaly.joblib")
vt_anom      = joblib.load(MODEL_DIR / "vt_v2_anomaly.joblib")
model_anom   = joblib.load(MODEL_DIR / "model_XGB_v2_anomaly.joblib")

feat_abs  = list(imputer_abs.feature_names_in_)
feat_anom = list(imputer_anom.feature_names_in_)
print(f"  Absolute model features : {len(feat_abs)}")
print(f"  Anomaly model features  : {len(feat_anom)}")

# ── 2. Load features ───────────────────────────────────────────────────────────
print("\nLoading admin_features_v2.parquet...")
df = pd.read_parquet(DATA_DIR / "admin_features_v2.parquet")
df = df[df["GID_0"].isin(THESIS_ISO3)].copy()
print(f"  Shape after country filter: {df.shape}")

# Identify the target column (log yield or yield)
candidates = [c for c in df.columns if "yield" in c.lower() or "target" in c.lower()]
print(f"  Candidate target columns: {candidates}")

# Try common target column names
for col in ["log_yield", "log_yield_kg_ha", "yield_kg_ha", "target", "log_maize_yield"]:
    if col in df.columns:
        TARGET_ABS = col
        break
else:
    # Fall back to first candidate
    if candidates:
        TARGET_ABS = candidates[0]
        print(f"  WARNING: using '{TARGET_ABS}' as absolute target — verify this is correct")
    else:
        raise ValueError("No target yield column found. Run:\n"
                         "  df.columns[df.columns.str.contains('yield|target', case=False)]")

for col in ["yield_anomaly", "anomaly", "yield_zscore", "zscore"]:
    if col in df.columns:
        TARGET_ANOM = col
        break
else:
    TARGET_ANOM = None
    print("  WARNING: no anomaly target column found — skipping anomaly evaluation")

print(f"  Absolute target : {TARGET_ABS}")
print(f"  Anomaly target  : {TARGET_ANOM}")

# Drop rows where target is missing
df_abs = df.dropna(subset=[TARGET_ABS]).copy()
print(f"  Rows with non-NA absolute target: {len(df_abs)}")

# Add missing features as NaN
for col in feat_abs:
    if col not in df_abs.columns:
        df_abs[col] = np.nan
if TARGET_ANOM:
    df_anom = df.dropna(subset=[TARGET_ANOM]).copy()
    for col in feat_anom:
        if col not in df_anom.columns:
            df_anom[col] = np.nan

# ── Helper: predict with pipeline ─────────────────────────────────────────────
def predict_abs(data):
    X = data[feat_abs].astype(float)
    return model_abs.predict(vt_abs.transform(imputer_abs.transform(X)))

def predict_anom(data):
    X = data[feat_anom].astype(float)
    return model_anom.predict(vt_anom.transform(imputer_anom.transform(X)))

# ── 3. In-sample R² and RMSE by country ───────────────────────────────────────
print("\n=== In-Sample Performance by Country ===")
insample_rows = []

for iso in sorted(THESIS_ISO3):
    d = df_abs[df_abs["GID_0"] == iso]
    if len(d) < 10:
        print(f"  {iso}: too few observations ({len(d)}), skipping")
        continue
    y_true = d[TARGET_ABS].values
    y_pred = predict_abs(d)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  {iso:3s}  n={len(d):5d}  R²={r2:+.3f}  RMSE={rmse:.3f}")
    insample_rows.append({"GID_0": iso, "n": len(d), "r2_insample_abs": r2, "rmse_insample_abs": rmse})

# ── 4. Spatial leave-one-admin1-out CV by country ─────────────────────────────
print("\n=== Spatial Leave-One-Admin1-Out CV by Country ===")
print("(Each GID_1 region held out in turn; model trained on remaining regions)")
cv_rows = []

for iso in sorted(THESIS_ISO3):
    d = df_abs[df_abs["GID_0"] == iso].copy()
    if "GID_1" not in d.columns:
        print(f"  {iso}: no GID_1 column — skipping CV")
        continue

    admin1s = d["GID_1"].unique()
    if len(admin1s) < 3:
        print(f"  {iso}: only {len(admin1s)} admin-1 regions — skipping CV")
        continue

    y_oof   = np.full(len(d), np.nan)
    idx_map = {gid: d.index.get_loc(d[d["GID_1"] == gid].index[0]) for gid in admin1s}

    oof_preds = []
    oof_true  = []

    for held_out in admin1s:
        train_mask = d["GID_1"] != held_out
        test_mask  = d["GID_1"] == held_out
        d_train = d[train_mask]
        d_test  = d[test_mask]
        if len(d_train) < 20 or len(d_test) < 2:
            continue

        # Refit imputer + VT + model on training fold
        from sklearn.impute import SimpleImputer
        from sklearn.feature_selection import VarianceThreshold
        import xgboost as xgb

        X_train = d_train[feat_abs].astype(float).values
        y_train = d_train[TARGET_ABS].values
        X_test  = d_test[feat_abs].astype(float).values
        y_test  = d_test[TARGET_ABS].values

        imp   = SimpleImputer(strategy="median").fit(X_train)
        vt    = VarianceThreshold(threshold=1e-5).fit(imp.transform(X_train))
        Xtr   = vt.transform(imp.transform(X_train))
        Xte   = vt.transform(imp.transform(X_test))

        m = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1
        )
        m.fit(Xtr, y_train)
        oof_preds.extend(m.predict(Xte).tolist())
        oof_true.extend(y_test.tolist())

    if len(oof_true) < 10:
        print(f"  {iso}: insufficient OOF predictions — skipping")
        continue

    r2_cv   = r2_score(oof_true, oof_preds)
    rmse_cv = np.sqrt(mean_squared_error(oof_true, oof_preds))
    n_adm1  = len(admin1s)
    print(f"  {iso:3s}  admin1_regions={n_adm1:3d}  OOF_n={len(oof_true):5d}"
          f"  Spatial_R²={r2_cv:+.3f}  RMSE={rmse_cv:.3f}")
    cv_rows.append({"GID_0": iso, "n_admin1": n_adm1, "oof_n": len(oof_true),
                    "r2_spatial_cv_abs": r2_cv, "rmse_spatial_cv_abs": rmse_cv})

# ── 5. Combine and save ────────────────────────────────────────────────────────
insample_df = pd.DataFrame(insample_rows).set_index("GID_0")
cv_df       = pd.DataFrame(cv_rows).set_index("GID_0") if cv_rows else pd.DataFrame()

if not cv_df.empty:
    eval_df = insample_df.join(cv_df, how="outer")
else:
    eval_df = insample_df

eval_df = eval_df.reset_index()
print("\n=== Summary ===")
print(eval_df.to_string(index=False))

eval_df.to_csv("eval_v2.csv", index=False)
print("\nSaved: eval_v2.csv")
print("Copy back: scp biadomako@armazi:~/lsms_crop_yields/eval_v2.csv ~/Desktop/Thesis/Data/")
