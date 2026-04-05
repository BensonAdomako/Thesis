"""
generate_predictions_v2.py
==========================
Generate district-level maize yield predictions using the v2 XGBoost models.
Produces both absolute (log yield level) and anomaly (yield z-score) predictions
for the 6 thesis countries: ETH, MWI, MLI, NGA, TZA, UGA.

Run on armazi from the lsms_crop_yields/ directory:
    python generate_predictions_v2.py

Output: predictions_v2.csv
    Columns: GID_0, GID_1, GID_2, NAME_1, NAME_2, year,
             predicted_yield_abs_kgha, log_predicted_yield_abs,
             predicted_yield_anomaly

Then copy back to thesis:
    scp biadomako@armazi:~/lsms_crop_yields/predictions_v2.csv ~/Desktop/Thesis/Data/
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("/home/ahobbs/lsms_crop_yields/data")
MODEL_DIR = Path("/home/ahobbs/lsms_crop_yields/models_v2")
OUT_FILE  = Path("predictions_v2.csv")

THESIS_ISO3 = {"ETH", "MWI", "MLI", "NGA", "TZA", "UGA"}
YEAR_MIN, YEAR_MAX = 2010, 2024

# ── 1. Load models ─────────────────────────────────────────────────────────────
print("Loading models and preprocessors...")

imputer_abs  = joblib.load(MODEL_DIR / "imputer_v2.joblib")
vt_abs       = joblib.load(MODEL_DIR / "vt_v2.joblib")
model_abs    = joblib.load(MODEL_DIR / "model_XGB_v2.joblib")

imputer_anom = joblib.load(MODEL_DIR / "imputer_v2_anomaly.joblib")
vt_anom      = joblib.load(MODEL_DIR / "vt_v2_anomaly.joblib")
model_anom   = joblib.load(MODEL_DIR / "model_XGB_v2_anomaly.joblib")

feat_names_abs  = list(imputer_abs.feature_names_in_)
feat_names_anom = list(imputer_anom.feature_names_in_)
print(f"  Absolute model : {len(feat_names_abs)} features")
print(f"  Anomaly model  : {len(feat_names_anom)} features")

# ── 2. Load pre-built features ────────────────────────────────────────────────
print("\nLoading admin_features_v2.parquet...")
df = pd.read_parquet(DATA_DIR / "admin_features_v2.parquet")
print(f"  Full shape: {df.shape}")

# Filter to thesis countries and year range
df = df[
    df["GID_0"].isin(THESIS_ISO3) &
    df["year"].between(YEAR_MIN, YEAR_MAX)
].copy()
print(f"  After filter (6 countries, {YEAR_MIN}-{YEAR_MAX}): {df.shape}")

# Check feature coverage
id_cols = ["GID_0", "GID_1", "GID_2", "NAME_1", "NAME_2", "year"]
all_needed = set(feat_names_abs) | set(feat_names_anom)
present    = set(df.columns)
missing    = all_needed - present
print(f"  Features present: {len(all_needed - missing)}/{len(all_needed)}")
print(f"  Missing (will be NaN → imputed): {sorted(missing)}")

# Add missing features as NaN (imputer fills with training median)
for col in missing:
    df[col] = np.nan

# ── 3. Predict — Absolute mode ────────────────────────────────────────────────
print("\nRunning absolute mode predictions...")

X_abs         = df[feat_names_abs].astype(float)
X_abs_imp     = imputer_abs.transform(X_abs)
X_abs_vt      = vt_abs.transform(X_abs_imp)
log_pred_abs  = model_abs.predict(X_abs_vt)

# Back-transform: model predicts log(tons/ha) → expm1 → tons/ha → kg/ha
pred_abs_kgha = np.expm1(log_pred_abs) * 1000
pred_abs_kgha = np.clip(pred_abs_kgha, 0, None)

print(f"  min={pred_abs_kgha.min():.0f}  mean={pred_abs_kgha.mean():.0f}  "
      f"max={pred_abs_kgha.max():.0f} kg/ha")

# ── 4. Predict — Anomaly mode ─────────────────────────────────────────────────
print("\nRunning anomaly mode predictions...")

X_anom     = df[feat_names_anom].astype(float)
X_anom_imp = imputer_anom.transform(X_anom)
X_anom_vt  = vt_anom.transform(X_anom_imp)
pred_anom  = model_anom.predict(X_anom_vt)

print(f"  min={pred_anom.min():.3f}  mean={pred_anom.mean():.3f}  "
      f"max={pred_anom.max():.3f} (z-score)")

# ── 5. Assemble and save ───────────────────────────────────────────────────────
out = df[id_cols].copy().reset_index(drop=True)
out["predicted_yield_abs_kgha"] = pred_abs_kgha
out["log_predicted_yield_abs"]  = log_pred_abs   # log scale for regression
out["predicted_yield_anomaly"]  = pred_anom

# Country name mapping (ISO3 → full name for conflict panel merge)
iso_to_name = {
    "ETH": "Ethiopia",
    "MWI": "Malawi",
    "MLI": "Mali",
    "NGA": "Nigeria",
    "TZA": "United Republic of Tanzania",
    "UGA": "Uganda",
}
out["country_name"] = out["GID_0"].map(iso_to_name)

print("\n=== Summary by Country ===")
print(out.groupby("GID_0").agg(
    districts = ("GID_2",                  "nunique"),
    obs       = ("predicted_yield_abs_kgha","count"),
    mean_kgha = ("predicted_yield_abs_kgha","mean"),
    mean_anom = ("predicted_yield_anomaly", "mean"),
).round(1).to_string())

out.to_csv(OUT_FILE, index=False)
print(f"\nSaved: {OUT_FILE}  ({len(out):,} rows x {len(out.columns)} cols)")
print("\nNext step — copy to thesis Data/ folder:")
print("  scp biadomako@armazi:~/lsms_crop_yields/predictions_v2.csv ~/Desktop/Thesis/Data/")
