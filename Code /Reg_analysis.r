# Reg_analysis.r
# Step 4 of the thesis pipeline.
#
# Inputs:
#   Data/conflict_yields_panel.csv   — Admin-2 x year panel (Step 3 output)
#
# Outputs:
#   Data/regression_results.tex          — LaTeX table, preferred specs
#   Data/regression_results_robustness.tex — LaTeX table, robustness checks
#
# Specifications:
#   Preferred: 3-month forward conflict window, TWFE OLS + Poisson,
#              log predicted yield + log population, SEs clustered at Admin-2
#   Robustness: 6-month and 12-month windows; binary (LPM); fatality count (Poisson)
#   All TWFE models: Admin-2 + year fixed effects
#   Admin-2 ID = ADM0_NAME | ADM1_NAME | ADM2_NAME (unique across countries)

# ── 1. Packages ───────────────────────────────────────────────────────────────
library(fixest)
library(dplyr)
library(readr)

# ── 2. Paths ──────────────────────────────────────────────────────────────────
PANEL_CSV   <- "Data/conflict_yields_panel.csv"
TEX_MAIN    <- "Data/regression_results.tex"
TEX_ROBUST  <- "Data/regression_results_robustness.tex"
TEX_IV      <- "Data/regression_results_iv.tex"

# ── 3. Load panel ─────────────────────────────────────────────────────────────
message("Loading panel...")
df <- read_csv(PANEL_CSV, show_col_types = FALSE)
message("  Rows: ", nrow(df), "  Cols: ", ncol(df))

# Unique Admin-2 identifier (ADM2_NAME alone is not unique across countries/states)
df <- df %>%
  mutate(
    adm2_id = paste(ADM0_NAME, ADM1_NAME, ADM2_NAME, sep = " | "),
    log_pop = log(population)
  )

# ── 4. Sanity checks ──────────────────────────────────────────────────────────
message("\n--- Pre-regression checks ---")
message("  Unique Admin-2 units : ", n_distinct(df$adm2_id))
message("  Years                : ", min(df$year), "-", max(df$year))
message("  NA log_pred_yield    : ", sum(is.na(df$log_pred_yield)))
message("  NA log_pop           : ", sum(is.na(df$log_pop)))
message("  NA conflict_3mo      : ", sum(is.na(df$conflict_3mo)))
message("  Share zero conflict  : ",
        round(100 * mean(df$conflict_3mo == 0), 1), "% (3-month window)")

if (any(is.na(df$log_pop))) {
  warning("NA values in log_pop — check population column")
}

# ── 5. Global rename dictionary for etable ───────────────────────────────────
setFixest_dict(c(
  log_pred_yield   = "Log pred. yield (kg/ha)",
  log_pop          = "Log population",
  conflict_3mo     = "Conflict events (3-month)",
  conflict_6mo     = "Conflict events (6-month)",
  conflict_12mo    = "Conflict events (12-month)",
  log_conflict_3mo = "Log conflict (3-month)",
  any_conflict_3mo = "Any conflict (3-month)",
  fatalities_3mo   = "Fatalities (3-month)",
  adm2_id          = "Admin-2",
  year             = "Year"
))

# ── 6. Preferred specification: 3-month forward conflict window ───────────────
message("\n=================================================================")
message("PREFERRED SPECIFICATION — 3-month forward conflict window")
message("=================================================================")

# Model 1: Pooled OLS, no FE, no controls (baseline / naive)
m1 <- feols(
  conflict_3mo ~ log_pred_yield,
  data = df
)

# Model 2: TWFE OLS — preferred main spec
m2 <- feols(
  conflict_3mo ~ log_pred_yield + log_pop | adm2_id + year,
  data    = df,
  cluster = ~adm2_id
)

# Model 3: TWFE Poisson — appropriate for count outcome (non-negative, zero-inflated)
m3 <- fepois(
  conflict_3mo ~ log_pred_yield + log_pop | adm2_id + year,
  data    = df,
  cluster = ~adm2_id
)

# Model 4: TWFE OLS log-log — log conflict on log yield
m4 <- feols(
  log_conflict_3mo ~ log_pred_yield + log_pop | adm2_id + year,
  data    = df,
  cluster = ~adm2_id
)

message("\n--- M1: Pooled OLS (no FE, no controls) ---")
print(summary(m1))
message("\n--- M2: TWFE OLS [preferred] ---")
print(summary(m2))
message("\n--- M3: TWFE Poisson ---")
print(summary(m3))
message("\n--- M4: TWFE OLS log-log ---")
print(summary(m4))

# ── 7. Robustness: alternative windows ───────────────────────────────────────
message("\n=================================================================")
message("ROBUSTNESS — 6-month forward conflict window")
message("=================================================================")

m5 <- feols(
  conflict_6mo ~ log_pred_yield + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
m6 <- fepois(
  conflict_6mo ~ log_pred_yield + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
print(summary(m5))
print(summary(m6))

message("\n=================================================================")
message("ROBUSTNESS — 12-month forward conflict window")
message("=================================================================")

m7 <- feols(
  conflict_12mo ~ log_pred_yield + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
m8 <- fepois(
  conflict_12mo ~ log_pred_yield + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
print(summary(m7))
print(summary(m8))

message("\n=================================================================")
message("ROBUSTNESS — Binary conflict (linear probability model)")
message("=================================================================")

m9 <- feols(
  any_conflict_3mo ~ log_pred_yield + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
print(summary(m9))

message("\n=================================================================")
message("ROBUSTNESS — Fatality count (Poisson)")
message("=================================================================")

m10 <- fepois(
  fatalities_3mo ~ log_pred_yield + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
print(summary(m10))

# ── 8. IV (2SLS) Regressions ─────────────────────────────────────────────────
# Endogenous: log_obs_yield (log of LSMS-observed yield)
# Instrument:  log_pred_yield (log of ML-predicted yield)
# Sample: district-years where mean_obs_yield is non-null (~18% of panel)
# FE: admin-2 + year (two-way); SEs clustered at admin-2
# fixest IV syntax: feols(y ~ exog | fe | endog ~ instr, ...)

message("\n=================================================================")
message("IV REGRESSIONS (2SLS) — instrument: log_pred_yield")
message("=================================================================")

# IV sample (district-years with observed yield)
df_iv <- df %>% filter(!is.na(mean_obs_yield))
message("\nIV sample (all 6 countries):")
message("  Observations : ", nrow(df_iv))
message("  Districts    : ", n_distinct(df_iv$adm2_id))
message("  Countries    : ", paste(sort(unique(df_iv$ADM0_NAME)), collapse = ", "))
message("  Share zero conflict (3-month): ",
        round(100 * mean(df_iv$conflict_3mo == 0), 1), "%")

# ── IV1: All 6 countries ──────────────────────────────────────────────────────
message("\n--- IV1: All 6 countries ---")
iv1 <- feols(
  conflict_3mo ~ 1 | adm2_id + year | log_obs_yield ~ log_pred_yield,
  data    = df_iv,
  cluster = ~adm2_id
)
message("Second stage:")
print(summary(iv1))
message("First stage:")
print(summary(iv1, stage = 1))
iv1_f <- fitstat(iv1, "ivf1")[[1]]$stat
message(sprintf("  First stage F-stat: %.2f", iv1_f))
if (iv1_f < 5) {
  message("  *** VERY WEAK INSTRUMENT (F < 5) — results uninterpretable ***")
} else if (iv1_f < 10) {
  message("  ** WARNING: weak instrument (F < 10) **")
}

# ── IV2: Drop Uganda (only 2010-2013, thin IV sample) ────────────────────────
message("\n--- IV2: Drop Uganda ---")
df_iv2 <- df_iv %>% filter(ADM0_NAME != "Uganda")
message("  N=", nrow(df_iv2), "  districts=", n_distinct(df_iv2$adm2_id),
        "  countries: ", paste(sort(unique(df_iv2$ADM0_NAME)), collapse=", "))
iv2 <- feols(
  conflict_3mo ~ 1 | adm2_id + year | log_obs_yield ~ log_pred_yield,
  data    = df_iv2,
  cluster = ~adm2_id
)
message("Second stage:")
print(summary(iv2))
message("First stage:")
print(summary(iv2, stage = 1))
iv2_f <- fitstat(iv2, "ivf1")[[1]]$stat
message(sprintf("  First stage F-stat: %.2f", iv2_f))
if (iv2_f < 5) {
  message("  *** VERY WEAK INSTRUMENT (F < 5) — results uninterpretable ***")
} else if (iv2_f < 10) {
  message("  ** WARNING: weak instrument (F < 10) **")
}

# ── IV3: Drop Uganda + Tanzania (Tanzania 2010-2013 only) ────────────────────
message("\n--- IV3: Drop Uganda + Tanzania ---")
df_iv3 <- df_iv %>%
  filter(!ADM0_NAME %in% c("Uganda", "United Republic of Tanzania"))
message("  N=", nrow(df_iv3), "  districts=", n_distinct(df_iv3$adm2_id),
        "  countries: ", paste(sort(unique(df_iv3$ADM0_NAME)), collapse=", "))
iv3 <- feols(
  conflict_3mo ~ 1 | adm2_id + year | log_obs_yield ~ log_pred_yield,
  data    = df_iv3,
  cluster = ~adm2_id
)
message("Second stage:")
print(summary(iv3))
message("First stage:")
print(summary(iv3, stage = 1))
iv3_f <- fitstat(iv3, "ivf1")[[1]]$stat
message(sprintf("  First stage F-stat: %.2f", iv3_f))
if (iv3_f < 5) {
  message("  *** VERY WEAK INSTRUMENT (F < 5) — results uninterpretable ***")
} else if (iv3_f < 10) {
  message("  ** WARNING: weak instrument (F < 10) **")
}

# ── IV4: Ethiopia + Malawi + Mali only (best model R²) ───────────────────────
message("\n--- IV4: Ethiopia + Malawi + Mali (highest-R² models) ---")
df_iv4 <- df_iv %>%
  filter(ADM0_NAME %in% c("Ethiopia", "Malawi", "Mali"))
message("  N=", nrow(df_iv4), "  districts=", n_distinct(df_iv4$adm2_id),
        "  countries: ", paste(sort(unique(df_iv4$ADM0_NAME)), collapse=", "))
iv4 <- feols(
  conflict_3mo ~ 1 | adm2_id + year | log_obs_yield ~ log_pred_yield,
  data    = df_iv4,
  cluster = ~adm2_id
)
message("Second stage:")
print(summary(iv4))
message("First stage:")
print(summary(iv4, stage = 1))
iv4_f <- fitstat(iv4, "ivf1")[[1]]$stat
message(sprintf("  First stage F-stat: %.2f", iv4_f))
if (iv4_f < 5) {
  message("  *** VERY WEAK INSTRUMENT (F < 5) — results uninterpretable ***")
} else if (iv4_f < 10) {
  message("  ** WARNING: weak instrument (F < 10) **")
}

# ── 9. LaTeX tables ───────────────────────────────────────────────────────────
message("\n=================================================================")
message("SAVING LaTeX TABLES")
message("=================================================================")

NOTE_MAIN <- paste(
  "Clustered standard errors at Admin-2 level in parentheses.",
  "All TWFE models include Admin-2 and year fixed effects.",
  "Key regressor is log of ML-predicted maize yield (kg/ha) from hybrid XGBoost models",
  "(Ethiopia, Tanzania, Uganda: country-specific; Mali, Nigeria: global model; Malawi: country-specific).",
  "Population control is log of gridded population sum per Admin-2;",
  "2021--2024 values carried forward from 2020.",
  "Niger excluded: negative out-of-sample R\\textsuperscript{2} for all yield models.",
  "* p<0.1, ** p<0.05, *** p<0.01."
)

NOTE_ROBUST <- paste(
  "Clustered standard errors at Admin-2 level in parentheses.",
  "All models include Admin-2 and year fixed effects.",
  "Columns (1)--(3) vary the forward conflict window (OLS).",
  "Column (4) is a linear probability model with binary conflict outcome.",
  "Column (5) uses fatality count as outcome (Poisson).",
  "* p<0.1, ** p<0.05, *** p<0.01."
)

# Table 1: Preferred specifications
etable(
  m1, m2, m3, m4,
  title    = "Effect of Predicted Crop Yield on Conflict (3-Month Forward Window)",
  headers  = list("(1) OLS\nNo FE" = 1, "(2) TWFE\nOLS" = 1,
                  "(3) TWFE\nPoisson" = 1, "(4) TWFE OLS\nLog-Log" = 1),
  se.below = TRUE,
  tex      = TRUE,
  file     = TEX_MAIN,
  replace  = TRUE,
  notes    = NOTE_MAIN
)
message("  Saved: ", TEX_MAIN)

# Table 2: Robustness checks
etable(
  m2, m5, m7, m9, m10,
  title    = "Robustness Checks: Alternative Windows and Outcomes",
  headers  = list("(1) 3-Month\nOLS" = 1, "(2) 6-Month\nOLS" = 1,
                  "(3) 12-Month\nOLS" = 1, "(4) Binary\nLPM" = 1,
                  "(5) Fatalities\nPoisson" = 1),
  se.below = TRUE,
  tex      = TRUE,
  file     = TEX_ROBUST,
  replace  = TRUE,
  notes    = NOTE_ROBUST
)
message("  Saved: ", TEX_ROBUST)

NOTE_IV <- paste(
  "2SLS estimates. Endogenous variable: log of LSMS-observed maize yield (kg/ha).",
  "Instrument: log of ML-predicted maize yield (kg/ha) from hybrid XGBoost models.",
  "Sample restricted to Admin-2 $\\times$ year cells where LSMS observed yield is available",
  "(approx. 18\\% of full panel, corresponding to survey years).",
  "All models include Admin-2 and year fixed effects.",
  "Standard errors clustered at Admin-2 level in parentheses.",
  "IV1: all 6 countries.",
  "IV2: drops Uganda (only 2010--2013 survey coverage).",
  "IV3: drops Uganda and Tanzania (Tanzania survey years 2010--2013 only).",
  "IV4: Ethiopia, Malawi, Mali only (countries with highest model R\\textsuperscript{2}).",
  "First-stage F-statistic reported. F $<$ 10 indicates potential weak instrument.",
  "* p<0.1, ** p<0.05, *** p<0.01."
)

setFixest_dict(c(
  getFixest_dict(),
  `fit_ivf1`          = "First-stage F-stat",
  `log_obs_yield`     = "Log obs. yield (kg/ha)",
  `log_obs_yield_hat` = "Log obs. yield (kg/ha) [IV]"
))

etable(
  iv1, iv2, iv3, iv4,
  title    = "IV (2SLS) Estimates: Effect of Crop Yield on Conflict (3-Month Forward Window)",
  headers  = list("(1) IV1\nAll 6" = 1, "(2) IV2\nDrop UGA" = 1,
                  "(3) IV3\nDrop UGA+TZA" = 1, "(4) IV4\nETH+MWI+MLI" = 1),
  fitstat  = ~ ivf1 + n + r2,
  se.below = TRUE,
  tex      = TRUE,
  file     = TEX_IV,
  replace  = TRUE,
  notes    = NOTE_IV
)
message("  Saved: ", TEX_IV)

message("\nDone. Results saved to:")
message("  ", TEX_MAIN)
message("  ", TEX_ROBUST)
message("  ", TEX_IV)
