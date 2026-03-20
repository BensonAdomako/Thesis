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

# ── 8. LaTeX tables ───────────────────────────────────────────────────────────
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

message("\nDone. Results saved to:")
message("  ", TEX_MAIN)
message("  ", TEX_ROBUST)
