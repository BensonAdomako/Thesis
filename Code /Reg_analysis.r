# Reg_analysis.r
# Step 4 of the thesis pipeline.
#
# Inputs:
#   Data/conflict_yields_panel_v2.csv  — Admin-2 x year panel with v2 XGBoost predictions
#
# Outputs:
#   Data/regression_results.tex              — preferred specs (3-month window)
#   Data/regression_results_robustness.tex   — alternative windows + fatalities
#   Data/regression_results_bycountry_ols.tex    — per-country log-log OLS
#   Data/regression_results_bycountry_poisson.tex — per-country Poisson
#
# Specifications:
#   Preferred: 3-month forward conflict window, TWFE OLS + Poisson + log-log OLS
#              key regressor: log_pred_yield_v2 (v2 XGBoost predictions)
#   Robustness: 6-month and 12-month windows; fatality count (Poisson)
#   Per-country: log-log OLS + Poisson, one column per country
#   All TWFE models: Admin-2 + year fixed effects, SEs clustered at Admin-2

# ── 1. Packages ───────────────────────────────────────────────────────────────
library(fixest)
library(dplyr)
library(readr)

# ── 2. Paths ──────────────────────────────────────────────────────────────────
PANEL_CSV        <- "Data/conflict_yields_panel_v2.csv"
TEX_MAIN         <- "Data/regression_results.tex"
TEX_ROBUST       <- "Data/regression_results_robustness.tex"
TEX_COUNTRY_OLS  <- "Data/regression_results_bycountry_ols.tex"
TEX_COUNTRY_POIS <- "Data/regression_results_bycountry_poisson.tex"

# ── 3. Load panel ─────────────────────────────────────────────────────────────
message("Loading panel...")
df <- read_csv(PANEL_CSV, show_col_types = FALSE)
message("  Rows: ", nrow(df), "  Cols: ", ncol(df))

# Unique Admin-2 identifier (ADM2_NAME alone is not unique across countries/states)
df <- df %>%
  mutate(
    adm2_id          = paste(ADM0_NAME, ADM1_NAME, ADM2_NAME, sep = " | "),
    log_pop          = log(population),
    log_conflict_3mo = log1p(conflict_3mo)
  )

# ── 4. Sanity checks ──────────────────────────────────────────────────────────
message("\n--- Pre-regression checks ---")
message("  Unique Admin-2 units    : ", n_distinct(df$adm2_id))
message("  Years                   : ", min(df$year), "-", max(df$year))
message("  NA log_pred_yield_v2    : ", sum(is.na(df$log_pred_yield_v2)))
message("  NA log_pop              : ", sum(is.na(df$log_pop)))
message("  NA conflict_3mo         : ", sum(is.na(df$conflict_3mo)))
message("  Share zero conflict     : ",
        round(100 * mean(df$conflict_3mo == 0), 1), "% (3-month window)")

if (any(is.na(df$log_pop))) {
  warning("NA values in log_pop — check population column")
}
if (sum(is.na(df$log_pred_yield_v2)) > 0) {
  message("  NOTE: Dropping ", sum(is.na(df$log_pred_yield_v2)),
          " rows with NA log_pred_yield_v2 from regressions")
  df <- df %>% filter(!is.na(log_pred_yield_v2))
}

# ── 5. Global rename dictionary for etable ───────────────────────────────────
setFixest_dict(c(
  log_pred_yield_v2 = "Log pred. yield v2 (kg/ha)",
  log_pop           = "Log population",
  conflict_3mo      = "Conflict events (3-month)",
  conflict_6mo      = "Conflict events (6-month)",
  conflict_12mo     = "Conflict events (12-month)",
  log_conflict_3mo  = "Log conflict (3-month)",
  fatalities_3mo    = "Fatalities (3-month)",
  adm2_id           = "Admin-2",
  year              = "Year"
))

# ── 6. Preferred specification: 3-month forward conflict window ───────────────
message("\n=================================================================")
message("PREFERRED SPECIFICATION — 3-month forward conflict window")
message("=================================================================")

# Model 1: Pooled OLS, no FE, no controls (baseline / naive)
m1 <- feols(
  conflict_3mo ~ log_pred_yield_v2,
  data = df
)

# Model 2: TWFE OLS
m2 <- feols(
  conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
  data    = df,
  cluster = ~adm2_id
)

# Model 3: TWFE Poisson — appropriate for count outcome (non-negative, zero-inflated)
m3 <- fepois(
  conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
  data    = df,
  cluster = ~adm2_id
)

# Model 4: TWFE OLS log-log — preferred for elasticity interpretation
m4 <- feols(
  log_conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
  data    = df,
  cluster = ~adm2_id
)

message("\n--- M1: Pooled OLS (no FE, no controls) ---")
print(summary(m1))
message("\n--- M2: TWFE OLS ---")
print(summary(m2))
message("\n--- M3: TWFE Poisson ---")
print(summary(m3))
message("\n--- M4: TWFE OLS log-log [preferred] ---")
print(summary(m4))

# ── 7. Robustness: alternative windows + fatalities ───────────────────────────
message("\n=================================================================")
message("ROBUSTNESS — 6-month forward conflict window")
message("=================================================================")

m5 <- feols(
  conflict_6mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
m6 <- fepois(
  conflict_6mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
print(summary(m5))
print(summary(m6))

message("\n=================================================================")
message("ROBUSTNESS — 12-month forward conflict window")
message("=================================================================")

m7 <- feols(
  conflict_12mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
m8 <- fepois(
  conflict_12mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
print(summary(m7))
print(summary(m8))

message("\n=================================================================")
message("ROBUSTNESS — Fatality count (Poisson)")
message("=================================================================")

m9 <- fepois(
  fatalities_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
  data = df, cluster = ~adm2_id
)
print(summary(m9))

# ── 8. Per-country specifications ─────────────────────────────────────────────
message("\n=================================================================")
message("PER-COUNTRY SPECIFICATIONS")
message("=================================================================")

country_map <- c(
  "Ethiopia"                    = "ETH",
  "Malawi"                      = "MWI",
  "Mali"                        = "MLI",
  "Nigeria"                     = "NGA",
  "United Republic of Tanzania" = "TZA",
  "Uganda"                      = "UGA"
)

country_ols     <- list()
country_poisson <- list()

for (cty_full in names(country_map)) {
  short <- country_map[[cty_full]]
  df_c  <- df %>% filter(ADM0_NAME == cty_full)
  message("\n--- ", short, " (n=", nrow(df_c), ") ---")

  ols_c <- tryCatch(
    feols(log_conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
          data = df_c, cluster = ~adm2_id),
    error = function(e) { message("  OLS failed: ", e$message); NULL }
  )
  pois_c <- tryCatch(
    fepois(conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
           data = df_c, cluster = ~adm2_id),
    error = function(e) { message("  Poisson failed: ", e$message); NULL }
  )

  if (!is.null(ols_c)) {
    print(summary(ols_c))
    country_ols[[short]] <- ols_c
  }
  if (!is.null(pois_c)) {
    print(summary(pois_c))
    country_poisson[[short]] <- pois_c
  }
}

# ── 9. LaTeX tables ───────────────────────────────────────────────────────────
message("\n=================================================================")
message("SAVING LaTeX TABLES")
message("=================================================================")

NOTE_MAIN <- paste(
  "Clustered standard errors at Admin-2 level in parentheses.",
  "All TWFE models include Admin-2 and year fixed effects.",
  "Key regressor is log of v2 ML-predicted maize yield (kg/ha) from XGBoost models",
  "trained on GROW-Africa survey data with richer satellite features (spatial R\\textsuperscript{2} $\\approx$ 0.35).",
  "Population control is log of gridded population sum per Admin-2;",
  "2021--2024 values carried forward from 2020.",
  "Niger excluded: negative out-of-sample R\\textsuperscript{2} for all yield models.",
  "* p<0.1, ** p<0.05, *** p<0.01."
)

NOTE_ROBUST <- paste(
  "Clustered standard errors at Admin-2 level in parentheses.",
  "All models include Admin-2 and year fixed effects.",
  "Columns (1)--(3) vary the forward conflict window (OLS).",
  "Column (4) uses fatality count as outcome (Poisson).",
  "Key regressor is log of v2 ML-predicted maize yield (kg/ha).",
  "* p<0.1, ** p<0.05, *** p<0.01."
)

NOTE_COUNTRY_OLS <- paste(
  "Clustered standard errors at Admin-2 level in parentheses.",
  "Each column is a separate TWFE OLS log-log regression for one country.",
  "Outcome: log(1 + conflict events) in the 3-month forward window.",
  "Regressor: log of v2 ML-predicted maize yield (kg/ha).",
  "All models include Admin-2 and year fixed effects.",
  "* p<0.1, ** p<0.05, *** p<0.01."
)

NOTE_COUNTRY_POIS <- paste(
  "Clustered standard errors at Admin-2 level in parentheses.",
  "Each column is a separate TWFE Poisson regression for one country.",
  "Outcome: conflict event count in the 3-month forward window.",
  "Regressor: log of v2 ML-predicted maize yield (kg/ha).",
  "All models include Admin-2 and year fixed effects.",
  "Districts with zero conflict in all years are dropped by Poisson (perfect prediction of zeros).",
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
  m2, m5, m7, m9,
  title    = "Robustness Checks: Alternative Windows and Outcomes",
  headers  = list("(1) 3-Month\nOLS" = 1, "(2) 6-Month\nOLS" = 1,
                  "(3) 12-Month\nOLS" = 1, "(4) Fatalities\nPoisson" = 1),
  se.below = TRUE,
  tex      = TRUE,
  file     = TEX_ROBUST,
  replace  = TRUE,
  notes    = NOTE_ROBUST
)
message("  Saved: ", TEX_ROBUST)

# Table 3: Per-country log-log OLS
if (length(country_ols) > 0) {
  etable(
    country_ols,
    title    = "Per-Country TWFE OLS Log-Log: Effect of Predicted Yield on Conflict (3-Month Window)",
    headers  = as.list(setNames(rep(1L, length(country_ols)), names(country_ols))),
    se.below = TRUE,
    tex      = TRUE,
    file     = TEX_COUNTRY_OLS,
    replace  = TRUE,
    notes    = NOTE_COUNTRY_OLS
  )
  message("  Saved: ", TEX_COUNTRY_OLS)
}

# Table 4: Per-country Poisson
if (length(country_poisson) > 0) {
  etable(
    country_poisson,
    title    = "Per-Country TWFE Poisson: Effect of Predicted Yield on Conflict (3-Month Window)",
    headers  = as.list(setNames(rep(1L, length(country_poisson)), names(country_poisson))),
    se.below = TRUE,
    tex      = TRUE,
    file     = TEX_COUNTRY_POIS,
    replace  = TRUE,
    notes    = NOTE_COUNTRY_POIS
  )
  message("  Saved: ", TEX_COUNTRY_POIS)
}

message("\nDone. Results saved to:")
message("  ", TEX_MAIN)
message("  ", TEX_ROBUST)
message("  ", TEX_COUNTRY_OLS)
message("  ", TEX_COUNTRY_POIS)
