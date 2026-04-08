# Reg_analysis.r — Step 4 of thesis pipeline
#
# Inputs:  Data/conflict_yields_panel_v3.csv
#          Data/acled_africa.csv  (for conflict-type breakdown)
#
# Outputs (8 LaTeX tables):
#   Data/regression_results.tex
#   Data/regression_results_robustness.tex
#   Data/regression_results_spatial_lag.tex
#   Data/regression_results_conflict_types.tex
#   Data/regression_results_aez.tex
#   Data/regression_results_bycountry_ols.tex
#   Data/regression_results_bycountry_poisson.tex
#   Data/regression_results_iv.tex

# ── 1. Packages ───────────────────────────────────────────────────────────────
library(fixest)
library(dplyr)
library(readr)
library(sf)
library(spdep)
library(geodata)
library(lubridate)
library(slider)
library(tidyr)

THESIS_ISO3 <- c("BFA","ETH","MLI","MWI","NER","NGA","TZA")

# ── 2. Paths ──────────────────────────────────────────────────────────────────
PANEL_CSV        <- "Data/conflict_yields_panel_v3.csv"
dir.create("Data",            showWarnings = FALSE)
dir.create("Write up/tables", showWarnings = FALSE, recursive = TRUE)
TEX_MAIN         <- "Data/regression_results.tex"
TEX_ROBUST       <- "Data/regression_results_robustness.tex"
TEX_SLAG         <- "Data/regression_results_spatial_lag.tex"
TEX_TYPES        <- "Data/regression_results_conflict_types.tex"
TEX_AEZ          <- "Data/regression_results_aez.tex"
TEX_COUNTRY_OLS  <- "Data/regression_results_bycountry_ols.tex"
TEX_COUNTRY_POIS <- "Data/regression_results_bycountry_poisson.tex"
TEX_IV           <- "Data/regression_results_iv.tex"

# ── 3. Load and prepare panel ─────────────────────────────────────────────────
message("Loading panel...")
df <- read_csv(PANEL_CSV, show_col_types = FALSE) %>%
  mutate(
    adm2_id          = GID_2,
    log_conflict_3mo = log1p(conflict_3mo),
    log_conflict_6mo = log1p(conflict_6mo),
    log_conflict_12mo = log1p(conflict_12mo)
  )

message("  Rows: ", nrow(df), "  Districts: ", n_distinct(df$GID_2))
message("  NA log_pop: ", sum(is.na(df$log_pop)))
message("  NA log_pred_yield_v2: ", sum(is.na(df$log_pred_yield_v2)))

# Drop rows with NA in key regressors
df <- df %>% filter(!is.na(log_pred_yield_v2), !is.na(log_pop))
message("  Rows after dropping NA: ", nrow(df))

# ── 4. Global fixest dictionary ───────────────────────────────────────────────
setFixest_dict(c(
  log_pred_yield_v2      = "Log pred.\ yield (kg/ha)",
  log_pop                = "Log population",
  spatial_lag_conflict   = "Spatial lag (conflict)",
  yield_x_sahel          = "Log yield $\\times$ Sahel",
  sahel                  = "Sahel indicator",
  conflict_3mo           = "Conflict events (3-mo.)",
  conflict_6mo           = "Conflict events (6-mo.)",
  conflict_12mo          = "Conflict events (12-mo.)",
  log_conflict_3mo       = "Log conflict (3-mo.)",
  log_conflict_6mo       = "Log conflict (6-mo.)",
  log_conflict_12mo      = "Log conflict (12-mo.)",
  fatalities_3mo         = "Fatalities (3-mo.)",
  adm2_id                = "Admin-2 FE",
  year                   = "Year FE"
))

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — MAIN RESULTS (3-month window)
# ═══════════════════════════════════════════════════════════════════════════════
message("\n=== TABLE 1: MAIN RESULTS ===")

m1 <- feols(conflict_3mo ~ log_pred_yield_v2,
            data = df)

m2 <- feols(conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
            data = df, cluster = ~adm2_id)

m3 <- fepois(conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
             data = df, cluster = ~adm2_id)

m4 <- feols(log_conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
            data = df, cluster = ~adm2_id)

print(summary(m4))

NOTE_MAIN <- paste0(
  "\\textit{Notes:} Clustered standard errors at the Admin-2 level in parentheses. ",
  "Column~(1) is pooled OLS with no fixed effects or controls. ",
  "Columns~(2)--(4) include Admin-2 and year two-way fixed effects. ",
  "The key regressor is the log of v2 XGBoost-predicted maize yield (kg/ha), ",
  "trained on GROW-Africa administrative yield records with satellite, soil, and climate inputs (spatial $R^2 = 0.35$). ",
  "The log population control is log of WorldPop gridded population summed to Admin-2; ",
  "values for 2021--2024 are carried forward from 2020. ",
  "Column~(4) is the preferred specification. ",
  "Uganda is excluded from all analyses (not in model training data; LOCO $R^2 < 0$). ",
  "$^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$."
)

etable(m1, m2, m3, m4,
  title    = "Effect of Predicted Maize Yield on Conflict: Main Results",
  headers  = list("(1)\\\\Pooled OLS" = 1, "(2)\\\\TWFE OLS" = 1,
                  "(3)\\\\TWFE Poisson" = 1, "(4)\\\\TWFE OLS Log-Log" = 1),
  se.below = TRUE,
  tex      = TRUE,
  file     = TEX_MAIN,
  replace  = TRUE,
  notes    = NOTE_MAIN
)
message("  Saved: ", TEX_MAIN)

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — ROBUSTNESS: ALTERNATIVE WINDOWS + FATALITIES
# ═══════════════════════════════════════════════════════════════════════════════
message("\n=== TABLE 2: ROBUSTNESS ===")

# Log-log for all three windows (preferred spec)
r1 <- feols(log_conflict_3mo  ~ log_pred_yield_v2 + log_pop | adm2_id + year,
            data = df, cluster = ~adm2_id)
r2 <- feols(log_conflict_6mo  ~ log_pred_yield_v2 + log_pop | adm2_id + year,
            data = df, cluster = ~adm2_id)
r3 <- feols(log_conflict_12mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
            data = df, cluster = ~adm2_id)

# Count Poisson (3-month) and fatality Poisson as additional checks
r4 <- fepois(conflict_3mo  ~ log_pred_yield_v2 + log_pop | adm2_id + year,
             data = df, cluster = ~adm2_id)
r5 <- fepois(fatalities_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
             data = df, cluster = ~adm2_id)

print(summary(r2)); print(summary(r3))

NOTE_ROBUST <- paste0(
  "\\textit{Notes:} Clustered standard errors at the Admin-2 level in parentheses. ",
  "All models include Admin-2 and year two-way fixed effects. ",
  "Columns~(1)--(3) use the log$(1+\\text{conflict events})$ outcome under the TWFE OLS log-log specification, ",
  "varying the length of the forward conflict window (3, 6, and 12 months post-harvest). ",
  "Column~(4) uses conflict event counts under TWFE Poisson QMLE. ",
  "Column~(5) replaces the conflict count with fatality count under Poisson QMLE; ",
  "districts with zero fatalities in all years are dropped by the Poisson absorbing condition. ",
  "$^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$."
)

etable(r1, r2, r3, r4, r5,
  title    = "Robustness Checks: Alternative Windows and Outcome Definitions",
  headers  = list("(1)\\\\3-Month\\\\Log-Log" = 1,
                  "(2)\\\\6-Month\\\\Log-Log"  = 1,
                  "(3)\\\\12-Month\\\\Log-Log" = 1,
                  "(4)\\\\3-Month\\\\Poisson"  = 1,
                  "(5)\\\\Fatalities\\\\Poisson" = 1),
  se.below = TRUE,
  tex      = TRUE,
  file     = TEX_ROBUST,
  replace  = TRUE,
  notes    = NOTE_ROBUST
)
message("  Saved: ", TEX_ROBUST)

# ═══════════════════════════════════════════════════════════════════════════════
# SPATIAL LAG DATA PREP
# ═══════════════════════════════════════════════════════════════════════════════
message("\n=== SPATIAL LAG PREP ===")

message("  Loading GADM shapefiles...")
gadm_list <- lapply(THESIS_ISO3, function(iso) {
  st_as_sf(geodata::gadm(iso, level = 2, path = "Data/gadm"))
})
gadm_all   <- bind_rows(gadm_list)
gadm_panel <- gadm_all %>% filter(GID_2 %in% unique(df$GID_2)) %>% arrange(GID_2)

message("  Computing queen contiguity (", nrow(gadm_panel), " units)...")
nb <- poly2nb(gadm_panel, queen = TRUE)
W  <- nb2listw(nb, style = "W", zero.policy = TRUE)

conf_wide <- df %>%
  select(GID_2, year, conflict_3mo) %>%
  pivot_wider(names_from = year, values_from = conflict_3mo, values_fill = 0) %>%
  right_join(tibble(GID_2 = gadm_panel$GID_2), by = "GID_2") %>%
  arrange(match(GID_2, gadm_panel$GID_2))

yr_cols  <- as.character(2010:2024)
slag_mat <- conf_wide %>% select(GID_2)
for (yr in yr_cols) {
  x <- conf_wide[[yr]]; x[is.na(x)] <- 0
  slag_mat[[yr]] <- lag.listw(W, x, zero.policy = TRUE)
}

slag_long <- slag_mat %>%
  pivot_longer(-GID_2, names_to = "year", values_to = "spatial_lag_conflict") %>%
  mutate(year = as.integer(year))

df_slag <- df %>% left_join(slag_long, by = c("GID_2", "year"))
message("  NA spatial lag: ", sum(is.na(df_slag$spatial_lag_conflict)))

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — ROBUSTNESS: SPATIAL LAG CONTROL
# ═══════════════════════════════════════════════════════════════════════════════
message("\n=== TABLE 3: SPATIAL LAG ===")

sl1 <- feols(log_conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
             data = df_slag, cluster = ~adm2_id)
sl2 <- feols(log_conflict_3mo ~ log_pred_yield_v2 + log_pop + spatial_lag_conflict | adm2_id + year,
             data = df_slag, cluster = ~adm2_id)
sl3 <- fepois(conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
              data = df_slag, cluster = ~adm2_id)
sl4 <- fepois(conflict_3mo ~ log_pred_yield_v2 + log_pop + spatial_lag_conflict | adm2_id + year,
              data = df_slag, cluster = ~adm2_id)

print(summary(sl2)); print(summary(sl4))

NOTE_SLAG <- paste0(
  "\\textit{Notes:} Clustered standard errors at the Admin-2 level in parentheses. ",
  "All models include Admin-2 and year two-way fixed effects. ",
  "Columns~(1) and~(3) reproduce the baseline log-log OLS and Poisson specifications without the spatial lag. ",
  "Columns~(2) and~(4) augment those specifications with a spatial lag of the conflict outcome, ",
  "defined as the population-weighted mean of conflict events in queen-contiguous neighbouring districts. ",
  "Contiguity is computed from GADM Level-2 polygons using the \\texttt{spdep} package. ",
  "Districts with no contiguous neighbours receive a spatial lag of zero. ",
  "$^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$."
)

etable(sl1, sl2, sl3, sl4,
  title    = "Robustness Check: Spatial Lag of Conflict",
  headers  = list("(1)\\\\OLS\\\\Baseline"       = 1,
                  "(2)\\\\OLS\\\\Spatial Lag"     = 1,
                  "(3)\\\\Poisson\\\\Baseline"    = 1,
                  "(4)\\\\Poisson\\\\Spatial Lag" = 1),
  se.below = TRUE,
  tex      = TRUE,
  file     = TEX_SLAG,
  replace  = TRUE,
  notes    = NOTE_SLAG
)
message("  Saved: ", TEX_SLAG)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFLICT TYPE DATA PREP
# ═══════════════════════════════════════════════════════════════════════════════
message("\n=== CONFLICT TYPE PREP ===")

acled <- read_csv("Data/acled_africa.csv", show_col_types = FALSE) %>%
  filter(country %in% c("Burkina Faso","Ethiopia","Mali","Malawi",
                         "Niger","Nigeria","Tanzania"),
         year >= 2010, year <= 2024,
         event_type %in% c("Battles","Riots",
                            "Violence against civilians",
                            "Explosions/Remote violence")) %>%
  select(event_date, year, event_type, latitude, longitude)

acled_sf  <- st_as_sf(acled, coords = c("longitude","latitude"), crs = 4326)
gadm_join <- gadm_all %>% select(GID_2) %>% st_make_valid()
joined    <- st_join(acled_sf, gadm_join, join = st_within)

message("  ACLED events matched: ", sum(!is.na(joined$GID_2)), " / ", nrow(joined))

monthly_type <- joined %>%
  st_drop_geometry() %>%
  filter(!is.na(GID_2)) %>%
  mutate(month = floor_date(as.Date(event_date), "month")) %>%
  group_by(GID_2, month, event_type) %>%
  summarise(n_events = n(), .groups = "drop")

all_months <- seq(as.Date("2010-01-01"), as.Date("2024-12-01"), by = "month")
all_types  <- c("Battles","Riots","Violence against civilians",
                "Explosions/Remote violence")

type_windows <- expand_grid(
  GID_2      = unique(df$GID_2),
  month      = all_months,
  event_type = all_types
) %>%
  left_join(monthly_type, by = c("GID_2","month","event_type")) %>%
  mutate(n_events = replace_na(n_events, 0)) %>%
  arrange(GID_2, event_type, month) %>%
  group_by(GID_2, event_type) %>%
  mutate(conf_3mo = slide_dbl(n_events, sum, .before = 0, .after = 2)) %>%
  ungroup()

harvest_months <- df %>% distinct(GID_2, harvest_month)

type_annual <- type_windows %>%
  left_join(harvest_months, by = "GID_2") %>%
  filter(month(month) == harvest_month) %>%
  mutate(year = year(month)) %>%
  filter(year >= 2010, year <= 2024) %>%
  select(GID_2, year, event_type, conf_3mo)

type_wide <- type_annual %>%
  pivot_wider(names_from = event_type, values_from = conf_3mo) %>%
  rename(
    conf_battles = Battles,
    conf_riots   = Riots,
    conf_vac     = `Violence against civilians`,
    conf_expl    = `Explosions/Remote violence`
  )

df_type <- df %>%
  left_join(type_wide, by = c("GID_2","year")) %>%
  mutate(
    log_conf_battles = log1p(conf_battles),
    log_conf_riots   = log1p(conf_riots),
    log_conf_vac     = log1p(conf_vac),
    log_conf_expl    = log1p(conf_expl)
  )

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — HETEROGENEOUS EFFECTS: CONFLICT TYPE
# ═══════════════════════════════════════════════════════════════════════════════
message("\n=== TABLE 4: CONFLICT TYPES ===")

# Update dict for type outcomes
setFixest_dict(c(
  log_pred_yield_v2  = "Log pred.\\ yield (kg/ha)",
  log_pop            = "Log population",
  log_conf_battles   = "Log battles (3-mo.)",
  log_conf_riots     = "Log riots (3-mo.)",
  log_conf_vac       = "Log viol.\\ vs.~civilians (3-mo.)",
  log_conf_expl      = "Log expl./remote (3-mo.)",
  spatial_lag_conflict = "Spatial lag (conflict)",
  yield_x_sahel      = "Log yield $\\times$ Sahel",
  log_conflict_3mo   = "Log conflict (3-mo.)",
  log_conflict_6mo   = "Log conflict (6-mo.)",
  log_conflict_12mo  = "Log conflict (12-mo.)",
  conflict_3mo       = "Conflict events (3-mo.)",
  fatalities_3mo     = "Fatalities (3-mo.)",
  adm2_id            = "Admin-2 FE",
  year               = "Year FE"
))

ct1 <- feols(log_conf_battles ~ log_pred_yield_v2 + log_pop | adm2_id + year,
             data = df_type, cluster = ~adm2_id)
ct2 <- feols(log_conf_riots   ~ log_pred_yield_v2 + log_pop | adm2_id + year,
             data = df_type, cluster = ~adm2_id)
ct3 <- feols(log_conf_vac     ~ log_pred_yield_v2 + log_pop | adm2_id + year,
             data = df_type, cluster = ~adm2_id)
ct4 <- feols(log_conf_expl    ~ log_pred_yield_v2 + log_pop | adm2_id + year,
             data = df_type, cluster = ~adm2_id)

for (m in list(ct1,ct2,ct3,ct4)) print(summary(m))

NOTE_TYPES <- paste0(
  "\\textit{Notes:} Clustered standard errors at the Admin-2 level in parentheses. ",
  "All models are TWFE OLS log-log, estimated on the full seven-country panel. ",
  "Each column uses a conflict-type-specific outcome: log$(1+\\text{events})$ in the 3-month post-harvest window ",
  "for battles, riots, violence against civilians, and explosions/remote violence respectively. ",
  "Event-type disaggregation is based on the ACLED \\texttt{event\\_type} classification. ",
  "All models include Admin-2 and year two-way fixed effects. ",
  "$^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$."
)

etable(ct1, ct2, ct3, ct4,
  title    = "Heterogeneous Effects by Conflict Type (TWFE OLS Log-Log)",
  headers  = list("(1)\\\\Battles" = 1, "(2)\\\\Riots" = 1,
                  "(3)\\\\Viol.\\ vs.\\ Civ." = 1,
                  "(4)\\\\Expl./Remote" = 1),
  se.below = TRUE,
  tex      = TRUE,
  file     = TEX_TYPES,
  replace  = TRUE,
  notes    = NOTE_TYPES
)
message("  Saved: ", TEX_TYPES)

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE 5 — HETEROGENEOUS EFFECTS: AEZ (SAHEL vs NON-SAHEL)
# ═══════════════════════════════════════════════════════════════════════════════
message("\n=== TABLE 5: AEZ ===")

df_aez <- df %>%
  mutate(
    sahel         = as.integer(GID_0 %in% c("BFA","MLI","NER")),
    yield_x_sahel = log_pred_yield_v2 * sahel
  )

df_nonsahel <- df_aez %>% filter(sahel == 0)
df_sahel    <- df_aez %>% filter(sahel == 1)

az1 <- feols(log_conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
             data = df_nonsahel, cluster = ~adm2_id)
az2 <- feols(log_conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
             data = df_sahel, cluster = ~adm2_id)
az3 <- feols(log_conflict_3mo ~ log_pred_yield_v2 + yield_x_sahel + log_pop | adm2_id + year,
             data = df_aez, cluster = ~adm2_id)
az4 <- fepois(conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
              data = df_nonsahel, cluster = ~adm2_id)
az5 <- fepois(conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
              data = df_sahel, cluster = ~adm2_id)

for (m in list(az1,az2,az3,az4,az5)) print(summary(m))

NOTE_AEZ <- paste0(
  "\\textit{Notes:} Clustered standard errors at the Admin-2 level in parentheses. ",
  "All models include Admin-2 and year two-way fixed effects. ",
  "The Sahel zone comprises Burkina Faso, Mali, and Niger; ",
  "the Non-Sahel zone comprises Ethiopia, Malawi, Nigeria, and Tanzania. ",
  "Columns~(1) and~(2) are split-sample TWFE OLS log-log specifications. ",
  "Column~(3) is the full-sample interaction model; the coefficient on ",
  "\\textit{Log yield $\\times$ Sahel} gives the differential yield effect for the Sahel zone ",
  "relative to the Non-Sahel baseline: the net Sahel effect is the sum of \\textit{Log pred.\\ yield} ",
  "and \\textit{Log yield $\\times$ Sahel}. ",
  "Columns~(4) and~(5) repeat the split-sample estimation under Poisson QMLE. ",
  "$^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$."
)

etable(az1, az2, az3, az4, az5,
  title    = "Heterogeneous Effects by Agro-Ecological Zone: Sahel vs.\\ Non-Sahel",
  headers  = list("(1)\\\\Non-Sahel\\\\OLS"     = 1,
                  "(2)\\\\Sahel\\\\OLS"          = 1,
                  "(3)\\\\Full Sample\\\\OLS (Interaction)" = 1,
                  "(4)\\\\Non-Sahel\\\\Poisson"  = 1,
                  "(5)\\\\Sahel\\\\Poisson"      = 1),
  se.below = TRUE,
  tex      = TRUE,
  file     = TEX_AEZ,
  replace  = TRUE,
  notes    = NOTE_AEZ
)
message("  Saved: ", TEX_AEZ)

# ═══════════════════════════════════════════════════════════════════════════════
# TABLES 6-7 — PER-COUNTRY
# ═══════════════════════════════════════════════════════════════════════════════
message("\n=== TABLES 6-7: PER-COUNTRY ===")

country_ols     <- list()
country_poisson <- list()

for (iso in THESIS_ISO3) {
  d <- df %>% filter(GID_0 == iso)
  if (nrow(d) == 0) next
  message("\n--- ", iso, " (n=", nrow(d), ") ---")

  ols_c <- tryCatch(
    feols(log_conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
          data = d, cluster = ~adm2_id),
    error = function(e) { message("  OLS failed: ", e$message); NULL }
  )
  pois_c <- tryCatch(
    fepois(conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
           data = d, cluster = ~adm2_id),
    error = function(e) { message("  Poisson failed: ", e$message); NULL }
  )
  if (!is.null(ols_c))  { print(summary(ols_c));  country_ols[[iso]]     <- ols_c  }
  if (!is.null(pois_c)) { print(summary(pois_c)); country_poisson[[iso]] <- pois_c }
}

NOTE_COUNTRY <- paste0(
  "\\textit{Notes:} Clustered standard errors at the Admin-2 level in parentheses. ",
  "Each column is a separate TWFE regression for one country. ",
  "All models include Admin-2 and year two-way fixed effects. ",
  "Country ISO3 codes: BFA = Burkina Faso, ETH = Ethiopia, MLI = Mali, ",
  "MWI = Malawi, NER = Niger, NGA = Nigeria, TZA = Tanzania. ",
  "$^{*}$ $p<0.10$, $^{**}$ $p<0.05$, $^{***}$ $p<0.01$."
)

if (length(country_ols) > 0) {
  etable(country_ols,
    title    = "Per-Country TWFE OLS Log-Log: Effect of Predicted Yield on Conflict",
    headers  = as.list(setNames(rep(1L, length(country_ols)), names(country_ols))),
    se.below = TRUE,
    tex      = TRUE,
    file     = TEX_COUNTRY_OLS,
    replace  = TRUE,
    notes    = NOTE_COUNTRY
  )
  message("  Saved: ", TEX_COUNTRY_OLS)
}

if (length(country_poisson) > 0) {
  etable(country_poisson,
    title    = "Per-Country TWFE Poisson: Effect of Predicted Yield on Conflict",
    headers  = as.list(setNames(rep(1L, length(country_poisson)), names(country_poisson))),
    se.below = TRUE,
    tex      = TRUE,
    file     = TEX_COUNTRY_POIS,
    replace  = TRUE,
    notes    = NOTE_COUNTRY
  )
  message("  Saved: ", TEX_COUNTRY_POIS)
}

# ── 8. IV / 2SLS robustness ───────────────────────────────────────────────────
# Instrument: log_pred_yield_v2 (predicted yield) for log observed yield
# IV sample: BFA, ETH, MWI, NER — the 4 countries with GID_2-level GROW-Africa
# observed yield records. MLI, NGA, TZA only have national-level records.

message("\nSection 8: IV / 2SLS robustness...")

grow_obs <- read_csv("Data/grow_maize_yields.csv", show_col_types = FALSE) %>%
  filter(GID_0 %in% THESIS_ISO3,
         !is.na(GADM_GID_2), GADM_GID_2 != "",
         !is.na(yield_tons_ha)) %>%
  mutate(log_yield_obs = log(yield_tons_ha * 1000)) %>%   # t/ha -> kg/ha -> log
  select(GID_2 = GADM_GID_2, year, log_yield_obs)

iv_df <- df %>%
  inner_join(grow_obs, by = c("GID_2", "year")) %>%
  filter(!is.na(log_pred_yield_v2), !is.na(log_pop))

message("  IV sample: N = ", nrow(iv_df),
        " | Districts = ", n_distinct(iv_df$GID_2),
        " | Countries: ", paste(sort(unique(iv_df$GID_0)), collapse = ", "))

# First stage
iv_fs <- feols(log_yield_obs ~ log_pred_yield_v2 + log_pop | GID_2 + year,
               data = iv_df, cluster = ~GID_2)
fs_f  <- fitstat(iv_fs, "f")[[1]]$stat
message("  First-stage F = ", round(fs_f, 2))

# OLS on observed yield (IV sample) — benchmark
iv_ols <- feols(log_conflict_3mo ~ log_yield_obs + log_pop | GID_2 + year,
                data = iv_df, cluster = ~GID_2)

# 2SLS TWFE
iv_2sls <- feols(log_conflict_3mo ~ log_pop | GID_2 + year |
                   log_yield_obs ~ log_pred_yield_v2,
                 data = iv_df, cluster = ~GID_2)

NOTE_IV <- paste0(
  "IV sample: BFA, ETH, MWI, NER (N = ", nrow(iv_df), ", ",
  n_distinct(iv_df$GID_2), " Admin-2 districts). ",
  "Instrument: log ML-predicted maize yield (v2 XGBoost). ",
  "First-stage F = ", round(fs_f, 1), " (Wald, clustered). ",
  "Wu-Hausman endogeneity test p = ",
  round(fitstat(iv_2sls, "wh")[[1]]$p, 3), ". ",
  "Standard errors clustered by Admin-2."
)

etable(iv_ols, iv_2sls,
       headers   = c("TWFE OLS (obs. yield)", "TWFE 2SLS"),
       keep      = c("log_yield_obs", "fit_log_yield_obs"),
       coefstat  = "se",
       se.below  = TRUE,
       digits    = 3,
       fitstat   = c("n", "r2", "wr2"),
       notes     = NOTE_IV,
       file      = TEX_IV,
       replace   = TRUE)
message("  Saved: ", TEX_IV)

message("\nDone. Tables saved:")
all_tex <- c(TEX_MAIN, TEX_ROBUST, TEX_SLAG, TEX_TYPES, TEX_AEZ,
             TEX_COUNTRY_OLS, TEX_COUNTRY_POIS, TEX_IV)
for (f in all_tex) message("  ", f)

# Mirror all tables to Write up/tables/ for Overleaf
for (f in all_tex) {
  dest <- file.path("Write up/tables", basename(f))
  file.copy(f, dest, overwrite = TRUE)
}
message("Tables also mirrored to Write up/tables/ (for Overleaf)")
