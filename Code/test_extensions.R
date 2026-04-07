# test_extensions.R
# Exploratory script — results only, does NOT modify main panel or LaTeX tables.
#
# Tests:
#   1. Spatial lag control (average conflict in neighbouring districts)
#   2. Heterogeneous effects by conflict type (Battles / Riots / Violence against civilians)
#   3. Heterogeneous effects by agro-ecological zone proxy (Sahel vs non-Sahel)

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

# ── Load panel ────────────────────────────────────────────────────────────────
df <- read_csv("Data/conflict_yields_panel_v3.csv", show_col_types = FALSE) %>%
  mutate(adm2_id = GID_2)

message("Panel: ", nrow(df), " rows, ", n_distinct(df$GID_2), " districts")

# ══════════════════════════════════════════════════════════════════════════════
# 1. SPATIAL LAG
# Spatial lag = average conflict_3mo among queen-contiguous neighbours
# ══════════════════════════════════════════════════════════════════════════════
message("\n=== 1. SPATIAL LAG ===")

# Load GADM shapefiles (cached from build_panel_v3.R run)
message("Loading GADM shapefiles...")
gadm_list <- lapply(THESIS_ISO3, function(iso) {
  g <- geodata::gadm(iso, level = 2, path = "Data/gadm")
  st_as_sf(g)
})
gadm_all <- bind_rows(gadm_list)

# Keep only districts that appear in panel
gadm_panel <- gadm_all %>%
  filter(GID_2 %in% unique(df$GID_2)) %>%
  arrange(GID_2)   # consistent ordering

message("  GADM districts in panel: ", nrow(gadm_panel))

# Queen contiguity neighbours
message("  Computing queen contiguity...")
nb <- poly2nb(gadm_panel, queen = TRUE)
W  <- nb2listw(nb, style = "W", zero.policy = TRUE)

# Spatial lag for each year
# Build a wide conflict matrix: rows = districts (in gadm_panel order), cols = years
conf_wide <- df %>%
  select(GID_2, year, conflict_3mo) %>%
  pivot_wider(names_from = year, values_from = conflict_3mo, values_fill = 0)

# Align to gadm_panel order
conf_mat <- conf_wide %>%
  right_join(tibble(GID_2 = gadm_panel$GID_2), by = "GID_2") %>%
  arrange(match(GID_2, gadm_panel$GID_2))

yr_cols <- as.character(2010:2024)

# Compute spatial lag year by year
slag_mat <- conf_mat %>% select(GID_2)
for (yr in yr_cols) {
  x <- conf_mat[[yr]]
  x[is.na(x)] <- 0
  slag_mat[[yr]] <- lag.listw(W, x, zero.policy = TRUE)
}

slag_long <- slag_mat %>%
  pivot_longer(-GID_2, names_to = "year", values_to = "spatial_lag_conflict") %>%
  mutate(year = as.integer(year))

df_slag <- df %>%
  left_join(slag_long, by = c("GID_2", "year"))

message("  NA spatial lag: ", sum(is.na(df_slag$spatial_lag_conflict)),
        " (border districts with no neighbours)")

# Regressions with spatial lag added
message("\n--- Spatial lag specs (3-month window) ---")

sl_ols <- feols(
  log_conflict_3mo ~ log_pred_yield_v2 + log_pop + spatial_lag_conflict | adm2_id + year,
  data = df_slag, cluster = ~adm2_id
)
sl_pois <- fepois(
  conflict_3mo ~ log_pred_yield_v2 + log_pop + spatial_lag_conflict | adm2_id + year,
  data = df_slag, cluster = ~adm2_id
)

message("\nTWFE OLS log-log + spatial lag:")
print(summary(sl_ols))
message("\nTWFE Poisson + spatial lag:")
print(summary(sl_pois))

message("\n--- Spatial lag: per-country OLS ---")
for (iso in THESIS_ISO3) {
  d <- df_slag %>% filter(GID_0 == iso)
  m <- tryCatch(
    feols(log_conflict_3mo ~ log_pred_yield_v2 + log_pop + spatial_lag_conflict | adm2_id + year,
          data = d, cluster = ~adm2_id),
    error = function(e) NULL
  )
  if (!is.null(m)) {
    b <- coef(m)["log_pred_yield_v2"]
    p <- pvalue(m)["log_pred_yield_v2"]
    message(sprintf("  %s  beta=%.3f  p=%.3f", iso, b, p))
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFLICT TYPE HETEROGENEITY
# Rebuild 3-month forward conflict counts by event_type from ACLED
# ══════════════════════════════════════════════════════════════════════════════
message("\n=== 2. CONFLICT TYPE HETEROGENEITY ===")

# Load ACLED and spatially join to GADM (reuse gadm_all)
message("Loading ACLED and joining to GADM...")
acled <- read_csv("Data/acled_africa.csv", show_col_types = FALSE) %>%
  filter(country %in% c("Burkina Faso","Ethiopia","Mali","Malawi",
                         "Niger","Nigeria","Tanzania"),
         year >= 2010, year <= 2024) %>%
  filter(event_type %in% c("Battles",
                            "Riots",
                            "Violence against civilians",
                            "Explosions/Remote violence")) %>%
  select(event_id_cnty, event_date, year, event_type, latitude, longitude, fatalities)

acled_sf <- st_as_sf(acled, coords = c("longitude","latitude"), crs = 4326)

gadm_join <- gadm_all %>% select(GID_0, GID_2) %>% st_make_valid()
joined    <- st_join(acled_sf, gadm_join, join = st_within)

message("  Matched ", sum(!is.na(joined$GID_2)), " / ", nrow(joined), " events")

# Build monthly counts by GID_2 × event_type
monthly_type <- joined %>%
  st_drop_geometry() %>%
  filter(!is.na(GID_2)) %>%
  mutate(month = floor_date(as.Date(event_date), "month")) %>%
  group_by(GID_2, month, event_type) %>%
  summarise(n_events = n(), .groups = "drop")

# Full monthly grid × event_type
all_gid2   <- unique(df$GID_2)
all_months <- seq(as.Date("2010-01-01"), as.Date("2024-12-01"), by = "month")
all_types  <- c("Battles","Riots","Violence against civilians","Explosions/Remote violence")

grid_type <- expand_grid(GID_2 = all_gid2, month = all_months, event_type = all_types) %>%
  left_join(monthly_type, by = c("GID_2","month","event_type")) %>%
  mutate(n_events = replace_na(n_events, 0)) %>%
  arrange(GID_2, event_type, month)

# 3-month forward window per type
message("  Computing 3-month forward windows by type...")
type_windows <- grid_type %>%
  group_by(GID_2, event_type) %>%
  mutate(conflict_3mo_type = slide_dbl(n_events, sum, .before = 0, .after = 2)) %>%
  ungroup()

# Join harvest_month and extract harvest-month rows
harvest_months <- df %>% distinct(GID_0, GID_2, harvest_month)

type_annual <- type_windows %>%
  left_join(harvest_months, by = "GID_2") %>%
  filter(month(month) == harvest_month) %>%
  mutate(year = year(month)) %>%
  filter(year >= 2010, year <= 2024) %>%
  select(GID_2, year, event_type, conflict_3mo_type)

# Pivot wide and merge to panel
type_wide <- type_annual %>%
  pivot_wider(names_from = event_type, values_from = conflict_3mo_type,
              names_prefix = "conf_") %>%
  rename_with(~ gsub(" ", "_", gsub("/", "_", .x)))

df_type <- df %>%
  left_join(type_wide, by = c("GID_2","year")) %>%
  mutate(
    log_conf_battles  = log1p(conf_Battles),
    log_conf_riots    = log1p(conf_Riots),
    log_conf_vac      = log1p(`conf_Violence_against_civilians`),
    log_conf_expl     = log1p(`conf_Explosions_Remote_violence`)
  )

message("\n--- By conflict type: TWFE OLS log-log ---")
type_specs <- list(
  Battles            = "log_conf_battles",
  Riots              = "log_conf_riots",
  `Viol. vs Civ.`    = "log_conf_vac",
  `Expl./Remote`     = "log_conf_expl"
)

for (nm in names(type_specs)) {
  outcome <- type_specs[[nm]]
  m <- tryCatch(
    feols(as.formula(paste(outcome,
          "~ log_pred_yield_v2 + log_pop | adm2_id + year")),
          data = df_type, cluster = ~adm2_id),
    error = function(e) NULL
  )
  if (!is.null(m)) {
    b <- coef(m)["log_pred_yield_v2"]
    p <- pvalue(m)["log_pred_yield_v2"]
    n <- nobs(m)
    message(sprintf("  %-22s  beta=%+.3f  p=%.3f  n=%d", nm, b, p, n))
  }
}

message("\n--- By conflict type: TWFE Poisson ---")
type_count_specs <- list(
  Battles          = "conf_Battles",
  Riots            = "conf_Riots",
  `Viol. vs Civ.`  = "conf_Violence_against_civilians",
  `Expl./Remote`   = "conf_Explosions_Remote_violence"
)

for (nm in names(type_count_specs)) {
  outcome <- paste0("`", type_count_specs[[nm]], "`")
  m <- tryCatch(
    fepois(as.formula(paste(outcome,
           "~ log_pred_yield_v2 + log_pop | adm2_id + year")),
           data = df_type, cluster = ~adm2_id),
    error = function(e) NULL
  )
  if (!is.null(m)) {
    b <- coef(m)["log_pred_yield_v2"]
    p <- pvalue(m)["log_pred_yield_v2"]
    n <- nobs(m)
    message(sprintf("  %-22s  beta=%+.3f  p=%.3f  n=%d", nm, b, p, n))
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# 3. AGRO-ECOLOGICAL ZONE PROXY
# Sahel (BFA, MLI, NER) = semi-arid, rain-fed, low rainfall
# Non-Sahel (ETH, MWI, NGA, TZA) = higher rainfall / highland systems
# ══════════════════════════════════════════════════════════════════════════════
message("\n=== 3. AGRO-ECOLOGICAL ZONE PROXY (Sahel vs Non-Sahel) ===")
message("    Sahel: BFA, MLI, NER  |  Non-Sahel: ETH, MWI, NGA, TZA")

df_aez <- df %>%
  mutate(aez = if_else(GID_0 %in% c("BFA","MLI","NER"), "Sahel", "Non-Sahel"))

for (zone in c("Sahel","Non-Sahel")) {
  d <- df_aez %>% filter(aez == zone)
  message("\n--- ", zone, " (n=", nrow(d), " obs, ",
          n_distinct(d$GID_2), " districts) ---")

  ols <- tryCatch(
    feols(log_conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
          data = d, cluster = ~adm2_id),
    error = function(e) { message("  OLS failed: ", e$message); NULL }
  )
  pois <- tryCatch(
    fepois(conflict_3mo ~ log_pred_yield_v2 + log_pop | adm2_id + year,
           data = d, cluster = ~adm2_id),
    error = function(e) { message("  Poisson failed: ", e$message); NULL }
  )

  if (!is.null(ols))  { message("  OLS log-log:"); print(summary(ols)) }
  if (!is.null(pois)) { message("  Poisson:");     print(summary(pois)) }
}

# Also run an interaction model: yield × Sahel dummy (full sample)
message("\n--- Interaction: yield × Sahel dummy (pooled) ---")
df_aez <- df_aez %>%
  mutate(sahel      = as.integer(GID_0 %in% c("BFA","MLI","NER")),
         yield_x_sahel = log_pred_yield_v2 * sahel)

int_ols <- feols(
  log_conflict_3mo ~ log_pred_yield_v2 + yield_x_sahel + log_pop | adm2_id + year,
  data = df_aez, cluster = ~adm2_id
)
print(summary(int_ols))

message("\nDone — test_extensions.R complete. No files were modified.")
