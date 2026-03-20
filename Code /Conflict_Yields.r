# Conflict_Yields.r
# Step 3 of the thesis pipeline.
#
# Inputs:
#   Data/Conflict.csv                     — ACLED raw conflict events
#   Data/all_data_with_predictions.csv    — ML yield predictions (Step 2b output)
#   Data/adm2_pop_area.csv                — Admin-2 population by year (control)
#   FAO/GAUL/2015/level2                  — Admin-2 boundaries via Earth Engine
#
# Output:
#   Data/conflict_yields_panel.csv        — Admin-2 × year panel ready for regression
#
# Notes:
#   Niger is excluded from scope (negative R² for all yield models).
#   Tanzania in GAUL/ACLED = "United Republic of Tanzania"; mapped automatically.
#   Forward conflict windows (3, 6, 12 months) start at harvest_end_month.
#   Population merge uses ADM0_NAME + ADM2_NAME + year from adm2_pop_area.csv.

# ── 1. Packages ───────────────────────────────────────────────────────────────
library(rgee)
library(sf)
library(dplyr)
library(readr)
library(slider)
library(lubridate)
library(tidyr)

# ── 2. Paths and constants ────────────────────────────────────────────────────
CONFLICT_CSV <- "Data/Conflict.csv"
YIELDS_CSV   <- "Data/all_data_with_predictions.csv"
POP_CSV      <- "Data/adm2_pop_area.csv"
OUTPUT_CSV   <- "Data/conflict_yields_panel.csv"

COUNTRIES_6 <- c("Ethiopia", "Malawi", "Mali", "Nigeria", "Tanzania", "Uganda")

# GAUL uses "United Republic of Tanzania"; ACLED uses "Tanzania"
GAUL_NAME_MAP <- c("Tanzania" = "United Republic of Tanzania")
to_gaul <- function(ctry) {
  if (ctry %in% names(GAUL_NAME_MAP)) GAUL_NAME_MAP[[ctry]] else ctry
}

# ── 3. Initialize Earth Engine ────────────────────────────────────────────────
message("Initializing Earth Engine...")
ee_Initialize()

# ── 4. Download Admin-2 boundaries for 6 countries ───────────────────────────
message("\nDownloading Admin-2 boundaries...")
admin_fc <- ee$FeatureCollection("FAO/GAUL/2015/level2")

admin_sf_list <- lapply(COUNTRIES_6, function(ctry) {
  gaul_name <- to_gaul(ctry)
  message("  Fetching: ", ctry)
  fc      <- admin_fc$filter(ee$Filter$eq("ADM0_NAME", gaul_name))
  sf_data <- ee_as_sf(fc)
  message("    ", nrow(sf_data), " polygons")
  sf_data
})

admin_sf <- bind_rows(admin_sf_list) %>%
  st_make_valid() %>%
  st_transform(4326)

message("Total Admin-2 polygons: ", nrow(admin_sf))

# ── 5. Load and filter ACLED conflict data ────────────────────────────────────
message("\nLoading conflict data...")
acled <- read_csv(CONFLICT_CSV, show_col_types = FALSE)
message("  Raw: ", nrow(acled), " events")

acled <- acled %>% filter(country %in% COUNTRIES_6)
message("  After 6-country filter: ", nrow(acled), " events")

acled_sf <- st_as_sf(acled, coords = c("longitude", "latitude"), crs = 4326)
if (any(!st_is_valid(acled_sf))) acled_sf <- st_make_valid(acled_sf)

# ── 6. Spatial join: conflict points → Admin-2 ───────────────────────────────
message("Spatial join: conflict → Admin-2...")
conf_joined <- st_join(
  acled_sf,
  admin_sf %>% select(ADM0_NAME, ADM1_NAME, ADM2_CODE, ADM2_NAME),
  join = st_within
)
matched_c <- sum(!is.na(conf_joined$ADM2_NAME))
message("  Matched ", matched_c, " / ", nrow(conf_joined), " events")

# ── 7. Build monthly conflict panel with forward windows ─────────────────────
message("Building monthly conflict panel...")
conf_monthly <- conf_joined %>%
  st_drop_geometry() %>%
  filter(!is.na(ADM2_NAME)) %>%
  mutate(
    event_date  = as.Date(event_date),
    event_month = floor_date(event_date, "month")
  ) %>%
  group_by(ADM0_NAME, ADM2_NAME, event_month) %>%
  summarize(
    conflict_events = n(),
    fatalities      = sum(fatalities, na.rm = TRUE),
    .groups = "drop"
  )

# Compute forward-summed windows via slider.
# .before=0, .after=N-1 → sums the current month plus N-1 following months.
conf_monthly <- conf_monthly %>%
  arrange(ADM0_NAME, ADM2_NAME, event_month) %>%
  group_by(ADM0_NAME, ADM2_NAME) %>%
  mutate(
    conflict_3mo_forward  = slide_dbl(conflict_events, sum, .before = 0, .after = 2,  .complete = FALSE),
    conflict_6mo_forward  = slide_dbl(conflict_events, sum, .before = 0, .after = 5,  .complete = FALSE),
    conflict_12mo_forward = slide_dbl(conflict_events, sum, .before = 0, .after = 11, .complete = FALSE),
    fatalities_3mo_forward  = slide_dbl(fatalities, sum, .before = 0, .after = 2,  .complete = FALSE),
    fatalities_12mo_forward = slide_dbl(fatalities, sum, .before = 0, .after = 11, .complete = FALSE)
  ) %>%
  ungroup()

message("  Monthly conflict rows: ", nrow(conf_monthly))

# ── 8. Load yield predictions ─────────────────────────────────────────────────
message("\nLoading yield predictions...")
yields <- read_csv(YIELDS_CSV, show_col_types = FALSE) %>%
  mutate(harvest_end_month = as.Date(harvest_end_month))

message("  Shape: ", nrow(yields), " rows x ", ncol(yields), " cols")
message("  Countries: ", paste(sort(unique(yields$country)), collapse = ", "))
message("  Years: ", min(yields$year), "-", max(yields$year))

# ── 9. Spatial join: yield points → Admin-2 ──────────────────────────────────
message("Spatial join: yield points -> Admin-2...")
yields_sf <- st_as_sf(
  yields,
  coords  = c("lon_modified", "lat_modified"),
  crs     = 4326,
  remove  = FALSE   # keep lat/lon columns
)

yields_admin <- st_join(
  yields_sf,
  admin_sf %>% select(ADM0_NAME, ADM1_NAME, ADM2_CODE, ADM2_NAME),
  join = st_within
) %>%
  st_drop_geometry()

matched_y <- sum(!is.na(yields_admin$ADM2_NAME))
message("  Matched ", matched_y, " / ", nrow(yields_admin), " yield points")

unmatched_summary <- yields_admin %>%
  group_by(country) %>%
  summarize(
    total     = n(),
    unmatched = sum(is.na(ADM2_NAME)),
    pct_unmatched = round(100 * unmatched / total, 1),
    .groups = "drop"
  )
print(unmatched_summary)

# ── 10. Merge conflict × yields ───────────────────────────────────────────────
message("\nMerging conflict x yields...")

# Both sides use GAUL ADM0_NAME after the spatial join.
# harvest_end_month (Date, day=01) matches event_month (floor_date, day=01).
yields_conf <- yields_admin %>%
  filter(!is.na(ADM2_NAME)) %>%
  left_join(
    conf_monthly,
    by = c("ADM0_NAME", "ADM2_NAME", "harvest_end_month" = "event_month")
  ) %>%
  mutate(
    conflict_events         = replace_na(conflict_events, 0),
    conflict_3mo_forward    = replace_na(conflict_3mo_forward, 0),
    conflict_6mo_forward    = replace_na(conflict_6mo_forward, 0),
    conflict_12mo_forward   = replace_na(conflict_12mo_forward, 0),
    fatalities_3mo_forward  = replace_na(fatalities_3mo_forward, 0),
    fatalities_12mo_forward = replace_na(fatalities_12mo_forward, 0)
  )

message("  Merged rows: ", nrow(yields_conf))

# ── 11. Aggregate to Admin-2 × year ──────────────────────────────────────────
message("Aggregating to Admin-2 x year...")

reg_data <- yields_conf %>%
  mutate(year = as.integer(year)) %>%
  group_by(ADM0_NAME, ADM2_NAME, year) %>%
  summarize(
    mean_pred_yield       = mean(predicted_yield_xgb, na.rm = TRUE),
    mean_obs_yield        = mean(observed_yield_kg,   na.rm = TRUE),  # NA for most rows (LSMS subset only)
    conflict_3mo          = sum(conflict_3mo_forward,  na.rm = TRUE),
    conflict_6mo          = sum(conflict_6mo_forward,  na.rm = TRUE),
    conflict_12mo         = sum(conflict_12mo_forward, na.rm = TRUE),
    fatalities_3mo        = sum(fatalities_3mo_forward,  na.rm = TRUE),
    fatalities_12mo       = sum(fatalities_12mo_forward, na.rm = TRUE),
    any_conflict_3mo      = as.integer(sum(conflict_3mo_forward,  na.rm = TRUE) > 0),
    any_conflict_6mo      = as.integer(sum(conflict_6mo_forward,  na.rm = TRUE) > 0),
    any_conflict_12mo     = as.integer(sum(conflict_12mo_forward, na.rm = TRUE) > 0),
    n_points              = n(),
    .groups = "drop"
  ) %>%
  mutate(
    log_pred_yield    = log1p(mean_pred_yield),
    log_obs_yield     = log1p(mean_obs_yield),
    log_conflict_3mo  = log1p(conflict_3mo),
    log_conflict_6mo  = log1p(conflict_6mo),
    log_conflict_12mo = log1p(conflict_12mo)
  )

message("  Reg data: ", nrow(reg_data), " Admin-2 x year rows")
message("  Admin-2 districts (unique names): ", n_distinct(paste(reg_data$ADM0_NAME, reg_data$ADM2_NAME)))

# ── 12. Merge population controls ────────────────────────────────────────────
message("\nMerging population controls...")
pop <- read_csv(POP_CSV, show_col_types = FALSE) %>%
  select(ADM0_NAME, ADM2_NAME, year, population = sum, adm2_area_km2) %>%
  mutate(year = as.integer(year))

reg_data <- reg_data %>%
  left_join(pop, by = c("ADM0_NAME", "ADM2_NAME", "year"))

matched_pop <- sum(!is.na(reg_data$population))
message("  Population matched: ", matched_pop, " / ", nrow(reg_data), " rows (",
        round(100 * matched_pop / nrow(reg_data), 1), "%)")

# ── 13. Sanity checks ─────────────────────────────────────────────────────────
message("\n--- Sanity checks ---")
message("Countries (GAUL): ", paste(sort(unique(reg_data$ADM0_NAME)), collapse = ", "))
message("Years: ", min(reg_data$year), "-", max(reg_data$year))
message("Total rows: ", nrow(reg_data))
message("Unique Admin-2: ", n_distinct(paste(reg_data$ADM0_NAME, reg_data$ADM2_NAME)))
message("NA pred_yield: ", sum(is.na(reg_data$mean_pred_yield)))
message("NA conflict_3mo: ", sum(is.na(reg_data$conflict_3mo)))
message("NA population: ", sum(is.na(reg_data$population)))

message("\nPer-country row counts:")
print(reg_data %>% count(ADM0_NAME, name = "n_adm2_year"))

message("\nConflict summary (conflict_3mo):")
print(summary(reg_data$conflict_3mo))

message("\nPred yield summary (kg/ha):")
print(summary(reg_data$mean_pred_yield))

# ── 14. Save ──────────────────────────────────────────────────────────────────
write_csv(reg_data, OUTPUT_CSV)
message("\nSaved: ", OUTPUT_CSV)
message("Shape: ", nrow(reg_data), " rows x ", ncol(reg_data), " cols")
message("Columns: ", paste(colnames(reg_data), collapse = ", "))
