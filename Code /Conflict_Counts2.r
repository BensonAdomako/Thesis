# 1. Load packages
# -----------------------------
library(rgee)
library(sf)
library(dplyr)
library(readr)
library(slider)
library(lubridate)
# -----------------------------
# 2. Initialize Earth Engine
# -----------------------------
ee_Initialize()

# -----------------------------
# 3. Specify countries
# -----------------------------
countries <- c("Nigeria", "Mali", "United Republic of Tanzania", "Uganda", "Ethiopia", "Malawi", "Niger")
message("Target countries: ", paste(countries, collapse = ", "))

# -----------------------------
# 4. Download admin boundaries by country
# -----------------------------
admin <- ee$FeatureCollection("FAO/GAUL/2015/level2")

message("Downloading admin boundaries by country...")
admin_sf_list <- list()

for (country in countries) {
  message("  Downloading: ", country)
  
  country_admin <- admin$filter(ee$Filter$eq("ADM0_NAME", country))
  
  tryCatch({
    admin_sf_list[[country]] <- ee_as_sf(country_admin)
    message("    ✓ Got ", nrow(admin_sf_list[[country]]), " polygons")
  }, error = function(e) {
    message("    ✗ Failed for ", country, ": ", e$message)
  })
}


# Combine all countries
admin_sf <- bind_rows(admin_sf_list)
message("Total admin polygons loaded: ", nrow(admin_sf))

# -----------------------------
# 5. Load local conflict points
# -----------------------------
acled <- read_csv("conflict.csv")
print(colnames(acled))

# Check what conflict types you have
message("Conflict types in data:")
print(table(acled$event_type))  # or whatever your conflict type column is called

acled_sf <- st_as_sf(
  acled,
  coords = c("longitude", "latitude"),
  crs = 4326
)
message("Conflict points loaded: ", nrow(acled_sf))

# -----------------------------
# 6. Ensure valid geometry
# -----------------------------
if (any(!st_is_valid(acled_sf))) {
  message("Fixing invalid geometries in conflict data...")
  acled_sf <- st_make_valid(acled_sf)
}

if (any(!st_is_valid(admin_sf))) {
  message("Fixing invalid geometries in admin data...")
  admin_sf <- st_make_valid(admin_sf)
}

# -----------------------------
# 7. Spatial join
# -----------------------------
message("Performing spatial join...")
joined <- st_join(acled_sf, admin_sf, join = st_within)
message("Joined dataset created with ", nrow(joined), " records.")

# Check how many points were successfully matched
matched <- sum(!is.na(joined$ADM2_NAME))
message("Successfully matched ", matched, " out of ", nrow(joined), " conflict points")


# ---- Load Yields data----
yields <- read_csv("predictions_all_models1.csv") 

yields <- yields %>%
  rename(country = country...1) %>%  # keep this as 'country'
  select(-country...7)               # drop the duplicate

# Create year colomn from Harvest_end_date
yields$year <- format(yields$harvest_end_month, "%Y")

# --- 1) Build predicted yield columns (country-specific) ---
yields_pred <- yields %>%
  mutate(
    pred_log_yield = dplyr::case_when(
      country == "Ethiopia" ~ Pred_Log_Ethiopia,
      country == "Malawi"   ~ Pred_Log_Malawi,
      country == "Mali"     ~ Pred_Log_Mali,
      country == "Nigeria"  ~ Pred_Log_Nigeria,
      country == "Tanzania" ~ Pred_Log_Tanzania,
      country == "Uganda"   ~ Pred_Log_Uganda
      
     
      
    )
    
  ) %>%
  dplyr::select(dplyr::all_of(c(
    "country",
    "lon_modified",
    "lat_modified",
    "year",
    "harvest_end_month",
    "True_Yield",
    "True_Log_Yield",
    "pred_log_yield"
    
  )))

# --- 2) Initialize Earth Engine and load GAUL Admin2 boundaries ---
ee_Initialize()

admin_fc <- ee$FeatureCollection("FAO/GAUL/2015/level2")
countries <- unique(yields_pred$country)

# fetch per-country admin2 polygons to avoid timeout

gaul_name_map <- c(
  "Tanzania" = "United Republic of Tanzania"
  # others only if they ever differ
)

get_admin_sf <- function(ctry) {
  gaul_name <- if (ctry %in% names(gaul_name_map)) gaul_name_map[[ctry]] else ctry
  fc <- admin_fc$filter(ee$Filter$eq("ADM0_NAME", gaul_name))
  ee_as_sf(fc)
}

admin_sf <- map_dfr(countries, get_admin_sf) %>%
  st_make_valid() %>%
  st_transform(4326)

# --- 4) Spatial join: assign each yield point to Admin2 polygon ---
yields_admin <- st_join(yields_sf, admin_sf, join = st_within)




# Build a Calendar Conflict Panel

conf_monthly <- joined %>%
  st_drop_geometry() %>%
  mutate(
    event_date  = as.Date(event_date),
    event_month = floor_date(event_date, "month")  # like 2011-10-01, 2011-11-01
  ) %>%
  group_by(ADM0_NAME, ADM2_NAME, event_month) %>%
  summarize(
    conflict_events = n(),
    .groups = "drop"
  )

#Compute Conflict After 3, 6 and 12 Months


conf_monthly <- conf_monthly %>%
  arrange(ADM0_NAME, ADM2_NAME, event_month) %>%
  group_by(ADM0_NAME, ADM2_NAME) %>%
  mutate(
    #  3-month window
    conflict_3mo_forward = slide_dbl(
      conflict_events,
      sum,
      .before = 0,
      .after  = 2,
      .complete = FALSE
    ),
    
    # 6-month window
    conflict_6mo_forward = slide_dbl(
      conflict_events,
      sum,
      .before = 0,
      .after  = 5,
      .complete = FALSE
    ),
    
    #  12-month window
    conflict_12mo_forward = slide_dbl(
      conflict_events,
      sum,
      .before = 0,
      .after  = 11,
      .complete = FALSE
    )
  ) %>%
  ungroup()

# Join with yields 

yields_conf <- yields_admin %>%
  left_join(
    conf_monthly,
    by = c(
      "ADM0_NAME",
      "ADM2_NAME",
      "harvest_end_month" = "event_month"
    )
  ) %>%
  mutate(
    conflict_events       = if_else(is.na(conflict_events),       0, conflict_events),
    conflict_3mo_forward  = if_else(is.na(conflict_3mo_forward),  0, conflict_3mo_forward),
    conflict_6mo_forward  = if_else(is.na(conflict_6mo_forward),  0, conflict_6mo_forward),
    conflict_12mo_forward = if_else(is.na(conflict_12mo_forward), 0, conflict_12mo_forward)
  )




# -----------------------------
# 8. Aggregate conflict counts per Admin X Year
# -----------------------------
reg_data <- yields_conf %>%
  mutate(
    pred_yield = expm1(pred_log_yield)
  ) %>%
  group_by(ADM0_NAME, ADM2_NAME, year) %>%
  summarize(
    mean_pred_yield       = mean(pred_yield, na.rm = FALSE),
    mean_true_logyield       = mean(True_Log_Yield, na.rm = FALSE),
    mean_true_yield       = mean(True_Yield, na.rm = FALSE),
    conflict_3mo  = sum(conflict_3mo_forward,  na.rm = FALSE),
    conflict_6mo  = sum(conflict_6mo_forward,  na.rm = FALSE),
    conflict_12mo = sum(conflict_12mo_forward, na.rm = FALSE),
    n_plots = n(),
    .groups = "drop"
  )

#Create log-versions of yield and conflict

reg_data <- reg_data %>%
  mutate(
    log_predyield          = log(mean_pred_yield + 1),          # in case of small values
    log_trueyield          = log(mean_true_yield + 1),
    log_conflict_3mo   = log1p(conflict_3mo),
    log_conflict_6mo   = log1p(conflict_6mo),
    log_conflict_12mo  = log1p(conflict_12mo)
  )











