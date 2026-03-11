library(dplyr)
library(readr)
library(sf)
library(purrr)
library(geojsonio)
library(rgee)

# ---- Load Yields data----
yields <- read_csv("all_data_with_predictions.csv")  # 

# create year' col, and construct LSMS wave-year mapping
if (!"year" %in% names(yields)) {
  wave_year_map <- list(
    Ethiopia = c(`1` = 2011, `2` = 2013, `3` = 2015, `4` = 2018, `5` = 2021),
    Malawi   = c(`2` = 2010, `3` = 2013, `4` = 2016),
    Mali     = c(`1` = 2014, `2` = 2017),
    Nigeria  = c(`3` = 2015, `4` = 2018),
    Tanzania = c(`2` = 2010, `3` = 2012),
    Uganda   = c(`2` = 2011, `3` = 2013)
  )
  
  yields <- yields %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      year = {
        m <- wave_year_map[[country]]
        if (is.null(m)) NA_integer_ else {
          y <- m[[as.character(wave)]]
          if (is.null(y)) NA_integer_ else y
        }
      }
    ) %>%
    dplyr::ungroup()
}


# --- 1) Build predicted yield columns (country-specific) ---
yields_pred <- yields %>%
  mutate(
    Pred_yield_log = dplyr::case_when(
      country == "Ethiopia" ~ pred_Ethiopia_log,
      country == "Malawi"   ~ pred_Malawi_log,
      country == "Mali"     ~ pred_Mali_log,
      country == "Nigeria"  ~ pred_Nigeria_log,
      country == "Tanzania" ~ pred_Tanzania_log,
      country == "Uganda"   ~ pred_Uganda_log,
      TRUE ~ pred_global_log   # fallback
    ),
    Pred_yield = exp(Pred_yield_log)   # normal mean yield
  ) %>%
  select(country, wave, year, season,
         lon_modified, lat_modified,mean_yield, log_yield, Pred_yield, Pred_yield_log)


# --- 2) Initialize Earth Engine and load GAUL Admin2 boundaries ---
ee_Initialize()

admin_fc <- ee$FeatureCollection("FAO/GAUL/2015/level2")
countries <- unique(yields_pred$country)

# fetch per-country admin2 polygons to avoid timeout
get_admin_sf <- function(ctry) {
  fc <- admin_fc$filter(ee$Filter$eq("ADM0_NAME", ctry))
  ee_as_sf(fc)
}

admin_sf <- map_dfr(countries, get_admin_sf) %>%
  st_make_valid() %>%
  st_transform(4326)

# --- 3) Convert yields to sf ---
yields_sf <- st_as_sf(
  yields_pred,
  coords = c("lon_modified", "lat_modified"),
  crs = 4326
)

# --- 4) Spatial join: assign each yield point to Admin2 polygon ---
yields_admin <- st_join(yields_sf, admin_sf, join = st_within)

# --- 5) Aggregate to Admin2 × year ---
yield_summary <- yields_admin %>%
  st_drop_geometry() %>%
  group_by(
    country = ADM0_NAME,
    ADM1_NAME,
    ADM2_NAME,
    year
  ) %>%
  summarise(
    mean_yield_pred = mean(Pred_yield, na.rm = TRUE),
    mean_log_yield_pred = mean(Pred_yield_log, na.rm = TRUE),
    mean_yield        = mean(mean_yield, na.rm = TRUE),
    mean_log_yield  = mean(log_yield,  na.rm = TRUE),
    n_points = n(),
    .groups = "drop"
  )

#Preparing Conflict data
conflicts <- read_csv("conflicts_admin_year_type.csv")

# Harmonize column names
conflicts <- conflicts %>%
  rename(
    country   = ADM0_NAME,   # <— GAUL country name → "country"
    ADM1_NAME = ADM1_NAME,
    ADM2_NAME = ADM2_NAME
  )

#Build total conflicts per ADMIN 2
conf_tot <- conflicts %>%
  group_by(country, ADM1_NAME, ADM2_NAME, year) %>%
  summarise(
    conflict_count = sum(conflict_count, na.rm = TRUE),
    .groups = "drop"
  )


#Build wide-by-type conflict variables
conf_wide <- conflicts %>%
  group_by(country, ADM1_NAME, ADM2_NAME, year, event_type) %>%
  summarise(n = sum(conflict_count, na.rm = TRUE), .groups = "drop") %>%
  tidyr::pivot_wider(
    names_from  = event_type,
    values_from = n,
    values_fill = 0,            # if a type is missing in a cell, treat as 0
    names_prefix = "conf_"
  )

#Merge with total conflict

merged_tot <- conf_tot %>%
  left_join(
    yield_summary,
    by = c("country","ADM1_NAME","ADM2_NAME","year")
  ) %>%
  mutate(
    conflict_count = ifelse(is.na(conflict_count), 0, conflict_count)
  )

#Merge with wide-by-type conflict variables
merged_full <- merged_tot %>%
  left_join(conf_wide, by = c("country","ADM1_NAME","ADM2_NAME","year"))

reg_data <- merged_full %>% filter(!is.na(mean_log_yield))

reg_data %>%
  count(country)

reg_data %>%
  count(ADM2_NAME)

reg_data %>%
  count(year)

write_csv(reg_data, "Conflict_Yields.csv")



