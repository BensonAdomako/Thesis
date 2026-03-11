# 1. Load packages
# -----------------------------
library(rgee)
library(sf)
library(dplyr)
library(readr)
# -----------------------------
# 2. Initialize Earth Engine
# -----------------------------
ee_Initialize()

# -----------------------------
# 3. Specify countries
# -----------------------------
countries <- c("Nigeria", "Mali", "Tanzania", "Uganda", "Ethiopia", "Malawi")
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
acled <- read_csv("Data/Conflict.csv")
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

# -----------------------------
# 8. Aggregate conflict counts per Admin, Year, and TYPE
# -----------------------------
conflict_summary <- joined %>%
  st_drop_geometry() %>%
  filter(!is.na(ADM2_NAME)) %>%  # only keep matched records
  group_by(ADM0_NAME, ADM1_NAME, ADM2_NAME, year, event_type) %>%  # ADD event_type here
  summarise(
    conflict_count = n(),
    total_fatalities = sum(fatalities, na.rm = TRUE),
    .groups = "drop"
  )

message("Summary contains ", nrow(conflict_summary), " admin-year-type combinations")

# Show breakdown by conflict type
message("\nConflict events by type:")
print(conflict_summary %>% 
        group_by(event_type) %>% 
        summarise(total_events = sum(conflict_count), .groups = "drop"))

# -----------------------------
# 9. Save to CSV
# -----------------------------
write_csv(conflict_summary, "Data/conflicts_admin_year_type.csv")
message("✅ Saved: conflicts_admin_year_type.csv")

# Optional: Check the results
head(conflict_summary, 20)
                 