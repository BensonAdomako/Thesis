# build_panel_v3.R
# Builds the conflict-yields panel from scratch using GADM boundaries,
# GROW-Africa harvest months, and v2 yield predictions.
#
# Key improvements over previous panel:
#   - GADM admin-2 spine (from predictions_v2.csv) — no LSMS survey point bias
#   - GROW-Africa harvest months — replaces LSMS-derived harvest months
#   - Full monthly conflict grid (zeros filled) — fixes sparse slider bug
#   - 7 countries: BFA ETH MLI MWI NER NGA TZA (Uganda excluded — not in training)
#
# Inputs:
#   Data/acled_africa.csv       — ACLED full Africa 2010-2024
#   Data/predictions_v2.csv     — v2 yield predictions (GADM admin-2 x year)
#   Data/grow_maize_yields.csv  — GROW-Africa harvest months
#   Data/adm2_pop_area.csv      — WorldPop population (GAUL admin-2 x year)
#
# Output:
#   Data/conflict_yields_panel_v3.csv

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tidyr)
  library(sf); library(geodata); library(terra)
  library(slider); library(lubridate); library(stringi)
})

THESIS_ISO3  <- c("BFA", "ETH", "MLI", "MWI", "NER", "NGA", "TZA")
YEAR_MIN     <- 2010L
YEAR_MAX     <- 2024L
GADM_DIR     <- "Data/gadm"
dir.create(GADM_DIR, showWarnings = FALSE)

# ── 1. GROW-Africa harvest months ─────────────────────────────────────────────
message("Step 1: Extracting harvest months from GROW-Africa...")

grow <- read_csv("Data/grow_maize_yields.csv", show_col_types = FALSE) %>%
  filter(GID_0 %in% THESIS_ISO3,
         !is.na(harvest_month)) %>%
  mutate(harvest_month = as.integer(harvest_month),
         areaHa        = as.numeric(areaHa),
         areaHa        = replace_na(areaHa, 1))

# For ETH and BFA: admin-2 level data available
# For others: admin-1 level — assign to all admin-2 within that admin-1
# Strategy: weighted mode by area

weighted_mode <- function(months, weights) {
  as.integer(names(which.max(tapply(weights, months, sum))))
}

# Admin-2 direct (ETH, BFA)
harvest_adm2 <- grow %>%
  filter(level == 2, !is.na(GADM_GID_2)) %>%
  group_by(GID_0, GID_2 = GADM_GID_2) %>%
  summarise(harvest_month = weighted_mode(harvest_month, areaHa),
            .groups = "drop")

# Admin-1 (others) — will join to GID_2 via predictions spine
harvest_adm1 <- grow %>%
  filter(level == 1, !is.na(GADM_GID_1)) %>%
  group_by(GID_0, GID_1 = GADM_GID_1) %>%
  summarise(harvest_month = weighted_mode(harvest_month, areaHa),
            .groups = "drop")

# National fallback (MWI, NER if no GID_1 match)
harvest_national <- grow %>%
  group_by(GID_0) %>%
  summarise(harvest_month = weighted_mode(harvest_month, areaHa),
            .groups = "drop")

message("  Harvest months by country:")
print(harvest_national)

# Load prediction spine to get GID_1 for each GID_2
pred <- read_csv("Data/predictions_v2.csv", show_col_types = FALSE)

# Assign harvest month: admin-2 > admin-1 > national
harvest_months <- pred %>%
  distinct(GID_0, GID_1, GID_2) %>%
  left_join(harvest_adm2, by = c("GID_0", "GID_2")) %>%
  rename(hm_adm2 = harvest_month) %>%
  left_join(harvest_adm1, by = c("GID_0", "GID_1")) %>%
  rename(hm_adm1 = harvest_month) %>%
  left_join(harvest_national, by = "GID_0") %>%
  rename(hm_national = harvest_month) %>%
  mutate(harvest_month = coalesce(hm_adm2, hm_adm1, hm_national)) %>%
  select(GID_0, GID_1, GID_2, harvest_month)

message("  Coverage:")
print(harvest_months %>% group_by(GID_0) %>%
      summarise(districts = n(), na_harvest = sum(is.na(harvest_month))))

# ── 2. Download GADM admin-2 shapefiles ──────────────────────────────────────
message("\nStep 2: Loading GADM admin-2 shapefiles...")

# ISO2 codes needed by geodata::gadm()
iso2_map <- c(BFA="BFA", ETH="ETH", MLI="MLI", MWI="MWI",
              NER="NER", NGA="NGA", TZA="TZA")

gadm_list <- lapply(THESIS_ISO3, function(iso) {
  fpath <- file.path(GADM_DIR, paste0("gadm41_", iso, "_2_pk.rds"))
  if (file.exists(fpath)) {
    message("  ", iso, ": loading cached")
    v <- readRDS(fpath)
  } else {
    message("  ", iso, ": downloading...")
    v <- geodata::gadm(iso2_map[iso], level = 2, path = GADM_DIR)
  }
  sf::st_as_sf(v) %>%
    select(GID_0, GID_1, GID_2, NAME_1, NAME_2, geometry)
})

gadm_sf <- bind_rows(gadm_list)
message("  Total GADM admin-2 units: ", nrow(gadm_sf))

# ── 3. Spatially join ACLED to GADM admin-2 ───────────────────────────────────
message("\nStep 3: Spatially joining ACLED events to GADM admin-2...")

# ACLED country names for our 7 countries
acled_countries <- c("Burkina Faso", "Ethiopia", "Mali", "Malawi",
                     "Niger", "Nigeria", "Tanzania")

acled_raw <- read_csv("Data/acled_africa.csv", show_col_types = FALSE) %>%
  filter(country %in% acled_countries,
         year >= YEAR_MIN, year <= YEAR_MAX,
         !is.na(latitude), !is.na(longitude))

message("  ACLED events after country/year filter: ", nrow(acled_raw))

acled_sf <- st_as_sf(acled_raw,
                     coords = c("longitude", "latitude"),
                     crs = 4326, remove = FALSE)

# Spatial join: tag each event with GID_2
message("  Running spatial join (this may take a minute)...")
acled_joined <- st_join(acled_sf,
                        gadm_sf %>% select(GID_0, GID_2),
                        join = st_within) %>%
  st_drop_geometry()

matched <- sum(!is.na(acled_joined$GID_2))
message("  Matched ", matched, " / ", nrow(acled_joined), " events (",
        round(100 * matched / nrow(acled_joined), 1), "%)")

# ── 4. Full monthly conflict grid + forward windows ───────────────────────────
message("\nStep 4: Building full monthly conflict grid...")

# Aggregate events to GID_2 x month
acled_monthly <- acled_joined %>%
  filter(!is.na(GID_2)) %>%
  mutate(event_month = floor_date(as.Date(event_date), "month")) %>%
  group_by(GID_2, event_month) %>%
  summarise(events     = n(),
            fatalities = sum(fatalities, na.rm = TRUE),
            .groups    = "drop")

# Full calendar grid: every GID_2 x every month 2010-01 to 2024-12
all_months <- seq(as.Date(paste0(YEAR_MIN, "-01-01")),
                  as.Date(paste0(YEAR_MAX, "-12-01")),
                  by = "month")
all_gid2   <- unique(acled_joined$GID_2[!is.na(acled_joined$GID_2)])

message("  Building ", length(all_gid2), " x ", length(all_months),
        " = ", length(all_gid2) * length(all_months), " cell grid...")

full_grid <- expand_grid(GID_2 = all_gid2, event_month = all_months) %>%
  left_join(acled_monthly, by = c("GID_2", "event_month")) %>%
  mutate(events     = replace_na(events, 0L),
         fatalities = replace_na(fatalities, 0L))

# Compute forward windows via slider over full calendar sequence
message("  Computing forward conflict windows...")
conflict_monthly <- full_grid %>%
  arrange(GID_2, event_month) %>%
  group_by(GID_2) %>%
  mutate(
    conflict_3mo  = slide_dbl(events, sum, .before = 0, .after = 2,  .complete = FALSE),
    conflict_6mo  = slide_dbl(events, sum, .before = 0, .after = 5,  .complete = FALSE),
    conflict_12mo = slide_dbl(events, sum, .before = 0, .after = 11, .complete = FALSE),
    fatalities_3mo  = slide_dbl(fatalities, sum, .before = 0, .after = 2,  .complete = FALSE),
    fatalities_12mo = slide_dbl(fatalities, sum, .before = 0, .after = 11, .complete = FALSE)
  ) %>%
  ungroup()

# ── 5. Build panel spine + assign harvest months ──────────────────────────────
message("\nStep 5: Building panel spine...")

# Drop non-administrative GADM units (lakes, water bodies)
lake_names <- c("Lake Manyara", "Lake Victoria", "Lake Tanganyika",
                "Lake Rukwa", "Lake Malawi")
pred <- pred %>% filter(!NAME_2 %in% lake_names)
message("  Dropped lake/water-body units: ",
        sum(pred$NAME_2 %in% lake_names), " rows")

panel_spine <- pred %>%
  select(GID_0, GID_1, GID_2, NAME_1, NAME_2, year,
         log_pred_yield_v2 = log_predicted_yield_abs,
         pred_yield_kgha   = predicted_yield_abs_kgha,
         pred_yield_anomaly = predicted_yield_anomaly) %>%
  filter(year >= YEAR_MIN, year <= YEAR_MAX) %>%
  left_join(harvest_months %>% select(GID_2, harvest_month), by = "GID_2")

message("  Spine rows: ", nrow(panel_spine))
message("  NA harvest_month: ", sum(is.na(panel_spine$harvest_month)))

# ── 6. Join conflict windows to panel ─────────────────────────────────────────
message("\nStep 6: Merging conflict windows into panel...")

# For each district-year, harvest date = year-harvest_month-01
# Join to conflict_monthly on that date
panel_spine <- panel_spine %>%
  mutate(harvest_date = as.Date(paste(year, harvest_month, "01", sep = "-")))

panel_conflict <- panel_spine %>%
  left_join(conflict_monthly %>%
              select(GID_2, event_month, conflict_3mo, conflict_6mo,
                     conflict_12mo, fatalities_3mo, fatalities_12mo),
            by = c("GID_2" = "GID_2", "harvest_date" = "event_month")) %>%
  mutate(across(c(conflict_3mo, conflict_6mo, conflict_12mo,
                  fatalities_3mo, fatalities_12mo),
                ~ replace_na(.x, 0L)))

message("  Panel rows after conflict merge: ", nrow(panel_conflict))
message("  Districts with any conflict (3mo): ",
        sum(panel_conflict$conflict_3mo > 0), " / ", nrow(panel_conflict))

# ── 7. Population: GAUL name match to GADM ───────────────────────────────────
message("\nStep 7: Merging population controls...")

normalise <- function(x) {
  x <- stri_trans_general(x, "Latin-ASCII")
  x <- tolower(trimws(x))
  x <- gsub("[[:punct:]]", " ", x)
  trimws(gsub("\\s+", " ", x))
}
strip_sfx <- function(x) trimws(gsub("\\s+", " ",
  gsub("\\b(boma|city|town|urban|municipal|municipality|township authority|rural)\\b", "", x)))

iso_to_fullname <- c(
  BFA = "Burkina Faso", ETH = "Ethiopia",   MLI = "Mali",
  MWI = "Malawi",       NER = "Niger",       NGA = "Nigeria",
  TZA = "United Republic of Tanzania"
)

# Manual crosswalk: GAUL ADM2_NAME → GADM NAME_2 (ETH Amharic translations + others)
eth_xwalk <- tribble(
  ~gaul_name,          ~gadm_name,
  # Amharic directional translations
  "Awi/Agew",          "Agew Awi",
  "East Gojam",        "Misraq Gojjam",
  "West Gojam",        "Mirab Gojjam",
  "North Gonder",      "Semen Gondar",
  "South Gonder",      "Debub Gondar",
  "North Wollo",       "Semen Wello",
  "South Wollo",       "Debub Wollo",
  "East Harerge",      "Misraq Harerge",
  "West Harerge",      "Mirab Hararghe",
  "East Shewa",        "Misraq Shewa",
  "West Shewa",        "Mirab Shewa",
  "South West Shewa",  "Debub Mirab Shewa",
  "East Wellega",      "Misraq Wellega",
  "West Wellega",      "Mirab Welega",
  "West Arsi",         "Mirab Arsi",
  "North Shewa(R3)",   "North Shewa",
  "Gedio",             "Gedeo",
  "Selti",             "Silti",
  "South Omo",         "Debub Omo",
  "KT",                "Kembata Tembaro",
  # Tigray zones
  "Central",           "Mehakelegnaw",
  "Eastern",           "Misraqawi",
  "North Western",     "Semien Mi'irabaw",
  "Southern",          "Debubawi",
  "Western",           "Mi'irabawi",
  # Afar zones
  "Zone 1",            "Afar Zone 1",
  "Zone 2",            "Afar Zone 2",
  "Zone 3",            "Afar Zone 3",
  "Zone 4",            "Afar Zone 4",
  "Zone 5",            "Afar Zone 5",
  # Addis Ababa
  "Region 14",         "Addis Abeba",
  # Segen People's Zone — split into sub-zones in GADM
  "Segen Peoples'",    "Alle",
  "Segen Peoples'",    "Amaro",
  "Segen Peoples'",    "Burji",
  "Segen Peoples'",    "Derashe",
  "Segen Peoples'",    "Konso",
  # Special Woreda
  "Special Woreda",    "Argoba"
) %>% mutate(gaul_norm = normalise(gaul_name),
             gadm_norm = normalise(gadm_name))

# NER manual crosswalk: GADM NAME_2 → GAUL ADM2_NAME
ner_xwalk <- tribble(
  ~gadm_name,       ~gaul_name,
  "Bkonni",         "Birni N'konni",
  "Tchin-Tabarade", "Tchin Tabaradene",
  "Tillabéry",      "Tillaberi",
  "Matameye",       "Matamey",
  "Mirriah",        "Miria"
) %>% mutate(gadm_norm = normalise(gadm_name),
             gaul_norm = normalise(gaul_name))

# NGA manual crosswalk: GADM NAME_2 → GAUL ADM2_NAME
nga_xwalk <- tribble(
  ~gadm_name,           ~gaul_name,
  "Yenagoa",            "Yenegoa",
  "Makurdi",            "Markurdi",
  "Otukpo",             "Oturkpo",
  "Vande Ikya",         "Vandeikya",
  "Obi Ngwa",           "Oboma Ngwa",
  "Emure/Ise/Orun",     "Emure",
  "Yewa North",         "Egbado /Yewa North",
  "Yewa South",         "Egbado /Yewa South",
  "Kiri Kasama",        "Kiri Kasamma"
) %>% mutate(gadm_norm = normalise(gadm_name),
             gaul_norm = normalise(gaul_name))

pop_raw_src <- read_csv("Data/admn2_pop.csv", show_col_types = FALSE) %>%
  rename(population = sum) %>%
  filter(ADM0_NAME %in% iso_to_fullname) %>%
  select(ADM0_NAME, ADM1_NAME, ADM2_NAME, year, population)

# NER: aggregate Niamey Commune 1/2/3 into a single "Niamey" entry
ner_niamey <- pop_raw_src %>%
  filter(ADM0_NAME == "Niger", ADM2_NAME %in% c("Commune 1","Commune 2","Commune 3")) %>%
  group_by(ADM0_NAME, ADM1_NAME, year) %>%
  summarise(population = sum(population, na.rm = TRUE), .groups = "drop") %>%
  mutate(ADM2_NAME = "Niamey")

pop_raw <- bind_rows(
  pop_raw_src %>% filter(!(ADM0_NAME == "Niger" &
                            ADM2_NAME %in% c("Commune 1","Commune 2","Commune 3"))),
  ner_niamey
) %>%
  mutate(name_norm  = normalise(ADM2_NAME),
         name_strip = strip_sfx(normalise(ADM2_NAME)))

# Carry forward to YEAR_MAX if needed
max_pop_year <- max(pop_raw$year)
if (max_pop_year < YEAR_MAX) {
  pop_extrap <- pop_raw %>% filter(year == max_pop_year) %>% select(-year)
  pop_raw <- bind_rows(pop_raw,
    purrr::map_dfr(seq(max_pop_year + 1, YEAR_MAX), ~ pop_extrap %>% mutate(year = .x)))
  message("  Population extrapolated from ", max_pop_year, " to ", YEAR_MAX)
}

# ── MWI special case: GADM level-2 = TAs; population at GAUL level = GADM NAME_1 ──
# Match: GAUL ADM2_NAME ↔ GADM NAME_1 (district), then assign to all TAs within district
mwi_districts <- panel_conflict %>%
  filter(GID_0 == "MWI") %>%
  distinct(GID_2, GID_1, NAME_1) %>%
  mutate(name_norm = normalise(NAME_1))

mwi_pop <- pop_raw %>%
  filter(ADM0_NAME == "Malawi") %>%
  select(name_norm, year, population)

mwi_pop_matched <- mwi_districts %>%
  left_join(mwi_pop, by = "name_norm") %>%
  select(GID_2, year, population)

message("  MWI: matched ",
        n_distinct(mwi_pop_matched$GID_2[!is.na(mwi_pop_matched$population)]),
        " / ", n_distinct(mwi_districts$GID_2), " TAs via parent district")

# ── ETH: normalised match + manual crosswalk for Amharic translations ──────────
eth_gadm <- panel_conflict %>%
  filter(GID_0 == "ETH") %>%
  distinct(GID_2, NAME_2) %>%
  mutate(name_norm = normalise(NAME_2))

eth_pop <- pop_raw %>% filter(ADM0_NAME == "Ethiopia")

# Pass 1: direct normalised match
eth_p1 <- eth_gadm %>%
  left_join(eth_pop %>% select(name_norm, year, population),
            by = "name_norm")

# Pass 2: manual crosswalk for unmatched
eth_unmatched <- eth_p1 %>% filter(is.na(population)) %>%
  distinct(GID_2, NAME_2, name_norm)

eth_p2 <- eth_unmatched %>%
  left_join(eth_xwalk %>% select(gadm_norm, gaul_norm), by = c("name_norm" = "gadm_norm")) %>%
  left_join(eth_pop %>% select(name_norm, year, population),
            by = c("gaul_norm" = "name_norm")) %>%
  select(GID_2, year, population)

eth_pop_matched <- bind_rows(
  eth_p1 %>% filter(!is.na(population)) %>% select(GID_2, year, population),
  eth_p2 %>% filter(!is.na(population))
) %>% distinct(GID_2, year, .keep_all = TRUE)

message("  ETH: matched ",
        n_distinct(eth_pop_matched$GID_2[!is.na(eth_pop_matched$population)]),
        " / ", n_distinct(eth_gadm$GID_2), " districts")

# ── All other countries: normalised name match + NGA manual crosswalk ──────────
other_gadm <- panel_conflict %>%
  filter(!GID_0 %in% c("MWI", "ETH")) %>%
  distinct(GID_0, GID_2, NAME_2) %>%
  mutate(country_full = iso_to_fullname[GID_0],
         name_norm    = normalise(NAME_2),
         name_strip   = strip_sfx(normalise(NAME_2)))

other_pop <- pop_raw %>%
  filter(!ADM0_NAME %in% c("Malawi", "Ethiopia")) %>%
  rename(country_full = ADM0_NAME)

# Pass 1: normalised exact
other_p1 <- other_gadm %>%
  left_join(other_pop %>% select(country_full, name_norm, year, population),
            by = c("country_full", "name_norm"),
            relationship = "many-to-many") %>%
  group_by(GID_0, GID_2, year) %>%
  summarise(population = first(na.omit(population)), .groups = "drop")

# Pass 2: suffix-stripped for unmatched
other_unmatched_ids <- other_p1 %>%
  filter(is.na(population)) %>%
  distinct(GID_0, GID_2) %>%
  left_join(other_gadm %>% select(GID_0, GID_2, country_full, name_strip),
            by = c("GID_0","GID_2"))

other_p2 <- other_unmatched_ids %>%
  left_join(other_pop %>% select(country_full, name_strip, year, population),
            by = c("country_full", "name_strip"),
            relationship = "many-to-many") %>%
  group_by(GID_0, GID_2, year) %>%
  summarise(population = first(na.omit(population)), .groups = "drop")

# Pass 3a: NGA manual crosswalk for known typos/renames
nga_unmatched <- bind_rows(other_p1, other_p2) %>%
  filter(is.na(population), GID_0 == "NGA") %>%
  distinct(GID_2) %>%
  left_join(other_gadm %>% filter(GID_0 == "NGA") %>%
              select(GID_2, name_norm), by = "GID_2") %>%
  left_join(nga_xwalk %>% select(gadm_norm, gaul_norm),
            by = c("name_norm" = "gadm_norm")) %>%
  left_join(other_pop %>% filter(country_full == "Nigeria") %>%
              select(gaul_norm = name_norm, year, population),
            by = "gaul_norm") %>%
  group_by(GID_2, year) %>%
  summarise(population = first(na.omit(population)), .groups = "drop") %>%
  mutate(GID_0 = "NGA")

# Pass 3b: NER manual crosswalk for spelling variants
ner_unmatched <- bind_rows(other_p1, other_p2) %>%
  filter(is.na(population), GID_0 == "NER") %>%
  distinct(GID_2) %>%
  left_join(other_gadm %>% filter(GID_0 == "NER") %>%
              select(GID_2, name_norm), by = "GID_2") %>%
  left_join(ner_xwalk %>% select(gadm_norm, gaul_norm),
            by = c("name_norm" = "gadm_norm")) %>%
  left_join(other_pop %>% filter(country_full == "Niger") %>%
              select(gaul_norm = name_norm, year, population),
            by = "gaul_norm") %>%
  group_by(GID_2, year) %>%
  summarise(population = first(na.omit(population)), .groups = "drop") %>%
  mutate(GID_0 = "NER")

other_pop_matched <- bind_rows(
  other_p1 %>% filter(!is.na(population)),
  other_p2 %>% filter(!is.na(population)),
  nga_unmatched %>% filter(!is.na(population)),
  ner_unmatched %>% filter(!is.na(population))
) %>% distinct(GID_0, GID_2, year, .keep_all = TRUE)

message("  Other countries matched: ",
        n_distinct(other_pop_matched$GID_2[!is.na(other_pop_matched$population)]),
        " / ", n_distinct(other_gadm$GID_2), " districts")

# ── Combine all population ─────────────────────────────────────────────────────
pop_matched <- bind_rows(
  mwi_pop_matched %>% mutate(GID_0 = "MWI"),
  eth_pop_matched %>% mutate(GID_0 = "ETH"),
  other_pop_matched
) %>% distinct(GID_0, GID_2, year, .keep_all = TRUE)

matched_pop <- n_distinct(pop_matched$GID_2[!is.na(pop_matched$population)])
message("  Total population matched: ", matched_pop, " / ",
        n_distinct(panel_conflict$GID_2), " districts")

panel_v3 <- panel_conflict %>%
  left_join(pop_matched, by = c("GID_0", "GID_2", "year")) %>%
  mutate(log_pop          = log(population),
         log_conflict_3mo = log1p(conflict_3mo))

# ── 8. Sanity checks + save ───────────────────────────────────────────────────
message("\n=== Sanity Checks ===")
message("Final panel rows : ", nrow(panel_v3))
message("Districts        : ", n_distinct(panel_v3$GID_2))
message("Years            : ", min(panel_v3$year), "-", max(panel_v3$year))
message("NA log_pred_yield: ", sum(is.na(panel_v3$log_pred_yield_v2)))
message("NA log_pop       : ", sum(is.na(panel_v3$log_pop)))
message("Zero conflict 3mo: ", round(100 * mean(panel_v3$conflict_3mo == 0), 1), "%")

message("\nBy country:")
panel_v3 %>%
  group_by(GID_0) %>%
  summarise(
    districts      = n_distinct(GID_2),
    obs            = n(),
    harvest_month  = first(harvest_month),
    pct_zero_conf  = round(100 * mean(conflict_3mo == 0), 1),
    mean_yield     = round(mean(pred_yield_kgha, na.rm = TRUE)),
    pct_pop_match  = round(100 * mean(!is.na(population)), 1),
    .groups = "drop"
  ) %>%
  print()

write_csv(panel_v3, "Data/conflict_yields_panel_v3.csv")
message("\nSaved: Data/conflict_yields_panel_v3.csv")
