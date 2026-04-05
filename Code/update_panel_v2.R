# update_panel_v2.R
# Merges v2 XGBoost predictions into the conflict panel.
# Replaces log_pred_yield / mean_pred_yield with v2 predictions.
# Also adds predicted_yield_anomaly column for robustness checks.
#
# Inputs:
#   Data/predictions_v2.csv          — v2 predictions (GADM GID_2 x year)
#   Data/conflict_yields_panel.csv   — existing panel (GAUL admin-2 x year)
# Output:
#   Data/conflict_yields_panel_v2.csv

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(stringi)
})

pred  <- read_csv("Data/predictions_v2.csv",        show_col_types = FALSE)
panel <- read_csv("Data/conflict_yields_panel.csv",  show_col_types = FALSE)

iso_map <- c(ETH = "Ethiopia", MWI = "Malawi", MLI = "Mali",
             NGA = "Nigeria",  TZA = "United Republic of Tanzania",
             UGA = "Uganda")
pred$country_name <- iso_map[pred$GID_0]

# ── Normalisation helpers ─────────────────────────────────────────────────────
normalise <- function(x) {
  x <- stri_trans_general(x, "Latin-ASCII")
  x <- tolower(trimws(x))
  x <- gsub("[[:punct:]]", " ", x)
  gsub("\\s+", " ", x)
}
strip_sfx <- function(x) {
  x <- gsub("\\b(boma|city|town|urban|municipal|municipality|township authority|rural)\\b", "", x)
  trimws(gsub("\\s+", " ", x))
}

# ── Manual crosswalk: GAUL ADM2_NAME → GADM NAME_2 ───────────────────────────
# Ethiopia: Amharic directional prefixes
#   Misraq = East, Mirab = West, Semen = North, Debub = South
manual_xwalk <- tribble(
  ~ADM0_NAME,                        ~ADM2_NAME,          ~gadm_name,
  # Ethiopia – directional translations
  "Ethiopia",  "Zone 1",                    "Afar Zone 1",
  "Ethiopia",  "Zone 2",                    "Afar Zone 2",
  "Ethiopia",  "Zone 3",                    "Afar Zone 3",
  "Ethiopia",  "Awi/Agew",                  "Agew Awi",
  "Ethiopia",  "East Gojam",                "Misraq Gojjam",
  "Ethiopia",  "West Gojam",                "Mirab Gojjam",
  "Ethiopia",  "North Gonder",              "Semen Gondar",
  "Ethiopia",  "South Gonder",              "Debub Gondar",
  "Ethiopia",  "North Wollo",               "Semen Wello",
  "Ethiopia",  "South Wollo",               "Debub Wollo",
  "Ethiopia",  "East Harerge",              "Misraq Harerge",
  "Ethiopia",  "West Harerge",              "Mirab Hararghe",
  "Ethiopia",  "East Shewa",                "Misraq Shewa",
  "Ethiopia",  "West Shewa",                "Mirab Shewa",
  "Ethiopia",  "South West Shewa",          "Debub Mirab Shewa",
  "Ethiopia",  "East Wellega",              "Misraq Wellega",
  "Ethiopia",  "West Wellega",              "Mirab Welega",
  "Ethiopia",  "West Arsi",                 "Mirab Arsi",
  "Ethiopia",  "North Shewa(R3)",           "North Shewa",
  "Ethiopia",  "Gedio",                     "Gedeo",
  "Ethiopia",  "Selti",                     "Silti",
  "Ethiopia",  "South Omo",                 "Debub Omo",
  "Ethiopia",  "KT",                        "Kembata Tembaro",
  # Ethiopia – Tigray zones (direction in Tigrinya)
  "Ethiopia",  "Central",                   "Debubawi",
  "Ethiopia",  "Eastern",                   "Misraqawi",
  "Ethiopia",  "North Western",             "Semien Mi'irabaw",
  "Ethiopia",  "Southern",                  "Debubawi",
  "Ethiopia",  "Western",                   "Mi'irabawi",
  # Nigeria – typos and spacing
  "Nigeria",   "Yenegoa",                   "Yenagoa",
  "Nigeria",   "Markurdi",                  "Makurdi",
  "Nigeria",   "Oturkpo",                   "Otukpo",
  "Nigeria",   "Vandeikya",                 "Vande Ikya",
  "Nigeria",   "Oboma Ngwa",               "Obi Ngwa",
  "Nigeria",   "Emure",                     "Emure/Ise/Orun",
  "Nigeria",   "Igbo-eze North",            "Igbo-Etiti",
  "Nigeria",   "Umuahia  North",            "Umuahia North"
)

# Resolve manual crosswalk: look up GID_2 for each gadm_name
pred_lookup <- pred %>% distinct(country_name, NAME_2, GID_2)

manual_resolved <- manual_xwalk %>%
  left_join(pred_lookup,
            by = c("ADM0_NAME" = "country_name", "gadm_name" = "NAME_2")) %>%
  select(ADM0_NAME, ADM2_NAME, GID_2) %>%
  filter(!is.na(GID_2))

cat("Manual crosswalk resolved:", nrow(manual_resolved), "districts\n")

# ── Automated matching ────────────────────────────────────────────────────────
pred_norm <- pred %>%
  distinct(country_name, GID_2, NAME_2) %>%
  mutate(name_norm  = normalise(NAME_2),
         name_strip = strip_sfx(normalise(NAME_2)))

panel_ids <- panel %>%
  distinct(ADM0_NAME, ADM2_NAME) %>%
  mutate(name_norm  = normalise(ADM2_NAME),
         name_strip = strip_sfx(normalise(ADM2_NAME)))

# Pass 1: normalised exact
auto1 <- panel_ids %>%
  left_join(pred_norm %>% select(country_name, GID_2, name_norm),
            by = c("ADM0_NAME" = "country_name", "name_norm"),
            relationship = "many-to-many") %>%
  group_by(ADM0_NAME, ADM2_NAME) %>%
  summarise(GID_2 = dplyr::first(na.omit(GID_2)), .groups = "drop")

# Pass 2: suffix-stripped
unmatched2 <- auto1 %>% filter(is.na(GID_2)) %>%
  select(ADM0_NAME, ADM2_NAME) %>%
  mutate(name_strip = strip_sfx(normalise(ADM2_NAME)))

auto2 <- unmatched2 %>%
  left_join(pred_norm %>% select(country_name, GID_2, name_strip),
            by = c("ADM0_NAME" = "country_name", "name_strip"),
            relationship = "many-to-many") %>%
  group_by(ADM0_NAME, ADM2_NAME) %>%
  summarise(GID_2 = dplyr::first(na.omit(GID_2)), .groups = "drop")

auto_matched <- bind_rows(
  auto1 %>% filter(!is.na(GID_2)),
  auto2 %>% filter(!is.na(GID_2))
) %>% distinct(ADM0_NAME, ADM2_NAME, GID_2)

# ── Combine auto + manual ─────────────────────────────────────────────────────
crosswalk <- bind_rows(auto_matched, manual_resolved) %>%
  distinct(ADM0_NAME, ADM2_NAME, .keep_all = TRUE)  # auto takes priority

total_districts <- n_distinct(paste(panel$ADM0_NAME, panel$ADM2_NAME))
cat("Final crosswalk coverage:", nrow(crosswalk), "/", total_districts, "districts (",
    round(100 * nrow(crosswalk) / total_districts, 1), "%)\n")

unresolved <- panel %>%
  distinct(ADM0_NAME, ADM2_NAME) %>%
  anti_join(crosswalk, by = c("ADM0_NAME", "ADM2_NAME"))
cat("Unresolved districts:", nrow(unresolved),
    "(will have NA yield — dropped from regression)\n")

# ── Aggregate v2 predictions to GAUL admin-2 x year ──────────────────────────
# Multiple GADM sub-districts can map to one GAUL district → take mean
pred_agg <- crosswalk %>%
  left_join(pred %>% select(GID_2, year, predicted_yield_abs_kgha,
                            log_predicted_yield_abs, predicted_yield_anomaly),
            by = "GID_2", relationship = "many-to-many") %>%
  group_by(ADM0_NAME, ADM2_NAME, year) %>%
  summarise(
    pred_yield_v2_kgha  = mean(predicted_yield_abs_kgha, na.rm = TRUE),
    log_pred_yield_v2   = mean(log_predicted_yield_abs,  na.rm = TRUE),
    pred_yield_anomaly  = mean(predicted_yield_anomaly,  na.rm = TRUE),
    .groups = "drop"
  )

# ── Merge into panel ──────────────────────────────────────────────────────────
panel_v2 <- panel %>%
  left_join(pred_agg, by = c("ADM0_NAME", "ADM2_NAME", "year"))

# Sanity checks
cat("\n=== Sanity Checks ===\n")
cat("Panel rows:", nrow(panel_v2), "\n")
cat("NA log_pred_yield_v2:", sum(is.na(panel_v2$log_pred_yield_v2)),
    "(", round(100 * mean(is.na(panel_v2$log_pred_yield_v2)), 1), "%)\n")
cat("Mean v2 yield (kg/ha):",
    round(mean(panel_v2$pred_yield_v2_kgha, na.rm = TRUE), 0), "\n")
cat("Mean old yield (kg/ha):",
    round(mean(exp(panel$log_pred_yield), na.rm = TRUE), 0), "\n")

cat("\nBy country:\n")
panel_v2 %>%
  group_by(ADM0_NAME) %>%
  summarise(
    n_obs          = n(),
    pct_matched    = round(100 * mean(!is.na(log_pred_yield_v2)), 1),
    mean_v2_kgha   = round(mean(pred_yield_v2_kgha, na.rm = TRUE), 0),
    mean_old_kgha  = round(mean(exp(log_pred_yield), na.rm = TRUE), 0),
    .groups = "drop"
  ) %>%
  print()

# Save
write_csv(panel_v2, "Data/conflict_yields_panel_v2.csv")
cat("\nSaved: Data/conflict_yields_panel_v2.csv\n")
