suppressPackageStartupMessages({
  library(dplyr); library(readr); library(stringi)
})

pred  <- read_csv("Data/predictions_v2.csv", show_col_types=FALSE)
panel <- read_csv("Data/conflict_yields_panel.csv", show_col_types=FALSE)

iso_map <- c(ETH="Ethiopia", MWI="Malawi", MLI="Mali",
             NGA="Nigeria", TZA="United Republic of Tanzania", UGA="Uganda")
pred$country_name <- iso_map[pred$GID_0]

# ── Normalisation helper ──────────────────────────────────────────────────
normalise <- function(x) {
  x <- stri_trans_general(x, "Latin-ASCII")   # strip diacritics
  x <- tolower(x)
  x <- gsub("[[:punct:]]", " ", x)
  x <- gsub("\\s+", " ", trimws(x))
  x
}

strip_suffixes <- function(x) {
  x <- gsub("\\b(boma|city|town|urban|municipal|municipality|township authority|rural)\\b", "", x)
  trimws(gsub("\\s+", " ", x))
}

# ── Build prediction lookup ───────────────────────────────────────────────
pred_norm <- pred %>%
  distinct(country_name, GID_2, NAME_2) %>%
  mutate(name_norm  = normalise(NAME_2),
         name_strip = strip_suffixes(normalise(NAME_2)))

# ── Panel normalised names ────────────────────────────────────────────────
panel_norm <- panel %>%
  distinct(ADM0_NAME, ADM2_NAME) %>%
  mutate(name_norm  = normalise(ADM2_NAME),
         name_strip = strip_suffixes(normalise(ADM2_NAME)))

# ── Pass 1: normalised exact match ────────────────────────────────────────
p1 <- panel_norm %>%
  left_join(pred_norm %>% select(country_name, GID_2, name_norm),
            by=c("ADM0_NAME"="country_name", "name_norm"),
            relationship="many-to-many") %>%
  group_by(ADM0_NAME, ADM2_NAME) %>%
  summarise(GID_2=first(GID_2), .groups="drop")

cat("After normalised match:", sum(!is.na(p1$GID_2)), "/", nrow(p1), "districts\n")

# ── Pass 2: suffix-stripped match ─────────────────────────────────────────
unmatched <- p1 %>%
  filter(is.na(GID_2)) %>%
  select(ADM0_NAME, ADM2_NAME) %>%
  mutate(name_strip = strip_suffixes(normalise(ADM2_NAME)))

p2 <- unmatched %>%
  left_join(pred_norm %>% select(country_name, GID_2, name_strip),
            by=c("ADM0_NAME"="country_name","name_strip"),
            relationship="many-to-many") %>%
  group_by(ADM0_NAME, ADM2_NAME) %>%
  summarise(GID_2 = dplyr::first(na.omit(GID_2)), .groups="drop")

cat("Suffix-strip adds:", sum(!is.na(p2$GID_2)), "more\n")

# ── Combine crosswalk ─────────────────────────────────────────────────────
crosswalk <- bind_rows(
  p1 %>% filter(!is.na(GID_2)),
  p2 %>% filter(!is.na(GID_2))
) %>% distinct(ADM0_NAME, ADM2_NAME, GID_2)

cat("Total matched:", nrow(crosswalk), "/", nrow(panel_norm), "districts\n")

still_out <- panel_norm %>%
  anti_join(crosswalk, by=c("ADM0_NAME","ADM2_NAME")) %>%
  select(ADM0_NAME, ADM2_NAME) %>%
  arrange(ADM0_NAME)

cat("Still unmatched:", nrow(still_out), "\n")
print(still_out, n=60)
