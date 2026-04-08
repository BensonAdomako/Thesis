# figures_gen.R
# Publication-quality figures for:
# "Crop Yields and Conflict in Sub-Saharan Africa: Satellite-Based Evidence"
#
# Generates 7 figures for 04_data.tex and 06_results.tex
# Run from project root: Rscript "Write up/figures_gen.R"
# -------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(sf)
  library(purrr)
  library(tidyr)
  library(scales)
  library(fixest)
})

# ---- Paths ---------------------------------------------------------------
panel_path  <- "Data/conflict_yields_panel_v3.csv"
gadm_dir    <- "Data/gadm/gadm"
output_dir  <- "Write up/figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

THESIS_ISO3 <- c("BFA", "ETH", "MLI", "MWI", "NER", "NGA", "TZA")

# ---- Country metadata ----------------------------------------------------
country_labels <- c(
  BFA = "Burkina Faso", ETH = "Ethiopia",   MLI = "Mali",
  MWI = "Malawi",       NER = "Niger",       NGA = "Nigeria",
  TZA = "Tanzania"
)

# AEZ classification
aez_map <- c(BFA = "Sahel", ETH = "Non-Sahel", MLI = "Sahel",
             MWI = "Non-Sahel", NER = "Sahel",  NGA = "Non-Sahel",
             TZA = "Non-Sahel")

# Colour palette — colour-blind-friendly (Okabe-Ito)
country_colours <- c(
  BFA = "#E69F00", ETH = "#56B4E9", MLI = "#009E73",
  MWI = "#F0E442", NER = "#CC79A7", NGA = "#D55E00",
  TZA = "#0072B2"
)
aez_colours <- c("Sahel" = "#D55E00", "Non-Sahel" = "#0072B2")

# ---- Load panel ----------------------------------------------------------
panel <- read_csv(panel_path, show_col_types = FALSE) %>%
  filter(GID_0 %in% THESIS_ISO3) %>%
  mutate(
    pred_yield_kgha = exp(log_pred_yield_v2),
    log_conflict    = log1p(conflict_3mo),
    aez             = aez_map[GID_0],
    country         = country_labels[GID_0],
    GID_0           = factor(GID_0, levels = THESIS_ISO3)
  )

# ---- Load GADM shapefiles -----------------------------------------------
load_gadm <- function(iso3) {
  rds <- file.path(gadm_dir, paste0("gadm41_", iso3, "_2_pk.rds"))
  obj <- readRDS(rds)
  # geodata stores as PackedSpatVector — unpack with terra then convert to sf
  if (inherits(obj, "PackedSpatVector")) {
    obj <- terra::vect(obj)
  }
  sf::st_as_sf(obj) %>% select(GID_2, GID_0, NAME_2)
}

message("Loading GADM shapefiles...")
gadm_all <- map_dfr(THESIS_ISO3, load_gadm) %>%
  mutate(GID_0 = as.character(GID_0),
         aez   = aez_map[GID_0])

# Join panel means to GADM
district_means <- panel %>%
  group_by(GID_2) %>%
  summarise(
    mean_log_conflict = mean(log_conflict, na.rm = TRUE),
    mean_yield        = mean(pred_yield_kgha, na.rm = TRUE),
    .groups = "drop"
  )

gadm_plot <- gadm_all %>% left_join(district_means, by = "GID_2")

# Country-level outlines for overlay
country_outlines <- gadm_all %>%
  group_by(GID_0) %>%
  summarise(geometry = st_union(geometry), .groups = "drop")

# =========================================================================
# Figure 1 — Study area map: AEZ classification
# =========================================================================
message("Generating Figure 1: Study area map...")

fig1 <- ggplot(gadm_plot) +
  geom_sf(aes(fill = aez), colour = NA, linewidth = 0) +
  geom_sf(data = country_outlines, fill = NA, colour = "white", linewidth = 0.5) +
  scale_fill_manual(
    values = aez_colours,
    name   = "Zone",
    na.value = "grey85"
  ) +
  labs(
    title    = NULL,
    subtitle = "Admin-2 districts coloured by agro-ecological zone (AEZ)"
  ) +
  theme_void(base_size = 11) +
  theme(
    legend.position  = c(0.12, 0.25),
    legend.title     = element_text(size = 9, face = "bold"),
    legend.text      = element_text(size = 8),
    plot.subtitle    = element_text(size = 9, hjust = 0.5, colour = "grey40"),
    plot.margin      = margin(5, 5, 5, 5)
  )

ggsave(file.path(output_dir, "fig1_study_area_map.pdf"),
       fig1, width = 10, height = 6, device = "pdf")
message("Saved: fig1_study_area_map.pdf")

# =========================================================================
# Figure 2 — Mean conflict intensity map
# =========================================================================
message("Generating Figure 2: Conflict intensity map...")

fig2 <- ggplot(gadm_plot) +
  geom_sf(aes(fill = mean_log_conflict), colour = NA, linewidth = 0) +
  geom_sf(data = country_outlines, fill = NA, colour = "white", linewidth = 0.5) +
  scale_fill_distiller(
    palette  = "YlOrRd",
    direction = 1,
    name     = "Mean\nlog(1+events)",
    na.value = "grey85",
    limits   = c(0, NA)
  ) +
  labs(subtitle = "Mean log(1 + ACLED events) in 3-month post-harvest window, 2010\u20132024") +
  theme_void(base_size = 11) +
  theme(
    legend.position = c(0.10, 0.25),
    legend.title    = element_text(size = 9, face = "bold"),
    legend.text     = element_text(size = 8),
    plot.subtitle   = element_text(size = 9, hjust = 0.5, colour = "grey40"),
    plot.margin     = margin(5, 5, 5, 5)
  )

ggsave(file.path(output_dir, "fig2_conflict_map.pdf"),
       fig2, width = 10, height = 6, device = "pdf")
message("Saved: fig2_conflict_map.pdf")

# =========================================================================
# Figure 3 — Conflict trends by country (2010–2024)
# =========================================================================
message("Generating Figure 3: Conflict trends...")

fig3_data <- panel %>%
  group_by(GID_0, year) %>%
  summarise(mean_conflict = mean(conflict_3mo, na.rm = TRUE), .groups = "drop")

fig3 <- ggplot(fig3_data,
               aes(x = year, y = mean_conflict,
                   colour = GID_0, group = GID_0)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  scale_y_continuous(
    trans  = "log1p",
    breaks = c(0, 1, 2, 5, 10, 20, 50),
    labels = label_comma(accuracy = 1)
  ) +
  scale_x_continuous(breaks = seq(2010, 2024, 2)) +
  scale_colour_manual(
    values = country_colours,
    name   = NULL,
    labels = country_labels
  ) +
  labs(
    x = "Year",
    y = "Mean conflict events per district (log scale)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    axis.text.x      = element_text(angle = 45, hjust = 1),
    legend.position  = "right",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(output_dir, "fig3_conflict_trends.pdf"),
       fig3, width = 8, height = 5, device = "pdf")
message("Saved: fig3_conflict_trends.pdf")

# =========================================================================
# Figure 4 — Predicted yield distribution by country (violin + boxplot)
# =========================================================================
message("Generating Figure 4: Yield distribution...")

fig4 <- ggplot(panel %>% filter(!is.na(pred_yield_kgha)),
               aes(x = GID_0, y = pred_yield_kgha, fill = GID_0)) +
  geom_violin(trim = TRUE, alpha = 0.6, colour = "grey40", linewidth = 0.4) +
  geom_boxplot(width = 0.12, outlier.size = 0.5, outlier.alpha = 0.3,
               colour = "grey30", fill = "white") +
  scale_fill_manual(values = country_colours, guide = "none") +
  scale_x_discrete(labels = country_labels) +
  scale_y_continuous(labels = label_comma(), limits = c(0, NA)) +
  labs(
    x = NULL,
    y = "Predicted maize yield (kg/ha)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x        = element_text(angle = 30, hjust = 1)
  )

ggsave(file.path(output_dir, "fig4_yield_distribution.pdf"),
       fig4, width = 8, height = 5, device = "pdf")
message("Saved: fig4_yield_distribution.pdf")

# =========================================================================
# Figure 5 — Share of district-years with zero conflict by country
# =========================================================================
message("Generating Figure 5: Zero-conflict shares...")

fig5_data <- panel %>%
  group_by(GID_0) %>%
  summarise(
    zero_share = mean(conflict_3mo == 0, na.rm = TRUE) * 100,
    .groups    = "drop"
  ) %>%
  arrange(zero_share) %>%
  mutate(GID_0 = factor(GID_0, levels = GID_0))

fig5 <- ggplot(fig5_data,
               aes(x = zero_share, y = GID_0, fill = as.character(GID_0))) +
  geom_col(width = 0.6, colour = "grey30") +
  geom_text(aes(label = sprintf("%.1f%%", zero_share)),
            hjust = -0.1, size = 3.4, fontface = "bold") +
  scale_fill_manual(values = country_colours, guide = "none") +
  scale_y_discrete(labels = country_labels) +
  scale_x_continuous(
    limits = c(0, 105),
    labels = label_percent(scale = 1),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    x = "Share of district-years with zero conflict (%)",
    y = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_blank()
  )

ggsave(file.path(output_dir, "fig5_zero_shares.pdf"),
       fig5, width = 7, height = 4.5, device = "pdf")
message("Saved: fig5_zero_shares.pdf")

# =========================================================================
# Figure 6 — Per-country coefficient forest plot (TWFE OLS log-log)
# =========================================================================
message("Generating Figure 6: Per-country coefficient plot...")

# Drop NA log_pop, then run per-country
panel_reg <- panel %>%
  filter(!is.na(log_pop), !is.na(log_pred_yield_v2)) %>%
  mutate(log_conflict = log1p(conflict_3mo))

coef_list <- map_dfr(THESIS_ISO3, function(iso) {
  dat <- panel_reg %>% filter(GID_0 == iso)
  if (n_distinct(dat$year) < 2 || n_distinct(dat$GID_2) < 2) return(NULL)
  m <- tryCatch(
    feols(log_conflict ~ log_pred_yield_v2 + log_pop | GID_2 + year,
          data = dat, cluster = ~GID_2),
    error = function(e) NULL
  )
  if (is.null(m)) return(NULL)
  cf <- coef(m)["log_pred_yield_v2"]
  se <- se(m)["log_pred_yield_v2"]
  tibble(GID_0 = iso, beta = cf, se = se,
         lo95 = cf - 1.96 * se, hi95 = cf + 1.96 * se,
         lo90 = cf - 1.645 * se, hi90 = cf + 1.645 * se)
}) %>%
  mutate(
    sig     = case_when(abs(beta / se) > 2.576 ~ "p<0.01",
                        abs(beta / se) > 1.960  ~ "p<0.05",
                        abs(beta / se) > 1.645  ~ "p<0.10",
                        TRUE                    ~ "n.s."),
    sig     = factor(sig, levels = c("p<0.01", "p<0.05", "p<0.10", "n.s.")),
    aez     = aez_map[GID_0],
    country = country_labels[GID_0],
    country = factor(country, levels = rev(country_labels[THESIS_ISO3]))
  )

fig6 <- ggplot(coef_list, aes(x = beta, y = country, colour = aez)) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50", linewidth = 0.5) +
  geom_errorbarh(aes(xmin = lo95, xmax = hi95), height = 0, linewidth = 1.0) +
  geom_errorbarh(aes(xmin = lo90, xmax = hi90), height = 0, linewidth = 1.8) +
  geom_point(aes(shape = sig), size = 3.5, fill = "white") +
  scale_colour_manual(values = aez_colours, name = "Zone") +
  scale_shape_manual(
    values = c("p<0.01" = 16, "p<0.05" = 16, "p<0.10" = 16, "n.s." = 1),
    name   = "Significance"
  ) +
  labs(
    x = expression(hat(beta) ~ "(log predicted yield, TWFE OLS log-log)"),
    y = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position    = "right"
  )

ggsave(file.path(output_dir, "fig6_country_coefs.pdf"),
       fig6, width = 8, height = 5, device = "pdf")
message("Saved: fig6_country_coefs.pdf")

# =========================================================================
# Figure 7 — AEZ binned scatter (Sahel vs Non-Sahel)
# =========================================================================
message("Generating Figure 7: AEZ binned scatter...")

# Partial out district and year FEs first, then bin
panel_demean <- panel_reg %>%
  group_by(GID_2) %>% mutate(yield_dm = log_pred_yield_v2 - mean(log_pred_yield_v2)) %>%
  ungroup() %>%
  group_by(year)   %>% mutate(yield_dm = yield_dm - mean(yield_dm)) %>%
  ungroup() %>%
  group_by(GID_2) %>% mutate(conf_dm = log_conflict - mean(log_conflict)) %>%
  ungroup() %>%
  group_by(year)   %>% mutate(conf_dm = conf_dm - mean(conf_dm)) %>%
  ungroup()

# Bin within each AEZ zone (20 bins)
binned <- panel_demean %>%
  filter(!is.na(aez)) %>%
  group_by(aez) %>%
  mutate(bin = ntile(yield_dm, 20)) %>%
  group_by(aez, bin) %>%
  summarise(
    x = mean(yield_dm, na.rm = TRUE),
    y = mean(conf_dm,  na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

fig7 <- ggplot(binned, aes(x = x, y = y, colour = aez)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60", linewidth = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60", linewidth = 0.4) +
  geom_point(aes(size = n), alpha = 0.85) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 0.9,
              aes(fill = aez), alpha = 0.15) +
  scale_colour_manual(values = aez_colours, name = "Zone") +
  scale_fill_manual(values = aez_colours, guide = "none") +
  scale_size_continuous(range = c(1.5, 5), guide = "none") +
  facet_wrap(~aez, nrow = 1) +
  labs(
    x = "Log predicted yield (demeaned by district + year FEs)",
    y = "Log(1 + conflict events) (demeaned)"
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position  = "none",
    strip.text       = element_text(face = "bold", size = 11)
  )

ggsave(file.path(output_dir, "fig7_aez_binscatter.pdf"),
       fig7, width = 9, height = 4.5, device = "pdf")
message("Saved: fig7_aez_binscatter.pdf")

message("\nAll 7 figures saved to: ", output_dir)
