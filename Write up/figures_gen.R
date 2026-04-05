# figures_gen.R
# Publication-quality figures for economics thesis:
# "Do Crop Yields Causally Affect Conflict in Sub-Saharan Africa?"
#
# Run from project root: Rscript "Write up/figures_gen.R"
# -------------------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(readr)
library(scales)

# ---- Paths ---------------------------------------------------------------
data_path  <- "Data/conflict_yields_panel.csv"
output_dir <- "Write up/figures"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Load data -----------------------------------------------------------
panel <- read_csv(data_path, show_col_types = FALSE)

# Country short-label lookup (exact ADM0_NAME values in data)
country_labels <- c(
  "Ethiopia"                    = "ETH",
  "Malawi"                      = "MWI",
  "Mali"                        = "MLI",
  "Nigeria"                     = "NGA",
  "Uganda"                      = "UGA",
  "United Republic of Tanzania" = "TZA"
)

# Colour palette — colour-blind-friendly
country_colours <- c(
  ETH = "#E69F00",
  MWI = "#56B4E9",
  MLI = "#009E73",
  NGA = "#F0E442",
  UGA = "#0072B2",
  TZA = "#D55E00"
)

# Add short label column
panel <- panel %>%
  mutate(
    country_short = recode(ADM0_NAME, !!!country_labels),
    country_short = factor(country_short,
                           levels = c("ETH", "MWI", "MLI", "NGA", "UGA", "TZA"))
  )

missing_countries <- panel %>% filter(is.na(country_short)) %>% distinct(ADM0_NAME)
if (nrow(missing_countries) > 0) {
  warning("Unmapped country names: ", paste(missing_countries$ADM0_NAME, collapse = ", "))
}

# =========================================================================
# Figure 1 — Mean conflict events per Admin-2 district by year
#            log1p y-axis so low-conflict countries (TZA, NGA) are visible
# =========================================================================

fig1_data <- panel %>%
  group_by(country_short, year) %>%
  summarise(mean_conflict = mean(conflict_3mo, na.rm = TRUE), .groups = "drop")

fig1 <- ggplot(fig1_data,
               aes(x = year, y = mean_conflict,
                   colour = country_short, group = country_short)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  scale_y_continuous(
    trans  = "log1p",
    breaks = c(0, 0.5, 1, 2, 5, 10, 20, 50, 100),
    labels = label_comma(accuracy = 0.1)
  ) +
  scale_x_continuous(breaks = 2010:2024) +
  scale_colour_manual(values = country_colours, name = "Country") +
  labs(
    title = "Figure 1: Mean Conflict Events per Admin-2 District by Year (3-Month Forward Window)",
    x     = "Year",
    y     = "Mean Conflict Events (log1p scale)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(size = 10, face = "bold", hjust = 0),
    axis.text.x      = element_text(angle = 45, hjust = 1),
    legend.position  = "right",
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(output_dir, "fig1_conflict_timeseries.pdf"),
  plot     = fig1,
  width    = 8, height = 6, device = "pdf"
)
message("Saved: fig1_conflict_timeseries.pdf")

# =========================================================================
# Figure 2 — Distribution of ML-predicted maize yield by country (boxplot)
# =========================================================================

panel <- panel %>%
  mutate(pred_yield_kgha = exp(log_pred_yield))

fig2 <- ggplot(panel %>% filter(!is.na(pred_yield_kgha)),
               aes(x = country_short, y = pred_yield_kgha, fill = country_short)) +
  geom_boxplot(outlier.size = 0.6, outlier.alpha = 0.4,
               width = 0.55, colour = "grey30") +
  scale_fill_manual(values = country_colours, guide = "none") +
  scale_y_continuous(labels = label_comma()) +
  labs(
    title = "Figure 2: Distribution of ML-Predicted Maize Yield by Country (2010\u20132024)",
    x     = "Country",
    y     = "Predicted Yield (kg/ha)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(size = 10, face = "bold", hjust = 0),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(output_dir, "fig2_yield_distribution.pdf"),
  plot     = fig2,
  width    = 8, height = 5, device = "pdf"
)
message("Saved: fig2_yield_distribution.pdf")

# =========================================================================
# Figure 3 — Log predicted yield vs log(1+conflict_3mo)
#            5% random sample of non-zero-conflict obs + loess smoother
# =========================================================================

set.seed(42)
fig3_data <- panel %>%
  filter(!is.na(log_pred_yield), conflict_3mo > 0) %>%
  slice_sample(prop = 0.05) %>%
  mutate(log_conflict = log1p(conflict_3mo))

fig3 <- ggplot(fig3_data,
               aes(x = log_pred_yield, y = log_conflict, colour = country_short)) +
  geom_point(size = 1.2, alpha = 0.55) +
  geom_smooth(method = "loess", se = TRUE,
              colour = "black", linewidth = 0.8,
              fill = "grey80", alpha = 0.4) +
  scale_colour_manual(values = country_colours, name = "Country") +
  labs(
    title = "Figure 3: Log Predicted Yield vs Log Conflict Events (3-Month Window)",
    x     = "Log Predicted Yield (log kg/ha)",
    y     = "Log(1 + Conflict Events)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(size = 10, face = "bold", hjust = 0),
    legend.position  = "right",
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(output_dir, "fig3_conflict_yield_scatter.pdf"),
  plot     = fig3,
  width    = 8, height = 5, device = "pdf"
)
message("Saved: fig3_conflict_yield_scatter.pdf")

# =========================================================================
# Figure 4 — Share of district-years with zero conflict by country (bar)
# =========================================================================

fig4_data <- panel %>%
  filter(!is.na(any_conflict_3mo)) %>%
  group_by(country_short) %>%
  summarise(
    zero_share = mean(any_conflict_3mo == 0, na.rm = TRUE) * 100,
    .groups = "drop"
  )

fig4 <- ggplot(fig4_data,
               aes(x = country_short, y = zero_share, fill = country_short)) +
  geom_col(width = 0.6, colour = "grey30") +
  geom_text(aes(label = sprintf("%.1f%%", zero_share)),
            vjust = -0.4, size = 3.6, fontface = "bold") +
  scale_fill_manual(values = country_colours, guide = "none") +
  scale_y_continuous(
    limits = c(0, 105),
    labels = label_percent(scale = 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    title = "Figure 4: Share of District-Years with Zero Conflict by Country (3-Month Window)",
    x     = "Country",
    y     = "Share with Zero Conflict (%)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title         = element_text(size = 10, face = "bold", hjust = 0),
    panel.grid.minor   = element_blank(),
    panel.grid.major.x = element_blank()
  )

ggsave(
  filename = file.path(output_dir, "fig4_zero_share.pdf"),
  plot     = fig4,
  width    = 8, height = 5, device = "pdf"
)
message("Saved: fig4_zero_share.pdf")

message("\nAll 4 figures saved to: ", output_dir)
