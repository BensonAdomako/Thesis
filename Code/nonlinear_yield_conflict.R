# nonlinear_yield_conflict.R
# ──────────────────────────────────────────────────────────────────────────────
# Frisch-Waugh residualization + non-linear (log-log) model
#
# Approach:
#   Step 1 — partial out admin-2 + year two-way FEs from both
#             log_conflict_3mo and log_pred_yield via separate feols calls
#             (Frisch-Waugh-Lovell theorem)
#   Step 2 — regress residualised outcome on residualised treatment using:
#               M_lin   : linear (baseline)
#               M_quad  : + quadratic term
#               M_cubic : + cubic term
#               M_ns3   : natural cubic spline, 3 df
#               M_ns5   : natural cubic spline, 5 df
#   Step 3 — F-tests for non-linearity vs linear baseline
#   Step 4 — bin-mean summary (20 equal-count bins) to inspect shape
# ──────────────────────────────────────────────────────────────────────────────

library(fixest)
library(dplyr)
library(readr)
library(splines)

PANEL_CSV <- "Data/conflict_yields_panel.csv"

# ── 1. Load panel ─────────────────────────────────────────────────────────────
df <- read_csv(PANEL_CSV, show_col_types = FALSE) %>%
  mutate(adm2_id = paste(ADM0_NAME, ADM1_NAME, ADM2_NAME, sep = " | "))

message("Panel: ", nrow(df), " rows, ", n_distinct(df$adm2_id), " districts")

# ── 2. Residualise via Frisch-Waugh (feols with FE only) ─────────────────────
message("\n--- Step 1: Partialling out admin-2 + year FEs ---")

fe_y <- feols(log_conflict_3mo ~ 1 | adm2_id + year, data = df)
fe_x <- feols(log_pred_yield   ~ 1 | adm2_id + year, data = df)

y_res <- residuals(fe_y)   # within-TWFE log conflict
x_res <- residuals(fe_x)   # within-TWFE log pred yield

res <- data.frame(y = y_res, x = x_res)

message("  Residual y  — mean: ", round(mean(y_res), 6),
        "  sd: ", round(sd(y_res), 4))
message("  Residual x  — mean: ", round(mean(x_res), 6),
        "  sd: ", round(sd(x_res), 4))
message("  N obs used  : ", nrow(res))

# ── 3. Non-linear models on residuals ─────────────────────────────────────────
message("\n--- Step 2: Non-linear models on residualised variables ---")

m_lin  <- lm(y ~ x,                                         data = res)
m_quad <- lm(y ~ x + I(x^2),                               data = res)
m_cub  <- lm(y ~ x + I(x^2) + I(x^3),                     data = res)
m_ns3  <- lm(y ~ ns(x, df = 3),                            data = res)
m_ns5  <- lm(y ~ ns(x, df = 5),                            data = res)

message("\n=== M_lin: Linear (Frisch-Waugh baseline) ===")
print(summary(m_lin))

message("\n=== M_quad: Quadratic ===")
print(summary(m_quad))

message("\n=== M_cubic: Cubic ===")
print(summary(m_cub))

message("\n=== M_ns3: Natural cubic spline (3 df) ===")
print(summary(m_ns3))

message("\n=== M_ns5: Natural cubic spline (5 df) ===")
print(summary(m_ns5))

# ── 4. F-tests for non-linearity ──────────────────────────────────────────────
message("\n--- Step 3: F-tests — non-linear specs vs linear baseline ---")

message("\nQuadratic vs Linear:")
print(anova(m_lin, m_quad))

message("\nCubic vs Linear:")
print(anova(m_lin, m_cub))

message("\nNatural spline (3df) vs Linear:")
print(anova(m_lin, m_ns3))

message("\nNatural spline (5df) vs Linear:")
print(anova(m_lin, m_ns5))

message("\nCubic vs Quadratic:")
print(anova(m_quad, m_cub))

# ── 5. Bin-mean summary (shape inspection) ───────────────────────────────────
message("\n--- Step 4: Bin-mean summary (20 equal-count bins of residual x) ---")

res <- res %>%
  mutate(bin = ntile(x, 20)) %>%
  group_by(bin) %>%
  mutate(x_mean = mean(x))

bin_summary <- res %>%
  group_by(bin) %>%
  summarise(
    x_mid  = mean(x),
    y_mean = mean(y),
    y_se   = sd(y) / sqrt(n()),
    n      = n(),
    .groups = "drop"
  )

message("  Bin means (residualised log pred yield → residualised log conflict):")
print(bin_summary, n = 20)

message("\nDone.")
