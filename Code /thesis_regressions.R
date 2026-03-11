# ═══════════════════════════════════════════════════════════════════════════════
#  THESIS REGRESSION ANALYSIS
#  Research Question: Do crop yields cause conflict?
#  Data: Admin-area × year panel, 7 Sub-Saharan African countries, 2010–2024
# ═══════════════════════════════════════════════════════════════════════════════

# ── 0. Packages & Setup ────────────────────────────────────────────────────────
library(fixest)       # feols, fepois, fenegbin — fast FE estimators
library(MASS)         # glm.nb (baseline NB without FE)
library(pscl)         # hurdle models
library(tidyverse)    # data wrangling
library(modelsummary) # publication-quality tables
library(kableExtra)   # table formatting

options(scipen = 999)
set.seed(42)

# ── 1. Load & Prepare Data ─────────────────────────────────────────────────────
df <- read_csv("reg_final.csv", show_col_types = FALSE)

df <- df |>
  mutate(
    # Core regressors
    log_yield    = mean_log_pred_yield,          # already log(kg/ha)
    log_pop      = log(pop_sum),
    log_dist     = log(dist_urban_km + 1),       # +1 avoids log(0) for areas at urban centre

    # Per-capita outcomes (for OLS)
    log_conf3_pc  = log1p(conflict_3mo_pc),
    log_conf6_pc  = log1p(conflict_6mo_pc),
    log_conf12_pc = log1p(conflict_12mo_pc),

    # Interaction term: does yield effect differ by remoteness?
    yield_x_dist  = log_yield * log_dist,

    # ID variables
    adm_id  = paste(ADM0_NAME, ADM2_NAME, sep = "_"),  # unique admin-area ID
    year_f  = as.character(year)
  ) |>
  # Sort panel properly
  arrange(adm_id, year)

# Compute lagged conflict WITHIN each admin area (for Model 5)
df <- df |>
  group_by(adm_id) |>
  mutate(
    lag_conf3_pc  = lag(log_conf3_pc,  order_by = year),
    lag_conf6_pc  = lag(log_conf6_pc,  order_by = year),
    lag_conf12_pc = lag(log_conf12_pc, order_by = year)
  ) |>
  ungroup()

cat("\n── Data summary ──────────────────────────────────────────────────────────────\n")
cat(sprintf("Observations      : %d\n",    nrow(df)))
cat(sprintf("Unique admin areas: %d\n",    n_distinct(df$adm_id)))
cat(sprintf("Countries         : %s\n",    paste(unique(df$ADM0_NAME), collapse = ", ")))
cat(sprintf("Years             : %d–%d\n", min(df$year), max(df$year)))
cat(sprintf("%% zero conflict (12mo): %.1f%%\n", mean(df$conflict_12mo == 0) * 100))

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — OLS Baseline  (Country + Year FE)
# ═══════════════════════════════════════════════════════════════════════════════
# Outcome: log(1 + conflict per 1,000 pop) — 12 months post-harvest
# The log transformation compresses extreme values and makes residuals more normal.
# Country and year FE absorb stable cross-country differences and common time shocks.
# Distance to urban and log population enter as explicit controls.
# SEs are clustered at the admin-area level to allow arbitrary serial correlation
# within districts across years.

m1_3  <- feols(log_conf3_pc  ~ log_yield + log_pop + log_dist |
                 ADM0_NAME + year_f, data = df,
               cluster = ~adm_id)

m1_6  <- feols(log_conf6_pc  ~ log_yield + log_pop + log_dist |
                 ADM0_NAME + year_f, data = df,
               cluster = ~adm_id)

m1_12 <- feols(log_conf12_pc ~ log_yield + log_pop + log_dist |
                 ADM0_NAME + year_f, data = df,
               cluster = ~adm_id)

cat("\n\n══════════════════════════════════════════════════════════════════════════════\n")
cat("MODEL 1 — OLS  |  log(1 + conflict pc)  |  Country + Year FE\n")
cat("══════════════════════════════════════════════════════════════════════════════\n")
etable(m1_3, m1_6, m1_12,
       headers = c("3-month", "6-month", "12-month"),
       coefstat = "se", digits = 4,
       signif.code = c("***"=.01,"**"=.05,"*"=.10))

cat("
DISCUSSION — Model 1 (OLS Baseline):
The OLS estimates serve as the entry point, mirroring specifications common in
the food-security and conflict literature. The coefficient on log_yield is
expected to be negative across all three horizons: higher predicted crop yields
should reduce the incidence and intensity of conflict in the months following
harvest. A one-unit increase in log yield (roughly a 2.7x increase in kg/ha)
is associated with a β-unit change in log(1 + conflict per 1,000 people).

The 12-month horizon typically shows the strongest absolute coefficient because
it captures a wider window of food-stress-induced grievances and resource
competition. The 3-month estimate captures more immediate post-harvest
channels — food availability, income effects, opportunity costs of fighting —
while the 6- and 12-month estimates may additionally reflect market price
adjustments and longer-term livelihood effects.

Distance to urban centre (log_dist) acts as a proxy for state capacity and
market integration. Remote areas with lower state presence and thinner markets
are hypothesised to be more vulnerable to conflict following yield shocks.
A positive coefficient on log_dist would be consistent with this mechanism.
Log population controls for the mechanical relationship between population
density and the raw number of events recorded.

IMPORTANT CAVEAT: OLS imposes linearity and symmetry, and is inappropriate for
a zero-inflated outcome (84.7% zeros). These estimates are presented for
comparability with existing work; the preferred specifications are Models 4–6.
")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Negative Binomial  (Country + Year FE)
# ═══════════════════════════════════════════════════════════════════════════════
# Outcome: raw conflict count — counts are the natural unit for NB regression.
# log(pop_sum) enters as an OFFSET (not a covariate): this constrains its
# coefficient to 1.0, exactly equivalent to modelling the rate (events per person).
# The NB dispersion parameter handles the overdispersion that would cause
# Poisson to underestimate standard errors.

m2_3  <- fenegbin(conflict_3mo  ~ log_yield + log_dist +
                    offset(log_pop) | ADM0_NAME + year_f,
                  data = df, cluster = ~adm_id)

m2_6  <- fenegbin(conflict_6mo  ~ log_yield + log_dist +
                    offset(log_pop) | ADM0_NAME + year_f,
                  data = df, cluster = ~adm_id)

m2_12 <- fenegbin(conflict_12mo ~ log_yield + log_dist +
                    offset(log_pop) | ADM0_NAME + year_f,
                  data = df, cluster = ~adm_id)

cat("\n\n══════════════════════════════════════════════════════════════════════════════\n")
cat("MODEL 2 — Negative Binomial  |  Raw conflict count  |  Country + Year FE\n")
cat("══════════════════════════════════════════════════════════════════════════════\n")
etable(m2_3, m2_6, m2_12,
       headers = c("3-month", "6-month", "12-month"),
       coefstat = "se", digits = 4,
       signif.code = c("***"=.01,"**"=.05,"*"=.10))

# Incidence Rate Ratios for interpretability
cat("\n── Incidence Rate Ratios (exp(β)) for Model 2 — 12-month ──────────────────\n")
coef_nb12 <- coef(m2_12)
irr_nb12  <- exp(coef_nb12)
se_nb12   <- se(m2_12)
cat(sprintf("  log_yield IRR = %.4f  (SE = %.4f)\n",
            irr_nb12["log_yield"], se_nb12["log_yield"]))
cat(sprintf("  log_dist  IRR = %.4f  (SE = %.4f)\n",
            irr_nb12["log_dist"],  se_nb12["log_dist"]))

cat("
DISCUSSION — Model 2 (Negative Binomial):
Moving to a count model respects the data-generating process. The Negative
Binomial directly models the number of conflict events as a non-negative integer,
with the NB dispersion parameter α capturing overdispersion (variance > mean)
that is clearly present in this dataset.

Coefficients here are interpreted as incidence rate ratios (IRRs) after
exponentiation: an IRR of 0.80 on log_yield would mean that a one-unit increase
in log predicted yield is associated with a 20% DECREASE in the expected
conflict count, holding all else constant.

Using population as an offset (rather than a covariate) is the technically
correct way to model per-capita rates in a count framework — it constrains the
elasticity of conflict with respect to population to exactly 1, which is the
natural benchmark for rate comparisons across districts of different sizes.

The inclusion of distance to urban centre here (unlike in the admin FE models
below) allows us to explicitly test the state-capacity mechanism: remote,
poorly-governed districts should show higher conflict rates conditional on yield
shocks. A significant positive IRR on log_dist would support this channel.
")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Hurdle Negative Binomial  (Country + Year dummies)
# ═══════════════════════════════════════════════════════════════════════════════
# The hurdle model separates two distinct processes:
#   Part 1 (Binomial logit): P(any conflict at all) — the ONSET decision
#   Part 2 (Truncated NB):   E[conflict | conflict > 0] — the INTENSITY level
# Note: pscl::hurdle() does not support proper panel FE; country + year dummies
# are included as explicit factor variables. Inference is approximate here.

df_hurdle <- df |>
  mutate(
    country_f = factor(ADM0_NAME),
    year_fac  = factor(year)
  )

m3_12 <- hurdle(
  conflict_12mo ~ log_yield + log_dist + log_pop + country_f + year_fac,
  data   = df_hurdle,
  dist   = "negbin",
  zero.dist = "binomial",
  link   = "logit"
)

cat("\n\n══════════════════════════════════════════════════════════════════════════════\n")
cat("MODEL 3 — Hurdle Negative Binomial  |  12-month conflict  |  Country + Year dummies\n")
cat("══════════════════════════════════════════════════════════════════════════════\n")

# Extract just the core coefficients (suppress country/year dummies)
coef_h    <- coef(m3_12)
se_h      <- sqrt(diag(vcov(m3_12)))
core_vars <- c("count_log_yield", "count_log_dist", "count_log_pop",
               "zero_log_yield",  "zero_log_dist",  "zero_log_pop")
core_vars <- core_vars[core_vars %in% names(coef_h)]

cat("\n  Count component (Truncated NB — Intensity conditional on conflict > 0):\n")
for (v in grep("^count_", core_vars, value=TRUE)) {
  cat(sprintf("    %-22s coef = %8.4f  (SE = %.4f)\n", v, coef_h[v], se_h[v]))
}
cat("\n  Zero component (Logit — Probability of ANY conflict):\n")
for (v in grep("^zero_", core_vars, value=TRUE)) {
  cat(sprintf("    %-22s coef = %8.4f  (SE = %.4f)\n", v, coef_h[v], se_h[v]))
}

cat("
DISCUSSION — Model 3 (Hurdle Negative Binomial):
The hurdle model is theoretically motivated by the idea that whether conflict
STARTS and how SEVERE it is may have different determinants. A yield shock in
a peaceful district might not trigger conflict at all (high barrier to onset),
while in a district with existing grievances or ethnic tensions the same shock
could escalate rapidly. The hurdle model tests these two channels separately.

ZERO COMPONENT (Logit): The coefficient on log_yield in the zero component
captures whether higher yields reduce the probability of ANY conflict occurring.
A negative coefficient here means higher yields lower the probability of
crossing the conflict threshold.

COUNT COMPONENT (Truncated NB): The coefficient on log_yield in the count
component captures whether, GIVEN that conflict occurs, its intensity is
affected by yield levels. This is a more nuanced question — once fighting has
started, does better harvest temper its scale?

If both components show a significant negative coefficient on log_yield, the
evidence for a yield-conflict nexus is particularly strong: yields affect both
onset likelihood AND intensity. If only the zero component is significant,
yields primarily operate through a conflict-prevention channel. If only the
count component, yields influence escalation dynamics more than initial onset.

NOTE: Country and year effects are modelled via explicit dummies (not true FE
dummies), so inference here is less efficient than in Models 4–6. Treat this
as a diagnostic/descriptive model rather than a causal estimate.
")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — PPML  (Admin-area + Year Two-Way FE)  *** MAIN SPECIFICATION ***
# ═══════════════════════════════════════════════════════════════════════════════
# Poisson Pseudo-Maximum Likelihood (Santos Silva & Tenreyro 2006).
# Admin-area FE sweep out ALL time-invariant district characteristics:
#   - distance to urban centre (absorbed)
#   - ethnicity, geography, historical conflict patterns
#   - soil quality, agro-ecological zone
# Year FE absorb global shocks: world food prices, climate trends, COVID-19.
# Identification is PURELY from within-district, within-year variation in yields.
# PPML is consistent even if the true DGP is not Poisson (robust to
# distributional misspecification), as long as the conditional mean is correctly
# specified — making it more robust than NB for the count component.

m4_3  <- fepois(conflict_3mo  ~ log_yield + offset(log_pop) | adm_id + year_f,
                data = df, cluster = ~adm_id)

m4_6  <- fepois(conflict_6mo  ~ log_yield + offset(log_pop) | adm_id + year_f,
                data = df, cluster = ~adm_id)

m4_12 <- fepois(conflict_12mo ~ log_yield + offset(log_pop) | adm_id + year_f,
                data = df, cluster = ~adm_id)

cat("\n\n══════════════════════════════════════════════════════════════════════════════\n")
cat("MODEL 4 — PPML (MAIN SPEC)  |  Raw conflict count  |  Admin + Year FE\n")
cat("══════════════════════════════════════════════════════════════════════════════\n")
etable(m4_3, m4_6, m4_12,
       headers = c("3-month", "6-month", "12-month"),
       coefstat = "se", digits = 4,
       signif.code = c("***"=.01,"**"=.05,"*"=.10))

# IRRs for main spec
cat("\n── Incidence Rate Ratios — Model 4 (PPML, Admin FE) ───────────────────────\n")
for (label in c("3-month","6-month","12-month")) {
  m_tmp <- list("3-month"=m4_3, "6-month"=m4_6, "12-month"=m4_12)[[label]]
  irr_tmp <- exp(coef(m_tmp)["log_yield"])
  se_tmp  <- se(m_tmp)["log_yield"]
  pct_chg <- (irr_tmp - 1) * 100
  cat(sprintf("  %s: IRR = %.4f  → a 10%% yield increase ≈ %.1f%% change in conflict\n",
              label, irr_tmp,
              (exp(coef(m_tmp)["log_yield"] * log(1.1)) - 1) * 100))
}

cat("
DISCUSSION — Model 4 (PPML, Main Specification):
This is the thesis's preferred causal estimate. By including admin-area fixed
effects, we compare a district to ITSELF in different years — effectively asking:
'In years when predicted yields were higher than average for THIS district, was
conflict lower than average for THIS district?' This within-estimator eliminates
all time-invariant confounders, including the distance to urban centre variable
from Models 1–2 (which does not vary over time within a district).

The year fixed effects ensure we are not simply picking up a global trend in
which yields and conflict both move together (e.g., a global food price shock
in 2011–2012 that simultaneously raised conflict across the continent and
reduced agricultural investment).

Because the predicted yields come from a machine-learning model trained on
satellite and weather data, they are unlikely to be contaminated by conflict
itself — a crucial advantage for causal identification. Conflict could
plausibly depress actual observed yields (reverse causality), but it is
unlikely to affect what our model PREDICTS yields would have been absent
conflict. This makes our predictor closer to an instrument than a raw covariate.

THE KEY COEFFICIENT: The sign, magnitude, and significance of log_yield in this
specification is the central empirical finding of the thesis. An IRR below 1.0
(negative coefficient) that survives admin + year FE is strong evidence that
yield shocks causally reduce conflict — not merely that poor-yield countries
are also conflict-prone.

The gradient across horizons (3 → 6 → 12 months) speaks to the MECHANISM:
if the 12-month effect is substantially larger than the 3-month effect, the
dominant channel is likely longer-run livelihood and income effects rather
than immediate food availability alone.
")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 5 — PPML + Lagged Conflict  (Admin + Year FE)
# ═══════════════════════════════════════════════════════════════════════════════
# Adds the lagged log conflict as a control to absorb conflict momentum/inertia.
# This is the most demanding robustness check: does yield STILL matter for
# conflict ABOVE AND BEYOND the district's own recent conflict history?
# Note: Including a lagged dependent variable in a FE panel model introduces
# Nickell (1981) bias, which attenuates the lagged coefficient toward zero.
# The effect is small when T is large (T=15 here), but worth acknowledging.

df_lag <- df |> filter(!is.na(lag_conf12_pc))  # loses one year per district

m5_12 <- fepois(conflict_12mo ~ log_yield + lag_conf12_pc +
                  offset(log_pop) | adm_id + year_f,
                data = df_lag, cluster = ~adm_id)

cat("\n\n══════════════════════════════════════════════════════════════════════════════\n")
cat("MODEL 5 — PPML + Lagged Conflict  |  12-month  |  Admin + Year FE\n")
cat("══════════════════════════════════════════════════════════════════════════════\n")
etable(m4_12, m5_12,
       headers      = c("Model 4 (Main)", "Model 5 (+ Lagged Conflict)"),
       coefstat     = "se", digits = 4,
       signif.code  = c("***"=.01,"**"=.05,"*"=.10))

cat("
DISCUSSION — Model 5 (Lagged Conflict Robustness):
Conflict exhibits strong path dependence: a district that experienced fighting
last year is more likely to experience it this year, independent of agricultural
conditions. Failing to account for this persistence would risk attributing
conflict momentum to yield shocks if the two happen to be correlated.

Model 5 adds the lagged log(1 + conflict per capita) from the PREVIOUS year as
a control variable. Comparing the yield coefficient in Model 5 with that in
Model 4 tells us how much of the estimated effect in the main spec simply
reflects yield–conflict co-movement driven by conflict persistence.

If the yield coefficient is stable across Models 4 and 5, the result is robust:
yields matter for conflict even after controlling for past conflict levels. If
the coefficient shrinks substantially, this suggests some of the estimated
effect in Model 4 was working through conflict persistence rather than
directly — still an indirect causal chain, but important to document.

The lagged conflict coefficient itself (expected to be positive and significant)
provides an estimate of conflict PERSISTENCE — the degree to which past conflict
predicts current conflict. A coefficient close to 1.0 would indicate near-
unit-root persistence (conflict is very hard to end once started); smaller
values indicate more transitory conflict patterns.
")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 6 — PPML + Yield × Distance Interaction  (Admin + Year FE)
# ═══════════════════════════════════════════════════════════════════════════════
# Tests whether the yield-conflict relationship is HETEROGENEOUS by remoteness.
# The mechanism: remote areas have weaker state presence, thinner food markets,
# and less ability to buffer yield shocks through trade or social insurance.
# If so, the yield-conflict link should be stronger (more negative) in districts
# farther from urban centres — even though distance itself is absorbed by admin FE,
# the INTERACTION of yield with distance is identified because yield varies by year.

m6_12 <- fepois(conflict_12mo ~ log_yield + yield_x_dist +
                  offset(log_pop) | adm_id + year_f,
                data = df, cluster = ~adm_id)

cat("\n\n══════════════════════════════════════════════════════════════════════════════\n")
cat("MODEL 6 — PPML + Yield × Distance Interaction  |  12-month  |  Admin + Year FE\n")
cat("══════════════════════════════════════════════════════════════════════════════\n")
etable(m4_12, m6_12,
       headers      = c("Model 4 (Main)", "Model 6 (+ Interaction)"),
       coefstat     = "se", digits = 4,
       signif.code  = c("***"=.01,"**"=.05,"*"=.10))

# Marginal effect at different distance quartiles
cat("\n── Marginal Effect of log_yield at Different Distances ──────────────────────\n")
b_yield <- coef(m6_12)["log_yield"]
b_inter <- coef(m6_12)["yield_x_dist"]
dist_quantiles <- quantile(df$log_dist, c(0.1, 0.25, 0.5, 0.75, 0.9), na.rm=TRUE)
cat("  Distance quantile | log_dist | dE/d(log_yield) | IRR\n")
cat("  ─────────────────────────────────────────────────────\n")
for (q in names(dist_quantiles)) {
  ld       <- dist_quantiles[q]
  me_coef  <- b_yield + b_inter * ld
  cat(sprintf("  %-17s | %7.3f  | %14.4f  | %.4f\n",
              q, ld, me_coef, exp(me_coef)))
}

cat("
DISCUSSION — Model 6 (Heterogeneous Effects by Remoteness):
This model tests the state-capacity mechanism explicitly. The key insight is
that while distance to urban centre is absorbed by admin FE (it does not vary
over time), the PRODUCT of yield and distance does vary over time — because
yields change year to year while distance remains fixed. This interaction is
therefore identified within the two-way FE framework.

The interaction coefficient β_interaction answers: 'Is the yield-conflict
elasticity more negative in remote districts?' A NEGATIVE interaction
coefficient would confirm the state-capacity mechanism: yield shocks in remote,
poorly-governed areas translate more strongly into conflict because the state
cannot mitigate food stress through market interventions or security deployments.

The marginal effect table above shows the total yield-conflict elasticity at
different points in the distribution of distance to urban centres, allowing us
to identify which types of districts drive the aggregate result. If the
relationship is concentrated among remote areas, targeted agricultural support
and market integration in those districts would be the most cost-effective
conflict-prevention policy.

POLICY IMPLICATION: If β_interaction is negative and significant, the finding
suggests that improving rural market access (reducing effective distance to
urban food markets) could substantially dampen the conflict-generating potential
of yield shocks — a complementary intervention to yield-improvement itself.
")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE — All 6 Models Side by Side (12-month horizon)
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n\n══════════════════════════════════════════════════════════════════════════════\n")
cat("SUMMARY — All 6 Models  |  12-month conflict horizon\n")
cat("══════════════════════════════════════════════════════════════════════════════\n")

# fixest models only (etable cannot handle pscl hurdle object)
etable(m1_12, m2_12, m4_12, m5_12, m6_12,
       headers = c("M1: OLS", "M2: NB", "M4: PPML",
                   "M5: PPML+Lag", "M6: PPML+Int"),
       coefstat    = "se",
       digits      = 4,
       keep        = c("log_yield", "log_dist", "lag_conf12_pc",
                       "yield_x_dist"),
       signif.code = c("***" = .01, "**" = .05, "*" = .10),
       notes       = paste0(
         "Notes: 12-month conflict horizon. ",
         "M1-M2: country + year FE. M4-M6: admin-area + year FE. ",
         "SEs clustered at admin-area. M3 (Hurdle) shown separately above."
       ))

# Hurdle model core coefficients printed separately above (not fixest-compatible)
cat("\n  [Model 3 Hurdle NB results shown in the section above]\n")

cat("
OVERALL DISCUSSION:
The six models form a progression from the least to most demanding specification,
allowing us to assess the robustness of the yield-conflict relationship along
multiple dimensions.

A coherent narrative would show:
  (1) The OLS baseline establishes a negative yield-conflict gradient, consistent
      with the descriptive evidence in the quartile analysis.
  (2) Moving to the Negative Binomial (Model 2) respects the count nature of the
      data and handles overdispersion — if the sign and significance are maintained,
      it confirms the result is not an artefact of OLS misspecification.
  (3) The Hurdle model (Model 3) reveals WHETHER yields operate through conflict
      prevention (onset) or conflict dampening (intensity), or both.
  (4) The PPML with admin FE (Model 4) is the key causal identification step.
      The within-estimator eliminates the possibility that we are simply comparing
      peaceful, high-yield countries (e.g., Tanzania) to conflict-prone, low-yield
      ones (e.g., Mali). If the coefficient survives this test, we have strong
      evidence for a causal yield-to-conflict channel.
  (5) Model 5's lagged conflict control tests whether the result is merely
      reflecting conflict momentum co-moving with yields, rather than a direct
      causal path.
  (6) Model 6's interaction reveals the geographic heterogeneity of the effect
      and points toward the state-capacity mechanism as a plausible channel.

THE GRADIENT ACROSS HORIZONS (Models 1–2) is itself informative about mechanism:
  - Immediate (3-month): food availability, post-harvest income, opportunity costs
  - Medium (6-month): market price adjustments, off-season coping behaviour
  - Long-run (12-month): structural livelihood effects, cumulative stress
A strengthening coefficient from 3→6→12 months would be consistent with
income/livelihood channels dominating over immediate food-availability channels.
")

cat("\n\nScript complete. Results printed above.\n")
