library(fixest)

feols(conflict_count ~ mean_log_yield_pred | ADM2_NAME + year, data = reg_data, notes = TRUE)

ols1 <- lm(conflict_count ~ mean_log_yield_pred, data = reg_data)
summary(ols1)

fe1 <- feols(
  conflict_count ~ mean_log_yield_pred | ADM2_NAME + year,
  data = reg_data,
  cluster = ~ ADM2_NAME   # cluster SEs at unit level
)
summary(fe1)

fp1 <- fepois(
  conflict_count ~ mean_log_yield_pred | ADM2_NAME + year,
  data = reg_data,
  cluster = ~ ADM2_NAME
)
summary(fp1)
