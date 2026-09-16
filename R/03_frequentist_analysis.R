# ==============================================================================
# 03_frequentist_analysis.R
#
# Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
#           with Frequentist, Bayesian, and Machine Learning Methods
# Author  : Cathy
# Purpose : Primary frequentist analysis of the composite disease-free
#           survival (DFS) endpoint and the secondary overall survival (OS)
#           endpoint: Kaplan-Meier estimation, log-rank tests, unadjusted and
#           multivariable-adjusted Cox models (multiple imputation pooling),
#           proportional-hazards diagnostics, absolute benefit measures, and
#           competing-risk sensitivity analyses (Aalen-Johansen, Fine-Gray).
#
# Inputs  : data/processed/colon_analysis.csv
#           data/processed/colon_mi_dfs.rds, colon_mi_os.rds
# Outputs : outputs/figures/fig_km_dfs.png, fig_km_os.png,
#                     fig_forest_adjusted.png, fig_zph.png, fig_cif.png
#           outputs/tables/table_logrank.csv, table_km_5yr.csv,
#                     table_cox_unadjusted.csv, table_cox_adjusted_mi.csv,
#                     table_cox_adjusted_cc.csv, table_zph.csv,
#                     table_finegray.csv, table_sensitivity.csv
# ==============================================================================

source("R/00_common.R")
suppressPackageStartupMessages(library(mice))  # required for with.mids dispatch
set_global_seed()
message("\n=== 03_frequentist_analysis.R ===\n")

colon_df <- read.csv(file.path(DATA_PROC, "colon_analysis.csv")) %>%
  mutate(rx = factor(rx, levels = ARM_LEVELS),
         sex = factor(sex), obstruct = factor(obstruct), perfor = factor(perfor),
         adhere = factor(adhere), differ = factor(differ, levels = c("Well", "Moderate", "Poor")),
         extent = factor(extent, levels = c("Submucosa", "Muscle", "Serosa", "Contiguous structures")),
         surg = factor(surg), node4 = factor(node4),
         rec_cr_f = factor(rec_cr, levels = 0:2,
                           labels = c("censored", "recurrence", "death w/o recurrence")))

mi_dfs <- readRDS(file.path(DATA_PROC, "colon_mi_dfs.rds"))
mi_os  <- readRDS(file.path(DATA_PROC, "colon_mi_os.rds"))

# ------------------------------------------------------------------------------
# 1. Kaplan-Meier curves with at-risk tables
# ------------------------------------------------------------------------------
km_figure <- function(dat, tvar, svar, title, subtitle, file, xmax = 8) {
  fit <- survfit(as.formula(sprintf("Surv(%s, %s) ~ rx", tvar, svar)),
                 data = dat, conf.type = "log-log")
  times_seq <- seq(0, xmax, 1)

  kd <- data.frame(
    time = fit$time, surv = fit$surv, lower = fit$lower, upper = fit$upper,
    arm = rep(names(fit$strata), fit$strata)
  ) %>%
    mutate(arm = factor(gsub("^rx=", "", arm), levels = ARM_LEVELS)) %>%
    bind_rows(data.frame(time = 0, surv = 1, lower = 1, upper = 1,
                         arm = factor(ARM_LEVELS, levels = ARM_LEVELS)))

  # At-risk numbers, robust to summary() ordering across strata
  rt <- summary(fit, times = times_seq, extend = TRUE)
  rtab <- data.frame(
    time = rt$time,
    n = rt$n.risk,
    arm = gsub("^rx=", "", as.character(rt$strata))
  ) %>% mutate(arm = factor(arm, levels = ARM_LEVELS))

  p1 <- ggplot(kd, aes(time, surv, colour = arm, fill = arm)) +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.10, colour = NA,
                show.legend = FALSE) +
    geom_step(linewidth = 0.75) +
    scale_colour_manual(values = ARM_COLS, labels = ARM_LABELS, drop = FALSE) +
    scale_fill_manual(values = ARM_COLS, guide = "none") +
    scale_x_continuous(breaks = times_seq, expand = expansion(mult = 0.01)) +
    scale_y_continuous(limits = c(0, 1.005), labels = scales::percent,
                       expand = expansion(mult = 0.005)) +
    labs(title = title, subtitle = subtitle,
         x = NULL, y = "Survival probability") +
    theme_trial +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

  p2 <- ggplot(rtab, aes(x = time, y = 1, label = n)) +
    geom_text(size = 2.9, colour = ARM_COLS[rtab$arm]) +
    scale_x_continuous(breaks = times_seq, limits = range(kd$time),
                       expand = expansion(mult = 0.01)) +
    scale_y_continuous(limits = c(0.6, 1.4), breaks = 1,
                       labels = "Number\nat risk") +
    labs(x = "Years since randomization", y = NULL) +
    theme_bw(base_size = 9) +
    theme(panel.grid = element_blank(), panel.border = element_blank(),
          axis.text.y = element_text(size = 8, hjust = 1),
          axis.title = element_text(size = 9))

  fig <- p1 / p2 + patchwork::plot_layout(heights = c(3.2, 1))
  save_fig(fig, file, width = 7, height = 5.8)
  fit
}

fit_km_dfs <- km_figure(colon_df, "dfs_time", "dfs_status",
  "Kaplan-Meier estimates of disease-free survival",
  "Composite endpoint: tumour recurrence or death from any cause",
  "fig_km_dfs")
fit_km_os <- km_figure(colon_df, "os_time", "os_status",
  "Kaplan-Meier estimates of overall survival",
  "Secondary endpoint: death from any cause",
  "fig_km_os")

# ------------------------------------------------------------------------------
# 2. Log-rank tests (overall and pairwise)
# ------------------------------------------------------------------------------
overall_lr <- survdiff(as.formula("Surv(dfs_time, dfs_status) ~ rx"), data = colon_df)
pair_lr <- combn(ARM_LEVELS, 2, simplify = FALSE)
lr_tab <- data.frame(
  comparison = c("Overall (3 df)", sapply(pair_lr, function(p) paste(p[2], "vs", p[1]))),
  statistic = c(overall_lr$chisq, sapply(pair_lr, function(p) {
    d <- colon_df[colon_df$rx %in% p, ]
    d$rx <- droplevels(factor(as.character(d$rx), levels = p))
    survdiff(Surv(dfs_time, dfs_status) ~ rx, data = d)$chisq
  })),
  p = c(1 - pchisq(overall_lr$chisq, 2), sapply(pair_lr, function(p) {
    d <- colon_df[colon_df$rx %in% p, ]
    d$rx <- droplevels(factor(as.character(d$rx), levels = p))
    1 - pchisq(survdiff(Surv(dfs_time, dfs_status) ~ rx, data = d)$chisq, 1)
  }))
) %>% mutate(p_fmt = fmt_p(p))
save_tab(lr_tab, "table_logrank")
print(lr_tab, row.names = FALSE)

# ------------------------------------------------------------------------------
# 3. Cox proportional-hazards models
# ------------------------------------------------------------------------------
covars_fml <- paste("rx + sex + age + obstruct + perfor + adhere + nodes +
                    differ + extent + surg", collapse = "")

# 3a. Unadjusted
fit_unadj <- coxph(Surv(dfs_time, dfs_status) ~ rx, data = colon_df, ties = "efron")
tidy_cox <- function(fit) {
  s <- summary(fit, conf.int = 0.95)
  data.frame(
    term = rownames(s$coefficients),
    HR = s$conf.int[, "exp(coef)"],
    lower = s$conf.int[, "lower .95"],
    upper = s$conf.int[, "upper .95"],
    se = s$coefficients[, "se(coef)"],
    z = s$coefficients[, "z"],
    p = s$coefficients[, "Pr(>|z|)"],
    row.names = NULL
  )
}
tab_unadj <- tidy_cox(fit_unadj) %>% mutate(across(where(is.numeric), ~ round(.x, 3)))
save_tab(tab_unadj, "table_cox_unadjusted")
print(tab_unadj, row.names = FALSE)

# 3b. Adjusted, complete case
fit_adj_cc <- coxph(as.formula(paste("Surv(dfs_time, dfs_status) ~", covars_fml)),
                    data = colon_df, ties = "efron", x = TRUE)
tab_adj_cc <- tidy_cox(fit_adj_cc) %>% mutate(across(where(is.numeric), ~ round(.x, 3)))
save_tab(tab_adj_cc, "table_cox_adjusted_cc")

# 3c. Adjusted, multiple imputation (Rubin's rules via mice::pool)
fit_adj_mi <- with(mi_dfs,
  coxph(Surv(dfs_time, dfs_status) ~ rx + sex + age + obstruct + perfor +
          adhere + nodes + differ + extent + surg, ties = "efron"))
pool_mi <- mice::pool(fit_adj_mi)
s_pool <- summary(pool_mi, exponentiate = TRUE, conf.int = 0.95)
tab_adj_mi <- s_pool %>%
  transmute(term, HR = estimate, lower = `2.5 %`, upper = `97.5 %`,
            p = p.value) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))
save_tab(tab_adj_mi, "table_cox_adjusted_mi")
print(tab_adj_mi, row.names = FALSE)

# 3d. Same for overall survival (adjusted, MI)
fit_os_mi <- with(mi_os,
  coxph(Surv(os_time, os_status) ~ rx + sex + age + obstruct + perfor +
          adhere + nodes + differ + extent + surg, ties = "efron"))
s_pool_os <- summary(mice::pool(fit_os_mi), exponentiate = TRUE, conf.int = 0.95)
tab_os_mi <- s_pool_os %>%
  transmute(term, HR = estimate, lower = `2.5 %`, upper = `97.5 %`,
            p = p.value) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))
save_tab(tab_os_mi, "table_cox_os_mi")

# ------------------------------------------------------------------------------
# 4. Proportional-hazards diagnostics
# ------------------------------------------------------------------------------
zph <- cox.zph(fit_adj_cc)
zph_tab <- as.data.frame(zph$table) %>%
  tibble::rownames_to_column("term") %>%
  mutate(p_fmt = fmt_p(p)) %>%
  select(term, chisq, df, p, p_fmt) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))
save_tab(zph_tab, "table_zph")
print(zph_tab, row.names = FALSE)

fig_zph <- data.frame(
  time = zph$x,
  y_rx = zph$y[, "rx"],
  y_nodes = zph$y[, "nodes"],
  y_obstruct = zph$y[, "obstruct"]
) %>%
  tidyr::pivot_longer(c(y_rx, y_nodes, y_obstruct),
                      names_to = "var", values_to = "resid") %>%
  mutate(var = factor(var, levels = c("y_rx", "y_nodes", "y_obstruct"),
                      labels = c("Treatment (rx)",
                                 "Positive nodes",
                                 "Obstruction (PH violation, p = 0.008)"))) %>%
  ggplot(aes(time, resid)) +
  geom_point(size = 0.8, alpha = 0.5, colour = "grey35") +
  geom_smooth(method = "loess", se = FALSE, colour = "#B22222", linewidth = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey20") +
  facet_wrap(~ var, scales = "free_y") +
  labs(title = "Proportional-hazards diagnostics",
       subtitle = "Scaled Schoenfeld residuals with loess smooth; a horizontal pattern supports proportional hazards",
       x = "Years since randomization", y = "Scaled Schoenfeld residual") +
  theme_trial
save_fig(fig_zph, "fig_zph", width = 7.5, height = 3.6)

# ------------------------------------------------------------------------------
# 5. Forest plot of the adjusted MI Cox model (DFS)
# ------------------------------------------------------------------------------
pretty_term <- function(t) {
  t <- gsub("rx", "", t); t <- gsub("^", "", t)
  recode <- c(
    "Lev" = "Levamisole vs observation",
    "Lev+5FU" = "Levamisole + 5-FU vs observation",
    "sexMale" = "Male vs female",
    "age" = "Age (per year)",
    "obstructYes" = "Obstruction: yes vs no",
    "perforYes" = "Perforation: yes vs no",
    "adhereYes" = "Adherence: yes vs no",
    "nodes" = "Positive nodes (per node)",
    "differModerate" = "Differentiation: moderate vs well",
    "differPoor" = "Differentiation: poor vs well",
    "extentMuscle" = "Extent: muscle vs submucosa",
    "extentSerosa" = "Extent: serosa vs submucosa",
    "extentContiguous structures" = "Extent: contiguous vs submucosa",
    "surgShort" = "Surgery interval: short vs long"
  )
  recode[t] %||% t
}

forest_df <- tab_adj_mi %>%
  filter(term != "(Intercept)") %>%
  mutate(label = pretty_term(term),
         sig = ifelse(lower > 1 | upper < 1, "Significant", "Not significant")) %>%
  arrange(HR)

fig_forest <- ggplot(forest_df, aes(HR, reorder(label, HR))) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey35") +
  geom_pointrange(aes(xmin = lower, xmax = upper, colour = sig),
                  fatten = 2, linewidth = 0.6) +
  scale_colour_manual(values = c("Significant" = "#B22222", "Not significant" = "grey35"),
                      guide = "none") +
  scale_x_log10() +
  annotation_logticks(sides = "b", size = 0.2) +
  labs(title = "Adjusted hazard ratios for disease-free survival",
       subtitle = "Multivariable Cox model, multiple imputation (m = 20), log scale",
       x = "Hazard ratio (95% CI, log scale)", y = NULL) +
  theme_trial +
  theme(axis.text.y = element_text(size = 8.5))
save_fig(fig_forest, "fig_forest_adjusted", width = 7, height = 5.6)

# ------------------------------------------------------------------------------
# 6. Absolute treatment benefit at 5 years
# ------------------------------------------------------------------------------
abs_5yr <- do.call(rbind, lapply(list(
  DFS = c("dfs_time", "dfs_status"),
  OS  = c("os_time", "os_status")), function(vs) {
  fit <- survfit(as.formula(sprintf("Surv(%s, %s) ~ rx", vs[1], vs[2])),
                 data = colon_df, conf.type = "log-log")
  s <- summary(fit, times = 5, extend = TRUE)
  data.frame(
    arm = gsub("^rx=", "", as.character(s$strata)),
    n_risk = s$n.risk,
    survival = s$surv,
    lower = s$lower,
    upper = s$upper
  )
})) %>%
  mutate(endpoint = rep(c("DFS", "OS"), each = length(ARM_LEVELS)),
         arm = factor(arm, levels = ARM_LEVELS))

abs_tab <- abs_5yr %>%
  transmute(endpoint, arm = ARM_LABELS[arm],
            `5yr_%` = round(100 * survival, 1),
            `5yr_lower%` = round(100 * lower, 1),
            `5yr_upper%` = round(100 * upper, 1))
save_tab(abs_tab, "table_km_5yr")
print(abs_tab, row.names = FALSE)

# Risk differences and NNT vs observation (bootstrap 95% CI)
set.seed(20260916)
rd_boot <- function(tvar, svar, idx) {
  d <- colon_df[idx, ]
  fit <- survfit(as.formula(sprintf("Surv(%s, %s) ~ rx", tvar, svar)), data = d)
  s <- summary(fit, times = 5, extend = TRUE)
  setNames(1 - s$surv, gsub("rx=", "", as.character(s$strata)))
}
rd_res <- t(sapply(1:2000, function(i) {
  rd_boot("dfs_time", "dfs_status", sample(nrow(colon_df), replace = TRUE))
}))
rd_tab <- data.frame(
  comparison = c("Lev vs Obs", "Lev+5FU vs Obs"),
  RD_5yr = c(mean(rd_res[, "Lev"] - rd_res[, "Obs"]),
             mean(rd_res[, "Lev+5FU"] - rd_res[, "Obs"])),
  lower = c(quantile(rd_res[, "Lev"] - rd_res[, "Obs"], .025),
            quantile(rd_res[, "Lev+5FU"] - rd_res[, "Obs"], .025)),
  upper = c(quantile(rd_res[, "Lev"] - rd_res[, "Obs"], .975),
            quantile(rd_res[, "Lev+5FU"] - rd_res[, "Obs"], .975))
) %>% mutate(NNT = round(1 / abs(RD_5yr), 1),
             across(where(is.numeric), ~ round(.x, 3)))
save_tab(rd_tab, "table_risk_difference")
print(rd_tab, row.names = FALSE)
message("  RD < 0 favours the active arm (fewer DFS events at 5 years)")

# ------------------------------------------------------------------------------
# 7. Competing-risk sensitivity analysis (recurrence endpoint)
# ------------------------------------------------------------------------------
# 7a. Aalen-Johansen cumulative incidence vs naive Kaplan-Meier
fit_aj <- survfit(Surv(rec_time, rec_cr_f) ~ rx, data = colon_df)
cif_df <- data.frame(
  time = fit_aj$time,
  cif = fit_aj$pstate[, "recurrence"],
  arm = rep(gsub("^rx=", "", names(fit_aj$strata)), fit_aj$strata)
) %>%
  mutate(arm = factor(arm, levels = ARM_LEVELS)) %>%
  bind_rows(data.frame(time = 0, cif = 0,
                       arm = factor(ARM_LEVELS, levels = ARM_LEVELS)))

naive_km <- survfit(Surv(rec_time, rec_status) ~ rx, data = colon_df)
naive_df <- data.frame(
  time = naive_km$time,
  cif = 1 - naive_km$surv,
  arm = rep(gsub("^rx=", "", names(naive_km$strata)), naive_km$strata)
) %>%
  mutate(arm = factor(arm, levels = ARM_LEVELS)) %>%
  bind_rows(data.frame(time = 0, cif = 0,
                       arm = factor(ARM_LEVELS, levels = ARM_LEVELS)))

fig_cif <- ggplot() +
  geom_step(data = cif_df, aes(time, cif, colour = arm), linewidth = 0.75) +
  geom_step(data = naive_df, aes(time, cif, colour = arm), linetype = "dashed",
            linewidth = 0.55, alpha = 0.65) +
  scale_colour_manual(values = ARM_COLS, labels = ARM_LABELS, drop = FALSE) +
  scale_x_continuous(breaks = 0:8) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Cumulative incidence of recurrence: competing-risk vs naive analysis",
       subtitle = "Solid: Aalen-Johansen cumulative incidence (death without recurrence is a competing event)\nDashed: one minus Kaplan-Meier (assumes death without recurrence is censoring)",
       x = "Years since randomization", y = "Cumulative incidence of recurrence") +
  theme_trial +
  theme(plot.subtitle = element_text(lineheight = 0.9))
save_fig(fig_cif, "fig_cif", width = 7, height = 4.6)

# 7b. Fine-Gray subdistribution hazard model
# NOTE: the RHS of finegray()'s formula specifies the covariates used for the
# censoring-distribution (IPCW) model, so it is restricted to the analysis
# covariates (passing `.` would leak event indicators and the patient id into
# the censoring model).
fg_dat <- finegray(Surv(rec_time, rec_cr_f) ~ rx + sex + age + obstruct +
                     perfor + adhere + nodes + differ + extent + surg,
                   data = colon_df, etype = "recurrence")
fit_fg <- coxph(Surv(fgstart, fgstop, fgstatus) ~ rx + sex + age + obstruct +
                  perfor + adhere + nodes + differ + extent + surg,
                data = fg_dat, weights = fgwt, ties = "efron")
tab_fg <- tidy_cox(fit_fg) %>%
  filter(grepl("^rx", term)) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)),
         term = pretty_term(term))
save_tab(tab_fg, "table_finegray")
print(tab_fg, row.names = FALSE)

# ------------------------------------------------------------------------------
# 8. Sensitivity: flexible functional form for age and nodes (splines vs linear)
# ------------------------------------------------------------------------------
cc <- colon_df[complete.cases(
  colon_df[, c("dfs_time", "dfs_status", "rx", "sex", "age", "obstruct",
               "perfor", "adhere", "nodes", "differ", "extent", "surg")]), ]
fit_lin <- coxph(Surv(dfs_time, dfs_status) ~ rx + sex + age + obstruct +
                   perfor + adhere + nodes + differ + extent + surg, data = cc)
fit_sp <- coxph(Surv(dfs_time, dfs_status) ~ rx + sex + pspline(age, df = 4) +
                  obstruct + perfor + adhere + pspline(nodes, df = 4) +
                  differ + extent + surg, data = cc)
sens_tab <- data.frame(
  model = c("Linear age and nodes (df = 1 each)",
            "Flexible splines (df = 4 each)"),
  loglik = round(c(fit_lin$loglik[2], fit_sp$loglik[2]), 2),
  AIC = c(round(AIC(fit_lin), 1), round(AIC(fit_sp), 1))
)
lrt <- anova(fit_lin, fit_sp)
sens_tab$LRT_p <- c(NA, fmt_p(lrt$`Pr(>|Chi|)`[2]))
save_tab(sens_tab, "table_sensitivity_splines")
print(sens_tab, row.names = FALSE)

message("\n=== 03_frequentist_analysis.R completed ===")
