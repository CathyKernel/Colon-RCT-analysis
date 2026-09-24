# ==============================================================================
# 05_trial_design.R
#
# Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
#           with Frequentist, Bayesian, and Machine Learning Methods
# Author  : Cathy
# Purpose : "Designing this trial today": (1) sample size for a two-arm
#           superiority trial on disease-free survival; (2) a group-sequential
#           design with O'Brien-Fleming efficacy spending and non-binding
#           beta-spending futility boundaries; (3) operating characteristics;
#           (4) a re-enactment of interim monitoring on the real trial data
#           (conditional power at the 50% interim analysis).
#
# Inputs  : data/processed/colon_analysis.csv
# Outputs : outputs/figures/fig_gs_boundaries.png, fig_power_curve.png,
#                     fig_gs_reenactment.png
#           outputs/tables/table_design_fixed.csv, table_design_gs.csv,
#                     table_design_oc.csv, table_gs_reenactment.csv,
#                     table_design_comparison.csv
# ==============================================================================

source("R/00_common.R")
suppressPackageStartupMessages(library(gsDesign))
set_global_seed()
message("\n=== 05_trial_design.R ===\n")

colon_df <- read.csv(file.path(DATA_PROC, "colon_analysis.csv")) %>%
  mutate(rx = factor(rx, levels = ARM_LEVELS)) %>%
  filter(rx %in% c("Obs", "Lev+5FU"))

# ------------------------------------------------------------------------------
# 1. Planning assumptions taken from the re-analysis
# ------------------------------------------------------------------------------
# Control-arm 5-year DFS and exponential hazard from the observed data
km_ctrl <- survfit(Surv(dfs_time, dfs_status) ~ rx, data = colon_df)
s_ctrl <- summary(km_ctrl, times = 5, extend = TRUE)
s5_ctrl <- s_ctrl$surv[as.character(s_ctrl$strata) == "rx=Obs"]
median_ctrl <- quantile(survfit(Surv(dfs_time, dfs_status) ~ 1,
  data = colon_df[colon_df$rx == "Obs", ]), 0.5)$quantile
lambda0 <- log(2) / as.numeric(median_ctrl)

HR_TARGET <- 0.65        # clinically meaningful target (close to observed 0.62)
ALPHA <- 0.025           # one-sided, standard for superiority
BETA <- 0.10             # 90% power
ACCRUAL_Y <- 3           # assumed accrual duration (years)
FUP_MIN <- 5             # minimum follow-up (years); total study T = 8y
RATIO <- 1               # 1:1 randomization

message(sprintf(
  "Assumptions: control 5-yr DFS = %.1f%%, median DFS = %.2f y, lambda0 = %.3f, HR = %.2f",
  100 * s5_ctrl, as.numeric(median_ctrl), lambda0, HR_TARGET))

# ------------------------------------------------------------------------------
# 2. Fixed-design sample size (Schoenfeld / Freedman approach via gsDesign)
# ------------------------------------------------------------------------------
fixed <- nSurv(lambdaC = lambda0, hr = HR_TARGET, hr0 = 1, etaE = 0,
               T = ACCRUAL_Y + FUP_MIN, minfup = FUP_MIN, ratio = RATIO,
               alpha = ALPHA, beta = BETA, sided = 1)
fixed_tab <- data.frame(
  Design = "Fixed design",
  Alpha = fixed$alpha, Power = 1 - fixed$beta,
  Events_required = ceiling(fixed$d),
  N_required = ceiling(fixed$n),
  HR_target = HR_TARGET,
  Study_duration = ACCRUAL_Y + FUP_MIN
)
save_tab(fixed_tab, "table_design_fixed")
print(fixed_tab, row.names = FALSE)

# Schoenfeld events formula, for transparency
ev_schoenfeld <- function(hr, alpha = ALPHA, power = 1 - BETA, p = 0.5) {
  4 * (qnorm(1 - alpha) + qnorm(power))^2 / log(hr)^2
}
message(sprintf("Schoenfeld events formula check: %.0f events (nSurv: %.0f)",
                ev_schoenfeld(HR_TARGET), ceiling(fixed$d)))

# ------------------------------------------------------------------------------
# 3. Group-sequential design (efficacy + non-binding futility)
# ------------------------------------------------------------------------------
gs <- gsSurv(
  k = 3, timing = c(0.5, 0.75), test.type = 4,
  alpha = ALPHA, beta = BETA, astar = 0,  # non-binding futility via beta spending
  sfu = sfLDOF, sfl = sfHSD, sflpar = -4,
  lambdaC = lambda0, hr = HR_TARGET, hr0 = 1, etaE = 0,
  T = ACCRUAL_Y + FUP_MIN, minfup = FUP_MIN, ratio = RATIO
)

gs_tab <- gsBoundSummary(gs) %>%
  as.data.frame() %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))
save_tab(gs_tab, "table_design_gs")
print(gs_tab, row.names = FALSE)

# Efficacy and futility boundaries on the HR scale at each analysis
hr_bounds <- t(sapply(1:gs$k, function(i) {
  c(analysis = i,
    events = ceiling(gs$n.I)[i],
    analysis_time_years = gs$T[i],
    eff_z = gs$upper$bound[i],
    eff_p = 1 - pnorm(gs$upper$bound[i]),
    eff_hr = gsHR(z = gs$upper$bound[i], i = i, x = gs, ratio = RATIO),
    fut_z = gs$lower$bound[i],
    fut_hr = gsHR(z = gs$lower$bound[i], i = i, x = gs, ratio = RATIO))
})) %>% as.data.frame()
hr_bounds[] <- lapply(hr_bounds, function(x) round(as.numeric(x), 3))
save_tab(hr_bounds, "table_design_oc")
print(hr_bounds, row.names = FALSE)

max_n_gs <- ceiling(sum(gs$gamma * gs$R))   # total planned enrolment
message(sprintf("Max events: %d | Max N: %d | Events saved vs fixed: %.0f (%.1f%%)",
                ceiling(max(gs$n.I)), max_n_gs,
                fixed$d - max(gs$n.I),
                100 * (fixed$d - max(gs$n.I)) / fixed$d))

# ---- Boundary plot (Z and p-value scale) -------------------------------------
f_seq <- gs$timing
if (length(f_seq) == gs$k - 1) f_seq <- c(f_seq, 1)
bnd <- data.frame(
  f = f_seq,
  eff_z = gs$upper$bound,
  fut_z = gs$lower$bound
) %>%
  mutate(eff_p = 1 - pnorm(eff_z), fut_p = 1 - pnorm(fut_z))

fig_gs <- ggplot(bnd, aes(x = f)) +
  geom_ribbon(aes(ymin = eff_z, ymax = 6), fill = "#B22222", alpha = 0.07) +
  geom_hline(yintercept = qnorm(1 - ALPHA), linetype = "dotted", colour = "grey30") +
  geom_line(aes(y = eff_z, colour = "Efficacy (O'Brien-Fleming spending)"),
            linewidth = 0.8) +
  geom_point(aes(y = eff_z), size = 1.8, colour = "#B22222") +
  geom_line(aes(y = fut_z, colour = "Futility (Hwang-Shih-DeCani, non-binding)"),
            linewidth = 0.8, linetype = "solid") +
  geom_point(aes(y = fut_z), size = 1.8, colour = "#0072B2") +
  geom_text(aes(y = eff_z, label = sprintf("p = %.4f", eff_p)),
            vjust = -1.0, size = 2.5, colour = "grey20") +
  scale_colour_manual(values = c("#B22222", "#0072B2"), name = NULL) +
  scale_x_continuous(labels = scales::percent, breaks = c(0.5, 0.75, 1)) +
  scale_y_continuous(limits = c(-1, 6), sec.axis = sec_axis(
    ~ 1 - pnorm(.), name = "One-sided p-value",
    breaks = c(0.5, 0.1, 0.01, 0.001), labels = c("0.5", "0.1", "0.01", "0.001"))) +
  labs(title = "Group-sequential design boundaries for the redesigned trial",
       subtitle = sprintf(
         "Two interim analyses at 50%% and 75%% of %d planned events; one-sided alpha = %.3f, 90%% power for HR = %.2f",
         ceiling(max(gs$n.I)), ALPHA, HR_TARGET),
       x = "Information fraction (proportion of planned DFS events)",
       y = "Z-statistic") +
  theme_trial
save_fig(fig_gs, "fig_gs_boundaries", width = 7, height = 4.8)

# ---- Sample size / power vs target HR ----------------------------------------
# Power of the FIXED design (sized for HR_TARGET, i.e. d_fixed events) as a
# function of the true hazard ratio, plus the event count that would be
# required to retain 90% power at each HR (Schoenfeld approximation).
hrs <- seq(0.55, 0.85, by = 0.025)
d_fixed <- ev_schoenfeld(HR_TARGET)          # events the fixed design accrues
power_curve <- sapply(hrs, function(h) {
  pnorm(((-log(h)) * sqrt(d_fixed) / 2) - qnorm(1 - ALPHA))
})
pw_df <- data.frame(HR = hrs, power = power_curve,
                    events = sapply(hrs, ev_schoenfeld))
save_tab(pw_df %>% mutate(across(where(is.numeric), ~ round(.x, 3))),
         "table_power_curve")

fig_pw <- ggplot(pw_df, aes(HR, power)) +
  geom_ribbon(aes(ymin = 0, ymax = power), fill = "#0072B2", alpha = 0.08) +
  geom_line(colour = "#0072B2", linewidth = 0.8) +
  geom_point(data = pw_df[pw_df$HR == 0.65, ], aes(HR, power), size = 2.5,
             colour = "#B22222") +
  geom_text(data = pw_df[pw_df$HR == 0.65, ],
            aes(HR, power, label = sprintf("HR = 0.65\n%.0f events", events)),
            vjust = -0.6, size = 3, colour = "#B22222") +
  geom_hline(yintercept = 0.9, linetype = "dashed", colour = "grey45") +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Power as a function of the target hazard ratio",
       subtitle = "Fixed design with 226 events (sized for HR = 0.65); annotations show events required for 90% power",
       x = "Target hazard ratio (treatment vs control)", y = "Power") +
  coord_cartesian(ylim = c(0, 1)) +
  theme_trial
save_fig(fig_pw, "fig_power_curve", width = 7, height = 4.4)

# ---- Comparison of spending functions ----------------------------------------
gs_obf  <- gsSurv(k = 3, timing = c(0.5, 0.75), test.type = 1, alpha = ALPHA,
                  beta = BETA, sfu = sfLDOF, lambdaC = lambda0, hr = HR_TARGET,
                  T = ACCRUAL_Y + FUP_MIN, minfup = FUP_MIN, ratio = RATIO)
gs_poc  <- gsSurv(k = 3, timing = c(0.5, 0.75), test.type = 1, alpha = ALPHA,
                  beta = BETA, sfu = sfLDPocock, lambdaC = lambda0,
                  hr = HR_TARGET, T = ACCRUAL_Y + FUP_MIN, minfup = FUP_MIN,
                  ratio = RATIO)
comp_tab <- data.frame(
  Design = c("Fixed design", "Group-sequential (O'Brien-Fleming)",
             "Group-sequential (Pocock)",
             "Group-sequential (OBF efficacy + futility)"),
  Max_events = c(ceiling(fixed$d), rep(ceiling(max(gs_obf$n.I)), 2),
                 ceiling(max(gs$n.I))),
  Final_alpha_spent = c(ALPHA, ALPHA, ALPHA, ALPHA),
  Final_efficacy_Z = c(qnorm(1 - ALPHA), gs_obf$upper$bound[3],
                       gs_poc$upper$bound[3], gs$upper$bound[3])
) %>% mutate(across(where(is.numeric), ~ round(.x, 3)))
save_tab(comp_tab, "table_design_comparison")
print(comp_tab, row.names = FALSE)

# ------------------------------------------------------------------------------
# 4. Re-enactment: interim monitoring on the real trial data
# ------------------------------------------------------------------------------
# Suppose the trial had been run with the redesigned monitoring plan: the first
# interim analysis is triggered when the planned number of events (ceiling of
# gs$n.I[1]) has accrued in the ACTUAL trial data. Because the public dataset
# lacks calendar dates, the interim snapshot is approximated by censoring every
# patient's follow-up at the time (years since randomization) of the n_int-th
# event; patients not yet eventing are administratively censored at that time.
# NOTE: gsDesign uses the convention that positive Z favours the experimental
# arm; for a hazard ratio below 1 we therefore analyse Z = -log(HR_hat)/SE.
d2 <- colon_df %>% mutate(treat = as.integer(rx == "Lev+5FU"))
n_int <- ceiling(gs$n.I)[1]              # planned events at interim 1
t_sorted <- sort(d2$dfs_time[d2$dfs_status == 1])
t_int <- t_sorted[n_int]                 # follow-up time of the n_int-th event
d_int <- d2 %>%
  mutate(
    dfs_time_i = pmin(dfs_time, t_int),
    dfs_status_i = as.integer(dfs_status == 1 & dfs_time <= t_int)
  )
fit_int <- coxph(Surv(dfs_time_i, dfs_status_i) ~ treat, data = d_int,
                 ties = "efron")
hr_int <- exp(coef(fit_int))
z_int <- -as.numeric(coef(fit_int)) / sqrt(vcov(fit_int)[1, 1])  # benefit > 0
info_frac <- n_int / ceiling(max(gs$n.I))

# Conditional power for the remaining analyses under three scenarios:
# (a) design alternative HR = 0.65, (b) observed trend, (c) null.
cp_fun <- function(theta, I_i, I_K, z_i, bound_K) {
  # conditional probability of crossing the final bound (normal approximation,
  # single remaining look formula; adequate because the final bound dominates)
  zK <- bound_K
  mu <- theta * (I_K - I_i)
  se <- sqrt(I_K - I_i)
  1 - pnorm((zK * sqrt(I_K) - z_i * sqrt(I_i) - mu) / se)
}
I_full <- ceiling(max(gs$n.I)) / 4  # information for log-HR ~ d / (p(1-p)), p = 1/2
I_int <- n_int / 4
bnd_final <- gs$upper$bound[3]
cp_tab <- data.frame(
  scenario = c("Null effect (HR = 1)", "Design alternative (HR = 0.65)",
               "Observed trend at interim"),
  conditional_power = c(
    cp_fun(0, I_int, I_full, z_int, bnd_final),
    cp_fun(-log(HR_TARGET), I_int, I_full, z_int, bnd_final),
    cp_fun(-log(hr_int), I_int, I_full, z_int, bnd_final)
  ),
  at_interim = sprintf("HR = %.2f, Z = %.2f, %d events (%.0f%% of planned information)",
                       hr_int, z_int, n_int, 100 * info_frac)
)
cp_tab$conditional_power <- round(cp_tab$conditional_power, 3)
save_tab(cp_tab, "table_gs_reenactment")
print(cp_tab, row.names = FALSE)

# Cross-check with gsDesign's own conditional power machinery
gs_check <- tryCatch({
  cp <- gsCP(x = gs, theta = c(0, gs$delta), i = 1, zi = z_int)
  # cp$upper$prob: (k-1) x n.theta matrix of crossing probabilities for the
  # remaining analyses, conditional on the interim result
  colSums(cp$upper$prob)
}, error = function(e) rep(NA_real_, 2))
message(sprintf(
  "gsCP cross-check: CP under null = %s | CP under design alternative = %s",
  ifelse(is.na(gs_check[1]), "unavailable", sprintf("%.3f", gs_check[1])),
  ifelse(is.na(gs_check[2]), "unavailable", sprintf("%.3f", gs_check[2]))))

# Decision at the interim under the redesign
eff_at_int <- gs$upper$bound[1]
fut_at_int <- gs$lower$bound[1]
decision <- ifelse(z_int >= eff_at_int, "STOP for efficacy",
            ifelse(z_int <= fut_at_int, "STOP for futility",
                   "CONTINUE (Z between futility and efficacy boundaries)"))
message(sprintf("\nRe-enacted interim decision: %s (Z = %.2f vs efficacy %.2f / futility %.2f)",
                decision, z_int, eff_at_int, fut_at_int))

# ---- Visual re-enactment -----------------------------------------------------
fig_re <- ggplot(bnd, aes(x = f)) +
  geom_line(aes(y = eff_z, colour = "Efficacy boundary"), linewidth = 0.8) +
  geom_line(aes(y = fut_z, colour = "Futility boundary"), linewidth = 0.8) +
  geom_point(aes(x = info_frac, y = z_int, colour = "Observed interim result"),
             size = 3, shape = 18) +
  geom_segment(aes(x = info_frac, xend = info_frac, y = -1, yend = z_int),
               linetype = "dotted", colour = "grey40") +
  geom_segment(aes(x = 0, xend = info_frac, y = z_int, yend = z_int),
               linetype = "dotted", colour = "grey40") +
  scale_colour_manual(values = c("Efficacy boundary" = "#B22222",
                                 "Futility boundary" = "#0072B2",
                                 "Observed interim result" = "#009E73"),
                      name = NULL) +
  scale_x_continuous(labels = scales::percent) +
  labs(title = "Re-enacted interim monitoring using the real trial data",
       subtitle = sprintf(
         "Interim after %d planned DFS events (Lev+5FU vs Obs): HR = %.2f, Z = %.2f -> %s",
         n_int, hr_int, z_int, decision),
       x = "Information fraction", y = "Z-statistic") +
  coord_cartesian(ylim = c(-1.5, 5)) +
  theme_trial
save_fig(fig_re, "fig_gs_reenactment", width = 7, height = 4.8)

message("\n=== 05_trial_design.R completed ===")
