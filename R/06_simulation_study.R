# ==============================================================================
# 06_simulation_study.R
#
# Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
#           with Frequentist, Bayesian, and Machine Learning Methods
# Author  : Cathy
# Purpose : Monte Carlo simulation studies that quantify the operating
#           characteristics behind the design choices:
#           Study A - efficiency of covariate adjustment in a randomized trial
#                     (variance, power, type I error) as a function of the
#                     prognostic strength of baseline covariates;
#           Study B - simulated operating characteristics of the module-5
#                     group-sequential design (type I error, power, stopping
#                     probabilities, expected events);
#           Study C - coverage of Wald vs bootstrap confidence intervals for
#                     the hazard ratio when proportional hazards is violated.
#
# Inputs  : data/processed/colon_analysis.csv
# Outputs : outputs/figures/fig_sim_efficiency.png, fig_sim_gs_oc.png,
#                     fig_sim_coverage.png
#           outputs/tables/table_sim_A.csv, table_sim_B.csv,
#                     table_sim_C.csv
#
# Runtime: approximately 8-12 minutes on a single core.
# ==============================================================================

source("R/00_common.R")
set_global_seed()
message("\n=== 06_simulation_study.R ===\n")

colon_df <- read.csv(file.path(DATA_PROC, "colon_analysis.csv")) %>%
  mutate(rx = factor(rx, levels = ARM_LEVELS),
         differ = factor(differ, levels = c("Well", "Moderate", "Poor")),
         extent = factor(extent, levels = c("Submucosa", "Muscle", "Serosa",
                                            "Contiguous structures")))

# Covariate pool for simulations: complete cases only (avoids NA propagation)
COLON_CC <- colon_df[complete.cases(
  colon_df[, c("age", "nodes", "obstruct", "perfor", "adhere", "sex",
               "differ", "extent")]), ]

# ---- Calibrated data-generating mechanism ------------------------------------
# Coefficients from the complete-case adjusted Cox model of module 3 (real data)
BETA_TRUE <- c(
  age = log(1.002), obstruct = log(1.250), perfor = log(1.068),
  adhere = log(1.171), nodes = log(1.081)
)
BETA_FACTOR <- list(
  sexFemale = log(1.091),
  differModerate = log(1.027), differPoor = log(1.338),
  extentMuscle = log(1.124), extentSerosa = log(1.875),
  `extentContiguous structures` = log(3.148)
)
LAMBDA0 <- log(2) / 2.96      # control exponential hazard (per year)
ACCRUAL_T <- 3                # years of uniform accrual
STUDY_T <- 8                  # administrative censoring time

# Build the prognostic linear predictor from resampled real covariates
make_linpred <- function(dat, strength) {
  lp <- with(dat,
    BETA_TRUE[["age"]] * strength * (age - 60) +
    BETA_TRUE[["nodes"]] * strength * nodes +
    BETA_TRUE[["obstruct"]] * strength * (obstruct == "Yes") +
    BETA_TRUE[["perfor"]] * strength * (perfor == "Yes") +
    BETA_TRUE[["adhere"]] * strength * (adhere == "Yes") +
    BETA_FACTOR$sexFemale * strength * (sex == "Female") +
    BETA_FACTOR$differModerate * strength * (differ == "Moderate") +
    BETA_FACTOR$differPoor * strength * (differ == "Poor") +
    BETA_FACTOR$extentMuscle * strength * (extent == "Muscle") +
    BETA_FACTOR$extentSerosa * strength * (extent == "Serosa") +
    BETA_FACTOR$`extentContiguous structures` * strength *
      (extent == "Contiguous structures"))
  as.numeric(lp)
}

# Simulate one replicate trial
sim_trial <- function(n, hr, strength, covs_for_adjust) {
  base <- COLON_CC[sample(nrow(COLON_CC), n, replace = TRUE), ]
  base$accrual <- runif(n, 0, ACCRUAL_T)
  lp <- make_linpred(base, strength)
  treat <- rep(0:1, length.out = n)              # alternating allocation
  lambda <- LAMBDA0 * exp(lp + log(hr) * treat)  # hr < 1 => lower hazard if treated
  T_event <- rexp(n, rate = lambda)
  T_obs <- pmin(T_event + base$accrual, STUDY_T) - base$accrual
  T_obs <- pmax(T_obs, 1 / 365.25)
  status <- as.integer(T_event + base$accrual <= STUDY_T)
  data.frame(time = T_obs, status = status, treat = treat,
             accrual = base$accrual,
             age = base$age, nodes = base$nodes,
             obstruct = base$obstruct, perfor = base$perfor,
             adhere = base$adhere, sex = base$sex,
             differ = base$differ, extent = base$extent)
}

fit_pair <- function(d) {
  # tryCatch guards against rare degenerate resamples (e.g. a factor level
  # absent by chance); such replicates are recorded as NA and excluded
  tryCatch({
    f1 <- coxph(Surv(time, status) ~ treat, data = d, ties = "efron")
    f2 <- coxph(Surv(time, status) ~ treat + age + nodes + obstruct + perfor +
                  adhere + sex + differ + extent, data = d, ties = "efron")
    lr <- survdiff(Surv(time, status) ~ treat, data = d)
    c(
      b_un = as.numeric(coef(f1)), se_un = as.numeric(sqrt(vcov(f1))),
      b_ad = as.numeric(coef(f2))[1], se_ad = sqrt(vcov(f2))[1, 1],
      z_lr = sqrt(lr$chisq) * sign(lr$exp[2] - lr$obs[2])  # signed log-rank Z
    )
  }, error = function(e) c(b_un = NA, se_un = NA, b_ad = NA, se_ad = NA,
                           z_lr = NA))
}

# ------------------------------------------------------------------------------
# Study A: efficiency of covariate adjustment
# ------------------------------------------------------------------------------
NSIM_A <- 1000
N_TRIAL <- 322
HR_A <- 0.72     # deliberately moderate effect: room to display power gains

message(sprintf("\nStudy A: %d replicates per scenario (n = %d, HR = %.2f)",
                NSIM_A, N_TRIAL, HR_A))
scen_A <- expand.grid(strength = c(0, 0.5, 1),
                      hr = c(1.0, HR_A)) %>%
  mutate(strength_lab = c("No prognostic covariates",
                          "Moderate prognostic strength",
                          "Full trial-calibrated strength")[match(strength, c(0, 0.5, 1))])

resA <- lapply(seq_len(nrow(scen_A)), function(s) {
  set.seed(20260916 + s)
  st <- scen_A$strength[s]; hr <- scen_A$hr[s]
  out <- t(replicate(NSIM_A, fit_pair(sim_trial(N_TRIAL, hr, st))))
  out <- out[!is.na(out[, "b_un"]), , drop = FALSE]
  data.frame(
    strength = st, strength_lab = scen_A$strength_lab[s], hr = hr,
    n_valid = nrow(out),
    power_un = mean(out[, "b_un"] / out[, "se_un"] < -qnorm(0.975)),
    power_ad = mean(out[, "b_ad"] / out[, "se_ad"] < -qnorm(0.975)),
    power_lr = mean(out[, "z_lr"] > qnorm(0.975)),
    sd_un = sd(out[, "b_un"]), mean_se_un = mean(out[, "se_un"]),
    sd_ad = sd(out[, "b_ad"]), mean_se_ad = mean(out[, "se_ad"]),
    # NB: the unadjusted Cox model targets the marginal HR, which is attenuated
    # toward 1 relative to the conditional HR because the hazard ratio is
    # non-collapsible; this is a key part of the Study A narrative
    bias_un = mean(out[, "b_un"]) - log(hr),
    bias_ad = mean(out[, "b_ad"]) - log(hr),
    events = NA
  )
})
resA <- do.call(rbind, resA)
# events per scenario (expectation, from one extra large run)
resA$events <- sapply(seq_len(nrow(scen_A)), function(s) {
  set.seed(99 + s)
  mean(replicate(50, sum(sim_trial(N_TRIAL, scen_A$hr[s],
                                   scen_A$strength[s])$status)))
})
tabA <- resA %>% mutate(across(where(is.numeric), ~ round(.x, 3))) %>%
  select(-strength)
save_tab(tabA, "table_sim_A")
print(tabA, row.names = FALSE)

# Prognostic strength descriptor: SD of the untreated linear predictor
lp_sd <- sapply(c(0, 0.5, 1), function(st) sd(make_linpred(COLON_CC, st)))
message("SD of prognostic index (none / half / full): ",
        paste(round(lp_sd, 3), collapse = " / "))

figA <- resA %>%
  filter(hr == HR_A) %>%
  transmute(strength_lab = factor(strength_lab, levels = unique(strength_lab)),
            Unadjusted = power_un, Adjusted = power_ad) %>%
  pivot_longer(c(Unadjusted, Adjusted), names_to = "Analysis", values_to = "power") %>%
  ggplot(aes(strength_lab, power, fill = Analysis)) +
  geom_col(position = position_dodge(0.75), width = 0.7) +
  geom_hline(yintercept = 0.8, linetype = "dashed", colour = "grey40") +
  scale_fill_manual(values = c("Unadjusted" = "#0072B2", "Adjusted" = "#009E73")) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(title = "Simulation Study A: power gain from covariate adjustment",
       subtitle = sprintf("%d replicates per scenario, n = %d, target HR = %.2f, one-sided alpha = 0.025; log-rank (dashed-free) is equivalent to unadjusted Cox",
                          NSIM_A, N_TRIAL, HR_A),
       x = "Prognostic strength of baseline covariates",
       y = "Empirical power") +
  theme_trial
save_fig(figA, "fig_sim_efficiency", width = 7, height = 4.6)

# ------------------------------------------------------------------------------
# Study B: operating characteristics of the group-sequential design
# ------------------------------------------------------------------------------
NSIM_B <- 3000
# event triggers and boundaries are read from the module-5 design output
oc <- read.csv(file.path(TAB_DIR, "table_design_oc.csv"))
EV_TRIGGER <- ceiling(oc$events)
EFF_B <- oc$eff_z
FUT_B <- oc$fut_z
message(sprintf("Design: events %s | efficacy Z %s | futility Z %s",
                paste(EV_TRIGGER, collapse = "/"),
                paste(round(EFF_B, 2), collapse = "/"),
                paste(round(FUT_B, 2), collapse = "/")))

# One simulated trial under the monitoring plan. Interim datasets are created
# by cutting the full data at the calendar time at which the planned number
# of events has accrued (patients still event-free are administratively
# censored at the cut).
sim_gs <- function(hr, adhere_futility = TRUE) {
  full <- sim_trial(335, hr, 1)
  cal_event <- sort(full$time[full$status == 1] + full$accrual)
  if (length(cal_event) < EV_TRIGGER[1]) {
    return(c(analysis = 3, decision = "continue", events = length(cal_event)))
  }
  for (k in 1:3) {
    m <- min(EV_TRIGGER[k], length(cal_event))
    cut <- cal_event[m]
    dk <- full %>%
      mutate(time_k = pmin(time, cut - accrual),
             status_k = as.integer(status == 1 & time + accrual <= cut)) %>%
      filter(time_k > 0)
    f <- suppressWarnings(tryCatch(
      coxph(Surv(time_k, status_k) ~ treat, data = dk, ties = "efron"),
      error = function(e) NULL))
    z <- if (is.null(f)) 0 else
      -as.numeric(coef(f)) / sqrt(vcov(f)[1, 1])
    if (z >= EFF_B[k])
      return(c(analysis = k, decision = "efficacy", events = m))
    if (k < 3 && adhere_futility && z <= FUT_B[k])
      return(c(analysis = k, decision = "futility", events = m))
    if (k == 3) return(c(analysis = 3, decision = "continue", events = m))
  }
}

message(sprintf("\nStudy B: %d replicates per scenario (group-sequential design)", NSIM_B))
set.seed(31415)
B_h1 <- replicate(NSIM_B, sim_gs(0.65), simplify = FALSE)
set.seed(27182)
B_h0 <- replicate(NSIM_B, sim_gs(1.0), simplify = FALSE)
set.seed(14142)
B_h0nb <- replicate(NSIM_B, sim_gs(1.0, adhere_futility = FALSE),
                    simplify = FALSE)

make_oc_tab <- function(sims, hr_lab) {
  ana <- as.numeric(sapply(sims, `[[`, "analysis"))
  dec <- sapply(sims, `[[`, "decision")
  ev <- as.numeric(sapply(sims, `[[`, "events"))
  eff <- dec == "efficacy"; fut <- dec == "futility"
  data.frame(
    scenario = hr_lab,
    efficacy_IA1 = mean(ana == 1 & eff),
    efficacy_IA2 = mean(ana == 2 & eff),
    efficacy_final = mean(ana == 3 & eff),
    total_efficacy = mean(eff),
    futility_IA1 = mean(ana == 1 & fut),
    futility_IA2 = mean(ana == 2 & fut),
    expected_events = mean(ev),
    expected_events_saved = 235 - mean(ev)
  )
}
tabB <- rbind(
  make_oc_tab(B_h1, "H1: HR = 0.65 (as designed)"),
  make_oc_tab(B_h0, "H0: HR = 1.00 (futility adhered)"),
  make_oc_tab(B_h0nb, "H0: HR = 1.00 (futility ignored)")
) %>% mutate(across(where(is.numeric), ~ round(.x, 3)))
save_tab(tabB, "table_sim_B")
print(tabB, row.names = FALSE)

# Design-theory comparison (from module 5 gsBoundSummary)
message("Design theory (gsDesign, non-binding): P(efficacy) H1 = 0.900 | H0 = 0.024")

figB <- tabB %>%
  filter(scenario != "H0: HR = 1.00 (futility ignored)") %>%
  mutate(scenario = factor(scenario, levels = c("H0: HR = 1.00 (futility adhered)",
                                                "H1: HR = 0.65 (as designed)"))) %>%
  pivot_longer(c(efficacy_IA1, efficacy_IA2, efficacy_final,
                 futility_IA1, futility_IA2),
               names_to = "stage", values_to = "prob") %>%
  mutate(type = ifelse(grepl("efficacy", stage), "Efficacy", "Futility"),
         stage = gsub("efficacy_IA|futility_IA", "IA ", stage),
         stage = ifelse(grepl("final", stage), "Final", stage),
         stage = factor(stage, levels = c("IA 1", "IA 2", "Final"))) %>%
  ggplot(aes(stage, prob, fill = type)) +
  geom_col(position = position_stack(), width = 0.62) +
  facet_wrap(~ scenario) +
  scale_fill_manual(values = c("Efficacy" = "#B22222", "Futility" = "#0072B2")) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Simulation Study B: simulated stopping probabilities of the group-sequential design",
       subtitle = sprintf("%d simulated trials per scenario; bars show probability of stopping at each analysis by reason", NSIM_B),
       x = NULL, y = "Probability") +
  theme_trial
save_fig(figB, "fig_sim_gs_oc", width = 7.5, height = 4.4)

# ------------------------------------------------------------------------------
# Study C: CI coverage under non-proportional hazards
# ------------------------------------------------------------------------------
NSIM_C <- 500
B_BOOT <- 100
message(sprintf("\nStudy C: %d replicates, %d bootstrap resamples (time-varying effect)", NSIM_C, B_BOOT))

sim_trial_tv <- function(n, b1 = log(0.45), b2 = log(0.80)) {
  # piecewise-constant log-hazard ratio: strong early effect, weaker later
  base <- COLON_CC[sample(nrow(COLON_CC), n, replace = TRUE), ]
  base$accrual <- runif(n, 0, ACCRUAL_T)
  lp <- make_linpred(base, 1)
  treat <- rep(0:1, length.out = n)
  T1 <- rexp(n, LAMBDA0 * exp(lp + b1 * treat))     # event time in era 1 (< 1y)
  T2 <- rexp(n, LAMBDA0 * exp(lp + b2 * treat))     # fresh clock for era 2
  # memorylessness: those surviving year 1 restart with the era-2 hazard
  T_event <- ifelse(T1 <= 1, T1, 1 + T2)
  T_obs <- pmin(T_event + base$accrual, STUDY_T) - base$accrual
  T_obs <- pmax(T_obs, 1 / 365.25)
  status <- as.integer(T_event + base$accrual <= STUDY_T)
  data.frame(time = T_obs, status = status, treat = treat)
}

set.seed(16180)
resC <- t(sapply(seq_len(NSIM_C), function(i) {
  d <- sim_trial_tv(335)
  f <- suppressWarnings(coxph(Surv(time, status) ~ treat, data = d,
                              ties = "efron"))
  b <- unname(as.numeric(coef(f)))
  se <- as.numeric(sqrt(vcov(f)))
  wald_lo <- b - 1.96 * se; wald_hi <- b + 1.96 * se
  # bootstrap percentile CI
  boots <- replicate(B_BOOT, {
    db <- d[sample(nrow(d), replace = TRUE), ]
    fb <- tryCatch(suppressWarnings(
      coxph(Surv(time, status) ~ treat, data = db, ties = "efron")),
      error = function(e) NULL)
    if (is.null(fb)) NA else unname(as.numeric(coef(fb)))
  })
  bt <- quantile(boots, c(.025, .975), na.rm = TRUE)
  c(b = b, wald_lo = wald_lo, wald_hi = wald_hi,
    boot_lo = unname(bt[1]), boot_hi = unname(bt[2]))
}))

# simulation-based estimand: mean estimate over all replicates
estimand <- mean(resC[, "b"])
tabC <- data.frame(
  method = c("Cox Wald CI", "Bootstrap percentile CI"),
  mean_logHR = round(c(mean(resC[, "b"]), mean(resC[, "b"])), 3),
  estimand_logHR = round(estimand, 3),
  empirical_sd = round(c(sd(resC[, "b"]), sd(resC[, "b"])), 3),
  coverage = round(c(
    mean(resC[, "wald_lo"] <= estimand & estimand <= resC[, "wald_hi"]),
    mean(resC[, "boot_lo"] <= estimand & estimand <= resC[, "boot_hi"])), 3),
  mean_width = round(c(mean(resC[, "wald_hi"] - resC[, "wald_lo"]),
                       mean(resC[, "boot_hi"] - resC[, "boot_lo"])), 3)
)
save_tab(tabC, "table_sim_C")
print(tabC, row.names = FALSE)

idx <- seq_len(NSIM_C)
figC_df <- rbind(
  data.frame(rep = idx, type = "Wald CI",
             lo = resC[, "wald_lo"], hi = resC[, "wald_hi"]),
  data.frame(rep = idx, type = "Bootstrap CI",
             lo = resC[, "boot_lo"], hi = resC[, "boot_hi"])
)
figC <- ggplot(figC_df, aes(x = rep, colour = type)) +
  geom_hline(yintercept = estimand, linetype = "dashed", colour = "grey20") +
  geom_errorbar(aes(ymin = lo, ymax = hi), position = position_dodge(0.6),
                linewidth = 0.35) +
  scale_colour_manual(values = c("Wald CI" = "#0072B2",
                                 "Bootstrap CI" = "#E69F00")) +
  labs(title = "Simulation Study C: interval estimates under non-proportional hazards",
       subtitle = sprintf("True effect is time-varying (HR 0.45 in year 1, HR 0.80 thereafter); dashed line = simulation estimand (%d replicates shown)", NSIM_C),
       x = "Simulation replicate", y = "log hazard ratio") +
  theme_trial
save_fig(figC, "fig_sim_coverage", width = 7, height = 4.4)

message("\n=== 06_simulation_study.R completed ===")
