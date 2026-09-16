# ==============================================================================
# 04_subgroup_analysis.R
#
# Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
#           with Frequentist, Bayesian, and Machine Learning Methods
# Author  : Cathy
# Purpose : Pre-specified subgroup analyses for the primary comparison
#           (Levamisole + 5-FU vs Observation) on disease-free survival:
#           subgroup-specific hazard ratios, tests of interaction, and a
#           summary forest plot with multiplicity considerations.
#
# Inputs  : data/processed/colon_analysis.csv
# Outputs : outputs/figures/fig_subgroup_forest.png
#           outputs/tables/table_subgroup.csv
# ==============================================================================

source("R/00_common.R")
set_global_seed()
message("\n=== 04_subgroup_analysis.R ===\n")

colon_df <- read.csv(file.path(DATA_PROC, "colon_analysis.csv")) %>%
  mutate(rx = factor(rx, levels = ARM_LEVELS),
         differ = factor(differ, levels = c("Well", "Moderate", "Poor")),
         extent = factor(extent, levels = c("Submucosa", "Muscle", "Serosa",
                                            "Contiguous structures"))) %>%
  # focus on the clinically effective comparison
  filter(rx %in% c("Obs", "Lev+5FU")) %>%
  mutate(treat = factor(ifelse(rx == "Lev+5FU", "Lev+5FU", "Obs"),
                        levels = c("Obs", "Lev+5FU")))
message(sprintf("Two-arm comparison subset: n = %d (Obs %d, Lev+5FU %d)",
                nrow(colon_df), sum(colon_df$treat == "Obs"),
                sum(colon_df$treat == "Lev+5FU")))

# ------------------------------------------------------------------------------
# 1. Subgroup definitions (pre-specified, clinically motivated)
# ------------------------------------------------------------------------------
subgroups <- list(
  list(name = "Age", var = "age_bin", levels = c("< 60 years", ">= 60 years"),
       label = "Age"),
  list(name = "Sex", var = "sex", levels = c("Male", "Female"), label = "Sex"),
  list(name = "Nodal status", var = "node4", levels = c("No", "Yes"),
       label = ">= 4 positive nodes"),
  list(name = "Obstruction", var = "obstruct", levels = c("No", "Yes"),
       label = "Obstruction"),
  list(name = "Local extent", var = "extent_grp",
       levels = c("Submucosa/muscle", "Serosa/contiguous"), label = "Local extent"),
  list(name = "Differentiation", var = "differ_grp",
       levels = c("Well/moderate", "Poor"), label = "Differentiation"),
  list(name = "Surgery interval", var = "surg", levels = c("Short", "Long"),
       label = "Surgery-to-registration")
)
colon_df <- colon_df %>%
  mutate(age_bin = ifelse(age < 60, "< 60 years", ">= 60 years"),
         extent_grp = ifelse(extent %in% c("Submucosa", "Muscle"),
                             "Submucosa/muscle", "Serosa/contiguous"),
         differ_grp = ifelse(differ == "Poor", "Poor", "Well/moderate"))

# ------------------------------------------------------------------------------
# 2. Subgroup-specific HRs, interaction tests, forest plot
# ------------------------------------------------------------------------------
fit_sub <- function(d) {
  f <- coxph(Surv(dfs_time, dfs_status) ~ treat, data = d, ties = "efron")
  s <- summary(fit <- f, conf.int = 0.95)
  data.frame(HR = s$conf.int[1, 1], lower = s$conf.int[1, 3],
             upper = s$conf.int[1, 4], p = s$coefficients[1, 5],
             n = nrow(d), events = sum(d$dfs_status))
}

rows <- list()
for (sg in subgroups) {
  d <- colon_df[!is.na(colon_df[[sg$var]]), ]
  # interaction test (1 df, Wald)
  f_int <- coxph(as.formula(sprintf(
    "Surv(dfs_time, dfs_status) ~ treat * %s + nodes", sg$var)),
    data = d, ties = "efron")
  int_terms <- grep(":", names(coef(f_int)), value = TRUE)
  wald <- coef(f_int)[int_terms] %*% solve(vcov(f_int)[int_terms, int_terms]) %*%
    coef(f_int)[int_terms]
  p_int <- 1 - pchisq(wald, length(int_terms))

  for (lv in sg$levels) {
    dl <- d[d[[sg$var]] == lv, ]
    if (nrow(dl) > 0 && sum(dl$dfs_status) > 0) {
      r <- fit_sub(dl)
      rows[[length(rows) + 1]] <- data.frame(
        subgroup = sg$label, level = lv, r, p_interaction = p_int)
    }
  }
}
sub_tab <- bind_rows(rows) %>%
  mutate(across(c(HR, lower, upper, p, p_interaction), ~ round(.x, 3))) %>%
  group_by(subgroup) %>%
  mutate(p_interaction = ifelse(row_number() == 1, p_interaction, NA)) %>%
  ungroup()

# Overall (all patients) row
r_all <- fit_sub(colon_df) %>%
  mutate(subgroup = "All patients", level = "", p_interaction = NA) %>%
  select(subgroup, level, HR, lower, upper, p, n, events, p_interaction)

sub_tab <- bind_rows(r_all, sub_tab) %>%
  mutate(across(where(is.numeric), ~ round(.x, 3)))
save_tab(sub_tab, "table_subgroup")
print(sub_tab, row.names = FALSE)

# ---- Forest plot -------------------------------------------------------------
fp_df <- sub_tab %>%
  mutate(row_idx = rev(seq_len(n())),
         label = ifelse(level == "", subgroup, paste0("   ", level)),
         txt = sprintf("%.2f (%.2f-%.2f)", HR, lower, upper))

fig_forest_sub <- ggplot(fp_df, aes(x = HR, y = reorder(label, row_idx))) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40") +
  geom_rect(aes(xmin = 0.30, xmax = 0.52, ymin = -Inf, ymax = Inf),
            fill = "grey96", alpha = 0.6) +
  geom_rect(aes(xmin = 1.75, xmax = 2.05, ymin = -Inf, ymax = Inf),
            fill = "grey96", alpha = 0.6) +
  geom_pointrange(aes(xmin = lower, xmax = upper), size = 0.32,
                  colour = "#B22222", fatten = 1.8) +
  geom_text(aes(x = 0.41, label = txt), hjust = 0, size = 2.7, colour = "grey15") +
  geom_text(aes(x = 1.90,
                label = ifelse(is.na(p_interaction), "",
                sprintf("int. p = %s", fmt_p(p_interaction)))),
            hjust = 1, size = 2.7, colour = "grey25") +
  scale_x_log10(limits = c(0.28, 2.2), breaks = c(0.3, 0.5, 0.7, 1, 1.5, 2)) +
  labs(title = "Subgroup analyses: disease-free survival, Lev+5-FU vs observation",
       subtitle = "Unadjusted Cox models within subgroups; interaction tests adjusted for nodal count. Right column: HR (95% CI)",
       x = "Hazard ratio (log scale)", y = NULL) +
  theme_trial +
  theme(axis.text.y = element_text(size = 9))
save_fig(fig_forest_sub, "fig_subgroup_forest", width = 7.5, height = 6)

message("\n=== 04_subgroup_analysis.R completed ===")
