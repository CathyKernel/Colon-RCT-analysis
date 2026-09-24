# ==============================================================================
# 02_descriptive.R
#
# Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
#           with Frequentist, Bayesian, and Machine Learning Methods
# Author  : Cathy
# Purpose : Baseline characteristics by randomized arm (Table 1), global and
#           pairwise balance diagnostics, and standardized mean differences.
#
# Inputs  : data/processed/colon_analysis.csv
# Outputs : outputs/tables/table1_baseline.csv
#           outputs/tables/smd_pairwise.csv
#           outputs/figures/fig_smd_loveplot.png
# ==============================================================================

source("R/00_common.R")
set_global_seed()
message("\n=== 02_descriptive.R ===\n")

colon_df <- read.csv(file.path(DATA_PROC, "colon_analysis.csv")) %>%
  mutate(rx = factor(rx, levels = ARM_LEVELS),
         across(c(sex, obstruct, perfor, adhere, differ, extent, surg, node4,
                  age_group, nodes_cat), function(x) {x[is.na(x)] <- NA; x}))

# Helper: dichotomize a "Yes" level as numeric for SMD computation
yes01 <- function(f) as.integer(f == "Yes")

# ------------------------------------------------------------------------------
# 1. Table 1: baseline characteristics by randomized arm
# ------------------------------------------------------------------------------
n_arm <- table(colon_df$rx)

row_cat <- function(var, label, show_missing = TRUE) {
  tab <- table(colon_df[[var]], colon_df$rx, useNA = "ifany")
  lev <- rownames(tab)
  out <- lapply(seq_along(lev), function(i) {
    n_i <- tab[i, ]
    pct <- 100 * n_i / n_arm
    if (lev[i] %in% c(NA_character_, "")) {
      lv <- "Missing"
    } else lv <- lev[i]
    data.frame(
      Characteristic = if (i == 1) label else "",
      Level = lv,
      Obs = sprintf("%d (%.1f)", n_i[1], pct[1]),
      Lev = sprintf("%d (%.1f)", n_i[2], pct[2]),
      `Lev+5FU` = sprintf("%d (%.1f)", n_i[3], pct[3]),
      check.names = FALSE, stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

row_num <- function(var, label) {
  s <- tapply(colon_df[[var]], colon_df$rx, function(x)
    c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE),
      med = median(x, na.rm = TRUE), q1 = quantile(x, .25, na.rm = TRUE),
      q3 = quantile(x, .75, na.rm = TRUE)))
  data.frame(
    Characteristic = label, Level = "",
    Obs = sprintf("%.1f (%.1f)", s[[1]]["mean"], s[[1]]["sd"]),
    Lev = sprintf("%.1f (%.1f)", s[[2]]["mean"], s[[2]]["sd"]),
    `Lev+5FU` = sprintf("%.1f (%.1f)", s[[3]]["mean"], s[[3]]["sd"]),
    check.names = FALSE, stringsAsFactors = FALSE
  )
}

rows_num_iqr <- function(var, label) {
  s <- tapply(colon_df[[var]], colon_df$rx, function(x)
    c(med = unname(median(x, na.rm = TRUE)),
      q1 = unname(quantile(x, .25, na.rm = TRUE)),
      q3 = unname(quantile(x, .75, na.rm = TRUE))))
  data.frame(
    Characteristic = label, Level = "",
    Obs = sprintf("%.0f [%.0f-%.0f]", s[[1]]["med"], s[[1]]["q1"], s[[1]]["q3"]),
    Lev = sprintf("%.0f [%.0f-%.0f]", s[[2]]["med"], s[[2]]["q1"], s[[2]]["q3"]),
    `Lev+5FU` = sprintf("%.0f [%.0f-%.0f]", s[[3]]["med"], s[[3]]["q1"], s[[3]]["q3"]),
    check.names = FALSE, stringsAsFactors = FALSE
  )
}

table1 <- rbind(
  data.frame(Characteristic = "Randomized (n)", Level = "",
             Obs = as.character(n_arm[1]), Lev = as.character(n_arm[2]),
             `Lev+5FU` = as.character(n_arm[3]), check.names = FALSE),
  row_num("age", "Age, years, mean (SD)"),
  row_cat("age_group", "Age group, n (%)"),
  row_cat("sex", "Sex, n (%)"),
  row_cat("obstruct", "Obstruction, n (%)"),
  row_cat("perfor", "Perforation, n (%)"),
  row_cat("adhere", "Adherence to organs, n (%)"),
  rows_num_iqr("nodes", "Positive nodes, median [IQR]"),
  row_cat("nodes_cat", "Node category, n (%)"),
  row_cat("differ", "Differentiation, n (%)"),
  row_cat("extent", "Local extent, n (%)"),
  row_cat("surg", "Surgery-to-registration, n (%)"),
  # outcome accountability
  {
    ev <- sapply(ARM_LEVELS, function(a) {
      d <- colon_df[colon_df$rx == a, ]
      sprintf("%d (%.1f)", sum(d$dfs_status), 100 * mean(d$dfs_status))
    })
    data.frame(Characteristic = "DFS events, n (%)", Level = "",
               Obs = ev[1], Lev = ev[2], `Lev+5FU` = ev[3], check.names = FALSE)
  },
  {
    ev <- sapply(ARM_LEVELS, function(a) {
      d <- colon_df[colon_df$rx == a, ]
      sprintf("%d (%.1f)", sum(d$os_status), 100 * mean(d$os_status))
    })
    data.frame(Characteristic = "Deaths, n (%)", Level = "",
               Obs = ev[1], Lev = ev[2], `Lev+5FU` = ev[3], check.names = FALSE)
  }
)

save_tab(table1, "table1_baseline")
message("Table 1 (first 12 rows):"); print(head(table1, 12), row.names = FALSE)

# ------------------------------------------------------------------------------
# 2. Global homogeneity tests (randomization balance diagnostics)
# ------------------------------------------------------------------------------
tests <- list(
  Age = broom::tidy(aov(age ~ rx, data = colon_df)) %>% filter(term == "rx"),
  Nodes = broom::tidy(kruskal.test(nodes ~ rx, data = colon_df))
)
chi_vars <- c("sex", "obstruct", "perfor", "adhere", "differ", "extent",
              "surg", "age_group", "nodes_cat")
chi_tests <- lapply(chi_vars, function(v) {
  tab <- table(colon_df[[v]], colon_df$rx)
  f <- tryCatch(fisher.test(tab, simulate.p.value = TRUE, B = 10000),
                error = function(e) chisq.test(tab))
  if (inherits(f, "htest")) {
    data.frame(variable = v, p = f$p.value)
  }
})
balance_tests <- rbind(
  data.frame(variable = "age (ANOVA)", p = tests$Age$p.value),
  data.frame(variable = "nodes (Kruskal-Wallis)", p = tests$Nodes$p.value),
  do.call(rbind, chi_tests)
) %>% mutate(p = as.numeric(p), p_fmt = fmt_p(p))
save_tab(balance_tests, "qc_balance_tests")
message("\nRandomization balance tests (all p-values):")
print(balance_tests, row.names = FALSE)

# ------------------------------------------------------------------------------
# 3. Standardized mean differences (pairwise, love plot)
# ------------------------------------------------------------------------------
# For categorical variables we use the binary-expansion approach common in
# propensity-score work: each level indicator contributes one SMD, summarized
# by the maximum over levels (Yang & Dalton 2012).
smd_num <- function(x, g1, g2) {
  x1 <- x[g1]; x2 <- x[g2]
  m <- function(z) mean(z, na.rm = TRUE)
  v <- function(z) var(z, na.rm = TRUE)
  abs(m(x1) - m(x2)) / sqrt((v(x1) + v(x2)) / 2)
}
smd_cat <- function(f, g1, g2) {
  p1 <- table(factor(f[g1])) / sum(g1)
  p2 <- table(factor(f[g2])) / sum(g2)
  lev <- union(names(p1), names(p2))
  p1 <- p1[lev]; p2 <- p2[lev]; p1[is.na(p1)] <- 0; p2[is.na(p2)] <- 0
  max(abs(p1 - p2) / sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2), na.rm = TRUE)
}

pairs_ <- list(
  c("Lev", "Obs"), c("Lev+5FU", "Obs"), c("Lev+5FU", "Lev")
)
smd_rows <- do.call(rbind, lapply(pairs_, function(pr) {
  g1 <- colon_df$rx == pr[1]; g2 <- colon_df$rx == pr[2]
  data.frame(
    variable = c("Age", "Male sex", "Obstruction", "Perforation", "Adherence",
                 "Positive nodes", "Poor differentiation", "Serosal/contiguous extent",
                 "Short surgery interval"),
    smd = c(
      smd_num(colon_df$age, g1, g2),
      smd_cat(as.character(colon_df$sex == "Male"), g1, g2),
      smd_num(yes01(colon_df$obstruct), g1, g2),
      smd_num(yes01(colon_df$perfor), g1, g2),
      smd_num(yes01(colon_df$adhere), g1, g2),
      smd_num(colon_df$nodes, g1, g2),
      smd_cat(as.character(colon_df$differ == "Poor"), g1, g2),
      smd_cat(as.character(colon_df$extent %in% c("Serosa", "Contiguous structures")), g1, g2),
      smd_num(yes01(factor(ifelse(colon_df$surg == "Short", 1, 0),
                           levels = c(0, 1), labels = c("No", "Yes"))), g1, g2)
    ),
    comparison = sprintf("%s vs %s", pr[1], pr[2])
  )
}))
save_tab(rounddf(smd_rows, 3), "smd_pairwise")

fig_smd <- ggplot(smd_rows,
                  aes(x = smd, y = reorder(variable, smd), colour = comparison)) +
  geom_vline(xintercept = c(-0.1, 0.1), linetype = "dashed", colour = "grey45") +
  geom_point(size = 2.2, position = position_dodge(width = 0.55)) +
  scale_colour_manual(values = c("#0072B2", "#D55E00", "#009E73")) +
  labs(title = "Baseline covariate balance across randomized arms",
       subtitle = "Absolute standardized mean differences; dashed lines mark |SMD| = 0.10",
       x = "Absolute standardized mean difference", y = NULL) +
  coord_cartesian(xlim = c(0, 0.25)) +
  theme_trial
save_fig(fig_smd, "fig_smd_loveplot", width = 7, height = 5)

message(sprintf("\nMax |SMD| across all pairwise comparisons: %.3f",
                max(smd_rows$smd, na.rm = TRUE)))
message("\n=== 02_descriptive.R completed ===")
