# ==============================================================================
# 01_data_curation.R
#
# Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
#           with Frequentist, Bayesian, and Machine Learning Methods
# Author  : Cathy
# Purpose : Import the raw trial data, run integrity and quality checks,
#           construct patient-level analysis datasets for the two co-primary
#           endpoints (disease-free survival and overall survival), profile
#           missing data, and create multiple imputations for adjusted models.
#
# Inputs  : data/raw/colon.csv
# Outputs : data/processed/colon_analysis.csv   (patient-level analysis file)
#           data/processed/colon_mi_dfs.rds     (mice object, DFS endpoint)
#           data/processed/colon_mi_os.rds      (mice object, OS endpoint)
#           outputs/tables/qc_*.csv             (quality-check tables)
#           outputs/figures/fig_missingness.png (missing-data profile)
# ==============================================================================

source("R/00_common.R")
suppressPackageStartupMessages(library(mice))
set_global_seed()
message("\n=== 01_data_curation.R ===\n")

# ------------------------------------------------------------------------------
# 1. Import and structure
# ------------------------------------------------------------------------------
raw <- read.csv(file.path(DATA_RAW, "colon.csv"), stringsAsFactors = FALSE)
raw$rownames <- NULL                      # drop artefact column from CSV export

stopifnot(nrow(raw) == 1858)              # 929 patients x 2 event types
stopifnot(length(unique(raw$study)) == 1) # single trial
message(sprintf("Raw records: %d rows | %d patients | %d variables",
                nrow(raw), length(unique(raw$id)), ncol(raw)))

# ---- Integrity check: covariates identical across the two records per patient
id_vars <- c("rx", "sex", "age", "obstruct", "perfor", "adhere",
             "nodes", "differ", "extent", "surg", "node4")
dups <- raw %>% group_by(id) %>% summarise(n_distinct = across(all_of(id_vars), n_distinct),
                                           .groups = "drop") %>%
  mutate(inconsistent = rowSums(across(-id, ~ .x > 1)))
stopifnot(max(dups$inconsistent) == 0)
message("Covariates consistent across duplicated patient records: PASS")

# ---- Pivot the two event-type records into one row per patient
wide <- raw %>%
  select(id, all_of(id_vars), etype, time, status) %>%
  pivot_wider(id_cols = c("id", all_of(id_vars)), names_from = etype,
              values_from = c(time, status), names_sep = "_e") %>%
  rename(dfs_time = time_e1, dfs_status = status_e1,
         os_time  = time_e2, os_status  = status_e2)

# ---- Cross-endpoint structure and composite DFS construction
# In this dataset `etype == 1` records time to RECURRENCE, with death before
# recurrence treated as censoring (a competing event). Because modern adjuvant
# colon-cancer trials define disease-free survival (DFS) as recurrence OR
# death, we construct a composite DFS endpoint and additionally retain the
# competing-risk coding for sensitivity analyses (Aalen-Johansen / Fine-Gray).
stopifnot(all(wide$dfs_time <= wide$os_time))
n_death_no_rec <- sum(wide$dfs_status == 0 & wide$os_status == 1)
stopifnot(all(wide$dfs_time[wide$dfs_status == 0 & wide$os_status == 1] ==
              wide$os_time[wide$dfs_status == 0 & wide$os_status == 1]))
message(sprintf(
  paste("Deaths without preceding recurrence (competing events): %d;",
        "composite DFS events = %d"),
  n_death_no_rec, sum(wide$dfs_status | wide$os_status)))

# ------------------------------------------------------------------------------
# 2. Recode variables with clinically meaningful labels
# ------------------------------------------------------------------------------
colon_df <- wide %>%
  transmute(
    id,
    rx        = factor(rx, levels = ARM_LEVELS),
    sex       = factor(sex, levels = c(1, 0), labels = c("Male", "Female")),
    age,
    age_group = cut(age, breaks = c(0, 59, 69, Inf),
                    labels = c("< 60", "60-69", ">= 70")),
    obstruct  = factor(obstruct, levels = c(0, 1), labels = c("No", "Yes")),
    perfor    = factor(perfor,   levels = c(0, 1), labels = c("No", "Yes")),
    adhere    = factor(adhere,   levels = c(0, 1), labels = c("No", "Yes")),
    nodes,
    nodes_cat = cut(nodes, breaks = c(-1, 0, 3, Inf),
                    labels = c("0", "1-3", ">= 4")),
    differ    = factor(differ, levels = 1:3,
                       labels = c("Well", "Moderate", "Poor")),
    extent    = factor(extent, levels = 1:4,
                       labels = c("Submucosa", "Muscle", "Serosa",
                                  "Contiguous structures")),
    surg      = factor(surg, levels = c(1, 0), labels = c("Short", "Long")),
    node4     = factor(node4, levels = c(0, 1), labels = c("No", "Yes")),
    # --- endpoints (years) ----------------------------------------------------
    # NOTE: compute recurrence variables BEFORE dfs_status is redefined as the
    # composite (recurrence or death) event indicator.
    rec_time  = dfs_time / 365.25,              # recurrence only
    rec_status = as.integer(dfs_status),
    rec_cr    = ifelse(dfs_status == 1, 1L,     # 1 = recurrence
                ifelse(os_status == 1, 2L, 0L)),# 2 = competing death, 0 = censored
    dfs_time  = dfs_time / 365.25,              # composite DFS: time
    dfs_status = as.integer(dfs_status | os_status),  # recurrence or death
    os_time   = os_time / 365.25,               # overall survival
    os_status
  )

stopifnot(!any(is.na(colon_df$rx) | is.na(colon_df$dfs_time) | is.na(colon_df$dfs_status)))
stopifnot(all(colon_df$dfs_time > 0))

# ------------------------------------------------------------------------------
# 3. Randomization balance and event summary (quality checks)
# ------------------------------------------------------------------------------
qc_balance <- colon_df %>%
  group_by(rx) %>%
  summarise(n = n(),
            dfs_events = sum(dfs_status),
            recurrences = sum(rec_status),
            deaths = sum(os_status),
            median_os_time = round(median(os_time), 2), .groups = "drop")
save_tab(qc_balance, "qc_randomization_balance")
print(qc_balance)

# Median follow-up via reverse Kaplan-Meier (censoring treated as events)
rev_km <- function(time, status) {
  q <- quantile(survfit(Surv(time, 1 - status) ~ 1), 0.5)
  as.numeric(q$quantile)
}
med_fu <- rev_km(colon_df$dfs_time, colon_df$dfs_status)
message(sprintf("Median follow-up (reverse KM): %.2f years", med_fu))

# ------------------------------------------------------------------------------
# 4. Missing-data profile
# ------------------------------------------------------------------------------
miss_prop <- colMeans(is.na(colon_df %>% select(-id, -dfs_time, -dfs_status,
                                                -os_time, -os_status)))
miss_tab <- tibble(
  variable = names(miss_prop),
  n_missing = colSums(is.na(colon_df[, names(miss_prop)])),
  pct_missing = round(100 * miss_prop, 2)
) %>% filter(n_missing > 0) %>%
  mutate(label = VAR_LABELS[variable])
save_tab(miss_tab, "qc_missingness")
print(miss_tab)
message(sprintf("Complete cases: %d / %d (%.1f%%)",
                sum(complete.cases(colon_df)), nrow(colon_df),
                100 * mean(complete.cases(colon_df))))

fig_missing <- colon_df %>%
  arrange(rx, age) %>%
  mutate(row = row_number()) %>%
  select(row, nodes, differ) %>%
  pivot_longer(cols = c(nodes, differ), names_to = "var", values_to = "val",
               values_transform = as.character) %>%
  mutate(var = factor(var, levels = c("nodes", "differ"),
                      labels = c("Positive lymph nodes", "Tumour differentiation"))) %>%
  ggplot(aes(x = row %% 2, y = row, colour = is.na(val))) +
  geom_point(size = 0.4, na.rm = TRUE) +
  scale_colour_manual(values = c("grey55", "#B22222"),
                      labels = c("Observed", "Missing"), name = NULL) +
  facet_wrap(~ var) +
  labs(title = "Missing-data pattern",
       subtitle = "Rows ordered by treatment arm and age; only variables with missing values shown",
       x = NULL, y = "Patient index") +
  scale_x_continuous(breaks = NULL) +
  guides(colour = guide_legend(override.aes = list(size = 3))) +
  theme_trial
save_fig(fig_missing, "fig_missingness", width = 7, height = 4)

# ------------------------------------------------------------------------------
# 5. Multiple imputation (FCS / mice)
# ------------------------------------------------------------------------------
# Following White & Royston (2009) and Sterne et al. (2009), the imputation
# models include the Nelson-Aalen cumulative hazard estimate and the event
# indicator of each endpoint, so that imputation is compatible with a
# subsequent Cox model on either DFS or OS.
make_mi <- function(dat, time_var, status_var, m = 20, seed = 20260916) {
  d <- dat
  # Remove the patient identifier and deterministic recodings of variables
  # already in the model: the identifier is pure noise as a predictor, and
  # the derived categories would be imputed independently of their
  # definitions. Neither is used by the downstream Cox models.
  d[c("id", "age_group", "nodes_cat", "node4")] <- NULL
  # Nelson-Aalen cumulative hazard evaluated at each patient's observed time
  haz <- basehaz(coxph(Surv(d[[time_var]], d[[status_var]]) ~ 1, ties = "efron"),
                 centered = FALSE)
  d$cumhaz <- approx(haz$time, haz$hazard, xout = d[[time_var]], rule = 2)$y
  d$ev <- d[[status_var]]

  mice(d, m = m, seed = seed, printFlag = FALSE,
       defaultMethod = c("pmm", "logreg", "polyreg", "polr"))
}

message("\nImputing missing values (m = 20, FCS with Nelson-Aalen predictors)...")
mi_dfs <- make_mi(colon_df, "dfs_time", "dfs_status")
mi_os  <- make_mi(colon_df, "os_time",  "os_status")

# Convergence / plausibility check: distributions of observed vs imputed values
chk <- mi_dfs$data %>%
  select(nodes, differ) %>%
  mutate(source = "Observed") %>%
  bind_rows(
    complete(mi_dfs, 1) %>% select(nodes, differ) %>%
      filter(is.na(mi_dfs$data$nodes) | is.na(mi_dfs$data$differ)) %>%
      mutate(source = "Imputed (m = 1)")
  )
chk_summary <- chk %>% group_by(source) %>%
  summarise(mean_nodes = mean(nodes, na.rm = TRUE),
            sd_nodes = sd(nodes, na.rm = TRUE),
            pct_poor_diff = mean(differ == "Poor", na.rm = TRUE), .groups = "drop")
save_tab(rounddf(chk_summary, 3), "qc_imputation_plausibility")
print(chk_summary)

saveRDS(mi_dfs, file.path(DATA_PROC, "colon_mi_dfs.rds"))
saveRDS(mi_os,  file.path(DATA_PROC, "colon_mi_os.rds"))

# ------------------------------------------------------------------------------
# 6. Export patient-level analysis file (missing values retained)
# ------------------------------------------------------------------------------
write.csv(colon_df, file.path(DATA_PROC, "colon_analysis.csv"), row.names = FALSE)
message(sprintf("\nSaved %d patients to data/processed/colon_analysis.csv",
                nrow(colon_df)))
message(sprintf("Event rates: composite DFS %.1f%% (n = %d) | recurrence %.1f%% | death %.1f%%",
                100 * mean(colon_df$dfs_status), sum(colon_df$dfs_status),
                100 * mean(colon_df$rec_status), 100 * mean(colon_df$os_status)))
message("\n=== 01_data_curation.R completed ===")
