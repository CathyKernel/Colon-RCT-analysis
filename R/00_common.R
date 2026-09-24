# ==============================================================================
# 00_common.R
#
# Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
#           with Frequentist, Bayesian, and Machine Learning Methods
# Author  : Cathy
# Purpose : Shared configuration: paths, labels, colour palette, plotting
#           theme, and helper functions used by all analysis scripts.
#
# Note    : All scripts are intended to be run from the repository root, e.g.
#             Rscript R/01_data_curation.R
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
  library(patchwork)
  library(survival)
})

# ---- Paths -------------------------------------------------------------------
FIG_DIR <- "outputs/figures"
TAB_DIR <- "outputs/tables"
DATA_RAW <- "data/raw"
DATA_PROC <- "data/processed"
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TAB_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- Trial arm configuration -------------------------------------------------
ARM_LEVELS <- c("Obs", "Lev", "Lev+5FU")
ARM_LABELS <- c(
  "Obs"     = "Observation",
  "Lev"     = "Levamisole",
  "Lev+5FU" = "Levamisole + 5-FU"
)
# Okabe-Ito colour-blind-safe palette
ARM_COLS <- c(
  "Obs"     = "#0072B2",
  "Lev"     = "#E69F00",
  "Lev+5FU" = "#009E73"
)
TREATMENT_COLS <- c("#454545", "#B22222")  # control / active in two-group plots

# ---- Variable labels ---------------------------------------------------------
VAR_LABELS <- c(
  rx      = "Treatment arm",
  sex     = "Sex",
  age     = "Age (years)",
  obstruct= "Obstruction",
  perfor  = "Perforation",
  adhere  = "Adherence to nearby organs",
  nodes   = "Positive lymph nodes",
  differ  = "Tumour differentiation",
  extent  = "Extent of local spread",
  surg    = "Timing of surgery to registration",
  node4   = ">= 4 positive nodes",
  time    = "Time (days)",
  status  = "Event indicator"
)

# ---- ggplot theme (academic, colour-blind safe) ------------------------------
theme_trial <- theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(linewidth = 0.25, colour = "grey88"),
    strip.background = element_rect(fill = "grey95", colour = NA),
    strip.text = element_text(face = "bold", size = 10),
    legend.position = "bottom",
    legend.title = element_blank(),
    axis.title = element_text(size = 10),
    plot.title = element_text(size = 11, face = "bold"),
    plot.subtitle = element_text(size = 9, colour = "grey30"),
    plot.caption = element_text(size = 7.5, colour = "grey40")
  )

# ---- Helpers -----------------------------------------------------------------
`%||%` <- function(a, b) if (is.null(a)) b else a

# Save a ggplot object as both PNG (for README) and PDF (for the report).
save_fig <- function(plot, name, width = 7, height = 5, dpi = 300) {
  stopifnot(grepl("\\.(png|pdf)$", name)[1] || TRUE)
  png_path <- file.path(FIG_DIR, paste0(name, ".png"))
  pdf_path <- file.path(FIG_DIR, paste0(name, ".pdf"))
  ggsave(png_path, plot, width = width, height = height, dpi = dpi, bg = "white")
  ggsave(pdf_path, plot, width = width, height = height, bg = "white")
  message("  figure saved: ", png_path)
  invisible(plot)
}

# Write a tidy data frame as CSV in outputs/tables.
save_tab <- function(df, name) {
  path <- file.path(TAB_DIR, paste0(name, ".csv"))
  write.csv(df, path, row.names = FALSE, na = "")
  message("  table saved: ", path)
  invisible(df)
}

# Round to fixed digits while keeping numeric class.
rounddf <- function(df, digits = 3) {
  df[] <- lapply(df, function(x) if (is.numeric(x)) round(x, digits) else x)
  df
}

# Format p-values in APA-ish style.
fmt_p <- function(p) {
  ifelse(is.na(p), NA_character_,
  ifelse(p < 0.001, "< 0.001",
  ifelse(p < 0.01, sprintf("%.3f", p),
  sprintf("%.2f", p))))
}

# ---- Reproducibility ---------------------------------------------------------
set_global_seed <- function(seed = 20260916) {
  set.seed(seed)
  invisible(seed)
}
