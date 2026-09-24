# Makefile - one-command reproduction of the full analysis
#
# Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
# Author  : Cathy
#
# Usage:
#   make all        # run the complete pipeline (R then Python)
#   make r          # run R modules 01-06
#   make python     # run Python modules 02-04
#   make module     # e.g. make module R=03_frequentist_analysis.R
#   make clean      # remove generated outputs (figures/tables/processed data)
#
# Runtime: ~20-25 minutes in total; R/06_simulation_study.R dominates.

RSCRIPT := Rscript
PYTHON  := python3

R_MODULES := R/01_data_curation.R \
             R/02_descriptive.R \
             R/03_frequentist_analysis.R \
             R/04_subgroup_analysis.R \
             R/05_trial_design.R \
             R/06_simulation_study.R

PY_MODULES := python/02_machine_learning.py \
              python/03_bayesian_analysis.py \
              python/04_cate_heterogeneity.py \
              python/05_rmst_analysis.py \
              python/06_figure_rebuilds.py

.PHONY: all r python clean module

all: r python
        @echo "=== Full pipeline complete: see outputs/ and report/ ==="

r:
        @for f in $(R_MODULES); do \
                echo "=== Running $$f ==="; \
                $(RSCRIPT) $$f || exit 1; \
        done

python: r
        @for f in $(PY_MODULES); do \
                echo "=== Running $$f ==="; \
                $(PYTHON) $$f || exit 1; \
        done

module:
        @if [ -n "$(R)" ]; then $(RSCRIPT) R/$(R); fi
        @if [ -n "$(PY)" ]; then $(PYTHON) python/$(PY); fi

clean:
        rm -f outputs/figures/*.png outputs/figures/*.pdf
        rm -f outputs/tables/*.csv
        rm -f data/processed/*
        @echo "Generated outputs removed. Run 'make all' to regenerate."
