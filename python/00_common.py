"""
00_common.py

Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
          with Frequentist, Bayesian, and Machine Learning Methods
Author  : Cathy
Purpose : Shared configuration for the Python analysis modules: paths,
          the Okabe-Ito colour-blind-safe palette, plotting style, and
          input/output helpers. Mirrors R/00_common.R for a coherent visual
          identity across the two languages.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---- Paths --------------------------------------------------------------------
FIG_DIR = "outputs/figures"
TAB_DIR = "outputs/tables"
DATA_PROC = "data/processed"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ---- Trial arm configuration (matches R/00_common.R) ---------------------------
ARM_LEVELS = ["Obs", "Lev", "Lev+5FU"]
ARM_LABELS = {
    "Obs": "Observation",
    "Lev": "Levamisole",
    "Lev+5FU": "Levamisole + 5-FU",
}
ARM_COLS = {
    "Obs": "#0072B2",
    "Lev": "#E69F00",
    "Lev+5FU": "#009E73",
}
TREATMENT_COLS = {"control": "#454545", "active": "#B22222"}
ACCENT = "#B22222"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
GREY = "#454545"

# ---- Global seed and version reporting ----------------------------------------
RANDOM_SEED = 20260916


def set_seed(seed: int = RANDOM_SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)


# ---- Plotting style ------------------------------------------------------------
def apply_style() -> None:
    """Apply a clean, academic ggplot-like style consistent with the R figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e2e2e2",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.autolayout": True,
    })


apply_style()


def save_fig(fig, name: str) -> None:
    """Save a matplotlib figure as PNG (README) and PDF (report)."""
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), facecolor="white")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), facecolor="white")
    print(f"  figure saved: {FIG_DIR}/{name}.png")
    plt.close(fig)


def save_tab(df: pd.DataFrame, name: str) -> None:
    """Save a tidy data frame as CSV in outputs/tables."""
    path = os.path.join(TAB_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  table saved: {path}")


def load_analysis_data() -> pd.DataFrame:
    """Load the patient-level analysis file produced by R/01_data_curation.R."""
    df = pd.read_csv(os.path.join(DATA_PROC, "colon_analysis.csv"))
    df["rx"] = pd.Categorical(df["rx"], categories=ARM_LEVELS)
    return df


def design_matrix(df: pd.DataFrame, drop_reference: bool = True) -> pd.DataFrame:
    """Build the modelling design matrix with clinically labelled columns."""
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"]
    X["nodes"] = df["nodes"]
    X["female"] = (df["sex"] == "Female").astype(int)
    X["obstruction"] = (df["obstruct"] == "Yes").astype(int)
    X["perforation"] = (df["perfor"] == "Yes").astype(int)
    X["adherence"] = (df["adhere"] == "Yes").astype(int)
    X["diff_moderate"] = (df["differ"] == "Moderate").astype(int)
    X["diff_poor"] = (df["differ"] == "Poor").astype(int)
    X["extent_muscle"] = (df["extent"] == "Muscle").astype(int)
    X["extent_serosa"] = (df["extent"] == "Serosa").astype(int)
    X["extent_contiguous"] = (df["extent"] == "Contiguous structures").astype(int)
    X["surgery_long"] = (df["surg"] == "Long").astype(int)
    return X.astype(float)


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "NA"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def percent_axis(ax, axis: str = "y") -> None:
    if axis == "y":
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    else:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
