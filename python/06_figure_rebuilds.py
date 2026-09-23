"""
06_figure_rebuilds.py

Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
          with Frequentist, Bayesian, and Machine Learning Methods
Author  : Cathy
Purpose : Publication-grade rebuilds of report figures whose original
          ggplot renderings had layout defects:

            1. fig_km_dfs / fig_km_os -- the at-risk table drew all three
               arms at the same y position, so the three numbers at every
               time point were printed on top of each other and illegible;
               round 2: left padding added so the t = 0 number clears the
               arm-label column, and all table text enlarged;
            2. fig_subgroup_forest    -- the in-panel "HR (95% CI)" and
               interaction-p annotations were placed inside the plotting
               area, where they collided with the point estimates and
               confidence intervals of low-HR subgroups; round 2: axes
               shifted right so the bold "Local extent" group name clears
               the "Submucosa/muscle" level label;
            3. fig_zph                -- the third facet title was truncated
               by the panel border, and the x axis displayed cox.zph's
               KM-transformed time (range 0-0.5) under a mislabelled
               "Years since randomization" axis;
            4. fig_sim_efficiency     -- round 2: rebuilt from
               table_sim_A.csv because the R/ggsave rendering clipped the
               long subtitle at the device edge ("...is equiva");
            5. fig_cif                -- round 3: the original ggplot
               rendering crowded the rotated y-axis title against the
               percentage tick labels and offered no explicit key for the
               two estimator line styles; rebuilt from the analysis data
               with the Aalen-Johansen estimator and the naive
               one-minus-Kaplan-Meier computed directly (validated against
               the report text: 38 competing deaths, AJ below naive for
               every arm, and an observation-arm gap of about one
               percentage point at eight years).

          The corresponding R scripts have been corrected in parallel
          (R/03_frequentist_analysis.R, R/04_subgroup_analysis.R).  Because
          the report figures must be exactly reproducible without an R
          runtime, this module regenerates the figures from the same
          data and saved outputs and is the authoritative source for the
          figures embedded in report/main.tex.

Inputs  : data/processed/colon_analysis.csv
          outputs/tables/table_subgroup.csv
          outputs/tables/table_sim_A.csv
Outputs : outputs/figures/fig_km_dfs.{png,pdf}
          outputs/figures/fig_km_os.{png,pdf}
          outputs/figures/fig_subgroup_forest.{png,pdf}
          outputs/figures/fig_zph.{png,pdf}
          outputs/figures/fig_sim_efficiency.{png,pdf}
          outputs/figures/fig_cif.{png,pdf}
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "python")
from importlib import import_module

common = import_module("00_common")

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.transforms import blended_transform_factory
from lifelines import CoxPHFitter
from statsmodels.nonparametric.smoothers_lowess import lowess

ARM_LEVELS = common.ARM_LEVELS          # ["Obs", "Lev", "Lev+5FU"]
ARM_LABELS = common.ARM_LABELS
ARM_COLS = common.ARM_COLS
ACCENT = common.ACCENT                  # "#B22222"
GREY_TEXT = "#484848"
Z95 = 1.959963984540054

# Reference values from outputs/tables/table_km_5yr.csv (fractions), used
# to validate the reimplemented Kaplan-Meier estimator and its log-log bands.
KM5_REF = {
    ("DFS", "Obs"): (0.424, 0.369, 0.478),
    ("DFS", "Lev"): (0.442, 0.386, 0.496),
    ("DFS", "Lev+5FU"): (0.592, 0.534, 0.645),
    ("OS", "Obs"): (0.526, 0.469, 0.579),
    ("OS", "Lev"): (0.535, 0.478, 0.589),
    ("OS", "Lev+5FU"): (0.634, 0.577, 0.685),
}


# ------------------------------------------------------------------------------
# 1. Kaplan-Meier machinery (matches R survfit with conf.type = "log-log")
# ------------------------------------------------------------------------------
def km_estimate(time, event):
    """KM survival curve with pointwise log-log 95% confidence bands.

    Returns arrays (t, S, lower, upper) starting at (0, 1, 1, 1).
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    ts, Ss, los, his = [0.0], [1.0], [1.0], [1.0]
    S, V = 1.0, 0.0
    for tt in np.unique(time[event == 1]):
        d = int(np.sum((time == tt) & (event == 1)))
        nr = int(np.sum(time >= tt))
        S *= 1.0 - d / nr
        V = V + d / (nr * (nr - d)) if nr > d else np.inf
        if not np.isfinite(V):
            lo, hi = 0.0, 1.0
        elif S <= 0.0:
            lo = hi = 0.0
        elif S >= 1.0:
            lo = hi = 1.0
        else:
            theta = np.log(-np.log(S))
            se = np.sqrt(V) / abs(np.log(S))
            hi = np.exp(-np.exp(theta - Z95 * se))
            lo = np.exp(-np.exp(theta + Z95 * se))
        ts.append(float(tt))
        Ss.append(S)
        los.append(lo)
        his.append(hi)
    return (np.array(ts), np.array(Ss), np.array(los), np.array(his))


def km_at(km, t):
    """Step-function lookup of (S, lower, upper) at time t."""
    ts, Ss, los, his = km
    i = np.searchsorted(ts, t, side="right") - 1
    i = int(np.clip(i, 0, len(ts) - 1))
    return Ss[i], los[i], his[i]


def n_at_risk(time, t):
    return int(np.sum(np.asarray(time, dtype=float) >= t))


def km_figure(df, tvar, svar, endpoint, title, subtitle, name):
    """KM curves with confidence bands + a one-row-per-arm risk table."""
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.0, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.6]}, layout="constrained")

    max_t = float(df[tvar].max())
    km_by_arm = {}
    for arm in ARM_LEVELS:
        sub = df[df["rx"] == arm]
        km = km_estimate(sub[tvar].values, sub[svar].values)
        km_by_arm[arm] = km
        c = ARM_COLS[arm]
        t, S, lo, hi = km
        ax.step(t, S, where="post", color=c, lw=1.5, label=ARM_LABELS[arm])
        ax.fill_between(t, lo, hi, step="post", color=c, alpha=0.12, lw=0)

    # Small left padding (shared by both panels) so that the risk-table
    # number at t = 0 sits fully inside the axes instead of spilling into
    # the arm-label column and colliding with it.
    ax.set_xlim(-0.6, max_t)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Survival probability")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper right", fontsize=10.5)
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.set_title(subtitle, fontsize=10, color=GREY_TEXT)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ---- risk table: one row per arm (the original drew all arms at the
    #      same y position, which stacked the three numbers on top of each
    #      other and made them unreadable) -----------------------------------
    times = np.arange(0, 9)
    n_arms = len(ARM_LEVELS)
    for i, arm in enumerate(ARM_LEVELS):
        sub = df[df["rx"] == arm]
        y = n_arms - 1 - i                       # Obs on top, Lev+5FU bottom
        vals = [n_at_risk(sub[tvar].values, tt) for tt in times]
        for tt, v in zip(times, vals):
            axr.text(tt, y, str(v), ha="center", va="center",
                     fontsize=11.5, color=ARM_COLS[arm])
    axr.set_yticks(range(n_arms))
    axr.set_yticklabels([ARM_LABELS[a] for a in ARM_LEVELS], fontsize=10.5)
    axr.tick_params(axis="y", pad=4)
    for lab, arm in zip(axr.get_yticklabels(), ARM_LEVELS):
        lab.set_color(ARM_COLS[arm])
    axr.set_ylim(-0.55, n_arms - 0.45)
    axr.set_xlim(-0.6, max_t)
    axr.set_xticks(times)
    axr.tick_params(axis="x", labelsize=10.5)
    axr.set_xlabel("Years since randomization", fontsize=11)
    axr.set_title("Number at risk", loc="left", fontsize=11,
                  fontweight="bold")
    axr.grid(False)
    for sp in axr.spines.values():
        sp.set_visible(False)
    axr.tick_params(length=0)

    common.save_fig(fig, name)

    # ---- validation against table_km_5yr.csv -------------------------------
    print(f"  [{name}] five-year validation (computed vs reference):")
    for arm in ARM_LEVELS:
        S, lo, hi = km_at(km_by_arm[arm], 5.0)
        ref = KM5_REF[(endpoint, arm)]
        ok = (abs(S - ref[0]) < 0.005 and abs(lo - ref[1]) < 0.005
              and abs(hi - ref[2]) < 0.005)
        print(f"    {ARM_LABELS[arm]:>18s}: "
              f"{100*S:5.1f} ({100*lo:4.1f}, {100*hi:4.1f}) vs "
              f"{100*ref[0]:5.1f} ({100*ref[1]:4.1f}, {100*ref[2]:4.1f})"
              f"  {'OK' if ok else 'MISMATCH'}")


# ------------------------------------------------------------------------------
# 2. Subgroup forest plot (text columns outside the plotting area)
# ------------------------------------------------------------------------------
def subgroup_forest():
    tab = pd.read_csv("outputs/tables/table_subgroup.csv")

    # Shorter display name so the bold name column stays clear of the
    # right-aligned level labels.
    tab["name"] = tab["subgroup"].replace(
        {"Surgery-to-registration": "Surgery interval"})
    first_of_group = ~tab["subgroup"].duplicated()

    n = len(tab)
    y_top = n - 1                     # first CSV row ("All patients") on top
    ys = y_top - np.arange(n)
    ylim = (-0.7, n + 0.1)

    fig = plt.figure(figsize=(7.4, 6.2), layout=None)
    # Axes shifted right (vs 0.345) so the longest level label
    # ("Submucosa/muscle") clears the bold "Local extent" group name.
    ax = fig.add_axes([0.375, 0.095, 0.255, 0.75])
    trans = blended_transform_factory(ax.transAxes, ax.transData)

    # light vertical grid only across the data rows
    for x in (0.3, 0.5, 0.7, 1.5, 2.0):
        ax.vlines(x, -0.55, y_top - 0.45, color="#e2e2e2", lw=0.7, zorder=1)
    ax.vlines(1.0, -0.55, y_top - 0.45, color="#737373", lw=1.0,
              linestyles="--", zorder=1)

    header_y = y_top + 0.8
    for i, row in tab.iterrows():
        y = ys[i]
        if i == 0:      # overall estimate: diamond marker, bold label
            ax.plot([row["lower"], row["upper"]], [y, y], color=ACCENT,
                    lw=1.6, solid_capstyle="butt", zorder=2)
            ax.plot(row["HR"], y, marker="D", ms=7, color=ACCENT, zorder=3)
        else:
            ax.plot([row["lower"], row["upper"]], [y, y], color=ACCENT,
                    lw=1.5, solid_capstyle="butt", zorder=2)
            ax.plot(row["HR"], y, marker="o", ms=5.5, color=ACCENT,
                    zorder=3)
        # text columns strictly outside the plotting area -> no collision
        ax.text(1.05, y, f'{row["HR"]:.2f} ({row["lower"]:.2f}-'
                         f'{row["upper"]:.2f})',
                transform=trans, fontsize=10, va="center", clip_on=False)
        if pd.notna(row["p_interaction"]):
            ax.text(1.86, y, common.fmt_p(row["p_interaction"]),
                    transform=trans, fontsize=10, va="center", clip_on=False)
        # bold subgroup name in the left margin, aligned with first level
        if i > 0 and first_of_group.iloc[i]:
            y_fig = 0.095 + 0.75 * (y - ylim[0]) / (ylim[1] - ylim[0])
            fig.text(0.02, y_fig, row["name"], fontsize=10.5,
                     fontweight="bold", color="#2b2b2b",
                     va="center", ha="left")

    ax.text(1.05, header_y, "HR (95% CI)", transform=trans, fontsize=10.5,
            fontweight="bold", va="center", clip_on=False)
    ax.text(1.86, header_y, "Interaction p", transform=trans, fontsize=10.5,
            fontweight="bold", va="center", clip_on=False)

    ax.set_xscale("log")
    ax.set_xlim(0.27, 2.35)
    ax.set_ylim(*ylim)
    ax.set_xticks([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    ax.set_xticklabels(["0.3", "0.5", "0.7", "1", "1.5", "2"])
    ax.xaxis.set_minor_locator(mtick.NullLocator())
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [("All patients" if lvl == "" else lvl) for lvl in tab["level"]],
        fontsize=10.5)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.spines["left"].set_visible(False)

    fig.text(0.02, 0.965,
             "Subgroup analyses: disease-free survival, Lev+5-FU vs "
             "observation",
             fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.925,
             "Unadjusted Cox models within subgroups; interaction tests "
             "adjusted for nodal count.\nRight: hazard ratios with 95% CIs "
             "and interaction p-values.",
             fontsize=9.5, color=GREY_TEXT, va="top")

    common.save_fig(fig, "fig_subgroup_forest")
    print(f"  [fig_subgroup_forest] {n} rows drawn; text columns placed "
          f"outside the axes (no in-panel annotations)")


# ------------------------------------------------------------------------------
# 3. Proportional-hazards diagnostics (scaled Schoenfeld residuals)
# ------------------------------------------------------------------------------
def zph_figure(df):
    covars = ["rx", "sex", "age", "obstruct", "perfor", "adhere", "nodes",
              "differ", "extent", "surg"]
    cc = df.dropna(subset=covars + ["dfs_time", "dfs_status"]).copy()

    # Dummy coding identical to R (reference levels: Obs, Female, No,
    # Well, Submucosa, Long) so the residuals match the R fit.
    X = pd.DataFrame(index=cc.index)
    X["rxLev"] = (cc["rx"] == "Lev").astype(float)
    X["rxLev+5FU"] = (cc["rx"] == "Lev+5FU").astype(float)
    X["sexMale"] = (cc["sex"] == "Male").astype(float)
    X["age"] = cc["age"].astype(float)
    X["obstructYes"] = (cc["obstruct"] == "Yes").astype(float)
    X["perforYes"] = (cc["perfor"] == "Yes").astype(float)
    X["adhereYes"] = (cc["adhere"] == "Yes").astype(float)
    X["nodes"] = cc["nodes"].astype(float)
    X["differModerate"] = (cc["differ"] == "Moderate").astype(float)
    X["differPoor"] = (cc["differ"] == "Poor").astype(float)
    X["extentMuscle"] = (cc["extent"] == "Muscle").astype(float)
    X["extentSerosa"] = (cc["extent"] == "Serosa").astype(float)
    X["extentContiguous"] = (cc["extent"] == "Contiguous structures").astype(float)
    X["surgShort"] = (cc["surg"] == "Short").astype(float)
    model = X.copy()
    model["dfs_time"] = cc["dfs_time"].values
    model["dfs_status"] = cc["dfs_status"].values

    cph = CoxPHFitter()
    cph.fit(model, duration_col="dfs_time", event_col="dfs_status")
    resid = cph.compute_residuals(model, kind="scaled_schoenfeld")
    resid = resid - resid.mean()          # 0 = no departure from constant effect
    time = cc.loc[resid.index, "dfs_time"].values

    panels = [
        ("rxLev", "Levamisole vs observation"),
        ("rxLev+5FU", "Levamisole + 5-FU vs observation"),
        ("nodes", "Positive nodes"),
        ("obstructYes", "Obstruction (PH violation, p = 0.008)"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4), layout="constrained",
                            sharex=True)
    for ax, (col, lab) in zip(axs.flat, panels):
        r = resid[col].values
        ax.scatter(time, r, s=9, alpha=0.45, color="#595959", lw=0)
        sm = lowess(r, time, frac=0.67)
        ax.plot(sm[:, 0], sm[:, 1], color=ACCENT, lw=1.7)
        ax.axhline(0.0, ls="--", color="#333333", lw=0.9)
        pad = 0.06 * (max(r.max(), sm[:, 1].max()) -
                      min(r.min(), sm[:, 1].min()))
        ax.set_ylim(min(r.min(), sm[:, 1].min()) - pad,
                    max(r.max(), sm[:, 1].max()) + pad)
        ax.set_title(lab, fontsize=11)
    for ax in axs[1]:
        ax.set_xlabel("Years since randomization")
    for ax in axs[:, 0]:
        ax.set_ylabel("Scaled Schoenfeld residual")
    fig.suptitle("Proportional-hazards diagnostics", fontsize=13,
                 fontweight="bold")

    common.save_fig(fig, "fig_zph")
    print(f"  [fig_zph] complete cases n={len(cc)}, events="
          f"{int(cc['dfs_status'].sum())}; x axis is untransformed years")
    sm_obs = lowess(resid["obstructYes"].values, time, frac=0.67)
    print(f"    obstruction loess: start {sm_obs[0, 1]:+.3f} -> "
          f"end {sm_obs[-1, 1]:+.3f} (declining trend expected)")


# ------------------------------------------------------------------------------
# 4. Simulation Study A rebuild (the original R/ggsave rendering clipped the
#    long subtitle at the device edge: "...is equiva" was cut off)
# ------------------------------------------------------------------------------
def sim_efficiency():
    tab = pd.read_csv("outputs/tables/table_sim_A.csv")
    tab = tab[tab["hr"] == 0.72].reset_index(drop=True)

    x = np.arange(len(tab))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 4.1), layout="constrained")
    ax.bar(x - w / 2, tab["power_un"], w, label="Unadjusted",
           color=common.BLUE)
    ax.bar(x + w / 2, tab["power_ad"], w, label="Adjusted",
           color=common.GREEN)
    for xi, (pu, pa) in zip(x, zip(tab["power_un"], tab["power_ad"])):
        ax.text(xi - w / 2, pu + 0.015, f"{pu:.3f}", ha="center",
                va="bottom", fontsize=9, color=common.BLUE)
        ax.text(xi + w / 2, pa + 0.015, f"{pa:.3f}", ha="center",
                va="bottom", fontsize=9, color=common.GREEN)
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.9)
    ax.text(-0.45, 0.812, "80% target", fontsize=9, color="grey",
            ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(tab["strength_lab"], fontsize=10.5)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Prognostic strength of baseline covariates")
    ax.set_ylabel("Empirical power")
    fig.suptitle("Simulation Study A: power gain from covariate adjustment",
                 fontsize=12.5, fontweight="bold")
    ax.set_title("1,000 replicates per scenario, n = 322, target HR = 0.72, "
                 "one-sided alpha = 0.025;\nlog-rank test (not shown) is "
                 "numerically equivalent to the unadjusted Cox test",
                 fontsize=9.5, color=GREY_TEXT)
    ax.legend(loc="upper left", fontsize=10)
    common.save_fig(fig, "fig_sim_efficiency")
    print("  [fig_sim_efficiency] rebuilt from table_sim_A.csv "
          f"({len(tab)} scenarios)")


# ------------------------------------------------------------------------------
# 5. Competing-risk cumulative-incidence rebuild (round 3)
# ------------------------------------------------------------------------------
def aj_and_naive(time, cause):
    """Aalen-Johansen CIF of cause 1 and the naive one-minus-KM curve.

    Parameters
    ----------
    time  : observed times (rec_time);
    cause : 0 = censored, 1 = recurrence, 2 = death without recurrence.

    Returns
    -------
    (t_aj, cif, t_naive, cif_naive) : step-function jump arrays, each
    starting at (0, 0).  The Aalen-Johansen increment at an event time
    t_j is S(t_{j-1}) * d_1(t_j) / n(t_j), where S is the Kaplan-Meier
    estimator of the overall event-free survival (events of either
    cause); the naive curve treats cause-2 events as censoring.
    """
    time = np.asarray(time, dtype=float)
    cause = np.asarray(cause, dtype=int)

    # --- Aalen-Johansen -----------------------------------------------------
    ts, cifs = [0.0], [0.0]
    S, cif = 1.0, 0.0                      # S = P(no event of any cause)
    for tt in np.unique(time[cause > 0]):  # jumps only at observed events
        n = int(np.sum(time >= tt))
        d1 = int(np.sum((time == tt) & (cause == 1)))
        d2 = int(np.sum((time == tt) & (cause == 2)))
        cif += S * d1 / n                  # S is the survival JUST BEFORE tt
        S *= 1.0 - (d1 + d2) / n
        ts.append(float(tt))
        cifs.append(cif)

    # --- naive 1 - KM (cause-2 events censored) -----------------------------
    tn, cn = [0.0], [0.0]
    Sn, cnaive = 1.0, 0.0
    for tt in np.unique(time[cause == 1]):
        n = int(np.sum(time >= tt))
        d1 = int(np.sum((time == tt) & (cause == 1)))
        Sn *= 1.0 - d1 / n
        cnaive = 1.0 - Sn
        tn.append(float(tt))
        cn.append(cnaive)

    return (np.array(ts), np.array(cifs),
            np.array(tn), np.array(cn))


def _step_at(t, y, q):
    """Right-continuous step lookup of y at time q (y jumps at t)."""
    i = int(np.clip(np.searchsorted(t, q, side="right") - 1, 0, len(t) - 1))
    return float(y[i])


def cif_figure(df):
    """Cumulative incidence of recurrence: AJ vs naive 1-KM, three arms."""
    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")

    max_t = 8.0
    curves = {}
    for arm in ARM_LEVELS:
        sub = df[df["rx"] == arm]
        t_aj, c_aj, t_nv, c_nv = aj_and_naive(sub["rec_time"].values,
                                               sub["rec_cr"].values)
        curves[arm] = (t_aj, c_aj, t_nv, c_nv)
        c = ARM_COLS[arm]
        ax.step(t_aj, c_aj, where="post", color=c, lw=1.6,
                label=ARM_LABELS[arm])
        ax.step(t_nv, c_nv, where="post", color=c, lw=1.0,
                linestyle=(0, (4, 2.4)), alpha=0.65)

    ax.set_xlim(0, max_t)
    ax.set_xticks(np.arange(0, max_t + 1))
    ax.set_ylim(0, 0.65)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Years since randomization", fontsize=11)
    ax.set_ylabel("Cumulative incidence of recurrence", fontsize=11)
    ax.tick_params(axis="both", labelsize=10.5)
    ax.grid(True, axis="y", color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)

    # Legend: colour encodes the arm; the two line styles are explained by
    # the subtitle (solid = Aalen-Johansen, dashed = naive 1-KM).
    ax.legend(loc="upper left", fontsize=10.5, title="Randomized arm",
              title_fontsize=10.5, framealpha=0.95, borderpad=0.6)

    ax.set_title("Solid: Aalen-Johansen cumulative incidence "
                 "(death without recurrence is a competing event)\n"
                 "Dashed: one minus Kaplan-Meier "
                 "(treats death without recurrence as censoring)",
                 fontsize=9.5, color=GREY_TEXT)
    fig.suptitle("Cumulative incidence of recurrence: "
                 "competing-risk vs naive analysis",
                 fontsize=13, fontweight="bold")
    common.save_fig(fig, "fig_cif")

    # ---- validation against the report text --------------------------------
    n_comp = int((df["rec_cr"] == 2).sum())
    print(f"  [fig_cif] competing deaths (rec_cr == 2): {n_comp} "
          f"(report text: 38)")
    for arm in ARM_LEVELS:
        t_aj, c_aj, t_nv, c_nv = curves[arm]
        grid = np.linspace(0, max_t, 161)
        viol = np.sum([_step_at(t_aj, c_aj, q) > _step_at(t_nv, c_nv, q) + 1e-12
                       for q in grid])
        a8 = _step_at(t_aj, c_aj, 8.0)
        n8 = _step_at(t_nv, c_nv, 8.0)
        print(f"    {arm:8s} AJ(8y)={a8:.4f}  naive(8y)={n8:.4f}  "
              f"gap={n8 - a8:+.4f}  AJ>naive violations: {viol}")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 06_figure_rebuilds.py ===")
    df = common.load_analysis_data()
    km_figure(df, "dfs_time", "dfs_status", "DFS",
              "Kaplan-Meier estimates of disease-free survival",
              "Composite endpoint: tumour recurrence or death from any "
              "cause", "fig_km_dfs")
    km_figure(df, "os_time", "os_status", "OS",
              "Kaplan-Meier estimates of overall survival",
              "Secondary endpoint: death from any cause", "fig_km_os")
    subgroup_forest()
    zph_figure(df)
    sim_efficiency()
    cif_figure(df)
    print("=== 06_figure_rebuilds.py completed ===")
