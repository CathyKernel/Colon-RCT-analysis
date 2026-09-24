"""
05_rmst_analysis.py

Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
          with Frequentist, Bayesian, and Machine Learning Methods
Author  : Cathy
Purpose : Restricted mean survival time (RMST) analysis of the primary
          comparison (Levamisole + 5-FU vs Observation) on disease-free
          survival, as a robustness analysis that does not require the
          proportional-hazards assumption:

            1. Unadjusted RMST and RMST difference / ratio at horizons
               tau = 3, 4, 5, 6, 7, 8 years, with bootstrap CIs
               (KM-based numerical integration of the survival function);
            2. Covariate-adjusted RMST difference at tau = 5 years via
               jackknife pseudo-observation regression (Andersen et al.
               2003), which - unlike the hazard ratio - targets a
               collapsible estimand, so the adjusted and unadjusted
               analyses estimate the SAME quantity;
            3. RMST difference within the obstruction stratum, the
              covariate with the strongest proportional-hazards violation
              (cox.zph p = 0.008), demonstrating conclusion robustness
              under non-proportional hazards.

The RMST difference is interpreted as the expected number of
disease-free years gained (within the horizon tau) under levamisole
plus 5-FU compared with observation.

Inputs  : data/processed/colon_analysis.csv
Outputs : outputs/figures/fig_rmst.png, fig_rmst.pdf
          outputs/tables/table_rmst.csv, table_rmst_adjusted.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, "python")
from importlib import import_module
common = import_module("00_common")

from lifelines import KaplanMeierFitter
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
common.apply_style()
common.set_seed()
print("=== 05_rmst_analysis.py ===\n")

TAU_GRID = [3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0]   # 5 appears twice: overall + adjusted
TAU_MAIN = 5.0
N_BOOT = 2000

# ------------------------------------------------------------------------------
# 1. Data: two-arm comparison
# ------------------------------------------------------------------------------
df = common.load_analysis_data()
df = df[df["rx"].isin(["Obs", "Lev+5FU"])].reset_index(drop=True)
df["treat"] = (df["rx"] == "Lev+5FU").astype(int)
n_all = len(df)
print(f"Two-arm sample: n = {n_all} "
      f"(Obs {int((df['treat'] == 0).sum())}, "
      f"Lev+5FU {int((df['treat'] == 1).sum())})")

t = df["dfs_time"].values.astype(float)
ev = df["dfs_status"].values.astype(bool)
treat = df["treat"].values

# ------------------------------------------------------------------------------
# 2. RMST estimation from the Kaplan-Meier estimator
# ------------------------------------------------------------------------------
def rmst_km(time, event, tau):
    """RMST up to `tau`: integral of the KM (product-limit) survival
    function, computed by exact step-function integration with numpy.
    Mathematically identical to integrating the lifelines KM estimate,
    but vectorized for the thousands of bootstrap refits below.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    n = len(time)
    ev_times = time[event & (time > 0) & (time <= tau)]
    if len(ev_times) == 0:
        return float(tau)                    # S(t) = 1 on [0, tau]
    ts, d = np.unique(ev_times, return_counts=True)
    times_sorted = np.sort(time)
    at_risk = n - np.searchsorted(times_sorted, ts, side="left")
    surv_at_events = np.cumprod(1.0 - d / at_risk)
    # survival is 1 before the first event, then constant between events
    knots = np.concatenate([[0.0], ts, [tau]])
    surv = np.concatenate([[1.0], surv_at_events])
    return float(np.sum(surv * np.diff(knots)))


def _km_check_lifelines(time, event, tau):
    """Reference implementation via lifelines (used only for the internal
    equivalence check below). The KM survival function is right-continuous,
    so the value on [t_j, t_{j+1}) is S(t_j): the integral uses the LEFT
    endpoint of each inter-event interval."""
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=event)
    ev_times = np.unique(time[np.asarray(event, dtype=bool)
                              & (time > 0) & (time <= tau)])
    knots = np.concatenate([[0.0], ev_times, [tau]])
    surv = kmf.survival_function_at_times(knots).values.ravel()
    return float(np.sum(surv[:-1] * np.diff(knots)))


# internal equivalence check: numpy KM integration == lifelines integration
for _tau in (3.0, 5.0, 8.0):
    for _mask in (treat == 0, treat == 1):
        _a = rmst_km(t[_mask], ev[_mask], _tau)
        _b = _km_check_lifelines(t[_mask], ev[_mask], _tau)
        assert abs(_a - _b) < 1e-8, ("RMST implementation mismatch", _a, _b)
print("  internal check: numpy KM integration matches lifelines to < 1e-8")


def rmst_boot_all(time, event, treat_vec, tau, n_boot=N_BOOT,
                  seed=common.RANDOM_SEED):
    """Bootstrap the RMST difference AND ratio in one pass (one bootstrap
    sample, one pair of KM refits) - percentile CIs for both."""
    r0 = rmst_km(time[treat_vec == 0], event[treat_vec == 0], tau)
    r1 = rmst_km(time[treat_vec == 1], event[treat_vec == 1], tau)
    rng = np.random.default_rng(seed)
    diffs, ratios = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(time), len(time))
        tb, eb, ab = time[idx], event[idx], treat_vec[idx]
        if (ab == 0).all() or (ab == 1).all():
            continue
        b0 = rmst_km(tb[ab == 0], eb[ab == 0], tau)
        b1 = rmst_km(tb[ab == 1], eb[ab == 1], tau)
        diffs.append(b1 - b0)
        ratios.append(b1 / b0)
    diffs = np.asarray(diffs); ratios = np.asarray(ratios)
    dlo, dhi = np.percentile(diffs, [2.5, 97.5])
    rlo, rhi = np.percentile(ratios, [2.5, 97.5])
    return (r1 - r0, dlo, dhi, r1 / r0, rlo, rhi)


# ---- RMST at each horizon ------------------------------------------------------
rows = []
for tau in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    r0 = rmst_km(t[treat == 0], ev[treat == 0], tau)
    r1 = rmst_km(t[treat == 1], ev[treat == 1], tau)
    d_hat, dlo, dhi, ratio, rlo, rhi = rmst_boot_all(t, ev, treat, tau)
    rows.append({
        "tau_years": tau,
        "rmst_obs": round(r0, 3), "rmst_lev5fu": round(r1, 3),
        "difference_years": round(d_hat, 3),
        "diff_ci_lo": round(dlo, 3), "diff_ci_hi": round(dhi, 3),
        "ratio": round(ratio, 3),
        "ratio_ci_lo": round(rlo, 3), "ratio_ci_hi": round(rhi, 3),
    })
rmst_tab = pd.DataFrame(rows)
common.save_tab(rmst_tab, "table_rmst")
print("\nRMST analysis (Lev+5FU vs Observation, DFS):")
print(rmst_tab.to_string(index=False))
print("NB: positive difference = disease-free years gained with Lev+5FU")

# ------------------------------------------------------------------------------
# 3. Covariate-adjusted RMST difference (pseudo-observation regression)
# ------------------------------------------------------------------------------
def rmst_pseudo_obs(time, event, tau):
    """Leave-one-out jackknife pseudo-observations for RMST(tau)."""
    n = len(time)
    theta_full = rmst_km(time, event, tau)
    ps = np.empty(n)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        ps[i] = n * theta_full - (n - 1) * rmst_km(time[keep], event[keep], tau)
    return ps


print("\nComputing leave-one-out pseudo-observations for RMST(tau = 5y) ...")
ps5 = rmst_pseudo_obs(t, ev, TAU_MAIN)
print(f"Pseudo-observations: mean = {ps5.mean():.3f}, "
      f"range = ({ps5.min():.3f}, {ps5.max():.3f})")

dfm = df.dropna(subset=["age", "nodes", "sex", "obstruct", "perfor", "adhere",
                        "differ", "extent", "surg"]).reset_index(drop=True)
X = common.design_matrix(dfm)
Xs = ((X - X.mean()) / X.std()).values
T = dfm["treat"].values.astype(float)
theta_map = dict(zip(df["id"].values, ps5))
theta_m = dfm["id"].map(theta_map).values.astype(float)
n_m = len(dfm)
print(f"Adjustment subset (complete cases): n = {n_m}")

# OLS of pseudo-observations on treatment + standardized covariates;
# bootstrap over patients for the CI (robust to the pseudo-obs dependence)
D = np.column_stack([np.ones(n_m), T, Xs])
beta = np.linalg.lstsq(D, theta_m, rcond=None)[0]
adj_hat = beta[1]

rng = np.random.default_rng(common.RANDOM_SEED + 2)
adj_boots = []
for _ in range(1000):
    idx = rng.integers(0, n_m, n_m)
    try:
        b = np.linalg.lstsq(D[idx], theta_m[idx], rcond=None)[0]
        adj_boots.append(b[1])
    except np.linalg.LinAlgError:
        continue
alo, ahi = np.percentile(adj_boots, [2.5, 97.5])
adj_tab = pd.DataFrame([{
    "estimator": "RMST difference at 5y, unadjusted (KM)",
    "estimate_years": round(float(rmst_tab.loc[rmst_tab.tau_years == 5.0,
                                               "difference_years"].iloc[0]), 3),
    "ci_lo": float(rmst_tab.loc[rmst_tab.tau_years == 5.0, "diff_ci_lo"].iloc[0]),
    "ci_hi": float(rmst_tab.loc[rmst_tab.tau_years == 5.0, "diff_ci_hi"].iloc[0]),
}, {
    "estimator": "RMST difference at 5y, covariate-adjusted (pseudo-obs)",
    "estimate_years": round(float(adj_hat), 3),
    "ci_lo": round(float(alo), 3), "ci_hi": round(float(ahi), 3),
}])
common.save_tab(adj_tab, "table_rmst_adjusted")
print("\nRMST difference at 5 years, unadjusted vs covariate-adjusted:")
print(adj_tab.to_string(index=False))
print("NB: both estimators target the same (collapsible) estimand; close "
      "agreement is expected and confirms the HR non-collapsibility contrast.")

# ------------------------------------------------------------------------------
# 4. RMST within the obstruction stratum (strongest PH violation)
# ------------------------------------------------------------------------------
ob = dfm["obstruct"].values == "Yes"
sub_rows = []
for lab, mask in [("No obstruction", ~ob), ("Obstruction", ob)]:
    tt = dfm["dfs_time"].values.astype(float)[mask]
    ee = dfm["dfs_status"].values.astype(bool)[mask]
    aa = T[mask].astype(int)
    d_hat, dlo, dhi, _ratio, _rlo, _rhi = rmst_boot_all(
        tt, ee, aa, TAU_MAIN, n_boot=1000, seed=common.RANDOM_SEED + 3)
    sub_rows.append({
        "stratum": lab, "n": int(mask.sum()),
        "rmst_diff_5y": round(d_hat, 3),
        "ci_lo": round(dlo, 3), "ci_hi": round(dhi, 3),
    })
sub_tab = pd.DataFrame(sub_rows)
common.save_tab(sub_tab, "table_rmst_subgroup")
print("\nRMST difference at 5 years by obstruction stratum (PH violator):")
print(sub_tab.to_string(index=False))

# ------------------------------------------------------------------------------
# 5. Figure: KM with RMST area + difference vs tau
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), layout="constrained")

# Left: KM curves with the RMST area (tau = 5y) shaded for both arms
grid_t = np.linspace(0, 9, 300)
for arm_val, arm_lab, colour, ls in [
        (0, "Observation", common.BLUE, "-"),
        (1, "Levamisole + 5-FU", common.GREEN, "-")]:
    sel = treat == arm_val
    kmf = KaplanMeierFitter()
    kmf.fit(t[sel], event_observed=ev[sel])
    sf = kmf.survival_function_at_times(grid_t).values.ravel()
    axes[0].plot(grid_t, sf, color=colour, linewidth=1.4, label=arm_lab,
                 linestyle=ls)
    if arm_val == 0:
        axes[0].fill_between(grid_t[grid_t <= TAU_MAIN], 0,
                             sf[grid_t <= TAU_MAIN], color=colour, alpha=0.12)
r0 = rmst_km(t[treat == 0], ev[treat == 0], TAU_MAIN)
r1 = rmst_km(t[treat == 1], ev[treat == 1], TAU_MAIN)
axes[0].axvline(TAU_MAIN, color="grey", linestyle="--", linewidth=0.9)
# Annotation placed in the empty lower-right region of the panel (below both
# KM curves, right of the tau line) with a white box so it can never collide
# with the curves; the original mid-panel position sat on the green curve.
axes[0].text(0.97, 0.05,
             f"tau = {TAU_MAIN:.0f} y\nRMST Obs = {r0:.2f} y\n"
             f"RMST Lev+5FU = {r1:.2f} y",
             transform=axes[0].transAxes, ha="right", va="bottom",
             fontsize=9.5, color="#333333", linespacing=1.45,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.9, edgecolor="#cccccc", linewidth=0.6))
axes[0].set_xlabel("Years since randomization")
axes[0].set_ylabel("Disease-free survival probability")
axes[0].set_title("RMST: area under the KM curve")
axes[0].set_ylim(0, 1.02)
# Legend in the empty upper-right corner (both curves have declined below
# 0.55 there); at "lower left" it collided with the RMST annotation block.
axes[0].legend(loc="upper right", fontsize=9)

# Right: RMST difference with 95% CI as a function of tau
tau_col = "tau_years"
axes[1].errorbar(rmst_tab[tau_col], rmst_tab["difference_years"],
                 yerr=[rmst_tab["difference_years"] - rmst_tab["diff_ci_lo"],
                       rmst_tab["diff_ci_hi"] - rmst_tab["difference_years"]],
                 fmt="o-", color=common.ACCENT, capsize=3, markersize=5,
                 linewidth=1.2)
axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.9)
axes[1].set_xlabel("Horizon tau (years)")
axes[1].set_ylabel("RMST difference (years, 95% CI)")
axes[1].set_title("Disease-free years gained vs observation")
axes[1].set_xticks([3, 4, 5, 6, 7, 8])

fig.savefig(f"{common.FIG_DIR}/fig_rmst.png", facecolor="white", dpi=300)
fig.savefig(f"{common.FIG_DIR}/fig_rmst.pdf", facecolor="white")
print(f"\n  figure saved: {common.FIG_DIR}/fig_rmst.png")
plt.close(fig)

print("\n=== 05_rmst_analysis.py completed ===")
