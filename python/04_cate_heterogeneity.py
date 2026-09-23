"""
04_cate_heterogeneity.py

Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
          with Frequentist, Bayesian, and Machine Learning Methods
Author  : Cathy
Purpose : Treatment-effect heterogeneity for the primary comparison
          (Levamisole + 5-FU vs Observation) on the absolute risk of a DFS
          event by 5 years:

            1. Jackknife pseudo-observations for the 5-year event risk are
               computed from the Kaplan-Meier estimator, converting the
               censored survival outcome into a continuous, regression-ready
               outcome (Andersen et al. 2003; Parner & Andersen 2010).
            2. Population-average treatment effect on 5-year risk:
               unadjusted Kaplan-Meier difference, cross-fitted
               G-computation, and a cross-fitted doubly robust AIPW
               estimator with influence-function standard errors.
            3. Conditional average treatment effects (CATE) via a
               cross-fitted T-learner (penalized outcome models), with the
               distribution of predicted absolute benefit, benefit-by-risk
               dependence, and exploratory Kaplan-Meier curves within
               predicted-benefit quartiles.

Inputs  : data/processed/colon_analysis.csv
Outputs : outputs/figures/fig_cate_distribution.png
                     fig_cate_km_quartiles.png
          outputs/tables/table_cate_ate.csv
                     table_cate_quartiles.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, "python")
from importlib import import_module
common = import_module("00_common")

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LassoCV, LinearRegression
from lifelines import KaplanMeierFitter
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
common.apply_style()
common.set_seed()
print("=== 04_cate_heterogeneity.py ===\n")

HORIZON = 5.0

# ------------------------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------------------------
df = common.load_analysis_data()
df = df[df["rx"].isin(["Obs", "Lev+5FU"])].reset_index(drop=True)
df["treat"] = (df["rx"] == "Lev+5FU").astype(int)
n_all = len(df)
print(f"Two-arm sample: n = {n_all} "
      f"(Obs {int((df['treat'] == 0).sum())}, Lev+5FU {int((df['treat'] == 1).sum())})")

# ------------------------------------------------------------------------------
# 2. Jackknife pseudo-observations for the 5-year DFS event risk
# ------------------------------------------------------------------------------
def km_risk_at(time, event, horizon, drop=None):
    """KM estimate of the event probability at `horizon`, optionally leaving
    out the single observation `drop` (leave-one-out jackknife)."""
    keep = np.ones(len(time), dtype=bool)
    if drop is not None:
        keep[drop] = False
    kmf = KaplanMeierFitter()
    kmf.fit(time[keep], event_observed=event[keep])
    return float(1.0 - kmf.predict(horizon))

t = df["dfs_time"].values.astype(float)
ev = df["dfs_status"].values.astype(bool)
theta_full = km_risk_at(t, ev, HORIZON)
print(f"Overall 5-year DFS event risk (KM): {theta_full:.3f}")

print("Computing leave-one-out pseudo-observations ...")
theta = np.empty(n_all)
for i in range(n_all):
    theta[i] = n_all * theta_full - (n_all - 1) * km_risk_at(t, ev, HORIZON, drop=i)
print(f"Pseudo-observations: mean = {theta.mean():.3f}, "
      f"range = ({theta.min():.3f}, {theta.max():.3f})")

# ------------------------------------------------------------------------------
# 3. Average treatment effect on the 5-year absolute risk
# ------------------------------------------------------------------------------
km1 = km_risk_at(t[df.treat == 1], ev[df.treat == 1], HORIZON)
km0 = km_risk_at(t[df.treat == 0], ev[df.treat == 0], HORIZON)
ate_km = km1 - km0

# Bootstrap CI for the KM difference
rng = np.random.default_rng(common.RANDOM_SEED)
boot = []
for _ in range(2000):
    idx = rng.integers(0, n_all, n_all)
    b_t = t[idx][df.treat.values[idx] == 1]
    b_e = ev[idx][df.treat.values[idx] == 1]
    c_t = t[idx][df.treat.values[idx] == 0]
    c_e = ev[idx][df.treat.values[idx] == 0]
    boot.append(km_risk_at(b_t, b_e, HORIZON) - km_risk_at(c_t, c_e, HORIZON))
ci_km = np.percentile(boot, [2.5, 97.5])

# ---- Cross-fitted G-computation and AIPW on the pseudo-observations ----------
dfm = df.dropna(subset=["age", "nodes", "sex", "obstruct", "perfor", "adhere",
                        "differ", "extent", "surg"]).reset_index(drop=True)
X = common.design_matrix(dfm)
Xs = (X - X.mean()) / X.std()
T = dfm["treat"].values
# align pseudo-observations with the modelling subset via patient id
ids_all = df["id"].values
theta_map = dict(zip(ids_all, theta))
theta_m = dfm["id"].map(theta_map).values.astype(float)
n_m = len(dfm)
print(f"\nModelling subset (complete cases): n = {n_m}, "
      f"{int(dfm['dfs_status'].sum())} events")

X1 = Xs.copy()
X1["treat"] = T
folds = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=common.RANDOM_SEED)
m1_hat = np.zeros(n_m)
m0_hat = np.zeros(n_m)
for tr, te in folds.split(Xs, T):
    d1 = X1.iloc[tr][X1.iloc[tr]["treat"] == 1].drop(columns="treat")
    d0 = X1.iloc[tr][X1.iloc[tr]["treat"] == 0].drop(columns="treat")
    y1 = theta_m[tr][T[tr] == 1]
    y0 = theta_m[tr][T[tr] == 0]
    f1 = LassoCV(cv=5, random_state=common.RANDOM_SEED).fit(d1, y1)
    f0 = LassoCV(cv=5, random_state=common.RANDOM_SEED).fit(d0, y0)
    m1_hat[te] = f1.predict(Xs.iloc[te])
    m0_hat[te] = f0.predict(Xs.iloc[te])

e = np.full(n_m, T.mean())          # randomized propensity (empirical)
aipw_if = (m1_hat - m0_hat
           + T * (theta_m - m1_hat) / e
           - (1 - T) * (theta_m - m0_hat) / (1 - e))
ate_aipw = aipw_if.mean()
se_aipw = aipw_if.std(ddof=1) / np.sqrt(n_m)
ate_gcomp = (m1_hat - m0_hat).mean()

# Influence-function CI for G-computation (plug-in, ignoring estimation
# uncertainty): report bootstrap instead for transparency
g_boot = []
for _ in range(1000):
    idx = rng.integers(0, n_m, n_m)
    g_boot.append((m1_hat[idx] - m0_hat[idx]).mean())

ate_tab = pd.DataFrame([
    {"estimator": "Kaplan-Meier difference (unadjusted)",
     "ATE_5yr_risk": round(ate_km, 4),
     "se": round(np.std(boot, ddof=1), 4),
     "ci_lo": round(ci_km[0], 4), "ci_hi": round(ci_km[1], 4)},
    {"estimator": "G-computation (cross-fitted, pseudo-obs)",
     "ATE_5yr_risk": round(ate_gcomp, 4),
     "se": round(np.std(g_boot, ddof=1), 4),
     "ci_lo": round(ate_gcomp - 1.96 * np.std(g_boot, ddof=1), 4),
     "ci_hi": round(ate_gcomp + 1.96 * np.std(g_boot, ddof=1), 4)},
    {"estimator": "AIPW (cross-fitted, doubly robust)",
     "ATE_5yr_risk": round(ate_aipw, 4),
     "se": round(se_aipw, 4),
     "ci_lo": round(ate_aipw - 1.96 * se_aipw, 4),
     "ci_hi": round(ate_aipw + 1.96 * se_aipw, 4)},
])
common.save_tab(ate_tab, "table_cate_ate")
print("\nAverage treatment effect on 5-year DFS event risk:")
print(ate_tab.to_string(index=False))
print("NB: negative ATE = risk reduction under Lev+5FU")

# ------------------------------------------------------------------------------
# 4. CATE: distribution of predicted absolute benefit
# ------------------------------------------------------------------------------
benefit = m1_hat - m0_hat               # predicted risk difference (negative = benefit)
base_risk = m0_hat                      # predicted control-arm 5-year risk

q = pd.qcut(benefit, 4, labels=False, duplicates="drop")
q_tab = []
for g in range(q.max() + 1):
    idx = (q == g)
    # observed (pseudo-observation based) group-specific ATE
    sub_T = T[idx]
    obs_ate = (theta_m[idx][sub_T == 1].mean() - theta_m[idx][sub_T == 0].mean())
    n1, n0 = int(sub_T.sum()), int((1 - sub_T).sum())
    q_tab.append({
        "quartile": f"Q{g + 1}",
        "n": int(idx.sum()),
        "mean_predicted_benefit": round(float(-benefit[idx].mean()), 4),
        "mean_baseline_risk": round(float(base_risk[idx].mean()), 4),
        "observed_risk_diff": round(float(obs_ate), 4),
        "n_treated": n1, "n_control": n0,
        "events": int(dfm["dfs_status"].values[idx].sum()),
    })
q_tab = pd.DataFrame(q_tab)
common.save_tab(q_tab, "table_cate_quartiles")
print("\nPredicted-benefit quartiles:")
print(q_tab.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
axes[0].hist(-benefit, bins=40, color=common.BLUE, alpha=0.8)
axes[0].axvline(0, color="grey", linestyle="--", linewidth=0.9)
axes[0].axvline(-ate_aipw, color=common.ACCENT, linewidth=1.3)
axes[0].text(-ate_aipw, axes[0].get_ylim()[1] * 0.95, " AIPW ATE",
             color=common.ACCENT, fontsize=9, va="top")
axes[0].set_xlabel("Predicted 5-year absolute benefit\n"
                   "(risk reduction, percentage points)", fontsize=10)
axes[0].set_ylabel("Patients")
axes[0].set_title("Distribution of predicted benefit")

sc = axes[1].scatter(base_risk * 100, -benefit * 100, c=T, s=10, alpha=0.55,
                     cmap=matplotlib.colors.ListedColormap(
                         [common.BLUE, common.GREEN]), linewidths=0)
axes[1].set_xlabel("Predicted baseline 5-year risk\n"
                   "under observation (%)", fontsize=10)
axes[1].set_ylabel("Predicted benefit (percentage points)")
axes[1].set_title("Benefit versus baseline risk")
handles = [plt.Line2D([], [], marker="o", ls="", color=common.BLUE,
                      label="Observation arm"),
           plt.Line2D([], [], marker="o", ls="", color=common.GREEN,
                      label="Lev+5FU arm")]
axes[1].legend(handles=handles, loc="upper right")
fig.tight_layout()
common.save_fig(fig, "fig_cate_distribution")

# ------------------------------------------------------------------------------
# 5. Exploratory Kaplan-Meier curves within extreme predicted-benefit quartiles
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(7.0, 3.6))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1, 2, wspace=0.25)
for j, g in enumerate([0, q.max()]):
    ax = fig.add_subplot(gs[0, j])
    idx = np.where(q == g)[0]
    for arm_val, arm_lab, colour in [
            (0, "Observation", common.BLUE),
            (1, "Levamisole + 5-FU", common.GREEN)]:
        sel = idx[T[idx] == arm_val]
        kmf = KaplanMeierFitter()
        kmf.fit(dfm["dfs_time"].values[sel],
                event_observed=dfm["dfs_status"].values[sel])
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colour,
                                   linewidth=1.4, label=arm_lab)
    ax.set_xlabel("Years since randomization")
    ax.set_ylabel("Disease-free survival")
    ax.set_ylim(0, 1.02)
    # Two-line title: the single-line titles were wider than the panels and
    # collided with each other at the figure centre.
    ax.set_title(f"{'Lowest' if j == 0 else 'Highest'} predicted-benefit\n"
                 f"quartile (n = {int((q == g).sum())})", fontsize=11)
    ax.legend(loc="lower left")
fig.tight_layout()
common.save_fig(fig, "fig_cate_km_quartiles")

# ---- Correlation between predicted benefit and baseline risk ------------------
corr = np.corrcoef(base_risk, -benefit)[0, 1]
print(f"\nCorrelation between baseline risk and predicted benefit: {corr:.3f}")
print("\n=== 04_cate_heterogeneity.py completed ===")
