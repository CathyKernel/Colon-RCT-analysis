"""
03_bayesian_analysis.py

Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
          with Frequentist, Bayesian, and Machine Learning Methods
Author  : Cathy
Purpose : Bayesian re-analysis of the primary comparison (Levamisole + 5-FU
          vs Observation) on disease-free survival using a Weibull
          proportional-hazards model fitted with NUTS (PyMC), with explicit
          handling of right censoring:

            1. Prior sensitivity analysis for the treatment effect:
               skeptical, weakly-informative, and enthusiastic priors;
            2. Bayesian sequential monitoring: posteriors updated at 25%,
               50%, 75%, and 100% of the observed DFS events;
            3. Posterior predictive checks (replicated Kaplan-Meier curves);
            4. Model criticism: Weibull vs exponential baseline via LOO.

          The Bayesian analysis deliberately mirrors the frequentist interim
          re-enactment of module 05 to contrast the two inferential
          paradigms on the same data.

Inputs  : data/processed/colon_analysis.csv
Outputs : outputs/figures/fig_bayes_priors.png
                     fig_bayes_sequential.png
                     fig_bayes_ppc.png
          outputs/tables/table_bayes_summary.csv
                     table_bayes_sequential.csv
                     table_bayes_model_compare.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, "python")
from importlib import import_module
common = import_module("00_common")

import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
common.apply_style()
common.set_seed()
print("=== 03_bayesian_analysis.py ===\n")
print(f"PyMC {pm.__version__} | ArviZ {az.__version__}\n")

# ------------------------------------------------------------------------------
# 1. Data: two-arm comparison, complete cases
# ------------------------------------------------------------------------------
df = common.load_analysis_data()
df = df[df["rx"].isin(["Obs", "Lev+5FU"])].copy()
df["treat"] = (df["rx"] == "Lev+5FU").astype(int)
covar_cols = ["age", "nodes", "sex", "obstruct", "extent", "surg"]
df = df.dropna(subset=covar_cols).reset_index(drop=True)
print(f"Bayesian analysis sample: n = {len(df)} "
      f"(Obs {int((df['treat'] == 0).sum())}, "
      f"Lev+5FU {int((df['treat'] == 1).sum())}), "
      f"{int(df['dfs_status'].sum())} DFS events")

t_obs = df["dfs_time"].values.astype(float)
event = df["dfs_status"].values.astype(bool)
treat = df["treat"].values.astype(float)

# Design matrix (standardized/simplified): keep the Bayesian model parsimonious
X = pd.DataFrame({
    "age_dec": (df["age"] - 60.0) / 10.0,
    "log_nodes": np.log1p(df["nodes"]),
    "female": (df["sex"] == "Female").astype(float),
    "obstruction": (df["obstruct"] == "Yes").astype(float),
    "extent_advanced": df["extent"].isin(["Serosa", "Contiguous structures"]).astype(float),
    "surgery_long": (df["surg"] == "Long").astype(float),
}).values
P = X.shape[1]
COV_NAMES = list(pd.DataFrame(X).columns)

# ------------------------------------------------------------------------------
# 2. Weibull PH model with right censoring via pm.Censored
# ------------------------------------------------------------------------------
# Parametrization: H(t | x) = (t / beta_i)^alpha with
# beta_i = beta0 * exp(-eta_i / alpha), eta_i = beta_treat * treat + x'gamma.
# Then H(t | x) = (t / beta0)^alpha * exp(eta_i): a Weibull PH model where
# exp(beta_treat) is the hazard ratio of the active arm.
# Uncensored observations use upper = +infinity (ordinary likelihood);
# censored observations use upper = t_i (contributes the survival term).

PRIORS = {
    # Skeptical: P(HR < 0.5 a priori) = 5%, i.e. sigma = |log 0.5| / 1.645
    "Skeptical": dict(mu=0.0, sigma=abs(np.log(0.5)) / 1.645),
    # Weakly informative: broad, centred at no effect
    "Weakly-informative": dict(mu=0.0, sigma=1.0),
    # Enthusiastic: centred at the design alternative HR = 0.65
    "Enthusiastic": dict(mu=np.log(0.65), sigma=abs(np.log(0.5)) / 1.645),
}


def fit_weibull(t, ev, tr, Xmat, prior, draws=2000, tune=2000,
                chains=2, seed=common.RANDOM_SEED):
    upper = np.where(ev, np.inf, t)
    with pm.Model() as model:
        alpha = pm.LogNormal("alpha", mu=np.log(1.2), sigma=0.4)
        beta0 = pm.LogNormal("beta0", mu=0.0, sigma=1.5)
        beta_treat = pm.Normal("beta_treat", mu=prior["mu"],
                               sigma=prior["sigma"])
        gamma = pm.Normal("gamma", mu=0.0, sigma=0.5, shape=P)
        eta = beta_treat * tr + pm.math.dot(Xmat, gamma)
        beta_i = beta0 * pm.math.exp(-eta / alpha)
        obs = pm.Censored(
            "obs", pm.Weibull.dist(alpha=alpha, beta=beta_i),
            lower=None, upper=upper, observed=t)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains,
                          target_accept=0.9, random_seed=seed,
                          progressbar=False, idata_kwargs={"log_likelihood": True})
    return idata


# ------------------------------------------------------------------------------
# 3. Prior sensitivity analysis
# ------------------------------------------------------------------------------
post_summaries = []
posteriors = {}
for name, prior in PRIORS.items():
    print(f"Sampling: {name} prior ...")
    idata = fit_weibull(t_obs, event, treat, X, prior)
    posteriors[name] = idata
    bt = idata.posterior["beta_treat"].values.ravel()
    hr = np.exp(bt)
    post_summaries.append({
        "prior": name,
        "prior_mu_logHR": round(prior["mu"], 3),
        "prior_sigma_logHR": round(prior["sigma"], 3),
        "posterior_mean_HR": round(float(hr.mean()), 3),
        "posterior_median_HR": round(float(np.median(hr)), 3),
        "cr2.5": round(float(np.percentile(hr, 2.5)), 3),
        "cr97.5": round(float(np.percentile(hr, 97.5)), 3),
        "P_HR_lt_1": round(float((hr < 1).mean()), 4),
        "P_HR_lt_0.8": round(float((hr < 0.8).mean()), 4),
    })
    print(f"  HR = {post_summaries[-1]['posterior_mean_HR']} "
          f"({post_summaries[-1]['cr2.5']}, {post_summaries[-1]['cr97.5']}), "
          f"P(HR<1) = {post_summaries[-1]['P_HR_lt_1']}")

# Frequentist comparator (module 3, adjusted MI Cox model)
freq_row = {
    "prior": "Frequentist (adj. Cox, MI) [reference]",
    "prior_mu_logHR": np.nan, "prior_sigma_logHR": np.nan,
    "posterior_mean_HR": 0.621, "posterior_median_HR": np.nan,
    "cr2.5": 0.496, "cr97.5": 0.776,
    "P_HR_lt_1": np.nan, "P_HR_lt_0.8": np.nan,
}
bayes_tab = pd.DataFrame(post_summaries + [freq_row])
common.save_tab(bayes_tab, "table_bayes_summary")
print("\n", bayes_tab.to_string(index=False))

# ---- Prior/posterior figure ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
grid = np.linspace(-2.2, 1.2, 500)
hr_grid = np.exp(grid)
colors = {"Skeptical": common.BLUE,
          "Weakly-informative": common.ORANGE,
          "Enthusiastic": common.GREEN}
for name, prior in PRIORS.items():
    d = np.exp(-0.5 * ((grid - prior["mu"]) / prior["sigma"]) ** 2) / \
        (prior["sigma"] * np.sqrt(2 * np.pi))
    axes[0].plot(hr_grid, d, color=colors[name], linewidth=1.3, label=name)
    bt = posteriors[name].posterior["beta_treat"].values.ravel()
    axes[1].hist(np.exp(bt), bins=60, density=True, histtype="step",
                 linewidth=1.3, color=colors[name])
axes[0].axvline(1.0, color="grey", linestyle="--", linewidth=0.9)
axes[0].set_xlim(0, 3.5)
axes[0].set_xlabel("Hazard ratio (prior scale)")
axes[0].set_ylabel("Prior density")
# Short titles: the previous long titles were wider than their panels and
# collided with each other at the figure centre.
axes[0].set_title("Priors for the treatment effect", fontsize=11)
axes[0].legend(fontsize=9)
axes[1].axvline(1.0, color="grey", linestyle="--", linewidth=0.9)
axes[1].axvline(0.621, color=common.ACCENT, linestyle=":", linewidth=1.3)
axes[1].text(0.621, axes[1].get_ylim()[1] * 0.95, " frequentist HR = 0.62",
             color=common.ACCENT, fontsize=9, va="top",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                       pad=1.5))
axes[1].set_xlim(0.2, 1.4)
axes[1].set_xlabel("Hazard ratio (posterior)")
axes[1].set_ylabel("Posterior density")
axes[1].set_title("Posterior of the treatment HR", fontsize=11)
fig.tight_layout()
common.save_fig(fig, "fig_bayes_priors")

# ------------------------------------------------------------------------------
# 4. Bayesian sequential monitoring (skeptical prior, event-driven looks)
# ------------------------------------------------------------------------------
frac_list = [0.25, 0.50, 0.75, 1.00]
n_events_tot = int(event.sum())
event_times = np.sort(t_obs[event])
seq_rows = []
print("\nSequential monitoring (skeptical prior, event-driven analyses):")
for f in frac_list:
    m = max(1, int(round(f * n_events_tot)))
    cut = event_times[m - 1]
    keep_t = np.minimum(t_obs, cut)
    keep_e = event & (t_obs <= cut)
    idata = fit_weibull(keep_t, keep_e, treat, X, PRIORS["Skeptical"],
                        draws=1000, tune=1000, seed=common.RANDOM_SEED + 7)
    bt = idata.posterior["beta_treat"].values.ravel()
    hr = np.exp(bt)
    seq_rows.append({
        "info_fraction": f,
        "events": int(keep_e.sum()),
        "posterior_mean_HR": round(float(hr.mean()), 3),
        "cr2.5": round(float(np.percentile(hr, 2.5)), 3),
        "cr97.5": round(float(np.percentile(hr, 97.5)), 3),
        "P_HR_lt_1": round(float((hr < 1).mean()), 4),
    })
    print(f"  {100 * f:>4.0f}% ({int(keep_e.sum()):>3d} events): "
          f"HR = {seq_rows[-1]['posterior_mean_HR']} "
          f"({seq_rows[-1]['cr2.5']}, {seq_rows[-1]['cr97.5']}), "
          f"P(HR<1) = {seq_rows[-1]['P_HR_lt_1']}")
seq_tab = pd.DataFrame(seq_rows)
common.save_tab(seq_tab, "table_bayes_sequential")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
axes[0].errorbar(seq_tab["info_fraction"], seq_tab["posterior_mean_HR"],
                 yerr=[seq_tab["posterior_mean_HR"] - seq_tab["cr2.5"],
                       seq_tab["cr97.5"] - seq_tab["posterior_mean_HR"]],
                 fmt="o-", color=common.ACCENT, capsize=3, markersize=5)
axes[0].axhline(1.0, color="grey", linestyle="--", linewidth=0.9)
axes[0].set_xlabel("Information fraction (DFS events)")
axes[0].set_ylabel("Posterior mean HR (95% CrI)")
axes[0].set_title("Sequential point estimates")
axes[0].set_ylim(0.3, 1.2)
axes[1].plot(seq_tab["info_fraction"], seq_tab["P_HR_lt_1"], "o-",
             color=common.BLUE, markersize=5)
axes[1].axhline(0.975, color=common.ACCENT, linestyle="--", linewidth=0.9)
axes[1].text(0.26, 0.955, "P(HR < 1) = 0.975 threshold",
             color=common.ACCENT, fontsize=9,
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                       pad=1.5))
axes[1].set_xlabel("Information fraction (DFS events)")
axes[1].set_ylabel("P(HR < 1 | data)")
axes[1].set_title("Bayesian monitoring statistic")
axes[1].set_ylim(0, 1.02)
fig.tight_layout()
common.save_fig(fig, "fig_bayes_sequential")

# ------------------------------------------------------------------------------
# 5. Posterior predictive check (control arm)
# ------------------------------------------------------------------------------
print("\nPosterior predictive check (control arm) ...")
idata_main = posteriors["Weakly-informative"]
post = idata_main.posterior
n_ctrl = int((treat == 0).sum())
draws_sel = np.random.choice(post["beta_treat"].values.ravel().shape[0],
                             size=100, replace=False)
alpha_s = post["alpha"].values.ravel()[draws_sel]
beta0_s = post["beta0"].values.ravel()[draws_sel]
gam_s = post["gamma"].values.reshape(-1, P)[draws_sel]
Xc = X[treat == 0]

from lifelines import KaplanMeierFitter
grid_t = np.linspace(0.01, 8.0, 120)
km_curves = np.zeros((len(draws_sel), len(grid_t)))
for s in range(len(draws_sel)):
    eta = Xc @ gam_s[s]
    beta_i = beta0_s[s] * np.exp(-eta / alpha_s[s])
    u = np.random.uniform(size=n_ctrl)
    t_rep = beta_i * (-np.log(u)) ** (1 / alpha_s[s])   # inverse survival fn
    kmf = KaplanMeierFitter()
    kmf.fit(t_rep)
    km_curves[s] = kmf.predict(grid_t).values
km_lo = np.percentile(km_curves, 2.5, axis=0)
km_hi = np.percentile(km_curves, 97.5, axis=0)

kmf_obs = KaplanMeierFitter()
kmf_obs.fit(t_obs[treat == 0], event_observed=event[treat == 0])

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.fill_between(grid_t, km_lo, km_hi, color=common.BLUE, alpha=0.18,
                label="Posterior predictive 95% band")
ax.plot(grid_t, km_curves.mean(axis=0), color=common.BLUE, linewidth=1.1,
        label="Posterior predictive mean")
km_obs_df = kmf_obs.survival_function_at_times(grid_t)
ax.step(grid_t, km_obs_df.values, where="post", color=common.ACCENT,
        linewidth=1.4, label="Observed (Observation arm)")
ax.set_xlabel("Years since randomization")
ax.set_ylabel("Disease-free survival probability")
ax.set_title("Posterior predictive check: replicated vs observed survival")
ax.legend(loc="lower left")
ax.set_ylim(0, 1.02)
fig.tight_layout()
common.save_fig(fig, "fig_bayes_ppc")

# ------------------------------------------------------------------------------
# 6. Model criticism: Weibull vs exponential baseline (LOO)
# ------------------------------------------------------------------------------
print("\nModel comparison (LOO): Weibull vs exponential baseline hazard ...")
upper = np.where(event, np.inf, t_obs)
with pm.Model() as m_exp:
    lam0 = pm.LogNormal("lam0", mu=np.log(0.25), sigma=1.5)
    beta_treat = pm.Normal("beta_treat", mu=0.0, sigma=1.0)
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5, shape=P)
    eta = beta_treat * treat + pm.math.dot(X, gamma)
    lam_i = lam0 * pm.math.exp(eta)
    obs = pm.Censored("obs", pm.Exponential.dist(lam=lam_i),
                      lower=None, upper=upper, observed=t_obs)
    idata_exp = pm.sample(draws=2000, tune=2000, chains=2, cores=2,
                          target_accept=0.9, random_seed=common.RANDOM_SEED,
                          progressbar=False,
                          idata_kwargs={"log_likelihood": True})

cmp = az.compare({"Weibull PH": idata_main, "Exponential PH": idata_exp})
cmp_tab = cmp.reset_index().rename(columns={"index": "model"})
cmp_cols = [c for c in ["model", "rank", "elpd_loo", "p_loo", "elpd_diff",
                        "weight"] if c in cmp_tab.columns]
common.save_tab(cmp_tab[cmp_cols].round(2), "table_bayes_model_compare")
print(cmp_tab[cmp_cols].round(2).to_string(index=False))

alpha_s = post["alpha"].values.ravel()
print(f"\nPosterior of Weibull shape alpha: mean = {alpha_s.mean():.3f}, "
      f"95% CrI = ({np.percentile(alpha_s, 2.5):.3f}, "
      f"{np.percentile(alpha_s, 97.5):.3f})")

print("\n=== 03_bayesian_analysis.py completed ===")
