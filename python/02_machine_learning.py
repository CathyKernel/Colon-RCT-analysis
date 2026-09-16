"""
02_machine_learning.py

Project : Re-analysis of the NCCTG Adjuvant Colon Cancer Randomized Trial
          with Frequentist, Bayesian, and Machine Learning Methods
Author  : Cathy
Purpose : Machine-learning prediction of the composite disease-free survival
          endpoint (prognostic modelling). Five models are compared with
          repeated stratified cross-validation:
            1. Clinical Cox model (main effects, benchmark)
            2. Elastic-net penalized Cox regression
            3. Random survival forest
            4. Gradient boosting with Cox partial-likelihood loss
            5. XGBoost with Cox objective (risk scores; C-index only)
          Discrimination is quantified with Harrell's C-index, dynamic AUC at
          5 years (IPCW), and prediction error with the integrated Brier
          score. The best model is calibrated graphically and interpreted via
          permutation importance.

Inputs  : data/processed/colon_analysis.csv
Outputs : outputs/figures/fig_ml_cv_performance.png
                     fig_ml_calibration_auc.png
                     fig_ml_importance.png
          outputs/tables/table_ml_performance.csv
                     table_ml_calibration.csv
                     table_ml_importance.csv

Note    : Prognostic prediction is a distinct task from treatment-effect
          estimation (which is addressed in 03 and 04): models here are fit
          WITHOUT the randomized treatment assignment as a predictor, so the
          models quantify baseline risk only.
"""

import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, "python")
from importlib import import_module
common = import_module("00_common")

from sklearn.model_selection import StratifiedKFold
from lifelines import CoxPHFitter
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored,
                            cumulative_dynamic_auc,
                            integrated_brier_score)
import xgboost as xgb

warnings.filterwarnings("ignore")
common.apply_style()
common.set_seed()
print("=== 02_machine_learning.py ===\n")

# ------------------------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------------------------
df = common.load_analysis_data()
feat_cols = ["age", "nodes", "sex", "obstruct", "perfor", "adhere", "differ",
             "extent", "surg"]
df_cc = df.dropna(subset=feat_cols).reset_index(drop=True)
print(f"Complete cases for prediction modelling: {len(df_cc)} / {len(df)}")
X = common.design_matrix(df_cc)
y_event = df_cc["dfs_status"].astype(bool).values
y_time = df_cc["dfs_time"].values.astype(float)
y = Surv.from_arrays(event=y_event, time=y_time)

EVAL_TIMES = np.arange(1.0, 6.0, 1.0)      # 1..5 years for the Brier score
HORIZON = 5.0                              # dynamic AUC horizon

# ------------------------------------------------------------------------------
# 2. Model zoo (treatment deliberately excluded: pure prognostic models)
# ------------------------------------------------------------------------------

def fit_clinical_cox(Xtr, ytr):
    d = Xtr.copy()
    d["time"] = ytr["time"]
    d["event"] = ytr["event"]
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(d, duration_col="time", event_col="event")
    return cph


def surv_matrix_lifelines(model, Xte, times):
    sf = model.predict_survival_function(Xte, times=times)
    return sf.values.T                      # (n, len(times))


def surv_matrix_sksurv(model, Xte, times, kind="survival"):
    if kind == "survival":
        fns = model.predict_survival_function(Xte)
    else:
        fns = model.predict_cumulative_hazard_function(Xte)
    if kind == "cumhaz":
        return np.array([[np.exp(-fn(t)) for t in times] for fn in fns])
    return np.array([[fn(t) for t in times] for fn in fns])


MODELS = {
    "Clinical Cox (main effects)": {
        "fit": fit_clinical_cox,
        # lifelines' partial hazard is exp(linear predictor): higher = riskier,
        # matching sksurv's convention (higher score = higher event risk)
        "risk": lambda m, X: m.predict_partial_hazard(X).values.ravel(),
        "surv": lambda m, X, t: surv_matrix_lifelines(m, X, t),
        "has_curves": True,
    },
    "Elastic-net Cox": {
        "fit": lambda Xtr, ytr: CoxnetSurvivalAnalysis(
            l1_ratio=0.5, alpha_min_ratio=0.05, n_alphas=50,
            fit_baseline_model=True).fit(Xtr, ytr),
        "risk": lambda m, X: m.predict(X).ravel(),
        "surv": lambda m, X, t: surv_matrix_sksurv(m, X, t, "survival"),
        "has_curves": True,
    },
    "Random survival forest": {
        "fit": lambda Xtr, ytr: RandomSurvivalForest(
            n_estimators=400, min_samples_leaf=8, max_features="sqrt",
            n_jobs=1, random_state=common.RANDOM_SEED).fit(Xtr, ytr),
        "risk": lambda m, X: m.predict(X),
        "surv": lambda m, X, t: surv_matrix_sksurv(m, X, t, "survival"),
        "has_curves": True,
    },
    "Gradient boosting (Cox loss)": {
        "fit": lambda Xtr, ytr: GradientBoostingSurvivalAnalysis(
            loss="coxph", learning_rate=0.05, n_estimators=250,
            min_samples_leaf=8, random_state=common.RANDOM_SEED).fit(Xtr, ytr),
        "risk": lambda m, X: m.predict(X),
        "surv": lambda m, X, t: surv_matrix_sksurv(m, X, t, "cumhaz"),
        "has_curves": True,
    },
    "XGBoost (Cox objective)": {
        "fit": lambda Xtr, ytr: xgb.XGBRegressor(
            objective="survival:cox", n_estimators=300, learning_rate=0.05,
            max_depth=3, subsample=0.8, colsample_bytree=0.8,
            random_state=common.RANDOM_SEED).fit(
                Xtr, np.where(ytr["event"], ytr["time"], -ytr["time"])),
        "risk": lambda m, X: m.predict(X),
        "surv": None,
        "has_curves": False,
    },
}

# ------------------------------------------------------------------------------
# 3. Repeated stratified cross-validation
# ------------------------------------------------------------------------------
N_SPLITS, N_REPEATS = 5, 5
print(f"Cross-validation: {N_REPEATS} x {N_SPLITS}-fold, stratified on event "
      f"indicator\n")

cv_records = []
oof_store = {name: np.full(len(df_cc), np.nan) for name in MODELS}

for rep in range(N_REPEATS):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=common.RANDOM_SEED + rep)
    for fold, (tr, te) in enumerate(skf.split(X, y_event)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        dtr = df_cc.iloc[tr]

        for name, spec in MODELS.items():
            model = spec["fit"](Xtr, ytr)
            risk = spec["risk"](model, Xte)
            c = concordance_index_censored(yte["event"], yte["time"], risk)[0]
            rec = {"model": name, "repeat": rep, "fold": fold, "c_index": c}
            if spec["has_curves"]:
                surv = spec["surv"](model, Xte, EVAL_TIMES)
                auc = cumulative_dynamic_auc(ytr, yte, risk, [HORIZON])[0][0]
                ibs = integrated_brier_score(ytr, yte, surv, EVAL_TIMES)
                rec["auc_5y"] = auc
                rec["ibs_1to5y"] = ibs
                # out-of-fold survival function at the 5-year horizon
                sf5 = surv[:, int(HORIZON) - 1]
                oof_store[name][te] = 1.0 - sf5
            cv_records.append(rec)

cv_df = pd.DataFrame(cv_records)
print(cv_df.groupby("model")[["c_index", "auc_5y", "ibs_1to5y"]]
      .agg(["mean", "std"]).round(4).to_string())

# ---- Apparent (in-sample) performance for the overfitting contrast -----------
print("\nApparent (in-sample) performance:")
apparent = []
for name, spec in MODELS.items():
    model = spec["fit"](X, y)
    risk = spec["risk"](model, X)
    c_app = concordance_index_censored(y_event, y_time, risk)[0]
    apparent.append({"model": name, "c_index_apparent": round(c_app, 4)})
    print(f"  {name:32s} C = {c_app:.4f}")

# ---- Summary table -------------------------------------------------------------
summary = (cv_df.groupby("model")
           .agg(c_index_cv=("c_index", "mean"),
                c_index_cv_sd=("c_index", "std"),
                auc_5y_cv=("auc_5y", "mean"),
                ibs_1to5y_cv=("ibs_1to5y", "mean"))
           .reset_index())
summary = summary.merge(pd.DataFrame(apparent), on="model")
summary = summary.round(4).sort_values("c_index_cv", ascending=False)
common.save_tab(summary, "table_ml_performance")
print("\n", summary.to_string(index=False))

# ------------------------------------------------------------------------------
# 4. CV performance figure
# ------------------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
order = (cv_df.groupby("model")["c_index"].mean()
         .sort_values(ascending=False).index.tolist())
data_box = [cv_df.loc[cv_df["model"] == m, "c_index"].values for m in order]
bp = axes[0].boxplot(data_box, vert=False, patch_artist=True, widths=0.55)
for patch in bp["boxes"]:
    patch.set_facecolor("#dceefb")
    patch.set_edgecolor(common.BLUE)
for med in bp["medians"]:
    med.set_color(common.ACCENT)
    med.set_linewidth(1.6)
for i, m in enumerate(order):
    axes[0].scatter(data_box[i], np.full(len(data_box[i]), i + 1),
                    s=12, color=common.BLUE, alpha=0.45, zorder=3)
axes[0].set_yticks(range(1, len(order) + 1))
axes[0].set_yticklabels(order)
axes[0].set_xlabel("Harrell's C-index (25 cross-validation folds)")
axes[0].set_title("Discrimination: cross-validated")

auc_tab = cv_df.dropna(subset=["auc_5y"]).groupby("model")["auc_5y"]
means = auc_tab.mean().loc[[m for m in order if m in auc_tab.mean().index]]
sds = auc_tab.std().loc[means.index]
axes[1].errorbar(range(len(means)), means.values, yerr=sds.values,
                 fmt="o", color=common.ACCENT, capsize=3, markersize=5)
axes[1].set_xticks(range(len(means)))
axes[1].set_xticklabels([m.replace(" (", "\n(") for m in means.index],
                        fontsize=8)
axes[1].set_ylabel(f"AUC at t = {HORIZON:.0f} years (IPCW)")
axes[1].set_title("Time-dependent discrimination")
axes[1].set_ylim(0.5, 1.0)
fig.suptitle("")
fig.tight_layout()
common.save_fig(fig, "fig_ml_cv_performance")

# ------------------------------------------------------------------------------
# 5. Calibration and horizon ROC for the best curve-capable model
# ------------------------------------------------------------------------------
best_name = summary.loc[summary["c_index_cv"].idxmax(), "model"]
print(f"\nBest model by CV C-index: {best_name}")

best_spec = MODELS[best_name]
yhat5 = oof_store[best_name]
assert not np.isnan(yhat5).any()

# 5-year "observed" event indicator with administrative censoring at 5 years
event5 = ((y_time <= HORIZON) & y_event).astype(int)
at_risk5 = y_time > HORIZON

# KM-based observed event probability within deciles of predicted risk
from lifelines import KaplanMeierFitter
q = pd.Series(pd.qcut(yhat5, 10, labels=False, duplicates="drop")).values
cal_rows = []
for g in sorted(np.unique(q)):
    idx = (q == g)
    kmf = KaplanMeierFitter()
    kmf.fit(y_time[idx], event_observed=y_event[idx])
    cal_rows.append({
        "decile": int(g) + 1,
        "n": int(idx.sum()),
        "events_by_5y": int(event5[idx].sum()),
        "predicted_risk": float(np.mean(yhat5[idx])),
        "observed_km_risk": float(1 - kmf.predict(HORIZON)),
    })
cal_tab = pd.DataFrame(cal_rows).round(4)
common.save_tab(cal_tab, "table_ml_calibration")

# ROC at 5 years (IPCW-free descriptive ROC on the 5-year binary outcome,
# restricted to patients whose horizon is observable)
mask = at_risk5 | (event5 == 1)
from sklearn.metrics import roc_curve, auc as sk_auc
fpr, tpr, _ = roc_curve(event5[mask], yhat5[mask])
roc_auc = sk_auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
axes[0].plot(fpr, tpr, color=common.ACCENT, linewidth=1.4,
             label=f"{best_name}\nAUC = {roc_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "--", color="grey", linewidth=0.9)
axes[0].set_xlabel("False-positive rate")
axes[0].set_ylabel("True-positive rate")
axes[0].set_title(f"5-year event classification")
axes[0].legend(loc="lower right")

lim = [0, 1]
axes[1].plot(lim, lim, "--", color="grey", linewidth=0.9)
axes[1].scatter(cal_tab["observed_km_risk"], cal_tab["predicted_risk"],
                s=28, color=common.BLUE, zorder=3, clip_on=False)
axes[1].set_xlabel("Observed 5-year event risk (Kaplan-Meier, by decile)")
axes[1].set_ylabel("Mean predicted 5-year risk")
axes[1].set_title("Calibration by risk decile")
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)
fig.tight_layout()
common.save_fig(fig, "fig_ml_calibration_auc")

# ------------------------------------------------------------------------------
# 6. Permutation importance of the best model (single 75/25 split)
# ------------------------------------------------------------------------------
rng = np.random.default_rng(common.RANDOM_SEED)
idx = np.arange(len(df_cc))
rng.shuffle(idx)
n_tr = int(0.75 * len(idx))
tr, te = idx[:n_tr], idx[n_tr:]
Xtr, Xte = X.iloc[tr], X.iloc[te]
ytr, yte = y[tr], y[te]
model = best_spec["fit"](Xtr, ytr)
base_c = concordance_index_censored(yte["event"], yte["time"],
                                    best_spec["risk"](model, Xte))[0]

n_perm = 20
imp_rows = []
for j, col in enumerate(X.columns):
    drops = []
    for _ in range(n_perm):
        Xp = Xte.copy()
        Xp[col] = Xp[col].sample(frac=1.0, random_state=int(rng.integers(1e9))).values
        drops.append(base_c - concordance_index_censored(
            yte["event"], yte["time"], best_spec["risk"](model, Xp))[0])
    imp_rows.append({"variable": col,
                     "mean_drop_c": float(np.mean(drops)),
                     "sd_drop_c": float(np.std(drops))})
imp_tab = (pd.DataFrame(imp_rows)
           .sort_values("mean_drop_c", ascending=False).round(4))
common.save_tab(imp_tab, "table_ml_importance")
print(imp_tab.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.4))
ax.barh(imp_tab["variable"][::-1], imp_tab["mean_drop_c"][::-1],
        xerr=imp_tab["sd_drop_c"][::-1], color=common.BLUE,
        error_kw=dict(lw=0.8, capsize=2), alpha=0.85)
ax.set_xlabel(f"Mean decrease in C-index upon permutation ({n_perm} shuffles)")
ax.set_title(f"Permutation importance: {best_name}")
fig.tight_layout()
common.save_fig(fig, "fig_ml_importance")

print("\n=== 02_machine_learning.py completed ===")
