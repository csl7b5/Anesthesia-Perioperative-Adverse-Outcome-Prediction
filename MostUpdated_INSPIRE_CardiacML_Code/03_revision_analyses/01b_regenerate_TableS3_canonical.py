#!/usr/bin/env python3
"""
Regenerate Supplementary Table S3 (MICE sensitivity) so its "Primary pipeline"
baseline matches the canonical S5 model (Table S5 / Table 2).

Root causes of the old S3 discrepancy that this script fixes:
  1. Old S3 used y_icu = icu_admit (any ICU admission) -> AUROC 0.903.
     Canonical outcome is extended ICU stay (icu_los_min >= 4320 min = 3 days).
  2. Old S3 built features from dynamic_features_vitals.csv only, not the S5
     pipeline. Here we reproduce the S5 feature pipeline exactly, then MICE-
     impute only the vital-sign columns (vasopressor/coupling stay zero).
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)

# ---- canonical parameters (identical to comment_5_performance) --------------
RANDOM_STATE, TEST_SIZE = 42, 0.30
ICU_THRESHOLD, DROP_FRAC, PRUNE_FRAC = 4320, 0.70, 0.85
N_ESTIMATORS, N_IMPUTATIONS = 100, 5
np.random.seed(RANDOM_STATE)

# ---- S5 feature pipeline ----------------------------------------------------
df          = pd.read_csv("model_combined_dataset.csv")
dyn_raw     = pd.read_csv("dynamic_features_vitals.csv")
dyn_missing = dyn_raw.isna().mean()

dynamic_prefixes = ("mean_", "std_", "slope_", "auc_", "min_", "max_",
                    "avg_rate_", "duration_", "num_events_", "total_dose_")
coupling_tokens  = ("_lag", "_slope", "_ri")
protected        = ("avg_rate_", "duration_", "num_events_", "total_dose_") + coupling_tokens
leak_cols = ["subject_id", "op_id", "died_inhospital", "icu_admit", "icu_los_min",
             "allcause_death_time", "icuin_time"]

op_id_all = df["op_id"].values
df_model  = df.drop(columns=[c for c in leak_cols if c in df.columns])
candidates = [c for c in df_model.columns
              if any(c.startswith(p) for p in dynamic_prefixes)
              or any(tok in c for tok in coupling_tokens)]
feature_cols = [f for f in candidates
                if (dyn_missing.get(f, 0) <= DROP_FRAC)
                or any(f.startswith(p) for p in protected)]
X_full = df_model[feature_cols].select_dtypes(include=[np.number]).fillna(0)
corr   = X_full.corr().abs()
upper  = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
final_cols = [c for c in X_full.columns if c not in
              {c for c in upper.columns if any(upper[c] > PRUNE_FRAC)}]
X_zero = X_full[final_cols].copy()
X_zero["op_id"] = op_id_all

# ---- outcomes (canonical) ---------------------------------------------------
df_cl = pd.read_csv("model_combined_dataset_clustered.csv")
outc  = df_cl[["op_id", "died_inhospital", "icu_los_min"]].copy()
outc["y_mort"] = outc["died_inhospital"].astype(int)
outc["y_icu"]  = (outc["icu_los_min"] >= ICU_THRESHOLD).astype(int)

merged = X_zero.merge(outc[["op_id", "y_mort", "y_icu"]], on="op_id", how="inner")
y_mort = merged["y_mort"].values
y_icu  = merged["y_icu"].values
op_ids = merged["op_id"].values
X_zero_arr = merged[final_cols].values.astype(float)

print(f"Feature matrix: {X_zero_arr.shape}")
print(f"Mortality events: {y_mort.sum()}/{len(y_mort)} ({y_mort.mean()*100:.1f}%)")
print(f"Extended ICU (>3d): {y_icu.sum()}/{len(y_icu)} ({y_icu.mean()*100:.1f}%)")

# ---- vital-sign columns to MICE-impute (others stay zero) -------------------
vital_final = [c for c in final_cols if c in set(dyn_raw.columns)]
vital_idx   = [final_cols.index(c) for c in vital_final]
print(f"Final features: {len(final_cols)} | vital (MICE-imputed): {len(vital_final)} | "
      f"vaso/coupling (kept 0): {len(final_cols)-len(vital_final)}")

# NaN-preserved vital values aligned to merged rows
dyn_aligned = (pd.DataFrame({"op_id": op_ids})
               .merge(dyn_raw[["op_id"] + vital_final], on="op_id", how="left"))
X_nan = X_zero_arr.copy()
X_nan[:, vital_idx] = dyn_aligned[vital_final].values.astype(float)

# ---- helpers ----------------------------------------------------------------
def cal_slope_intercept(y_true, y_prob):
    lp = np.log(np.clip(y_prob, 1e-6, 1-1e-6) / (1 - np.clip(y_prob, 1e-6, 1-1e-6)))
    lr = LogisticRegression(fit_intercept=True, max_iter=1000, solver="lbfgs")
    lr.fit(lp.reshape(-1, 1), y_true)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])

def run_rf(X, y, seed=RANDOM_STATE):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                          stratify=y, random_state=seed)
    Xtr_s, ytr_s = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed)
    clf.fit(Xtr_s, ytr_s)
    p = clf.predict_proba(Xte)[:, 1]
    inter, slope = cal_slope_intercept(yte.astype(int), p)
    return dict(auroc=roc_auc_score(yte, p), brier=brier_score_loss(yte, p),
                bal=balanced_accuracy_score(yte, (p >= 0.5).astype(int)),
                slope=slope, intercept=inter)

# ---- baseline (zero-imputation, canonical) ----------------------------------
print("\n=== Primary pipeline (zero-imputation) ===")
base = {"mortality": run_rf(X_zero_arr, y_mort), "icu": run_rf(X_zero_arr, y_icu)}
for k, v in base.items():
    print(f"  {k:10s} AUROC={v['auroc']:.3f} Brier={v['brier']:.3f} "
          f"Bal={v['bal']:.3f} slope={v['slope']:.3f} int={v['intercept']:.3f}")

# ---- MICE (impute vitals only, m=5) -----------------------------------------
print(f"\n=== MICE (m={N_IMPUTATIONS}) — imputing vital columns ===")
mice = {"mortality": [], "icu": []}
vital_block = X_nan[:, vital_idx]
for m in range(N_IMPUTATIONS):
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10,
                           sample_posterior=True, random_state=m*37+13, skip_complete=True)
    imputed_vitals = imp.fit_transform(vital_block)
    X_imp = X_zero_arr.copy()
    X_imp[:, vital_idx] = imputed_vitals
    mice["mortality"].append(run_rf(X_imp, y_mort))
    mice["icu"].append(run_rf(X_imp, y_icu))
    print(f"  m={m+1} mort AUROC={mice['mortality'][-1]['auroc']:.3f} "
          f"icu AUROC={mice['icu'][-1]['auroc']:.3f}")

def pool(rows, key):
    vals = [r[key] for r in rows]
    return np.mean(vals), np.std(vals, ddof=1)

# ---- build S3 comparison table ----------------------------------------------
recs = []
name = {"mortality": "In-hospital mortality", "icu": "Extended ICU stay"}
for k in ["mortality", "icu"]:
    b = base[k]
    recs.append({"Outcome": name[k], "Imputation": "Primary pipeline",
                 "AUROC": f"{b['auroc']:.3f}", "Brier score": f"{b['brier']:.3f}",
                 "Balanced accuracy": f"{b['bal']:.3f}",
                 "Calibration slope": f"{b['slope']:.3f}",
                 "Calibration intercept": f"{b['intercept']:.3f}"})
    am, asd = pool(mice[k], "auroc"); bm, bsd = pool(mice[k], "brier")
    lm, lsd = pool(mice[k], "bal");   sm, ssd = pool(mice[k], "slope")
    im, isd = pool(mice[k], "intercept")
    recs.append({"Outcome": name[k], "Imputation": f"MICE m = {N_IMPUTATIONS}, mean (SD)",
                 "AUROC": f"{am:.3f} ({asd:.3f})", "Brier score": f"{bm:.3f} ({bsd:.3f})",
                 "Balanced accuracy": f"{lm:.3f} ({lsd:.3f})",
                 "Calibration slope": f"{sm:.3f} ({ssd:.3f})",
                 "Calibration intercept": f"{im:.3f} ({isd:.3f})"})

out = pd.DataFrame(recs)
out.to_csv("reviewer_responses/comment_1_missing_data/mice_vs_zero_comparison.csv", index=False)
print("\n=== Supplementary Table S3 (regenerated) ===")
print(out.to_string(index=False))
print("\nSaved -> reviewer_responses/comment_1_missing_data/mice_vs_zero_comparison.csv")
