"""Regenerate model_comparison_results.csv (random + temporal splits, full bootstrap CIs)."""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INSPIRE_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
os.chdir(INSPIRE_ROOT)
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
OUTPUT_DIR = _SCRIPT_DIR

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.30
ICU_THRESHOLD = 4320
DROP_FRAC = 0.70
PRUNE_FRAC = 0.85
N_BOOTSTRAP = 500
np.random.seed(RANDOM_STATE)

MODEL_COMPARISON_COLS = [
    "split",
    "outcome",
    "model",
    "auroc",
    "auroc_lo",
    "auroc_hi",
    "brier",
    "brier_lo",
    "brier_hi",
    "bal_acc",
    "bal_lo",
    "bal_hi",
]


def delong_ci(y_true, y_score, alpha=0.05):
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n1, n0 = len(pos), len(neg)
    auc = roc_auc_score(y_true, y_score)
    vx = np.array([(pos[i] > neg).mean() + 0.5 * (pos[i] == neg).mean() for i in range(n1)])
    vy = np.array([(neg[i] < pos).mean() + 0.5 * (neg[i] == pos).mean() for i in range(n0)])
    var = np.var(vx, ddof=1) / n1 + np.var(vy, ddof=1) / n0
    z = stats.norm.ppf(1 - alpha / 2)
    return auc, max(0.0, auc - z * np.sqrt(var)), min(1.0, auc + z * np.sqrt(var))


def bstrap_ci(y_true, y_score, fn, n=N_BOOTSTRAP):
    vals = []
    for _ in range(n):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        yb, pb = y_true[idx], y_score[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        try:
            vals.append(fn(yb, pb))
        except Exception:
            pass
    return np.percentile(vals, [2.5, 97.5]) if vals else [np.nan, np.nan]


# Feature pipeline (same as notebook)
df = pd.read_csv("model_combined_dataset.csv")
dyn_missing = pd.read_csv("dynamic_features_vitals.csv").isna().mean()
ops_dates = pd.read_csv("data/extracted_operations.csv", usecols=["op_id", "opdate"]).dropna()

dynamic_prefixes = (
    "mean_",
    "std_",
    "slope_",
    "auc_",
    "min_",
    "max_",
    "avg_rate_",
    "duration_",
    "num_events_",
    "total_dose_",
)
coupling_tokens = ("_lag", "_slope", "_ri")
leak_cols = [
    "subject_id",
    "op_id",
    "died_inhospital",
    "icu_admit",
    "icu_los_min",
    "allcause_death_time",
    "icuin_time",
]
df_model = df.drop(columns=[c for c in leak_cols if c in df.columns])
protected = ("avg_rate_", "duration_", "num_events_", "total_dose_") + tuple(coupling_tokens)
candidates = [
    c
    for c in df_model.columns
    if any(c.startswith(p) for p in dynamic_prefixes) or any(tok in c for tok in coupling_tokens)
]
feature_cols = [
    f
    for f in candidates
    if (dyn_missing.get(f, 0) <= DROP_FRAC) or any(f.startswith(p) for p in protected)
]
X_full = df_model[feature_cols].select_dtypes(include=[np.number]).fillna(0)
corr = X_full.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
X = X_full.drop(columns=[c for c in upper.columns if any(upper[c] > PRUNE_FRAC)])
X_arr = X.values

df_cl = pd.read_csv("model_combined_dataset_clustered.csv")
y_mort = df_cl["died_inhospital"].astype(int).values
y_icu = (df_cl["icu_los_min"] >= ICU_THRESHOLD).astype(int).values
op_ids = df_cl["op_id"].values


def make_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Logistic Regression (L2)": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
        ),
    }


def maybe_scale(name, Xtr, Xte):
    if "Logistic" in name:
        sc = StandardScaler()
        return sc.fit_transform(Xtr), sc.transform(Xte)
    return Xtr, Xte


all_rows = []
for oname, y in [("mortality", y_mort), ("icu_extended", y_icu)]:
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)), test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    Xtr, Xte, ytr, yte = X_arr[idx_tr], X_arr[idx_te], y[idx_tr], y[idx_te]
    for mname, model in make_models().items():
        Xtr_m, Xte_m = maybe_scale(mname, Xtr.copy(), Xte.copy())
        try:
            Xtr_s, ytr_s = SMOTE(random_state=RANDOM_STATE).fit_resample(Xtr_m, ytr)
        except Exception:
            Xtr_s, ytr_s = Xtr_m, ytr
        model.fit(Xtr_s, ytr_s)
        proba = model.predict_proba(Xte_m)[:, 1]
        auc, lo, hi = delong_ci(yte, proba)
        brier = brier_score_loss(yte, proba)
        bal = balanced_accuracy_score(yte, (proba >= 0.5).astype(int))
        b_ci = bstrap_ci(yte, proba, brier_score_loss)
        ba_ci = bstrap_ci(
            yte, proba, lambda yt, yp: balanced_accuracy_score(yt, (yp >= 0.5).astype(int))
        )
        all_rows.append(
            {
                "split": "random",
                "outcome": oname,
                "model": mname,
                "auroc": round(auc, 4),
                "auroc_lo": round(lo, 4),
                "auroc_hi": round(hi, 4),
                "brier": round(brier, 4),
                "brier_lo": round(b_ci[0], 4),
                "brier_hi": round(b_ci[1], 4),
                "bal_acc": round(bal, 4),
                "bal_lo": round(ba_ci[0], 4),
                "bal_hi": round(ba_ci[1], 4),
            }
        )

results_df = pd.DataFrame(all_rows)

# Temporal split
df_dates = pd.DataFrame({"op_id": op_ids}).merge(ops_dates, on="op_id", how="left")
opdate = df_dates["opdate"].fillna(df_dates["opdate"].median()).values
sort_idx = np.argsort(opdate)
n_train = int(len(sort_idx) * 0.70)
idx_tr_t, idx_te_t = sort_idx[:n_train], sort_idx[n_train:]

temp_rows = []
for oname, y in [("mortality", y_mort), ("icu_extended", y_icu)]:
    Xtr_t, Xte_t = X_arr[idx_tr_t], X_arr[idx_te_t]
    ytr_t, yte_t = y[idx_tr_t], y[idx_te_t]
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    try:
        Xtr_s, ytr_s = SMOTE(random_state=RANDOM_STATE).fit_resample(Xtr_t, ytr_t)
    except Exception:
        Xtr_s, ytr_s = Xtr_t, ytr_t
    rf.fit(Xtr_s, ytr_s)
    proba = rf.predict_proba(Xte_t)[:, 1]
    auc, lo, hi = delong_ci(yte_t, proba)
    brier = brier_score_loss(yte_t, proba)
    bal = balanced_accuracy_score(yte_t, (proba >= 0.5).astype(int))
    b_ci = bstrap_ci(yte_t, proba, brier_score_loss)
    ba_ci = bstrap_ci(
        yte_t, proba, lambda yt, yp: balanced_accuracy_score(yt, (yp >= 0.5).astype(int))
    )
    temp_rows.append(
        {
            "split": "temporal",
            "outcome": oname,
            "model": "Random Forest",
            "auroc": round(auc, 4),
            "auroc_lo": round(lo, 4),
            "auroc_hi": round(hi, 4),
            "brier": round(brier, 4),
            "brier_lo": round(b_ci[0], 4),
            "brier_hi": round(b_ci[1], 4),
            "bal_acc": round(bal, 4),
            "bal_lo": round(ba_ci[0], 4),
            "bal_hi": round(ba_ci[1], 4),
        }
    )

combined = pd.concat([results_df, pd.DataFrame(temp_rows)], ignore_index=True)
combined = combined[MODEL_COMPARISON_COLS]
out_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
combined.to_csv(out_path, index=False)
print("Wrote", out_path)
print(combined.to_string(index=False))
sys.exit(0)
