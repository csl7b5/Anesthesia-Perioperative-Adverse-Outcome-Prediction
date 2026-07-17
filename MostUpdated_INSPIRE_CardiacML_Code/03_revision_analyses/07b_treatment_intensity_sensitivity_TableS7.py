#!/usr/bin/env python3
"""
Confounding-by-indication sensitivity (Reviewer 3): retrain the canonical model
with vs. without treatment-intensity features (vasopressor/inotrope dosing +
drug-hemodynamic coupling, RBC transfusion, urine output, ventilator settings).
Same 224-feature pipeline, seed, split, and outcomes as Table S5.
"""
import os, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore")

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT)
RS, TEST, ICU_THR, DROP_FRAC, PRUNE, NTREE = 42, 0.30, 4320, 0.70, 0.85, 100
np.random.seed(RS)

df = pd.read_csv("model_combined_dataset.csv")
dyn_missing = pd.read_csv("dynamic_features_vitals.csv").isna().mean()
dyn_prefixes = ("mean_","std_","slope_","auc_","min_","max_","avg_rate_","duration_","num_events_","total_dose_")
coupling = ("_lag","_slope","_ri")
protected = ("avg_rate_","duration_","num_events_","total_dose_") + coupling
leak = ["subject_id","op_id","died_inhospital","icu_admit","icu_los_min","allcause_death_time","icuin_time"]
op_id_all = df["op_id"].values
dfm = df.drop(columns=[c for c in leak if c in df.columns])
cand = [c for c in dfm.columns if any(c.startswith(p) for p in dyn_prefixes) or any(t in c for t in coupling)]
feat = [f for f in cand if (dyn_missing.get(f,0) <= DROP_FRAC) or any(f.startswith(p) for p in protected)]
Xf = dfm[feat].select_dtypes(include=[np.number]).fillna(0)
corr = Xf.corr().abs(); upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
final = [c for c in Xf.columns if c not in {c for c in upper.columns if any(upper[c]>PRUNE)}]
X = Xf[final].copy(); X["op_id"] = op_id_all

cl = pd.read_csv("model_combined_dataset_clustered.csv")
o = cl[["op_id","died_inhospital","icu_los_min"]].copy()
o["y_mort"] = o["died_inhospital"].astype(int)
o["y_icu"] = (o["icu_los_min"]>=ICU_THR).astype(int)
m = X.merge(o[["op_id","y_mort","y_icu"]], on="op_id", how="inner")
y = {"mortality": m["y_mort"].values, "icu": m["y_icu"].values}
Xall = m[final].values.astype(float)

# treatment-intensity tokens named by Reviewer 3
DRUGS = ("dobui","dopai","eph","epi","epii","mlni","nepi","pepi","phe","vaso","ntgi")
VENT  = ("peep","pip","pmean","pplat","minvol","vt","fio2")
OTHER = ("rbc","uo")
TX = DRUGS + VENT + OTHER
def is_tx(f): return any(t in f for t in TX)
tx_cols = [c for c in final if is_tx(c)]
keep_cols = [c for c in final if not is_tx(c)]
idx_keep = [final.index(c) for c in keep_cols]
print(f"Total features: {len(final)} | treatment-intensity dropped: {len(tx_cols)} | retained: {len(keep_cols)}")

def boot_ci(yt, yp, n=1000):
    v=[]
    for _ in range(n):
        i=np.random.choice(len(yt),len(yt),replace=True)
        if yt[i].sum() in (0,len(yt)): continue
        v.append(roc_auc_score(yt[i],yp[i]))
    return np.percentile(v,[2.5,97.5])

def run(Xmat, yv):
    Xtr,Xte,ytr,yte = train_test_split(Xmat,yv,test_size=TEST,stratify=yv,random_state=RS)
    Xs,ys = SMOTE(random_state=RS).fit_resample(Xtr,ytr)
    clf = RandomForestClassifier(n_estimators=NTREE,random_state=RS).fit(Xs,ys)
    p = clf.predict_proba(Xte)[:,1]
    return roc_auc_score(yte,p), boot_ci(yte,p), brier_score_loss(yte,p)

rows=[]
for name in ["mortality","icu"]:
    fa,fci,fb = run(Xall, y[name])
    ra,rci,rb = run(Xall[:,idx_keep], y[name])
    print(f"\n{name}:")
    print(f"  Full model   AUROC={fa:.3f} [{fci[0]:.3f}-{fci[1]:.3f}] Brier={fb:.3f}")
    print(f"  No-treatment AUROC={ra:.3f} [{rci[0]:.3f}-{rci[1]:.3f}] Brier={rb:.3f}")
    print(f"  delta AUROC = {ra-fa:+.3f}")
    rows.append({"Outcome":name,"Full_AUROC":round(fa,3),"Full_CI":f"{fci[0]:.3f}-{fci[1]:.3f}",
                 "NoTx_AUROC":round(ra,3),"NoTx_CI":f"{rci[0]:.3f}-{rci[1]:.3f}","delta":round(ra-fa,3)})
pd.DataFrame(rows).to_csv("reviewer_responses/comment_7_confounding/treatment_intensity_sensitivity.csv",index=False)
print("\nDropped features:", tx_cols[:40])
print("\nSaved -> reviewer_responses/comment_7_confounding/treatment_intensity_sensitivity.csv")
