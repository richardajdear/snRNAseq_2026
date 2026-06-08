#!/usr/bin/env python3
"""
Y2b — controlled separability baseline: child-vs-adol AUC on the STAGE-1
all-cell latent, for the SAME principled-ExN cells used in the ExN-only latent.
Isolates 'ExN-only embedding vs all-cell embedding' holding cells/cohort fixed
(the 0.62/0.67 numbers were from the older shipped integration).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.stats as stats
sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR
ALLCELL = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"
AGE_LO, AGE_HI, BOUND = 1.0, 25.0, 10.0
SRC = {"PsychAD-V3": "PSYCHAD", "Velmeshev-V3": "VELMESHEV"}

def grouped_auc(Z, y, groups, max_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    npos=len(np.unique(groups[y==1])); nneg=len(np.unique(groups[y==0]))
    if npos<2 or nneg<2: return np.nan
    gkf=GroupKFold(n_splits=min(max_splits,npos,nneg)); aucs=[]
    for tr,te in gkf.split(Z,y,groups=groups):
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
        clf=LogisticRegression(max_iter=1000).fit(Z[tr],y[tr])
        aucs.append(roc_auc_score(y[te],clf.predict_proba(Z[te])[:,1]))
    return float(np.mean(aucs)) if aucs else np.nan

a=ad.read_h5ad(ALLCELL, backed="r")
exn=set(pd.read_parquet(OUT_DIR/"y4_principledexn_cellids_VELMESHEV.parquet").index.astype(str)) | \
    set(pd.read_parquet(OUT_DIR/"y4_principledexn_cellids_PSYCHAD.parquet").index.astype(str))
obs=a.obs; Z=np.asarray(a.obsm["X_scVI"][:])
isexn=np.asarray(obs.index.astype(str).isin(exn))
age=pd.to_numeric(obs["age_years"],errors="coerce").values
src=obs["source"].astype(str).values
don=obs["individual"].astype(str).values
rows=[]
for g,s in SRC.items():
    m=isexn & (src==s) & (age>=AGE_LO)&(age<AGE_HI)&np.isfinite(age)
    y=(age[m]<BOUND).astype(int)
    auc=grouped_auc(Z[m],y,don[m])
    rows.append({"group":g,"latent":"all-cell (same ExN cells)","n_cells":int(m.sum()),"cv_auc":auc})
    print(f"{g}: all-cell-latent AUC on {m.sum():,} principled-ExN cells = {auc:.3f}")
pd.DataFrame(rows).to_csv(OUT_DIR/"y2b_allcell_baseline.csv",index=False)
print("saved y2b_allcell_baseline.csv")
