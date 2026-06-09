#!/usr/bin/env python3
"""
Y8b — measure the controlled one-variable test: PsychAD capped at 200k
(VelPsychAD_V3_allcell_dev30_psyCAP200k) vs the uncapped dev30. Same metric
and classifier/C3+ projection as y8, on the cap run's X_scVI.

Compare mixing to: dev30 uncapped 0.112/0.188 (all/postnatal); good 0.223/0.300.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR, build_c3plus_table
CAP = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30_psyCAP200k/scvi_output/integrated.h5ad"
MODULE_ENS = ["ENSG00000171532","ENSG00000127152","ENSG00000119042","ENSG00000081189",
              "ENSG00000104722","ENSG00000100285","ENSG00000067715","ENSG00000132639","ENSG00000078018"]
K=30
def batch_mixing(Z, batch, n_sample=40000, seed=0):
    rng=np.random.default_rng(seed); idx=rng.choice(Z.shape[0],min(n_sample,Z.shape[0]),replace=False)
    nn=NearestNeighbors(n_neighbors=K+1).fit(Z); _,ind=nn.kneighbors(Z[idx])
    b=pd.factorize(batch)[0]; bi=b[idx]; neigh=b[ind[:,1:]]
    obs_cross=1-(neigh==bi[:,None]).mean(1).mean()
    p=pd.Series(batch).value_counts(normalize=True).values
    return float(obs_cross/(1-np.sum(p**2)))
a=ad.read_h5ad(CAP, backed="r")
Z=np.asarray(a.obsm["X_scVI"][:]); obs=a.obs
bk="source-chemistry" if "source-chemistry" in obs else "source"
batch=obs[bk].astype(str).values; age=pd.to_numeric(obs["age_years"],errors="coerce").values
print(f"psyCAP200k n={a.n_obs:,} batches={sorted(set(batch))}")
print(f"  batch-mixing ALL={batch_mixing(Z,batch):.3f}  POSTNATAL={batch_mixing(Z[age>=0],batch[age>=0]):.3f}")
print("  (compare: dev30 uncapped 0.112/0.188 ; good 0.223/0.300)")
# classifier + C3+
counts=sp.csr_matrix(a.X[:]); tot=np.asarray(counts.sum(1)).ravel(); inv=1.0/np.where(tot>0,tot,1)
var=pd.Index(a.var_names.astype(str)); c3w=build_c3plus_table().set_index("ensembl_id")["weight"]
hit=var.intersection(c3w.index); c3=np.asarray(counts[:,[var.get_loc(g) for g in hit]].multiply(inv[:,None]).dot(c3w.reindex(hit).values)).ravel()*1e6
midx=[var.get_loc(g) for g in MODULE_ENS if g in var]; mod=np.asarray(np.log1p(counts[:,midx].multiply(inv[:,None])*1e4).todense()).mean(1)
don=obs["individual"].astype(str).values
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
for g,tag in {"PsychAD-V3":"PSYCHAD-V3","Velmeshev-V3":"VELMESHEV-V3"}.items():
    m=(batch==tag)&(age>=1)&(age<25)&np.isfinite(age)
    if m.sum()<100: continue
    y=(age[m]<10).astype(int); dg=don[m]; npos=len(np.unique(dg[y==1])); nneg=len(np.unique(dg[y==0])); auc=np.nan
    if npos>=2 and nneg>=2:
        aucs=[]
        for tr,te in GroupKFold(n_splits=min(5,npos,nneg)).split(Z[m],y,dg):
            if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
            aucs.append(roc_auc_score(y[te],LogisticRegression(max_iter=1000).fit(Z[m][tr],y[tr]).predict_proba(Z[m][te])[:,1]))
        auc=float(np.mean(aucs)) if aucs else np.nan
    clf=LogisticRegression(max_iter=2000).fit(Z[m],y); axis=-(Z[m]@clf.coef_.ravel())
    print(f"{g}: AUC={auc:.3f} rho(axis,C3+)={stats.spearmanr(axis,c3[m]).correlation:+.2f} "
          f"rho(age,C3+)={stats.spearmanr(age[m],c3[m]).correlation:+.2f} rho(module,C3+)={stats.spearmanr(mod[m],c3[m]).correlation:+.2f}")
print("DONE")
