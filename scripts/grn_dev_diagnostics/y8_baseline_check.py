#!/usr/bin/env python3
"""
Y8 — diagnose the batch-integration regression and test the good run.

(1) Quantitative batch-mixing metric (kNN-based, iLISI-like) on:
      - the BAD new run  VelPsychAD_V3_allcell_dev30
      - the GOOD old run VelWangPsychAD_200k_prepost_V3only_tuning5
    overall and POSTNATAL-only (to separate a real failure from the
    prenatal-only Ramos block that has no PsychAD counterpart).
    mixing in [0,1]: 1 = neighbours are batch-random (well mixed),
    ~0 = neighbours are same-batch (poor integration).
(2) On the GOOD run: per-cohort child-vs-adol grouped-CV AUC and how external
    C3+ projects onto the age axis / actual age / maturation module.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=01:30:00 --mem=250G --cpus-per-task=16 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y8_baseline_check.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR, build_c3plus_table

BASE = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated"
BAD = f"{BASE}/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"
GOOD = f"{BASE}/VelWangPsychAD_200k_prepost_V3only_tuning5/scvi_output/integrated.h5ad"
MODULE_ENS = ["ENSG00000171532", "ENSG00000127152", "ENSG00000119042",
              "ENSG00000081189", "ENSG00000104722", "ENSG00000100285",
              "ENSG00000067715", "ENSG00000132639", "ENSG00000078018"]
K = 30


def batch_mixing(Z, batch, n_sample=40000, seed=0):
    """Mean fraction of kNN from a DIFFERENT batch, normalised by the
    batch-random expectation (1 - sum p_b^2). Returns score in ~[0,1]."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    idx = rng.choice(n, min(n_sample, n), replace=False)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(Z)
    _, ind = nn.kneighbors(Z[idx])
    b = pd.factorize(batch)[0]
    bi = b[idx]
    neigh = b[ind[:, 1:]]                       # drop self
    same = (neigh == bi[:, None]).mean(1)       # fraction same-batch
    obs_cross = 1 - same.mean()
    p = pd.Series(batch).value_counts(normalize=True).values
    exp_cross = 1 - np.sum(p ** 2)
    return float(obs_cross / exp_cross), float(obs_cross), float(exp_cross)


def percell_c3_module(a):
    counts = sp.csr_matrix(a.X[:])
    tot = np.asarray(counts.sum(1)).ravel(); inv = 1.0 / np.where(tot > 0, tot, 1)
    var = pd.Index(a.var_names.astype(str))
    c3w = build_c3plus_table().set_index("ensembl_id")["weight"]
    hit = var.intersection(c3w.index); cidx = [var.get_loc(g) for g in hit]
    c3 = np.asarray(counts[:, cidx].multiply(inv[:, None]).dot(c3w.reindex(hit).values)).ravel() * 1e6
    midx = [var.get_loc(g) for g in MODULE_ENS if g in var]
    mod = np.asarray(np.log1p(counts[:, midx].multiply(inv[:, None]) * 1e4).todense()).mean(1)
    return c3, mod, float(c3w.reindex(hit).sum() / c3w.sum())


def main():
    rows = []
    for name, path in [("BAD_dev30_V3", BAD), ("GOOD_200k_VelWangPsychAD", GOOD)]:
        if not Path(path).exists():
            print(f"{name}: missing {path}"); continue
        a = ad.read_h5ad(path, backed="r")
        Z = np.asarray(a.obsm["X_scVI"][:])
        bk = "source-chemistry" if "source-chemistry" in a.obs else "source"
        batch = a.obs[bk].astype(str).values
        age = pd.to_numeric(a.obs["age_years"], errors="coerce").values
        m_all, oc, ec = batch_mixing(Z, batch)
        post = age >= 0
        m_post, _, _ = batch_mixing(Z[post], batch[post])
        print(f"\n[{name}] n={a.n_obs:,} batches={sorted(set(batch))}")
        print(f"  batch-mixing ALL = {m_all:.3f} (obs cross {oc:.2f} / exp {ec:.2f})")
        print(f"  batch-mixing POSTNATAL-only = {m_post:.3f}")
        rows.append({"run": name, "n": a.n_obs, "n_batches": len(set(batch)),
                     "mixing_all": round(m_all, 3), "mixing_postnatal": round(m_post, 3)})
        del a
    pd.DataFrame(rows).to_csv(OUT_DIR / "y8_batch_mixing.csv", index=False)

    # ---- science on the GOOD run ----
    if not Path(GOOD).exists():
        return
    print("\n" + "=" * 60 + "\nGOOD run — child-vs-adol AUC + C3+ projection\n" + "=" * 60)
    a = ad.read_h5ad(GOOD, backed="r")
    Z = np.asarray(a.obsm["X_scVI"][:])
    obs = a.obs
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    bk = "source-chemistry" if "source-chemistry" in obs else "source"
    sc = obs[bk].astype(str).values
    don = obs["individual"].astype(str).values if "individual" in obs else obs.index.astype(str)
    c3, mod, cov = percell_c3_module(a)
    print(f"C3+ coverage {cov*100:.0f}% weight")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    cohorts = {"PsychAD-V3": "PSYCHAD-V3", "Velmeshev-V3": "VELMESHEV-V3"}
    res = []
    for g, tag in cohorts.items():
        m = (sc == tag) & (age >= 1) & (age < 25) & np.isfinite(age)
        if m.sum() < 100:
            print(f"{g}: too few ({m.sum()})"); continue
        y = (age[m] < 10).astype(int); dg = don[m]
        npos = len(np.unique(dg[y == 1])); nneg = len(np.unique(dg[y == 0]))
        auc = np.nan
        if npos >= 2 and nneg >= 2:
            aucs = []
            for tr, te in GroupKFold(n_splits=min(5, npos, nneg)).split(Z[m], y, dg):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                clf = LogisticRegression(max_iter=1000).fit(Z[m][tr], y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(Z[m][te])[:, 1]))
            auc = float(np.mean(aucs)) if aucs else np.nan
        # supervised axis + C3+ projection
        clf = LogisticRegression(max_iter=2000).fit(Z[m], y)
        axis = -(Z[m] @ clf.coef_.ravel())
        r_ax_c3 = stats.spearmanr(axis, c3[m]).correlation
        r_age_c3 = stats.spearmanr(age[m], c3[m]).correlation
        r_mod_c3 = stats.spearmanr(mod[m], c3[m]).correlation
        res.append({"cohort": g, "n_cells": int(m.sum()), "n_donors": len(np.unique(dg)),
                    "cv_auc": round(auc, 3), "rho_axis_c3": round(r_ax_c3, 3),
                    "rho_age_c3": round(r_age_c3, 3), "rho_module_c3": round(r_mod_c3, 3)})
        print(f"{g}: AUC={auc:.3f}  rho(axis,C3+)={r_ax_c3:+.2f}  "
              f"rho(age,C3+)={r_age_c3:+.2f}  rho(module,C3+)={r_mod_c3:+.2f}")
    pd.DataFrame(res).to_csv(OUT_DIR / "y8_goodrun_classifier_c3.csv", index=False)
    print(f"\nsaved y8_batch_mixing.csv, y8_goodrun_classifier_c3.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
