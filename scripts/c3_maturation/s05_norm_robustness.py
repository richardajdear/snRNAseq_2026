#!/usr/bin/env python3
"""Normalisation robustness (sbatch) — does the within-EN C3 trend depend on how
pseudobulks are aggregated?

Two aggregation schemes, computed per (donor x EN subtype) from per-cell counts:
  SUM  : sum raw counts across cells -> CPM -> log1p     (current; deep-cell biased)
  MEAN : per cell CPM -> log1p -> mean across cells       (equal cell weight)
Then score C3 (signed_logcpm) on each and compare the within-EN age trend per
cohort (PsychAD-V3, Herring-V3, U01-V2).

Reads the big integrated.h5ad objects in sequential chunks (memory-safe).
SUBMIT (do NOT run on login node):
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=02:00:00 --mem=200G scripts/run_script.sh scripts/c3_maturation/s05_norm_robustness.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L
from s04_within_en_cohorts import (VEL_MATURE, PSY_MATURE, AGE_LO_POST, AGE_LO_PSY, AGE_HI)
from s01a_within_celltype_trajectory import partial_spearman

DATASETS = {
    "Velmeshev": dict(
        path=L.B / "Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad",
        mature=VEL_MATURE, age_lo=AGE_LO_POST),
    "PsychAD": dict(
        path=L.B / "PsychAD_noage_tuning5/scvi_output/integrated.h5ad",
        mature=PSY_MATURE, age_lo=AGE_LO_PSY),
}
CHUNK = 50_000


def process(name, cfg, c3_ens):
    print(f"\n=== {name}: {cfg['path']}", flush=True)
    a = ad.read_h5ad(cfg["path"], backed="r")
    n = a.n_obs
    print(f"  n_obs={n:,} n_var={a.n_var:,}", flush=True)
    obs = a.obs
    stcol = "cell_type_aligned"
    subtype = obs[stcol].astype(str).values
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    indiv = obs["individual"].astype(str).values
    chem = obs["chemistry"].astype(str).values if "chemistry" in obs else np.array(["?"] * n)
    dset = obs["dataset"].astype(str).values if "dataset" in obs else np.array(["?"] * n)

    keep = np.isin(subtype, cfg["mature"]) & (age >= cfg["age_lo"]) & (age < AGE_HI)
    print(f"  EN-mature postnatal cells: {keep.sum():,}", flush=True)

    # C3 gene column positions
    var = a.var_names
    pos = {g: i for i, g in enumerate(var)}
    c3_cols = np.array([pos[g] for g in c3_ens.index if g in pos])
    c3_w = c3_ens.loc[[g for g in c3_ens.index if g in pos]].values.astype(float)
    print(f"  C3 genes present: {len(c3_cols)}", flush=True)

    # group keys
    gkey = np.array([f"{i}||{s}" for i, s in zip(indiv, subtype)])
    groups = {}  # key -> dict(sum_counts[full], sum_logcpm_c3, ncells, meta)
    nG = a.n_var

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        m = keep[start:stop]
        if not m.any():
            continue
        X = a.layers["counts"][start:stop]
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        X = X[m]
        tot = np.asarray(X.sum(1)).ravel().astype(np.float64)
        tot[tot == 0] = 1.0
        # per-cell log1p CPM on C3 cols
        Xc = X[:, c3_cols].toarray().astype(np.float64)
        logcpm_c3 = np.log1p(Xc * (1e6 / tot[:, None]))
        idxs = np.where(m)[0] + start
        ks = gkey[idxs]
        Xcsc = X  # full for SUM
        for k in np.unique(ks):
            gm = ks == k
            g = groups.get(k)
            if g is None:
                first = idxs[gm][0]
                g = dict(sum_counts=np.zeros(nG), sum_logcpm=np.zeros(len(c3_cols)),
                         ncells=0,
                         meta=dict(individual=indiv[first], subtype=subtype[first],
                                   age=age[first], chem=chem[first], subsource=dset[first]))
                groups[k] = g
            g["sum_counts"] += np.asarray(Xcsc[gm].sum(0)).ravel()
            g["sum_logcpm"] += logcpm_c3[gm].sum(0)
            g["ncells"] += int(gm.sum())
        if start % (CHUNK * 20) == 0:
            print(f"    ...{stop:,}/{n:,}", flush=True)

    rows = []
    for k, g in groups.items():
        if g["ncells"] < 20:
            continue
        sc = g["sum_counts"]
        tot = sc.sum()
        if tot <= 0:
            continue
        cpm = sc[c3_cols] * (1e6 / tot)
        score_sum = float(np.dot(np.log1p(cpm), c3_w))
        score_mean = float(np.dot(g["sum_logcpm"] / g["ncells"], c3_w))
        rows.append(dict(dataset=name, **g["meta"], ncells=g["ncells"],
                         log10_total=np.log10(tot),
                         score_sum=score_sum, score_mean=score_mean))
    return pd.DataFrame(rows)


def main():
    c3 = L.c3_signed()
    out = pd.concat([process(nm, cfg, c3) for nm, cfg in DATASETS.items()], ignore_index=True)
    out.to_csv(L.OUT_DIR / "s05_norm_scores.csv", index=False)

    # cohort: PsychAD -> PsychAD-V3; Velmeshev -> sub-source (Herring/U01/Ramos)
    sub_map = {"Herring": "Herring-V3", "U01": "U01-V2", "Ramos": "Ramos-V3"}
    out["cohort"] = np.where(out["dataset"] == "PsychAD", "PsychAD-V3",
                             out["subsource"].map(sub_map).fillna(out["subsource"]))
    print("\n--- within-EN age trend: SUM vs MEAN aggregation, per cohort ---")
    res = []
    for coh, d in out.groupby("cohort"):
        if d["individual"].nunique() < 6 or coh in (None, "nan"):
            continue
        r_sum = partial_spearman(d["score_sum"].values, d["age"].values, d["log10_total"].values)
        r_mean = partial_spearman(d["score_mean"].values, d["age"].values, d["log10_total"].values)
        rr = stats.spearmanr(d["score_sum"], d["score_mean"]).statistic
        res.append(dict(cohort=coh, n=len(d), donors=d["individual"].nunique(),
                        rho_age_SUM=r_sum, rho_age_MEAN=r_mean, score_corr=rr))
    res = pd.DataFrame(res)
    print(res.round(3).to_string(index=False))
    res.to_csv(L.OUT_DIR / "s05_norm_robustness_summary.csv", index=False)
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
