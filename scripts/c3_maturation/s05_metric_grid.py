#!/usr/bin/env python3
"""Metric grid (sbatch) — 8 C3 scores crossing pole x transform x aggregation.

The C3 score has three independent binary choices:
  pole        : C3+ (positive weights only)        vs  signed (C3+ minus C3-)
  transform   : linear CPM                          vs  log1p(CPM)
  aggregation : AGG  = sum counts across cells -> CPM   (deep-cell weighted)
                CELL = per-cell CPM -> mean across cells (equal cell weight)
=> 2 x 2 x 2 = 8 metrics.

AGG needs only the summed-count pseudobulk; CELL needs per-cell counts, so this
runs from the integrated objects (sbatch). For each (donor x mature-EN subtype)
pseudobulk we accumulate, over cells:
  sum_counts[all genes]                    -> AGG (cpm from the sum)
  sum_cpm_c3[C3 genes]  = Σ_c cpm_{c,g}     -> CELL linear  (mean cpm)
  sum_logcpm_c3[C3 genes]= Σ_c log1p(cpm)   -> CELL log     (mean log1p cpm)

Then per cohort (PsychAD-V3, Herring-V3, U01-V2) and per metric we report:
  depth robustness  rho(score, log10 depth | age)   [want ~0]
  within-EN trend   rho(score, age | log10 depth)
so we can see whether the conclusion is robust to the metric / aggregation choice.

SUBMIT:
  sbatch --time=03:00:00 --mem=200G scripts/run_script.sh scripts/c3_maturation/s05_metric_grid.py
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
    "Velmeshev": dict(path=L.B / "Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad",
                      mature=VEL_MATURE, age_lo=AGE_LO_POST),
    "PsychAD":   dict(path=L.B / "PsychAD_noage_tuning5/scvi_output/integrated.h5ad",
                      mature=PSY_MATURE, age_lo=AGE_LO_PSY),
}
CHUNK = 50_000
CPM = 1e6


def process(name, cfg, c3):
    print(f"\n=== {name}: {cfg['path']}", flush=True)
    a = ad.read_h5ad(cfg["path"], backed="r")
    n = a.shape[0]
    print(f"  n_obs={n:,} n_var={a.shape[1]:,}", flush=True)
    obs = a.obs
    subtype = obs["cell_type_aligned"].astype(str).values
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    indiv = obs["individual"].astype(str).values
    dset = obs["dataset"].astype(str).values if "dataset" in obs else np.array(["?"] * n)
    keep = np.isin(subtype, cfg["mature"]) & (age >= cfg["age_lo"]) & (age < AGE_HI)
    print(f"  EN-mature postnatal cells: {keep.sum():,}", flush=True)

    var = a.var_names
    pos = {g: i for i, g in enumerate(var)}
    c3_cols = np.array([pos[g] for g in c3.index if g in pos])
    w = c3.loc[[g for g in c3.index if g in pos]].values.astype(float)
    wpos = np.where(w > 0, w, 0.0)
    nG = a.shape[1]
    gkey = np.array([f"{i}||{s}" for i, s in zip(indiv, subtype)])

    G = {}
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        m = keep[start:stop]
        if not m.any():
            continue
        X = a.layers["counts"][start:stop]
        X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        X = X[m]
        tot = np.asarray(X.sum(1)).ravel().astype(np.float64); tot[tot == 0] = 1.0
        cpm_c3 = X[:, c3_cols].toarray().astype(np.float64) * (CPM / tot[:, None])
        logcpm_c3 = np.log1p(cpm_c3)
        idxs = np.where(m)[0] + start
        ks = gkey[idxs]
        for k in np.unique(ks):
            gm = ks == k
            g = G.get(k)
            if g is None:
                first = idxs[gm][0]
                g = dict(sum_counts=np.zeros(nG), sum_cpm=np.zeros(len(c3_cols)),
                         sum_logcpm=np.zeros(len(c3_cols)), ncells=0,
                         meta=dict(individual=indiv[first], subtype=subtype[first],
                                   age=age[first], subsource=dset[first]))
                G[k] = g
            g["sum_counts"] += np.asarray(X[gm].sum(0)).ravel()
            g["sum_cpm"] += cpm_c3[gm].sum(0)
            g["sum_logcpm"] += logcpm_c3[gm].sum(0)
            g["ncells"] += int(gm.sum())
        if start % (CHUNK * 20) == 0:
            print(f"    ...{stop:,}/{n:,}", flush=True)

    rows = []
    for k, g in G.items():
        if g["ncells"] < 20:
            continue
        sc = g["sum_counts"]; tot = sc.sum()
        if tot <= 0:
            continue
        cpm_agg = sc[c3_cols] * (CPM / tot)
        mean_cpm = g["sum_cpm"] / g["ncells"]
        mean_logcpm = g["sum_logcpm"] / g["ncells"]
        r = dict(dataset=name, **g["meta"], ncells=g["ncells"], log10_total=np.log10(tot))
        # 8 metrics
        r["AGG_pos_lin"]   = float(np.dot(cpm_agg, wpos))
        r["AGG_pos_log"]   = float(np.dot(np.log1p(cpm_agg), wpos))
        r["AGG_sign_lin"]  = float(np.dot(cpm_agg, w))
        r["AGG_sign_log"]  = float(np.dot(np.log1p(cpm_agg), w))
        r["CELL_pos_lin"]  = float(np.dot(mean_cpm, wpos))
        r["CELL_pos_log"]  = float(np.dot(mean_logcpm, wpos))
        r["CELL_sign_lin"] = float(np.dot(mean_cpm, w))
        r["CELL_sign_log"] = float(np.dot(mean_logcpm, w))
        rows.append(r)
    return pd.DataFrame(rows)


METRICS = ["AGG_pos_lin", "AGG_pos_log", "AGG_sign_lin", "AGG_sign_log",
           "CELL_pos_lin", "CELL_pos_log", "CELL_sign_lin", "CELL_sign_log"]


def main():
    c3 = L.c3_signed()
    out = pd.concat([process(nm, cfg, c3) for nm, cfg in DATASETS.items()], ignore_index=True)
    out.to_csv(L.OUT_DIR / "s05_metric_grid_scores.csv", index=False)
    sub_map = {"Herring": "Herring-V3", "U01": "U01-V2", "Ramos": "Ramos-V3"}
    out["cohort"] = np.where(out["dataset"] == "PsychAD", "PsychAD-V3",
                             out["subsource"].map(sub_map).fillna(out["subsource"]))

    res = []
    for coh, d in out.groupby("cohort"):
        if d["individual"].nunique() < 6 or coh in ("Ramos-V3",):
            continue
        for mtr in METRICS:
            v = d[mtr].values
            r_depth = partial_spearman(v, d["log10_total"].values, d["age"].values)
            r_age = partial_spearman(v, d["age"].values, d["log10_total"].values)
            res.append(dict(cohort=coh, metric=mtr, n=len(d),
                            rho_depth_ctrl_age=r_depth, rho_age_ctrl_depth=r_age))
    res = pd.DataFrame(res)
    res.to_csv(L.OUT_DIR / "s05_metric_grid_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n===== depth robustness rho(score,depth|age) per metric x cohort =====")
    print(res.pivot(index="metric", columns="cohort", values="rho_depth_ctrl_age").round(3).to_string())
    print("\n===== within-EN age trend rho(score,age|depth) per metric x cohort =====")
    print(res.pivot(index="metric", columns="cohort", values="rho_age_ctrl_depth").round(3).to_string())
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
