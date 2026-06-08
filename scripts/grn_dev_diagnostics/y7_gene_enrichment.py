#!/usr/bin/env python3
"""
Y7 — re-test the §4.2 gene-level enrichment on the PRINCIPLED ExN set.
Per cohort, build donor pseudobulk by summing counts over each donor's
principled-ExN cells (1-25y), compute per-gene child(<10) vs adol(>=10)
Cohen's d, and test whether C3+ genes are over-represented among the top-300
childhood-elevated genes (hypergeometric), with the genome-wide background.
Compares to the W-stage (marker-rule) result.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=00:40:00 --mem=200G --cpus-per-task=8 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y7_gene_enrichment.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR, build_c3plus_table, AGE_LO, AGE_HI

ALLCELL = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"
GROUPS = ["PsychAD-V3", "Velmeshev-V3"]
SRC = {"PsychAD-V3": "PSYCHAD", "Velmeshev-V3": "VELMESHEV"}
EXCLUDE = {"Donor_1400"}
BOUND = 10.0


def cohens_d_cols(X, ischild):
    c, a = X[ischild], X[~ischild]
    n1, n2 = len(c), len(a)
    if n1 < 2 or n2 < 2:
        return np.full(X.shape[1], np.nan)
    v1, v2 = c.var(0, ddof=1), a.var(0, ddof=1)
    spv = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(spv > 0, (c.mean(0) - a.mean(0)) / spv, np.nan)


def main():
    a = ad.read_h5ad(ALLCELL, backed="r")
    obs = a.obs
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    src = obs["source"].astype(str).values
    don = obs["individual"].astype(str).values
    names = obs.index.astype(str)
    counts = sp.csr_matrix(a.X[:])
    var = a.var_names.astype(str).values
    c3 = set(build_c3plus_table()["ensembl_id"])
    is_c3 = np.isin(var, list(c3))

    exn = np.asarray(names.isin(
        set(pd.read_parquet(OUT_DIR / "y4_principledexn_cellids_VELMESHEV.parquet").index.astype(str)) |
        set(pd.read_parquet(OUT_DIR / "y4_principledexn_cellids_PSYCHAD.parquet").index.astype(str))))

    rows = []
    for g in GROUPS:
        s = SRC[g]
        m = exn & (src == s) & (age >= AGE_LO) & (age < AGE_HI) & np.isfinite(age) \
            & ~pd.Series(don).isin(EXCLUDE).values
        di = pd.Series(don[m])
        # donor pseudobulk (sum counts) -> CPM log1p
        donors = di.unique()
        rowsX = []; donor_age = []
        cm = counts[np.where(m)[0]]
        for d in donors:
            sel = (di.values == d)
            if sel.sum() < 5:
                continue
            v = np.asarray(cm[sel].sum(0)).ravel()
            tot = v.sum()
            rowsX.append(np.log1p(v / (tot if tot > 0 else 1) * 1e6))
            donor_age.append(age[m][sel][0])
        X = np.vstack(rowsX); da = np.array(donor_age)
        ischild = da < BOUND
        d = cohens_d_cols(X, ischild)  # + = higher in child
        ser = pd.Series(d, index=var)
        bg = d[~is_c3 & np.isfinite(d)]
        c3d = d[is_c3 & np.isfinite(d)]
        # enrichment in top-300 child-elevated
        valid = np.isfinite(d)
        order = np.argsort(-d[valid])
        top = np.where(valid)[0][order[:300]]
        k = int(is_c3[top].sum()); K = int(is_c3[valid].sum()); N = int(valid.sum())
        hp = stats.hypergeom.sf(k-1, N, K, 300)
        rows.append({"cohort": g, "n_child_donors": int(ischild.sum()),
                     "n_adol_donors": int((~ischild).sum()),
                     "C3+_mean_age_d": round(float(np.nanmean(c3d)), 3),
                     "background_mean_age_d": round(float(np.nanmean(bg)), 3),
                     "top300_C3+_count": k, "expected": round(300*K/N, 1),
                     "hypergeom_p": hp})
        print(f"{g}: child={ischild.sum()} adol={(~ischild).sum()}; "
              f"C3+ mean d={np.nanmean(c3d):+.3f} bg={np.nanmean(bg):+.3f}; "
              f"top300 C3+={k} (exp {300*K/N:.0f}) p={hp:.1e}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "y7_gene_enrichment_principled.csv", index=False)
    print("\nW-stage (marker-rule) was: Vel-V3 115 vs 63 exp p=3e-12; PsychAD 30 vs 29 p=0.47")
    print(f"saved y7_gene_enrichment_principled.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
