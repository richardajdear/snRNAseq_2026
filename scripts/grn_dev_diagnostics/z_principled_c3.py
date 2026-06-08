#!/usr/bin/env python3
"""
Z — re-run the core C3+ developmental result under the PRINCIPLED ExN
definition, and compare head-to-head with the OLD marker-rule ExN definition
on the SAME joint V3 cells. This isolates how much the report's headline
(maturity-q0 child->adol C3+ drop) depended on the marker rule's over-calling
of immature DLX+/LHX6+ interneurons into ExN.

On the Stage-1 all-cell joint V3 embedding (PsychAD-V3 + Velmeshev-V3; in the
1-25y window Velmeshev-V3 == Herring, since Ramos is prenatal-only):
  - per-cell C3+ (external AHBA weights) and the 9-gene maturity module.
  - For each ExN definition (marker-rule y3 vs principled y4) and cohort:
      * all-ExN fuzzy d (child vs adol, 1-25y)
      * maturity-q0 (least-mature quintile) fuzzy d
      * combined V3-pair (cohort-centred)
      * cross-sectional rho(maturity module, C3+)
Donor_1400 excluded (as in the report).

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=01:00:00 --mem=200G --cpus-per-task=8 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/z_principled_c3.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR, build_c3plus_table, fuzzy_d_from_donor_scores, AGE_LO, AGE_HI

ALLCELL = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"
GROUPS = ["PsychAD-V3", "Velmeshev-V3"]
SRC = {"PsychAD-V3": "PSYCHAD", "Velmeshev-V3": "VELMESHEV"}
EXCLUDE = {"Donor_1400"}
MIN_Q0 = 5
MODULE_ENS = ["ENSG00000171532", "ENSG00000127152", "ENSG00000119042",
              "ENSG00000081189", "ENSG00000104722", "ENSG00000100285",
              "ENSG00000067715", "ENSG00000132639", "ENSG00000078018"]


def donor_fuzzy(c3, age, don, mask):
    """Donor-mean C3+ over `mask` cells -> fuzzy child/adol d."""
    df = pd.DataFrame({"c3": c3[mask], "age": age[mask], "don": don[mask]})
    g = df.groupby("don").agg(score=("c3", "mean"), n=("c3", "size"),
                              age=("age", "first"))
    g = g[g["n"] >= MIN_Q0]
    r = fuzzy_d_from_donor_scores(g["age"].values, g["score"].values)
    return r["mean_d"], len(g), g


def main():
    print(f"Loading {ALLCELL} (backed) ...")
    a = ad.read_h5ad(ALLCELL, backed="r")
    obs = a.obs
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    src = obs["source"].astype(str).values
    don = obs["individual"].astype(str).values
    names = obs.index.astype(str)

    # per-cell C3+ and 9-gene module from HVG counts
    var_ens = pd.Index(a.var_names.astype(str))
    c3w = build_c3plus_table().set_index("ensembl_id")["weight"]
    counts = sp.csr_matrix(a.X[:])
    tot = np.asarray(counts.sum(1)).ravel()
    inv = 1.0 / np.where(tot > 0, tot, 1)
    hit = var_ens.intersection(c3w.index)
    cov = float(c3w.reindex(hit).sum() / c3w.sum())
    cidx = [var_ens.get_loc(g) for g in hit]
    c3 = np.asarray(counts[:, cidx].multiply(inv[:, None]).dot(c3w.reindex(hit).values)).ravel() * 1e6
    midx = [var_ens.get_loc(g) for g in MODULE_ENS if g in var_ens]
    module = np.asarray(np.log1p(counts[:, midx].multiply(inv[:, None]) * 1e4).todense()).mean(1)
    print(f"C3+ coverage on all-cell HVG: {len(hit)}/{len(c3w)} ({100*cov:.0f}% weight); module {len(midx)}/9")

    defs = {
        "marker_rule": {s: set(pd.read_parquet(OUT_DIR / f"y3_markerexn_cellids_{s}.parquet").index.astype(str))
                        for s in ["VELMESHEV", "PSYCHAD"]},
        "principled": {s: set(pd.read_parquet(OUT_DIR / f"y4_principledexn_cellids_{s}.parquet").index.astype(str))
                       for s in ["VELMESHEV", "PSYCHAD"]},
    }

    rows = []
    combined_scores = {}  # (defn) -> list of cohort-centred donor frames
    for defn, idsets in defs.items():
        cc = []
        for g in GROUPS:
            s = SRC[g]
            exn = np.asarray(names.isin(idsets[s]))
            base = exn & (src == s) & (age >= AGE_LO) & (age < AGE_HI) & np.isfinite(age) & ~pd.Series(don).isin(EXCLUDE).values
            # all-ExN fuzzy d
            d_all, n_all, _ = donor_fuzzy(c3, age, don, base)
            # maturity q0 (least-mature quintile of the module, within this cohort's ExN)
            mvals = module[base]
            if len(mvals) > 100:
                q0_thr = np.quantile(mvals, 0.2)
                q0 = base.copy()
                q0[base] = module[base] <= q0_thr
            else:
                q0 = base
            d_q0, n_q0, g_q0 = donor_fuzzy(c3, age, don, q0)
            # cross-sectional rho(module, c3) within ExN (all ages of this cohort)
            ce = exn & (src == s) & np.isfinite(age)
            rho_mat = stats.spearmanr(module[ce], c3[ce]).correlation
            rows.append({"exn_def": defn, "cohort": g, "n_exn_cells": int(base.sum()),
                         "d_allExN": round(d_all, 3), "n_don_all": n_all,
                         "d_q0": round(d_q0, 3), "n_don_q0": n_q0,
                         "rho_module_c3": round(rho_mat, 2)})
            gd = g_q0.reset_index(); gd["cohort"] = g
            gd["score_c"] = gd["score"] - gd["score"].mean()  # cohort-centre
            cc.append(gd)
        # combined V3-pair (cohort-centred q0 donors)
        allc = pd.concat(cc, ignore_index=True)
        r = fuzzy_d_from_donor_scores(allc["age"].values, allc["score_c"].values)
        rows.append({"exn_def": defn, "cohort": "COMBINED-V3pair", "n_exn_cells": np.nan,
                     "d_allExN": np.nan, "n_don_all": np.nan,
                     "d_q0": round(r["mean_d"], 3), "n_don_q0": len(allc),
                     "rho_module_c3": np.nan})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "z_principled_c3.csv", index=False)
    print("\n" + "=" * 70)
    print("ExN-definition effect on the C3+ developmental result")
    print("(d>0 = C3+ drops child->adol; Velmeshev-V3 in 1-25y == Herring)")
    print("=" * 70)
    print(res.to_string(index=False))
    print("\nReport headline (OLD marker-rule, prior integration): "
          "PsychAD-V3 q0 +0.49, Vel-V3 q0 +0.49, combined V3 +0.46")


if __name__ == "__main__":
    main()
