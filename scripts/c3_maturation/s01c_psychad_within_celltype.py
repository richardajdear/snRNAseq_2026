#!/usr/bin/env python3
"""Step 1A replication in PsychAD (clean V3, postnatal, many donors).

Velmeshev showed a within-mature-EN postnatal C3 DECLINE that replicated within
both sub-datasets. PsychAD is all-V3 (no V2 depth artefact) and postnatal with
far more donors -> the decisive clean test. Restrict to age>=5 (PsychAD <5y
labels are unreliable) and condition on EN subtype + depth.

PsychAD excitatory_by_celltype is ~450 MB: borderline for inline; run via sbatch
if killed (see CLAUDE.md):
  sbatch --time=00:20:00 --mem=32G scripts/run_script.sh \
    scripts/c3_maturation/s01c_psychad_within_celltype.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L
from s01a_within_celltype_trajectory import cluster_ols, partial_spearman

AGE_LO, AGE_HI = 5.0, 30.0   # PsychAD <5y labels unreliable; cap at 30 for dev window


def main():
    path = L.PB["PsychAD_ExN_by_donor"]
    exc = L.B / "PsychAD_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad"
    use = exc if exc.exists() else path
    print(f"Loading {use} ...", flush=True)
    a = ad.read_h5ad(use)
    print(f"  shape {a.shape}  obs cols: {list(a.obs.columns)}", flush=True)

    # detect subtype column
    st_col = next((c for c in ["cell_type_aligned", "cell_type_raw", "subclass",
                               "cell_class"] if c in a.obs.columns), None)
    print(f"  subtype column: {st_col}")
    print("  subtype counts:", dict(a.obs[st_col].astype(str).value_counts().head(20)))

    w = L.c3_signed(a)
    cpm = L.cpm_matrix(a)
    score = L.score_weighted_cpm(cpm, a.var_names, w, log1p=True)
    dm = L.depth_metrics(a)
    df = pd.DataFrame({
        "individual": a.obs["individual"].astype(str).values,
        "subtype": a.obs[st_col].astype(str).values,
        "age_years": pd.to_numeric(a.obs["age_years"], errors="coerce").values,
        "n_cells": a.obs.get("n_cells", pd.Series(np.nan, index=a.obs_names)).values,
        "score": score, "log10_total": dm["log10_total"].values,
    })
    df = df[(df["age_years"] >= AGE_LO) & (df["age_years"] < AGE_HI)
            & (df["n_cells"] >= 20)].copy()
    print(f"\n  postnatal [{AGE_LO},{AGE_HI}) rows >=20 cells: {len(df)} "
          f"donors: {df['individual'].nunique()}", flush=True)

    # mature EN subtypes by pattern (PsychAD uses EN_L2_3 / 'L2/3 IT' etc.)
    pat = df["subtype"].str.contains(r"(?:EN.?L|L[2-6]|IT|ET|CT|NP|UpperLayer|DeepLayer)",
                                     case=False, regex=True)
    excl = df["subtype"].str.contains(r"Imm|Newborn|SP|Progenitor|IN_|Inhib|Interneuron",
                                      case=False, regex=True)
    mature = df[pat & ~excl].copy()
    print(f"  mature-EN subtypes used: {sorted(mature['subtype'].unique())}")
    print(f"  mature-EN rows: {len(mature)} donors: {mature['individual'].nunique()}")

    print("\n--- per-subtype partial Spearman(score, age | depth) ---")
    for st, sub in mature.groupby("subtype"):
        if sub["individual"].nunique() < 8:
            continue
        r = partial_spearman(sub["score"].values, sub["age_years"].values,
                             sub["log10_total"].values)
        print(f"  {st:18s} n={len(sub):4d} donors={sub['individual'].nunique():4d} "
              f"rho_age|depth={r:+.3f}")

    # pooled mature-EN slope, controlling subtype + depth, donor-clustered
    sts = sorted(mature["subtype"].unique())[1:]
    cols = [np.ones(len(mature)), mature["age_years"].values, mature["log10_total"].values]
    names = ["const", "age", "log10_total"]
    for s in sts:
        cols.append((mature["subtype"] == s).astype(float).values); names.append(f"st[{s}]")
    X = np.column_stack(cols)
    beta, se, t, p = cluster_ols(mature["score"].values, X, mature["individual"].values)
    ai = names.index("age")
    print(f"\n--- HEADLINE PsychAD pooled MATURE-EN age slope (V3, age>=5) ---")
    print(f"  n={len(mature)} donors={mature['individual'].nunique()}  "
          f"slope={beta[ai]:+.4f} SE={se[ai]:.4f} t={t[ai]:+.2f} p={p[ai]:.2e}")

    rho = partial_spearman(mature["score"].values, mature["age_years"].values,
                           mature["log10_total"].values)
    print(f"  pooled rho_age|depth = {rho:+.3f}")
    print("\n  >>> Velmeshev gave rho~-0.28 (decline). Does PsychAD V3 agree?")

    out = mature[["individual", "subtype", "age_years", "score", "log10_total"]].copy()
    out.to_csv(L.OUT_DIR / "s01c_psychad_mature_en_scores.csv", index=False)
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
