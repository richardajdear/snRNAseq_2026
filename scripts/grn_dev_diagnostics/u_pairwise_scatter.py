#!/usr/bin/env python3
"""
U — pairwise scatter matrix of the four per-cell variables that drive the
C3+ developmental analysis: age, UMI depth, maturity module, C3+ score.

Six relationships, each as a 3-panel figure in cohort order
PsychAD-V3 / Velmeshev-V3 / Velmeshev-V2, with Spearman ρ annotated:
  1 depth   vs age
  2 maturity vs age
  3 maturity vs depth
  4 C3+      vs age
  5 C3+      vs depth
  6 C3+      vs maturity
Produced twice: (a) all ExN cells, (b) upper-layer ExN only.

Outputs: u_all_<n>_*.png, u_upper_<n>_*.png, u_correlation_summary.csv
and the report PAIRWISE_RELATIONS.md (written separately).

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=00:25:00 --mem=64G \
     scripts/run_script.sh scripts/grn_dev_diagnostics/u_pairwise_scatter.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR

CACHE_V4 = OUT_DIR / "r_per_cell_cache_v4.parquet"
COHORTS = ["PsychAD-V3", "Velmeshev-V3", "Velmeshev-V2"]   # user-requested order
MODULE_MATURE = ["NEUROD2", "BCL11B", "SATB2", "MEF2C", "NEFM", "NEFH",
                 "SYT1", "SNAP25", "MAP2"]
EXCLUDE_DONORS = {"Donor_1400"}

# name, xcol, ycol, xlabel, ylabel, xlog, ylog
PAIRS = [
    ("1_depth_vs_age",      "age_years",     "total_umi",     "age (years)",      "UMI depth",        False, True),
    ("2_maturity_vs_age",   "age_years",     "mature_module", "age (years)",      "maturity module",  False, False),
    ("3_maturity_vs_depth", "total_umi",     "mature_module", "UMI depth",        "maturity module",  True,  False),
    ("4_c3_vs_age",         "age_years",     "per_cell_c3",   "age (years)",      "C3+ (CPM)",        False, False),
    ("5_c3_vs_depth",       "total_umi",     "per_cell_c3",   "UMI depth",        "C3+ (CPM)",        True,  False),
    ("6_c3_vs_maturity",    "mature_module", "per_cell_c3",   "maturity module",  "C3+ (CPM)",        False, False),
]


def add_module(df):
    cp = [f"cp_{m}" for m in MODULE_MATURE if f"cp_{m}" in df.columns]
    df = df.copy()
    df["mature_module"] = df[cp].mean(axis=1)
    return df


def make_pair_fig(df, pair, subset_tag, subset_desc):
    name, xc, yc, xl, yl, xlog, ylog = pair
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    rows = []
    for ax, g in zip(axes, COHORTS):
        sub = df[df["group"] == g]
        x = sub[xc].values.astype(float)
        y = sub[yc].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) > 5:
            rho, p = stats.spearmanr(x, y)
        else:
            rho, p = np.nan, np.nan
        X = np.log10(np.clip(x, 1, None)) if xlog else x
        Y = np.log10(np.clip(y, 1, None)) if ylog else y
        ax.hexbin(X, Y, gridsize=42, cmap="viridis", bins="log", mincnt=1)
        ax.set_title(f"{g}\nSpearman ρ = {rho:+.2f}   (n={len(x):,})",
                     fontsize=10)
        ax.set_xlabel(("log10 " if xlog else "") + xl)
        if g == COHORTS[0]:
            ax.set_ylabel(("log10 " if ylog else "") + yl)
        rows.append({"relationship": name, "subset": subset_tag, "cohort": g,
                     "spearman_rho": rho, "p": p, "n_cells": int(len(x))})
    pretty = name.split("_", 1)[1].replace("_", " ")
    fig.suptitle(f"{name.split('_')[0]}. {pretty}  —  {subset_desc}",
                 fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / f"u_{subset_tag}_{name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out.name}", flush=True)
    return rows


def main():
    df = pd.read_parquet(CACHE_V4)
    df = df[~df["individual"].isin(EXCLUDE_DONORS)].reset_index(drop=True)
    df = add_module(df)
    print(f"loaded {len(df):,} cells", flush=True)

    subsets = [
        ("all",   df,                              "all ExN cells"),
        ("upper", df[df["layer"] == "upper"],      "upper-layer ExN only"),
    ]
    all_rows = []
    for tag, sub, desc in subsets:
        print(f"\n=== subset: {tag} ({len(sub):,} cells) ===", flush=True)
        for pair in PAIRS:
            all_rows += make_pair_fig(sub, pair, tag, desc)

    summ = pd.DataFrame(all_rows)
    summ.to_csv(OUT_DIR / "u_correlation_summary.csv", index=False)
    print("\n=== Spearman ρ summary (pivot: relationship × cohort) ===")
    for tag, _, _ in subsets:
        print(f"\n--- {tag} ---")
        piv = (summ[summ["subset"] == tag]
               .pivot(index="relationship", columns="cohort",
                      values="spearman_rho")[COHORTS].round(2))
        print(piv.to_string())
    print(f"\nAll U outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
