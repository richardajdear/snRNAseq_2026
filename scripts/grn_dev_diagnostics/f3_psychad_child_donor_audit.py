#!/usr/bin/env python3
"""
F3 — audit the 11 PsychAD child donors that anchor the developmental
window. Where do they come from (HBCC vs Aging cohort)? What is the
recorded cause of death / PMI / brain mass / donor diagnosis?

Per-donor C3+ aggregate score is computed and ranked, to see whether
the negative aggregate is driven by 1-2 outlier donors or is uniform
across all 11.

Uses ONLY the pseudobulk h5ad (small, login-node safe).

Outputs:
  f3_psychad_child_donors.csv       per-donor metadata + per-donor C3+ score
  f3_donor_score_plot.png           strip plot of per-donor C3+ score by stage
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (
    load_pseudobulk, cpm_from_counts, subset_age_window,
    project_score, build_c3plus_table, OUT_DIR,
)

DUMP_KEYS = [
    "individual", "donor_id", "age_years", "sex", "PMI",
    "dataset", "source", "cohort", "cohort_name",
    "disease", "disease_ontology_term_id", "self_reported_ethnicity",
    "development_stage", "n_cells", "region",
    "tissue", "chemistry", "source-chemistry",
    "cause_of_death", "Diagnosis", "DiagnosisGroup", "study",
]


def main():
    a = load_pseudobulk("PsychAD")
    print("PsychAD pseudobulk:", a.shape)
    print("obs cols:", list(a.obs.columns))

    obs = a.obs.copy()

    # Project the C3+ aggregate score per donor
    weights = build_c3plus_table().set_index("ensembl_id")["weight"]
    cpm = cpm_from_counts(a)
    win = subset_age_window(cpm)   # 1-25 y
    score = project_score(win, weights)
    obs_win = win.obs.copy()
    obs_win["c3plus_score"] = score.values

    keep_cols = [c for c in DUMP_KEYS if c in obs.columns]
    print(f"\nDumping columns: {keep_cols}")

    # All donors in the developmental window
    dev = obs_win[keep_cols + ["c3plus_score", "stage"]].copy()
    dev = dev.sort_values(["stage", "age_years"]).reset_index(drop=True)
    print("\n=== All donors in 1-25 y window ===")
    print(dev.to_string(index=False))
    dev.to_csv(OUT_DIR / "f3_psychad_devwindow_donors.csv", index=False)

    # Focus on children
    children = dev[dev["stage"] == "child"]
    print(f"\n=== {len(children)} PsychAD CHILD donors (1-9 y) — full metadata ===")
    for col in keep_cols + ["c3plus_score"]:
        if col in children.columns:
            print(f"  {col}: {children[col].tolist()}")

    # Distribution: child vs adol score
    fig, ax = plt.subplots(figsize=(8, 5))
    for stage, color in [("child", "C3"), ("adol", "C0")]:
        sub = dev[dev["stage"] == stage]
        ax.scatter(sub["age_years"], sub["c3plus_score"], color=color,
                    s=40, alpha=0.75, label=f"{stage} (n={len(sub)})")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("PsychAD C3+ weighted score (per donor)")
    ax.set_title("PsychAD: per-donor C3+ score across the developmental window\n"
                 "(red = childhood donors anchoring the d=-0.44 statistic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "f3_donor_score_plot.png", dpi=150)
    plt.close(fig)

    # Score summary by potential confounders we found
    for stratum in ["sex", "dataset", "source", "cohort", "cohort_name", "study",
                     "Diagnosis", "DiagnosisGroup", "cause_of_death"]:
        if stratum not in dev.columns:
            continue
        print(f"\n  --- by {stratum} ---")
        ss = (dev.groupby([stratum, "stage"], observed=True)
                  .agg(n=("c3plus_score", "size"),
                       mean=("c3plus_score", "mean"))
                  .unstack(fill_value=np.nan))
        print(ss.to_string())

    # Identify outliers
    if len(children) >= 3:
        med = children["c3plus_score"].median()
        mad = (children["c3plus_score"] - med).abs().median()
        print(f"\n  child score median={med:.0f}  MAD={mad:.0f}")
        for _, row in children.iterrows():
            z = (row["c3plus_score"] - med) / max(mad, 1.0)
            tag = "  outlier ↓" if z < -3 else ("  outlier ↑" if z > 3 else "")
            ind = row.get("individual", row.get("donor_id", "?"))
            print(f"  {ind}  age={row['age_years']:.1f}  "
                  f"score={row['c3plus_score']:.0f}  z={z:+.2f}{tag}")

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
