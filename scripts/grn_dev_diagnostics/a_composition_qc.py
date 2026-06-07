#!/usr/bin/env python3
"""
Group A — composition & power QC for the AHBA C3+ PsychAD-vs-Velmeshev
disagreement. Reads the two ExN_manual_by_donor.h5ad pseudobulks and
produces:

    a1_donor_density.png         donor count vs age, by dataset
    a2_per_donor_qc.png          n_cells, total counts, n_genes vs age
    a4_region_chemistry.csv      region/chemistry tally per dataset & stage

Run:
    python scripts/grn_dev_diagnostics/a_composition_qc.py
"""

from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated")
INPUTS = {
    "PsychAD":   ROOT / "PsychAD_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
    "Velmeshev": ROOT / "Vel_prepost_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
}
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Age windows match the §3 best-grid in grn_dev_multi.md
CHILD = (1.0, 9.0)
ADOL  = (9.0, 25.0)

def stage(age):
    if CHILD[0] <= age < CHILD[1]: return "child"
    if ADOL[0]  <= age < ADOL[1]:  return "adol"
    return "other"


def build_donor_df(adatas):
    rows = []
    for name, a in adatas.items():
        counts = a.layers["counts"]
        # per-donor total counts and n_genes_detected (gene with any count)
        total = np.asarray(counts.sum(axis=1)).ravel()
        if hasattr(counts, "getnnz"):
            n_genes = counts.getnnz(axis=1)
        else:
            n_genes = (counts > 0).sum(axis=1)
        d = a.obs.copy()
        d["dataset_name"] = name
        d["total_counts"] = total
        d["n_genes_detected"] = n_genes
        d["stage"] = d["age_years"].apply(stage)
        rows.append(d)
    return pd.concat(rows, axis=0, ignore_index=False)


def plot_a1(donors):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 91, 2)
    for name, color in [("PsychAD", "C0"), ("Velmeshev", "C1")]:
        sub = donors[donors["dataset_name"] == name]
        ax.hist(sub["age_years"], bins=bins, alpha=0.55, color=color,
                label=f"{name} (n={len(sub)})")
    ax.axvspan(*CHILD, color="green", alpha=0.10, label="childhood 1–9")
    ax.axvspan(*ADOL,  color="purple", alpha=0.10, label="adolescence 9–25")
    ax.set_xlabel("age (years)")
    ax.set_ylabel("donor count")
    ax.set_title("A1. Donor age density by dataset")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "a1_donor_density.png", dpi=150)
    plt.close(fig)

    # zoom 0-30
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 31, 1)
    for name, color in [("PsychAD", "C0"), ("Velmeshev", "C1")]:
        sub = donors[(donors["dataset_name"] == name) & (donors["age_years"] <= 30)]
        ax.hist(sub["age_years"], bins=bins, alpha=0.55, color=color,
                label=f"{name} (n={len(sub)})")
    ax.axvspan(*CHILD, color="green", alpha=0.10)
    ax.axvspan(*ADOL,  color="purple", alpha=0.10)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("donor count")
    ax.set_title("A1. Donor age density (0–30 y zoom)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "a1_donor_density_zoom.png", dpi=150)
    plt.close(fig)


def plot_a2(donors):
    metrics = [
        ("n_cells",          "ExN cells per donor (pseudobulk input)"),
        ("total_counts",     "Total raw counts per donor"),
        ("n_genes_detected", "Genes detected per donor (>=1 count)"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.0 * len(metrics)),
                             sharex=True)
    young = donors[donors["age_years"] <= 30].copy()
    for ax, (col, title) in zip(axes, metrics):
        for name, marker in [("PsychAD", "o"), ("Velmeshev", "s")]:
            sub = young[young["dataset_name"] == name]
            for chem, c in [("V2", "tab:red"), ("V3", "tab:blue")]:
                ss = sub[sub["chemistry"] == chem]
                if len(ss) == 0: continue
                ax.scatter(ss["age_years"], ss[col], alpha=0.65, s=22,
                           marker=marker, color=c,
                           label=f"{name} {chem} (n={len(ss)})")
        ax.set_yscale("log")
        ax.set_ylabel(col)
        ax.set_title("A2. " + title)
        ax.axvspan(*CHILD, color="green", alpha=0.08)
        ax.axvspan(*ADOL,  color="purple", alpha=0.08)
        ax.legend(loc="best", fontsize=7, ncol=2)
    axes[-1].set_xlabel("age (years)")
    fig.tight_layout()
    fig.savefig(OUT / "a2_per_donor_qc.png", dpi=150)
    plt.close(fig)


def tally_a4(donors):
    # region × chemistry × stage tally
    tab = (donors
           .groupby(["dataset_name", "stage", "region", "chemistry"], observed=True)
           .size()
           .rename("n_donors")
           .reset_index()
           .sort_values(["dataset_name", "stage", "region", "chemistry"]))
    tab.to_csv(OUT / "a4_region_chemistry.csv", index=False)

    # stage summary (counts, library, n_genes — medians)
    stages = (donors[donors["stage"] != "other"]
              .groupby(["dataset_name", "stage", "chemistry"], observed=True)
              .agg(n_donors=("individual", "count"),
                   median_n_cells=("n_cells", "median"),
                   median_total_counts=("total_counts", "median"),
                   median_n_genes=("n_genes_detected", "median"))
              .reset_index()
              .sort_values(["dataset_name", "stage", "chemistry"]))
    stages.to_csv(OUT / "a4_stage_summary.csv", index=False)
    return tab, stages


def main():
    adatas = {n: ad.read_h5ad(p) for n, p in INPUTS.items()}
    donors = build_donor_df(adatas)
    donors.to_csv(OUT / "a_donor_table.csv", index_label="obs_name")

    plot_a1(donors)
    plot_a2(donors)
    tab, stages = tally_a4(donors)

    print("=== A1/A2 donor table written ===")
    print(donors.groupby(["dataset_name", "stage"], observed=True).size().to_string())
    print("\n=== A4 region/chemistry tally ===")
    print(tab.to_string(index=False))
    print("\n=== A4 stage summary ===")
    print(stages.to_string(index=False))
    print(f"\nOutputs in: {OUT}")


if __name__ == "__main__":
    main()
