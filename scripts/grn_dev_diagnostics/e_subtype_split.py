#!/usr/bin/env python3
"""
Group E1 — split the ExN_manual bin by marker subtype.

Uses the per-donor × marker_annotation pseudobulks at
    .../pseudobulk_output/by_cell_class_manual.h5ad

For each subtype in {ExN_mature, ExN_immature, ExN_weak}, recompute
the aggregate AHBA C3+ weighted GRN score per donor, then donor-level
Cohen's d (child vs adol). Also reports the per-donor marker
composition shift across stages — does PsychAD's ExN bin contain
different proportions of {mature, immature, weak} than Velmeshev's
in childhood vs adolescence?

Outputs:
    e_subtype_aggregate_d.csv
    e_subtype_composition.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (
    cpm_from_counts, subset_age_window, project_score,
    build_c3plus_table, cohens_d, OUT_DIR, CHILD, ADOL,
)

INPUTS = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/by_cell_class_manual.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class_manual.h5ad",
}

SUBTYPES = ["ExN_mature", "ExN_immature", "ExN_weak"]


def aggregate_d_subtype(adata_by_class, subtype, weights, dataset_name):
    sub = adata_by_class[adata_by_class.obs["marker_annotation"] == subtype].copy()
    if sub.n_obs < 4:
        return None
    cpm = cpm_from_counts(sub)
    win = subset_age_window(cpm)
    if (win.obs["stage"] == "child").sum() < 2 or (win.obs["stage"] == "adol").sum() < 2:
        return None
    s = project_score(win, weights)
    sw = win.obs[["stage", "chemistry"]].join(s).dropna(subset=["score"])
    d_all = cohens_d(sw[sw["stage"] == "child"]["score"].values,
                     sw[sw["stage"] == "adol"]["score"].values)
    out = {
        "dataset": dataset_name,
        "subtype": subtype,
        "chemistry": "all",
        "n_child": int((sw["stage"] == "child").sum()),
        "n_adol":  int((sw["stage"] == "adol").sum()),
        "mean_child": float(sw[sw["stage"] == "child"]["score"].mean()),
        "mean_adol":  float(sw[sw["stage"] == "adol"]["score"].mean()),
        "cohens_d": float(d_all),
    }
    rows = [out]
    # also by chemistry
    for chem in sorted(sw["chemistry"].unique()):
        ss = sw[sw["chemistry"] == chem]
        nc = (ss["stage"] == "child").sum(); na = (ss["stage"] == "adol").sum()
        if nc < 2 or na < 2: continue
        rows.append({
            "dataset": dataset_name,
            "subtype": subtype,
            "chemistry": chem,
            "n_child": int(nc),
            "n_adol":  int(na),
            "mean_child": float(ss[ss["stage"] == "child"]["score"].mean()),
            "mean_adol":  float(ss[ss["stage"] == "adol"]["score"].mean()),
            "cohens_d": float(cohens_d(
                ss[ss["stage"] == "child"]["score"].values,
                ss[ss["stage"] == "adol"]["score"].values)),
        })
    return rows


def composition(adata_by_class, dataset_name):
    """For each donor, fraction of cells in each ExN subtype."""
    df = adata_by_class.obs[["individual", "age_years", "chemistry",
                              "marker_annotation", "n_cells"]].copy()
    df = df[df["marker_annotation"].isin(SUBTYPES)]
    tot = (df.groupby(["individual", "age_years", "chemistry"], observed=True)["n_cells"]
             .sum().rename("total_exn"))
    df = df.merge(tot, on=["individual", "age_years", "chemistry"])
    df["fraction"] = df["n_cells"] / df["total_exn"]
    df["dataset"] = dataset_name
    age = df["age_years"]
    df["stage"] = np.where((age >= CHILD[0]) & (age < CHILD[1]), "child",
                  np.where((age >= ADOL[0])  & (age < ADOL[1]),  "adol", "other"))
    return df


def main():
    weights = build_c3plus_table().set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes")

    all_rows = []
    comp_rows = []
    for name, p in INPUTS.items():
        a = ad.read_h5ad(p)
        print(f"\n=== {name} ({a.shape}) ===")
        for sub in SUBTYPES:
            r = aggregate_d_subtype(a, sub, weights, name)
            if r is None:
                print(f"  {sub}: skipped (too few donors in stage)")
                continue
            all_rows.extend(r)
        comp_rows.append(composition(a, name))

    res = pd.DataFrame(all_rows)
    res = res.sort_values(["dataset", "subtype", "chemistry"]).reset_index(drop=True)
    res.to_csv(OUT_DIR / "e_subtype_aggregate_d.csv", index=False)

    print("\n=== Aggregate Cohen's d per ExN subtype (child vs adol) ===")
    print(res.to_string(index=False))

    comp = pd.concat(comp_rows, ignore_index=True)
    comp.to_csv(OUT_DIR / "e_subtype_composition_per_donor.csv", index=False)

    # composition summary: mean fraction per stage × subtype
    csum = (comp[comp["stage"] != "other"]
            .groupby(["dataset", "stage", "marker_annotation", "chemistry"], observed=True)
            .agg(mean_fraction=("fraction", "mean"),
                 median_fraction=("fraction", "median"),
                 n_donors=("individual", "nunique"))
            .reset_index())
    csum.to_csv(OUT_DIR / "e_subtype_composition_summary.csv", index=False)
    print("\n=== ExN subtype composition (mean fraction per stage) ===")
    print(csum.to_string(index=False))

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
