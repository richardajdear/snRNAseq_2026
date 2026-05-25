"""
D4 recovery: EN/IN marker expression in PsychAD <1y cells.

D4 in diagnose_psychad_relabel.py failed because the integrated h5ad stores
var with Ensembl IDs, not gene symbols.  This script uses the correct IDs.

Outputs to scripts/relabel_comparison/no_age_run/marker_means.csv
and prints a summary table to stdout.

Run with:
    sbatch --time=00:30:00 scripts/run_script.sh scripts/d4_marker_expression.py
"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path

RDS = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
H5AD = f"{RDS}/Cam_snRNAseq/integrated/VelWangPsychAD_semisup3_tuning5/scvi_output/integrated.h5ad"
OUT_DIR = Path("/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/relabel_comparison/no_age_run")

# Ensembl IDs for the 12 marker genes
EN_ENSG = {
    "SLC17A7": "ENSG00000104888",
    "NEUROD2":  "ENSG00000171532",
    "NEUROD6":  "ENSG00000197905",
    "RBFOX3":   "ENSG00000128266",
    "SATB2":    "ENSG00000119042",
    "TBR1":     "ENSG00000136535",
}
IN_ENSG = {
    "GAD1":    "ENSG00000128683",
    "GAD2":    "ENSG00000136750",
    "SLC32A1": "ENSG00000101438",
    "DLX1":    "ENSG00000144355",
    "DLX2":    "ENSG00000115844",
    "LHX6":    "ENSG00000106688",
}
ALL_ENSG = {**EN_ENSG, **IN_ENSG}

print("=" * 70)
print("D4 marker expression (Ensembl ID fix)")
print(f"  Input:  {H5AD}")
print(f"  Output: {OUT_DIR}/marker_means.csv")
print("=" * 70)

print("\nLoading h5ad (backed mode)...")
adata = ad.read_h5ad(H5AD, backed="r")
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"  var index sample: {list(adata.var_names[:5])}")

# Resolve marker columns
found_cols = {}
for sym, ensg in ALL_ENSG.items():
    if ensg in adata.var_names:
        found_cols[sym] = ensg
    else:
        print(f"  WARNING: {sym} ({ensg}) not found in var")
print(f"  Markers found: {list(found_cols.keys())}")

obs = adata.obs.copy()
obs["age_bin"] = pd.cut(
    obs["age_years"],
    bins=[-np.inf, 0, 1, 5, 18, 30, 50, np.inf],
    labels=["prenatal", "<1y", "1-5y", "5-18y", "18-30y", "30-50y", "50+y"],
    right=False,
)

# Groups of interest
groups = {
    "PSYCHAD_under1y": (obs["source-chemistry"] == "PSYCHAD-V3") & (obs["age_years"] >= 0) & (obs["age_years"] < 1),
    "PSYCHAD_1_5y":    (obs["source-chemistry"] == "PSYCHAD-V3") & (obs["age_years"] >= 1) & (obs["age_years"] < 5),
    "PSYCHAD_5_18y":   (obs["source-chemistry"] == "PSYCHAD-V3") & (obs["age_years"] >= 5) & (obs["age_years"] < 18),
    "WANG_prenatal":   (obs["source-chemistry"] == "WANG-multiome") & (obs["age_years"] < 0),
    "WANG_under1y":    (obs["source-chemistry"] == "WANG-multiome") & (obs["age_years"] >= 0) & (obs["age_years"] < 1),
    "VEL_V3_prenatal": (obs["source-chemistry"] == "VELMESHEV-V3") & (obs["age_years"] < 0),
    "VEL_V3_under1y":  (obs["source-chemistry"] == "VELMESHEV-V3") & (obs["age_years"] >= 0) & (obs["age_years"] < 1),
}

# Also break PsychAD <1y by predicted cell class
print("\nComputing mean scanvi_normalized per group and marker...")
rows = []
for grp_name, mask in groups.items():
    idx = np.where(mask.values)[0]
    n = len(idx)
    if n == 0:
        continue
    # Fetch only the needed columns from scanvi_normalized
    ensg_ids = list(found_cols.values())
    col_idx = [adata.var_names.get_loc(e) for e in ensg_ids]
    # Read layer slice: subset rows then columns
    layer = adata.layers["scanvi_normalized"]
    if sp.issparse(layer):
        block = layer[np.ix_(idx, col_idx)]
        if sp.issparse(block):
            block = block.toarray()
    else:
        block = np.array(layer[idx][:, col_idx])
    for j, sym in enumerate(found_cols.keys()):
        rows.append({
            "group": grp_name,
            "n_cells": n,
            "marker": sym,
            "marker_type": "EN" if sym in EN_ENSG else "IN",
            "mean_expr": float(block[:, j].mean()),
            "frac_nonzero": float((block[:, j] > 0).mean()),
        })
    print(f"  {grp_name}: n={n:,}  EN mean={block[:, :len(EN_ENSG)].mean():.4f}  IN mean={block[:, len(EN_ENSG):].mean():.4f}")

df = pd.DataFrame(rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_DIR / "marker_means.csv", index=False)

# Pivot for easy reading
pivot = df.pivot_table(index="group", columns="marker", values="mean_expr")
print("\nMean scanvi_normalized expression per group:")
print(pivot.to_string())

# Summary: EN vs IN mean per group
summary = df.groupby(["group", "marker_type"])["mean_expr"].mean().unstack()
summary.columns.name = None
summary["EN_vs_IN_ratio"] = summary.get("EN", 0) / (summary.get("IN", 1) + 1e-9)
print("\nEN vs IN mean expression per group:")
print(summary.to_string())

# Critical check: PsychAD <1y EN marker expression vs Wang/Vel <1y
print("\n" + "=" * 70)
print("KEY QUESTION: Do PsychAD <1y cells express EN markers?")
for sym in EN_ENSG:
    if sym not in found_cols:
        continue
    sub = df[df["marker"] == sym][["group", "mean_expr", "frac_nonzero"]].set_index("group")
    pa  = sub.loc["PSYCHAD_under1y", "mean_expr"] if "PSYCHAD_under1y" in sub.index else float("nan")
    wng = sub.loc["WANG_under1y",    "mean_expr"] if "WANG_under1y"    in sub.index else float("nan")
    vel = sub.loc["VEL_V3_under1y",  "mean_expr"] if "VEL_V3_under1y"  in sub.index else float("nan")
    print(f"  {sym:12s}  PsychAD<1y={pa:.4f}  Wang<1y={wng:.4f}  Vel<1y={vel:.4f}")

print(f"\nWritten: {OUT_DIR}/marker_means.csv")
print("Done.")
