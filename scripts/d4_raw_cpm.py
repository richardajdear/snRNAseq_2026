"""
D4 supplement: EN/IN marker expression in raw counts (library-size normalised).

Compares marker expression BEFORE scVI batch correction, to determine whether
the low EN/IN ratio in PsychAD <1y (~0.32) vs Vel-V3 <1y (~3.05) seen in
scanvi_normalized is a scVI artefact or reflects the underlying data.

Uses the `counts` layer (raw integer counts) from the integrated h5ad.
Library-size normalisation is computed within the 15k HVG subspace — this
gives a consistent relative comparison across groups, though not true CPM
(which would require the full transcriptome per cell).

Run with:
    sbatch --time=00:30:00 scripts/run_script.sh scripts/d4_raw_cpm.py
"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from pathlib import Path

RDS = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
H5AD = f"{RDS}/Cam_snRNAseq/integrated/VelWangPsychAD_semisup3_tuning5/scvi_output/integrated.h5ad"
OUT_DIR = Path("/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/relabel_comparison/no_age_run")

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
    "DLX2":    "ENSG00000115844",
    "LHX6":    "ENSG00000106688",
}
ALL_ENSG = {**EN_ENSG, **IN_ENSG}

print("=" * 70)
print("D4 raw CPM check — marker expression before scVI batch correction")
print(f"  Layer: counts (raw integer counts, library-size normalised within 15k HVGs)")
print(f"  Input: {H5AD}")
print("=" * 70)

print("\nLoading h5ad obs and var (backed mode)...")
adata = ad.read_h5ad(H5AD, backed="r")
print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"  Layers: {list(adata.layers.keys())}")

# Resolve marker columns
found_cols = {}
for sym, ensg in ALL_ENSG.items():
    if ensg in adata.var_names:
        found_cols[sym] = ensg
    else:
        print(f"  WARNING: {sym} ({ensg}) not found in var")
col_idx = [adata.var_names.get_loc(e) for e in found_cols.values()]
print(f"  Markers found: {list(found_cols.keys())}")

obs = adata.obs.copy()

groups = {
    "PSYCHAD_under1y": (obs["source-chemistry"] == "PSYCHAD-V3") & (obs["age_years"] >= 0) & (obs["age_years"] < 1),
    "PSYCHAD_1_5y":    (obs["source-chemistry"] == "PSYCHAD-V3") & (obs["age_years"] >= 1) & (obs["age_years"] < 5),
    "PSYCHAD_5_18y":   (obs["source-chemistry"] == "PSYCHAD-V3") & (obs["age_years"] >= 5) & (obs["age_years"] < 18),
    "WANG_prenatal":   (obs["source-chemistry"] == "WANG-multiome") & (obs["age_years"] < 0),
    "WANG_under1y":    (obs["source-chemistry"] == "WANG-multiome") & (obs["age_years"] >= 0) & (obs["age_years"] < 1),
    "VEL_V3_prenatal": (obs["source-chemistry"] == "VELMESHEV-V3") & (obs["age_years"] < 0),
    "VEL_V3_under1y":  (obs["source-chemistry"] == "VELMESHEV-V3") & (obs["age_years"] >= 0) & (obs["age_years"] < 1),
}

print("\nComputing per-cell library-size normalised expression from counts layer...")
print("  (normalisation = count / per-cell total counts × 1e4 — within 15k HVG subspace)")

rows = []
for grp_name, mask in groups.items():
    idx = np.where(mask.values)[0]
    n = len(idx)
    if n == 0:
        continue
    # Read counts block: rows=idx, all columns (need total for library size)
    counts_layer = adata.layers["counts"]
    if sp.issparse(counts_layer):
        block_all = counts_layer[idx, :]
        if sp.issparse(block_all):
            block_all = block_all.toarray()
    else:
        block_all = np.array(counts_layer[idx, :])

    # Library size normalise (within-HVG): divide each cell by its total counts * 1e4
    lib_size = block_all.sum(axis=1, keepdims=True).astype(float)
    lib_size[lib_size == 0] = 1.0  # avoid divide-by-zero for empty cells
    norm_block = block_all / lib_size * 1e4

    # Extract marker columns
    marker_block = norm_block[:, col_idx]
    en_cols = [i for i, sym in enumerate(found_cols.keys()) if sym in EN_ENSG]
    in_cols = [i for i, sym in enumerate(found_cols.keys()) if sym in IN_ENSG]
    en_mean = marker_block[:, en_cols].mean()
    in_mean = marker_block[:, in_cols].mean()
    ratio   = en_mean / (in_mean + 1e-9)

    print(f"  {grp_name}: n={n:,}  EN mean={en_mean:.4f}  IN mean={in_mean:.4f}  EN/IN={ratio:.3f}")

    for j, sym in enumerate(found_cols.keys()):
        col = marker_block[:, j]
        rows.append({
            "group":       grp_name,
            "n_cells":     n,
            "marker":      sym,
            "marker_type": "EN" if sym in EN_ENSG else "IN",
            "mean_norm":   float(col.mean()),
            "median_norm": float(np.median(col)),
            "frac_nonzero": float((col > 0).mean()),
        })

df = pd.DataFrame(rows)
out_csv = OUT_DIR / "marker_means_raw_cpm.csv"
df.to_csv(out_csv, index=False)

# Summary table
pivot = df.pivot_table(index="group", columns="marker", values="mean_norm")
print("\nMean library-size-normalised counts per group (×1e4):")
print(pivot.to_string())

summary = df.groupby(["group", "marker_type"])["mean_norm"].mean().unstack()
summary.columns.name = None
summary["EN_vs_IN_ratio"] = summary.get("EN", 0) / (summary.get("IN", 1) + 1e-9)
print("\nEN vs IN ratio — raw counts normalised:")
print(summary.to_string())

# Side-by-side comparison with scanvi_normalized
print("\n" + "=" * 70)
print("COMPARISON: raw counts-normalised vs scanvi_normalized EN/IN ratio")
print("  (scanvi_normalized from job 29623375)")
scanvi_ratios = {
    "PSYCHAD_under1y": 0.320,
    "PSYCHAD_1_5y":    0.708,
    "PSYCHAD_5_18y":   0.804,
    "WANG_under1y":    1.304,
    "VEL_V3_prenatal": 3.789,
    "VEL_V3_under1y":  3.047,
}
print(f"  {'Group':<22}  {'raw EN/IN':>10}  {'scanvi EN/IN':>12}  {'ratio change':>12}")
for grp, scanvi_r in scanvi_ratios.items():
    if grp in summary.index:
        raw_r = summary.loc[grp, "EN_vs_IN_ratio"]
        change = raw_r / (scanvi_r + 1e-9)
        print(f"  {grp:<22}  {raw_r:>10.3f}  {scanvi_r:>12.3f}  {change:>12.2f}×")

print(f"\nWritten: {out_csv}")
print("Done.")
