"""
D10_v2 — HVG-intersection sensitivity test, FIXED version.

Uses the integrated h5ad's `counts` layer (verified raw integer counts) for
all three sources. Avoids the .X/.raw.X confusion in per-dataset h5ads.

Approach (within the 15k HVG subspace of integrated.h5ad):
1. Load integrated h5ad; identify <1y cells per source.
2. Per-cell QC on raw counts: total UMI, n_genes detected (within 15k HVG).
3. Per-source HVG selection (seurat_v3 on counts) at sizes 1000, 2500, 5000,
   10000.  (Capped at 15000 since that's the size of the integrated HVG set.)
4. Intersect per-source HVG sets at each size.
5. Renormalise CPM within intersection only.
6. Compute EN/IN marker means and EN/IN ratio per group at each size.
7. Reference: full 15k HVG baseline (matches D4 raw CPM).

User hypothesis: PsychAD has higher UMI/cell (2× confirmed from sanity check).
If the EN/IN gap narrows as HVG intersection focuses on common variable genes,
the gap is partly a CPM denominator artefact. If it persists, it is real
biology.

Run with:
  sbatch --time=01:00:00 scripts/run_script.sh scripts/d10_v2_hvg_intersection.py
"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scanpy as sc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

RDS = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
INTEGRATED = f"{RDS}/Cam_snRNAseq/integrated/VelWangPsychAD_semisup3_tuning5/scvi_output/integrated.h5ad"
OUT = Path("/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/relabel_comparison/hvg_intersection_v2")
OUT.mkdir(parents=True, exist_ok=True)

EN_ENSG = {
    "SLC17A7": "ENSG00000104888",
    "NEUROD2": "ENSG00000171532",
    "NEUROD6": "ENSG00000197905",
    "RBFOX3":  "ENSG00000128266",
    "SATB2":   "ENSG00000119042",
    "TBR1":    "ENSG00000136535",
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
HVG_SIZES = [1000, 2500, 5000, 10000]

print("=" * 70)
print("D10_v2 — HVG intersection (uses integrated counts layer, verified raw)")
print(f"  Output: {OUT}")
print("=" * 70)

# ─── 1. Identify <1y cells per source ────────────────────────────────────
print("\n[1] Reading integrated h5ad obs...")
ai = ad.read_h5ad(INTEGRATED, backed="r")
obs = ai.obs
age = obs["age_years"].values.astype(float)
src = obs["source-chemistry"].astype(str).values

groups = {
    "PSYCHAD": (src == "PSYCHAD-V3")     & (age >= 0) & (age < 1),
    "WANG":    (src == "WANG-multiome")  & (age >= 0) & (age < 1),
    "VEL":     (src == "VELMESHEV-V3")   & (age >= 0) & (age < 1),
}
indices = {k: np.where(m)[0] for k, m in groups.items()}
for k, idx in indices.items():
    print(f"  {k} <1y: {len(idx):,}")

# ─── 2. Load counts layer to memory per group ────────────────────────────
print("\n[2] Loading counts blocks per group (counts layer, integer raw)...")
# Counts is sparse in layers
counts_layer = ai.layers["counts"]
print(f"  counts layer type: {type(counts_layer).__name__}")
# Backed sparse needs slicing
def load_block(idx):
    """Load a (n_idx, n_vars) dense block from counts layer."""
    if hasattr(counts_layer, 'toarray'):
        # already in-memory sparse
        return counts_layer[idx, :]
    else:
        # backed sparse - use chunked loading
        return counts_layer[idx, :]

blocks = {}
for k, idx in indices.items():
    # use sorted indices for sparse slicing
    sidx = np.sort(idx)
    print(f"  Loading {k}: {len(sidx):,} cells...")
    blk = load_block(sidx)
    # Ensure sparse CSR
    if not sp.issparse(blk):
        blk = sp.csr_matrix(blk)
    elif not sp.isspmatrix_csr(blk):
        blk = blk.tocsr()
    blocks[k] = blk
    print(f"    shape={blk.shape}, nnz={blk.nnz:,}")

var_names = ai.var_names.values
ai.file.close()
del ai

# Verify integer-like
print("\n  Verify raw integer counts:")
for k, blk in blocks.items():
    sample = blk.data[:1000] if blk.data.size > 1000 else blk.data
    is_int = np.allclose(sample, np.round(sample))
    print(f"    {k}: integer-like={is_int}, sample max={sample.max():.1f}, mean={sample.mean():.2f}")

# ─── 3. Per-cell QC ──────────────────────────────────────────────────────
print("\n[3] Per-cell QC (in 15k HVG subspace of integrated.h5ad)")
qc_rows = []
for k, blk in blocks.items():
    total = np.asarray(blk.sum(axis=1)).flatten()
    ngenes = np.asarray((blk > 0).sum(axis=1)).flatten()
    qc_rows.append({
        "group":            k,
        "n_cells":          int(blk.shape[0]),
        "total_counts_mean":   float(total.mean()),
        "total_counts_median": float(np.median(total)),
        "total_counts_P25":    float(np.percentile(total, 25)),
        "total_counts_P75":    float(np.percentile(total, 75)),
        "n_genes_mean":      float(ngenes.mean()),
        "n_genes_median":    float(np.median(ngenes)),
        "frac_genes_detected": float(ngenes.mean() / blk.shape[1]),
    })
    print(f"  {k}: total_UMI mean={total.mean():,.0f}  median={np.median(total):,.0f}  "
          f"P25-P75=[{np.percentile(total,25):,.0f}, {np.percentile(total,75):,.0f}]")
    print(f"      n_genes  mean={ngenes.mean():,.0f}  median={np.median(ngenes):,.0f}  "
          f"frac_detected={ngenes.mean()/blk.shape[1]:.3f}")

pd.DataFrame(qc_rows).to_csv(OUT / "per_cell_qc.csv", index=False)

# ─── 4. Per-source HVG selection (within 15k HVG subspace) ──────────────
print("\n[4] Per-source HVG selection at multiple sizes")
def compute_hvg(blk, n_top, name):
    """Return set of var_names that are HVG."""
    a = ad.AnnData(X=blk.copy())
    a.var_names = pd.Index(var_names)
    try:
        sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor="seurat_v3")
        hvg = set(a.var_names[a.var["highly_variable"].values].tolist())
    except Exception as e:
        print(f"    HVG failed for {name}: {e}")
        hvg = set()
    return hvg

hvg_records = []
hvg_intersections = {}
for size in HVG_SIZES:
    print(f"\n  --- HVG size = {size} ---")
    hvg_sets = {}
    for k, blk in blocks.items():
        hvg = compute_hvg(blk, size, k)
        hvg_sets[k] = hvg
        print(f"    {k}: {len(hvg):,}")
    inter = set.intersection(*hvg_sets.values()) if all(hvg_sets.values()) else set()
    union = set.union(*hvg_sets.values()) if any(hvg_sets.values()) else set()
    en_present = [s for s, e in EN_ENSG.items() if e in inter]
    in_present = [s for s, e in IN_ENSG.items() if e in inter]
    print(f"    INTERSECTION: {len(inter):,}")
    print(f"    UNION:        {len(union):,}")
    print(f"    EN markers in intersection: {en_present}")
    print(f"    IN markers in intersection: {in_present}")
    hvg_records.append({
        "hvg_size":     size,
        "psychad_hvg":  len(hvg_sets["PSYCHAD"]),
        "wang_hvg":     len(hvg_sets["WANG"]),
        "vel_hvg":      len(hvg_sets["VEL"]),
        "intersection": len(inter),
        "union":        len(union),
        "en_markers_in_intersection": ",".join(en_present),
        "in_markers_in_intersection": ",".join(in_present),
    })
    hvg_intersections[size] = (inter, en_present, in_present)

pd.DataFrame(hvg_records).to_csv(OUT / "hvg_sizes_per_dataset.csv", index=False)

# ─── 5. Marker means within intersection ─────────────────────────────────
print("\n[5] EN/IN marker means at each HVG intersection size")

var_pos = {g: i for i, g in enumerate(var_names)}

def markers_in_subset(blk, inter_genes, en_syms, in_syms):
    """CPM-normalise within `inter_genes`, return EN/IN means and library stats."""
    inter_idx = np.array([var_pos[g] for g in inter_genes if g in var_pos])
    if len(inter_idx) == 0:
        return None
    en_idx = [var_pos[EN_ENSG[s]] for s in en_syms if EN_ENSG[s] in var_pos]
    in_idx = [var_pos[IN_ENSG[s]] for s in in_syms if IN_ENSG[s] in var_pos]

    # library size = sum within intersection
    inter_block = blk[:, inter_idx]
    lib = np.asarray(inter_block.sum(axis=1)).flatten().astype(float)
    lib[lib == 0] = 1.0

    en_block = blk[:, en_idx].toarray() if en_idx else np.zeros((blk.shape[0], 0))
    in_block = blk[:, in_idx].toarray() if in_idx else np.zeros((blk.shape[0], 0))
    en_norm = float((en_block / lib[:, None] * 1e4).mean()) if en_idx else 0.0
    in_norm = float((in_block / lib[:, None] * 1e4).mean()) if in_idx else 0.0
    return {
        "en_mean": en_norm,
        "in_mean": in_norm,
        "en_over_in": en_norm / (in_norm + 1e-9),
        "lib_mean": float(lib.mean()),
        "lib_median": float(np.median(lib)),
        "n_en_markers": len(en_idx),
        "n_in_markers": len(in_idx),
    }

marker_records = []

# Reference: full 15k HVG (matches D4 raw CPM)
print("\n  [Reference] Full 15k HVG (matches D4 raw CPM):")
full_genes = set(var_names)
en_full = [s for s in EN_ENSG if EN_ENSG[s] in full_genes]
in_full = [s for s in IN_ENSG if IN_ENSG[s] in full_genes]
for k, blk in blocks.items():
    res = markers_in_subset(blk, full_genes, en_full, in_full)
    print(f"    {k}: n={blk.shape[0]:,}  lib_mean={res['lib_mean']:,.0f}  "
          f"EN={res['en_mean']:.4f}  IN={res['in_mean']:.4f}  EN/IN={res['en_over_in']:.3f}")
    marker_records.append({"hvg_size": "FULL_15K", "intersection_size": len(full_genes),
                           "group": k, "n_cells": int(blk.shape[0]), **res})

# Each HVG size
for size in HVG_SIZES:
    inter, en_present, in_present = hvg_intersections[size]
    if not inter:
        print(f"\n  HVG={size}: empty intersection")
        continue
    print(f"\n  HVG={size}, intersection={len(inter)}, EN markers={len(en_present)}, IN markers={len(in_present)}")
    for k, blk in blocks.items():
        res = markers_in_subset(blk, inter, en_present, in_present)
        if res is None:
            continue
        print(f"    {k}: n={blk.shape[0]:,}  lib_mean={res['lib_mean']:,.0f}  "
              f"EN={res['en_mean']:.4f}  IN={res['in_mean']:.4f}  EN/IN={res['en_over_in']:.3f}")
        marker_records.append({"hvg_size": size, "intersection_size": len(inter),
                               "group": k, "n_cells": int(blk.shape[0]), **res})

df = pd.DataFrame(marker_records)
df.to_csv(OUT / "markers_by_hvg_size.csv", index=False)

# ─── 6. Summary ──────────────────────────────────────────────────────────
print("\n--- Summary: EN/IN ratio by HVG size ---")
pivot_ratio = df.pivot_table(index="group", columns="hvg_size", values="en_over_in")
print(pivot_ratio.to_string())

print("\n--- Library size in intersection by HVG size ---")
pivot_lib = df.pivot_table(index="group", columns="hvg_size", values="lib_mean")
print(pivot_lib.to_string())

# Key comparison: does PsychAD EN/IN narrow toward Wang/Vel as HVG narrows?
print("\n--- Key test: PsychAD EN/IN as fraction of Wang and Vel EN/IN ---")
for size in ["FULL_15K"] + HVG_SIZES:
    sub = df[df["hvg_size"] == size].set_index("group")
    if not {"PSYCHAD", "WANG", "VEL"}.issubset(sub.index):
        continue
    psy_r = sub.loc["PSYCHAD", "en_over_in"]
    w_r   = sub.loc["WANG", "en_over_in"]
    v_r   = sub.loc["VEL",  "en_over_in"]
    print(f"  size={str(size):<10}  PsychAD={psy_r:.3f}  Wang={w_r:.3f}  Vel={v_r:.3f}  "
          f"PsychAD/Wang={psy_r/(w_r+1e-9):.3f}  PsychAD/Vel={psy_r/(v_r+1e-9):.3f}")

# Write summary.md
summary_lines = [
    "# D10_v2 — HVG intersection (raw counts from integrated h5ad)",
    "",
    "## Per-cell QC (in 15k HVG subspace)",
    "",
    "| Group | n_cells | total_UMI mean | total_UMI median | n_genes mean | n_genes median | frac_detected |",
    "|-------|--------:|---------------:|-----------------:|-------------:|---------------:|--------------:|",
]
for r in qc_rows:
    summary_lines.append(
        f"| {r['group']} | {r['n_cells']:,} | {r['total_counts_mean']:,.0f} | "
        f"{r['total_counts_median']:,.0f} | {r['n_genes_mean']:,.0f} | "
        f"{r['n_genes_median']:,.0f} | {r['frac_genes_detected']:.3f} |"
    )
summary_lines += [
    "",
    "## HVG sizes per source and intersection",
    "",
    "| HVG size | PsychAD HVG | Wang HVG | Vel HVG | Intersection | EN markers | IN markers |",
    "|---------:|------------:|---------:|--------:|-------------:|:-----------|:-----------|",
]
for r in hvg_records:
    summary_lines.append(
        f"| {r['hvg_size']} | {r['psychad_hvg']} | {r['wang_hvg']} | {r['vel_hvg']} | "
        f"{r['intersection']} | {r['en_markers_in_intersection']} | {r['in_markers_in_intersection']} |"
    )
summary_lines += [
    "",
    "## EN/IN ratio by HVG intersection size",
    "",
    pivot_ratio.round(3).to_markdown(),
    "",
    "## Library size (sum in intersection) by HVG size",
    "",
    pivot_lib.round(0).to_markdown(),
    "",
    "## Decision rule",
    "",
    "If PsychAD EN/IN rises substantially as intersection narrows (toward Wang/Vel ratios), the gap is a CPM/gene-pool artefact. If PsychAD/Wang and PsychAD/Vel ratios stay flat (~0.08-0.12), the gap is genuine biology.",
]
with open(OUT / "summary.md", "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nWritten:")
print(f"  {OUT / 'per_cell_qc.csv'}")
print(f"  {OUT / 'hvg_sizes_per_dataset.csv'}")
print(f"  {OUT / 'markers_by_hvg_size.csv'}")
print(f"  {OUT / 'summary.md'}")
print("\nDone.")
