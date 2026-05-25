"""
D10 — Test whether the PsychAD <1y vs Wang/Vel <1y EN/IN gap is a CPM
normalization artifact driven by differing gene pools across datasets.

User hypothesis: PsychAD nuclei have a larger expressed gene pool (broader
transcription per cell), so per-cell library-size normalisation inflates
the denominator and deflates per-marker expression. Restricting to a
common gene set across all three datasets — particularly HVGs intersected
across datasets — should yield comparable marker ratios.

Approach:
1. Load integrated h5ad obs to find <1y barcodes per source (PsychAD-V3,
   Wang-multiome, Vel-V3).
2. Load each per-dataset h5ad (FULL transcriptome) and subset to those
   barcodes.
3. Per-cell QC: total UMI, n_genes detected (full transcriptome), per group.
4. Find common gene set across all three datasets (intersection of var_names).
5. Per-dataset HVG selection (seurat_v3 on counts, within common gene set)
   at sizes 2k, 5k, 10k.
6. Intersect HVG sets across datasets at each size.
7. CPM-normalise within intersection (denominator = sum of counts within
   intersection only).
8. Compute EN/IN marker means and EN/IN ratio per group at each HVG size.

Sensitivity test: if PsychAD <1y EN/IN rises substantially as the HVG
intersection narrows, the gap is a CPM artefact. If it stays at ~0.16,
the gap is genuine signal in the data.

Run with:
  sbatch --mem=200G --time=02:00:00 scripts/run_script.sh scripts/d10_hvg_intersection.py
"""
import sys
import warnings
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

DATASETS = {
    "WANG":     f"{RDS}/Cam_snRNAseq/wang/wang.h5ad",
    "VEL":      f"{RDS}/Cam_snRNAseq/velmeshev/velmeshev.h5ad",
    "PSY_HBCC": f"{RDS}/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad",
    "PSY_AGE":  f"{RDS}/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad",
}

OUT = Path("/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/relabel_comparison/hvg_intersection")
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

HVG_SIZES = [2000, 5000, 10000]

print("=" * 70)
print("D10 — HVG intersection test")
print(f"  Output dir: {OUT}")
print("=" * 70)


# ─── 1. Get <1y barcodes per source from integrated h5ad ────────────────────
print("\n--- 1. Reading integrated h5ad metadata ---")
adata_int = ad.read_h5ad(INTEGRATED, backed="r")
obs = adata_int.obs
age = obs["age_years"].values.astype(float)
src = obs["source-chemistry"].astype(str).values
obs_names_int = adata_int.obs_names.values

mask = {}
mask["PSYCHAD"] = (src == "PSYCHAD-V3") & (age >= 0) & (age < 1)
mask["WANG"]    = (src == "WANG-multiome") & (age >= 0) & (age < 1)
mask["VEL"]     = (src == "VELMESHEV-V3") & (age >= 0) & (age < 1)

barcodes = {k: set(obs_names_int[m]) for k, m in mask.items()}
print(f"  PsychAD <1y: {len(barcodes['PSYCHAD']):,} cells")
print(f"  Wang <1y:    {len(barcodes['WANG']):,} cells")
print(f"  Vel-V3 <1y:  {len(barcodes['VEL']):,} cells")
adata_int.file.close()


# ─── 2. Load per-dataset h5ads and subset to matching barcodes ──────────────
def load_subset(name, path, target_barcodes):
    print(f"\n  Loading {name}: {path}")
    a = ad.read_h5ad(path, backed="r")
    print(f"    full: {a.n_obs:,} × {a.n_vars:,}")
    # Match by obs_names
    common = np.array([b in target_barcodes for b in a.obs_names])
    idx = np.where(common)[0]
    print(f"    matched: {len(idx):,} cells")
    if len(idx) == 0:
        a.file.close()
        return None
    sub = a[idx].to_memory()
    a.file.close()
    return sub


print("\n--- 2. Loading per-dataset h5ads (full transcriptome) ---")
wang = load_subset("WANG", DATASETS["WANG"], barcodes["WANG"])
vel  = load_subset("VEL",  DATASETS["VEL"],  barcodes["VEL"])

# PsychAD: try both cohorts and concatenate
psy_parts = []
for cohort in ["PSY_HBCC", "PSY_AGE"]:
    p = load_subset(cohort, DATASETS[cohort], barcodes["PSYCHAD"])
    if p is not None and p.n_obs > 0:
        psy_parts.append(p)

if not psy_parts:
    print("  ERROR: no PsychAD cells matched")
    sys.exit(1)

psy = ad.concat(psy_parts, join="inner", merge="same") if len(psy_parts) > 1 else psy_parts[0]
print(f"  PsychAD <1y total: {psy.n_obs:,} cells × {psy.n_vars:,} genes")
print(f"  Wang <1y:           {wang.n_obs:,} × {wang.n_vars:,}")
print(f"  Vel-V3 <1y:         {vel.n_obs:,} × {vel.n_vars:,}")

# Make sure .X is counts (per-dataset files have no 'counts' layer; .X is raw)
def get_counts(a):
    X = a.X
    if sp.issparse(X):
        # Verify integer-like
        sample = X.data[:1000] if X.data.size > 1000 else X.data
        is_int = np.allclose(sample, np.round(sample))
        if not is_int:
            print(f"  WARNING: .X does not look like raw integer counts")
    return X


# ─── 3. Per-cell QC (full transcriptome) ────────────────────────────────────
print("\n--- 3. Per-cell QC (full transcriptome) ---")
qc_rows = []
for name, a in [("PSYCHAD", psy), ("WANG", wang), ("VEL", vel)]:
    X = get_counts(a)
    total = np.asarray(X.sum(axis=1)).flatten()
    if sp.issparse(X):
        ngenes = np.asarray((X > 0).sum(axis=1)).flatten()
    else:
        ngenes = (X > 0).sum(axis=1)
    qc_rows.append({
        "group": name,
        "n_cells": a.n_obs,
        "n_vars": a.n_vars,
        "total_counts_mean": float(total.mean()),
        "total_counts_median": float(np.median(total)),
        "total_counts_P25": float(np.percentile(total, 25)),
        "total_counts_P75": float(np.percentile(total, 75)),
        "n_genes_mean": float(ngenes.mean()),
        "n_genes_median": float(np.median(ngenes)),
        "n_genes_P25": float(np.percentile(ngenes, 25)),
        "n_genes_P75": float(np.percentile(ngenes, 75)),
    })
    print(f"  {name}: total UMI mean={total.mean():,.0f}  median={np.median(total):,.0f}  P25-P75=[{np.percentile(total,25):,.0f}, {np.percentile(total,75):,.0f}]")
    print(f"         n_genes mean={ngenes.mean():,.0f}  median={np.median(ngenes):,.0f}  P25-P75=[{np.percentile(ngenes,25):,.0f}, {np.percentile(ngenes,75):,.0f}]")

pd.DataFrame(qc_rows).to_csv(OUT / "per_cell_qc.csv", index=False)


# ─── 4. Common gene set across the three datasets ───────────────────────────
print("\n--- 4. Common gene set ---")
common = sorted(set(psy.var_names) & set(wang.var_names) & set(vel.var_names))
print(f"  PsychAD vars: {psy.n_vars:,}")
print(f"  Wang vars:    {wang.n_vars:,}")
print(f"  Vel vars:     {vel.n_vars:,}")
print(f"  Common:       {len(common):,}")

# Check markers in common
en_in_common = [s for s, e in EN_ENSG.items() if e in common]
in_in_common = [s for s, e in IN_ENSG.items() if e in common]
print(f"  EN markers in common: {en_in_common}")
print(f"  IN markers in common: {in_in_common}")

# Subset each to common genes
print("\n  Subsetting to common genes...")
psy_c  = psy[:,  common].copy()
wang_c = wang[:, common].copy()
vel_c  = vel[:,  common].copy()


# ─── 5. Per-dataset HVG within common gene set ──────────────────────────────
print("\n--- 5. Per-dataset HVG selection ---")
hvg_results = {}
for size in HVG_SIZES:
    print(f"\n  HVG size = {size}")
    hvg_sets = {}
    for name, a in [("PSYCHAD", psy_c), ("WANG", wang_c), ("VEL", vel_c)]:
        try:
            # seurat_v3 expects counts in .X
            sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=size, inplace=False)
            # The above is inplace=False; recompute with inplace to get var col
            sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=size)
            hvg = set(a.var_names[a.var["highly_variable"].values])
        except Exception as e:
            print(f"    {name} HVG failed: {e}")
            hvg = set()
        hvg_sets[name] = hvg
        print(f"    {name}: HVG={len(hvg):,}")
    intersection = set.intersection(*hvg_sets.values()) if all(hvg_sets.values()) else set()
    print(f"    INTERSECTION: {len(intersection):,}")
    # Marker presence in intersection
    en_present = [s for s, e in EN_ENSG.items() if e in intersection]
    in_present = [s for s, e in IN_ENSG.items() if e in intersection]
    print(f"    EN markers in intersection: {en_present}")
    print(f"    IN markers in intersection: {in_present}")
    hvg_results[size] = {
        "hvg_sets": hvg_sets,
        "intersection": intersection,
        "en_present": en_present,
        "in_present": in_present,
    }


# ─── 6. CPM within intersection, compute markers ────────────────────────────
print("\n--- 6. Marker EN/IN per group at each HVG size ---")
results = []

# Reference: full common-gene CPM (no HVG filter)
print("\n  [Reference] FULL common-gene CPM (no HVG filter):")
for name, a in [("PSYCHAD", psy_c), ("WANG", wang_c), ("VEL", vel_c)]:
    X = get_counts(a)
    lib = np.asarray(X.sum(axis=1)).flatten().astype(float)
    lib[lib == 0] = 1.0
    en_idx = [a.var_names.get_loc(EN_ENSG[s]) for s in en_in_common]
    in_idx = [a.var_names.get_loc(IN_ENSG[s]) for s in in_in_common]
    en_block = X[:, en_idx]
    in_block = X[:, in_idx]
    if sp.issparse(en_block): en_block = en_block.toarray()
    if sp.issparse(in_block): in_block = in_block.toarray()
    en_norm = (en_block / lib[:, None] * 1e4).mean()
    in_norm = (in_block / lib[:, None] * 1e4).mean()
    ratio = en_norm / (in_norm + 1e-9)
    results.append({
        "hvg_size": "FULL_COMMON",
        "intersection_size": len(common),
        "group": name, "n_cells": a.n_obs,
        "EN_mean": en_norm, "IN_mean": in_norm, "EN_over_IN": ratio,
    })
    print(f"    {name}: n={a.n_obs:,}  EN={en_norm:.4f}  IN={in_norm:.4f}  EN/IN={ratio:.3f}")

for size in HVG_SIZES:
    res = hvg_results[size]
    intersection = res["intersection"]
    en_present = res["en_present"]
    in_present = res["in_present"]
    if not intersection:
        print(f"\n  HVG={size}: empty intersection, skipping")
        continue
    inter_list = sorted(intersection)
    print(f"\n  HVG={size}, intersection={len(intersection)}, EN markers={len(en_present)}, IN markers={len(in_present)}")
    for name, a in [("PSYCHAD", psy_c), ("WANG", wang_c), ("VEL", vel_c)]:
        # Find intersection-set column indices in this dataset
        # Build position map once
        var_pos = {g: i for i, g in enumerate(a.var_names.values)}
        inter_idx = np.array([var_pos[g] for g in inter_list if g in var_pos])
        en_idx = [var_pos[EN_ENSG[s]] for s in en_present if EN_ENSG[s] in var_pos]
        in_idx = [var_pos[IN_ENSG[s]] for s in in_present if IN_ENSG[s] in var_pos]
        # CPM within intersection only — slice block, compute per-cell sum, normalise
        X = get_counts(a)
        # Slice the intersection block (sparse-friendly)
        inter_block = X[:, inter_idx]
        if sp.issparse(inter_block):
            lib = np.asarray(inter_block.sum(axis=1)).flatten().astype(float)
        else:
            lib = inter_block.sum(axis=1).astype(float)
        lib[lib == 0] = 1.0
        # Marker blocks
        en_block = X[:, en_idx] if en_idx else None
        in_block = X[:, in_idx] if in_idx else None
        if en_block is not None and sp.issparse(en_block): en_block = en_block.toarray()
        if in_block is not None and sp.issparse(in_block): in_block = in_block.toarray()
        en_norm = float((en_block / lib[:, None] * 1e4).mean()) if en_block is not None else 0.0
        in_norm = float((in_block / lib[:, None] * 1e4).mean()) if in_block is not None else 0.0
        ratio = en_norm / (in_norm + 1e-9)
        results.append({
            "hvg_size": size,
            "intersection_size": len(intersection),
            "group": name, "n_cells": a.n_obs,
            "EN_mean": en_norm, "IN_mean": in_norm, "EN_over_IN": ratio,
        })
        print(f"    {name}: n={a.n_obs:,}  EN={en_norm:.4f}  IN={in_norm:.4f}  EN/IN={ratio:.3f}")


# ─── 7. Save & summary ──────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUT / "markers_by_hvg_size.csv", index=False)

print("\n--- Summary: EN/IN ratio by HVG size ---")
pivot_ratio = df.pivot_table(index="group", columns="hvg_size", values="EN_over_IN")
print(pivot_ratio.to_string())

print("\n--- Summary: EN mean by HVG size ---")
pivot_en = df.pivot_table(index="group", columns="hvg_size", values="EN_mean")
print(pivot_en.to_string())

print("\n--- Summary: IN mean by HVG size ---")
pivot_in = df.pivot_table(index="group", columns="hvg_size", values="IN_mean")
print(pivot_in.to_string())

# Key comparison: does PsychAD EN/IN rise relative to Wang/Vel as HVG narrows?
print("\n--- Key test: PsychAD EN/IN as fraction of Wang and Vel EN/IN ---")
for size in ["FULL_COMMON"] + HVG_SIZES:
    sub = df[df["hvg_size"] == size].set_index("group")
    if "PSYCHAD" not in sub.index or "WANG" not in sub.index or "VEL" not in sub.index:
        continue
    psy_r = sub.loc["PSYCHAD", "EN_over_IN"]
    wang_r = sub.loc["WANG", "EN_over_IN"]
    vel_r = sub.loc["VEL", "EN_over_IN"]
    pw = psy_r / (wang_r + 1e-9)
    pv = psy_r / (vel_r + 1e-9)
    print(f"  size={size:<12}  PsychAD/Wang={pw:.3f}  PsychAD/Vel={pv:.3f}  (PsychAD={psy_r:.3f}, Wang={wang_r:.3f}, Vel={vel_r:.3f})")

print(f"\nWritten:")
print(f"  {OUT / 'per_cell_qc.csv'}")
print(f"  {OUT / 'markers_by_hvg_size.csv'}")
print("\nDone.")
