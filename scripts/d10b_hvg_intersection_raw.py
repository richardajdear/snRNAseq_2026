"""
D10b — CORRECTED HVG intersection test.

Fix from D10: the original used `.X` directly. For Wang/Vel CellxGene-format
h5ads, `.X` is LOG-NORMALIZED expression; the raw integer counts live in
`.raw.X` (with their own var). PsychAD has `.X` = raw counts and no .raw.

D10's QC numbers for Wang/Vel were therefore meaningless (sums of log-
normalised values, not UMI counts). This script:
  • uses .raw.X for Wang and Vel,
  • uses .X for PsychAD (which IS raw counts),
  • verifies integer-likeness for each before proceeding.

Then repeats:
  1. Per-cell QC (true raw UMI, n_genes detected, full transcriptome)
  2. Common gene set across all three (intersection of var_names)
  3. Per-dataset HVG at sizes 2k, 5k, 10k within common gene set
  4. Intersect HVG sets across datasets
  5. CPM-normalise within intersection
  6. Marker EN/IN per group

Output: scripts/relabel_comparison/hvg_intersection/d10b_*
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
    "SLC17A7": "ENSG00000104888", "NEUROD2": "ENSG00000171532",
    "NEUROD6": "ENSG00000197905", "RBFOX3":  "ENSG00000128266",
    "SATB2":   "ENSG00000119042", "TBR1":    "ENSG00000136535",
}
IN_ENSG = {
    "GAD1":    "ENSG00000128683", "GAD2":    "ENSG00000136750",
    "SLC32A1": "ENSG00000101438", "DLX1":    "ENSG00000144355",
    "DLX2":    "ENSG00000115844", "LHX6":    "ENSG00000106688",
}
ALL_ENSG = {**EN_ENSG, **IN_ENSG}
HVG_SIZES = [2000, 5000, 10000]

print("=" * 70)
print("D10b — HVG intersection test (RAW counts, fixed)")
print("=" * 70)


def extract_raw_anndata(a, label):
    """Return a fresh AnnData whose .X is the raw integer counts.

    For Wang/Vel (CellxGene format) raw counts live in a.raw.X with their
    own var. For PsychAD raw counts live in a.X.
    """
    if a.raw is not None:
        # raw.X has its own var_names — build a new AnnData
        Xr = a.raw.X
        # raw.var has a different index, but we want gene Ensembl IDs.
        # raw.var._index should match raw.X columns.
        var_raw = a.raw.var.copy()
        new = ad.AnnData(X=Xr, obs=a.obs.copy(), var=var_raw)
        # Sanity: integer
        sample = new.X.data[:1000] if sp.issparse(new.X) else new.X.flatten()[:1000]
        is_int = np.allclose(sample, np.round(sample))
        print(f"  {label}: using .raw.X — n_vars={new.n_vars:,}, integer-like={is_int}, max sample={sample.max():.1f}")
        if not is_int:
            print(f"  WARNING: {label} raw.X is NOT integer!")
        return new
    else:
        sample = a.X.data[:1000] if sp.issparse(a.X) else a.X.flatten()[:1000]
        is_int = np.allclose(sample, np.round(sample))
        print(f"  {label}: using .X directly (no .raw) — n_vars={a.n_vars:,}, integer-like={is_int}, max sample={sample.max():.1f}")
        if not is_int:
            print(f"  WARNING: {label} .X is NOT integer!")
        return a


# ─── 1. Get <1y barcodes per source ──────────────────────────────────
print("\n--- 1. Reading integrated h5ad metadata ---")
adata_int = ad.read_h5ad(INTEGRATED, backed="r")
obs = adata_int.obs
age = obs["age_years"].values.astype(float)
src = obs["source-chemistry"].astype(str).values
obs_names_int = adata_int.obs_names.values

barcodes = {
    "PSYCHAD": set(obs_names_int[(src == "PSYCHAD-V3") & (age >= 0) & (age < 1)]),
    "WANG":    set(obs_names_int[(src == "WANG-multiome") & (age >= 0) & (age < 1)]),
    "VEL":     set(obs_names_int[(src == "VELMESHEV-V3") & (age >= 0) & (age < 1)]),
}
print(f"  PsychAD <1y: {len(barcodes['PSYCHAD']):,}")
print(f"  Wang <1y:    {len(barcodes['WANG']):,}")
print(f"  Vel-V3 <1y:  {len(barcodes['VEL']):,}")
adata_int.file.close()


# ─── 2. Load <1y cells from per-dataset h5ads ──────────────────────
def load_raw_subset(label, path, target_barcodes):
    print(f"\n  [{label}] loading {path}")
    a = ad.read_h5ad(path, backed="r")
    print(f"    full: {a.n_obs:,} × {a.n_vars:,}; .raw={'YES' if a.raw is not None else 'NO'}")
    mask = a.obs.index.isin(target_barcodes)
    idx = np.where(mask)[0]
    print(f"    matched: {len(idx):,}")
    if len(idx) == 0:
        a.file.close()
        return None
    sub = a[idx].to_memory()
    a.file.close()
    return extract_raw_anndata(sub, label)


print("\n--- 2. Loading per-dataset h5ads (RAW counts via .raw.X where needed) ---")
wang = load_raw_subset("WANG", DATASETS["WANG"], barcodes["WANG"])
vel  = load_raw_subset("VEL",  DATASETS["VEL"],  barcodes["VEL"])

psy_parts = []
for cohort in ["PSY_HBCC", "PSY_AGE"]:
    p = load_raw_subset(cohort, DATASETS[cohort], barcodes["PSYCHAD"])
    if p is not None and p.n_obs > 0:
        psy_parts.append(p)
if not psy_parts:
    print("ERROR: no PsychAD cells matched")
    sys.exit(1)

if len(psy_parts) > 1:
    psy = ad.concat(psy_parts, join="inner", merge="same")
else:
    psy = psy_parts[0]
print(f"\n  PsychAD <1y total: {psy.n_obs:,} × {psy.n_vars:,}")
print(f"  Wang <1y:           {wang.n_obs:,} × {wang.n_vars:,}")
print(f"  Vel-V3 <1y:         {vel.n_obs:,} × {vel.n_vars:,}")


# ─── 3. Per-cell QC on TRUE raw counts ─────────────────────────────
print("\n--- 3. Per-cell QC (full transcriptome, RAW counts) ---")
qc_rows = []
for name, a in [("PSYCHAD", psy), ("WANG", wang), ("VEL", vel)]:
    X = a.X
    if sp.issparse(X):
        total = np.asarray(X.sum(axis=1)).flatten()
        ngenes = np.asarray((X > 0).sum(axis=1)).flatten()
    else:
        total = X.sum(axis=1)
        ngenes = (X > 0).sum(axis=1)
    qc_rows.append({
        "group": name, "n_cells": int(a.n_obs), "n_vars": int(a.n_vars),
        "total_UMI_mean":   float(total.mean()),
        "total_UMI_median": float(np.median(total)),
        "total_UMI_P25":    float(np.percentile(total, 25)),
        "total_UMI_P75":    float(np.percentile(total, 75)),
        "n_genes_mean":     float(ngenes.mean()),
        "n_genes_median":   float(np.median(ngenes)),
    })
    print(f"  {name}: total UMI mean={total.mean():,.0f}  median={np.median(total):,.0f}  P25-P75=[{np.percentile(total,25):,.0f}, {np.percentile(total,75):,.0f}]")
    print(f"         n_genes  mean={ngenes.mean():,.0f}  median={np.median(ngenes):,.0f}")
pd.DataFrame(qc_rows).to_csv(OUT / "d10b_per_cell_qc.csv", index=False)


# ─── 4. Common gene set ────────────────────────────────────────────
print("\n--- 4. Common gene set ---")
common = sorted(set(psy.var_names) & set(wang.var_names) & set(vel.var_names))
print(f"  PsychAD vars: {psy.n_vars:,}")
print(f"  Wang vars:    {wang.n_vars:,}")
print(f"  Vel vars:     {vel.n_vars:,}")
print(f"  Common:       {len(common):,}")

en_in_common = [s for s, e in EN_ENSG.items() if e in common]
in_in_common = [s for s, e in IN_ENSG.items() if e in common]
print(f"  EN markers in common: {en_in_common}")
print(f"  IN markers in common: {in_in_common}")

print("\n  Subsetting to common genes...")
psy_c  = psy[:,  common].copy()
wang_c = wang[:, common].copy()
vel_c  = vel[:,  common].copy()


# ─── 5. Per-dataset HVG ────────────────────────────────────────────
print("\n--- 5. Per-dataset HVG (seurat_v3 on raw counts in common gene set) ---")
hvg_results = {}
for size in HVG_SIZES:
    print(f"\n  HVG size = {size}")
    hvg_sets = {}
    for name, a in [("PSYCHAD", psy_c), ("WANG", wang_c), ("VEL", vel_c)]:
        try:
            sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=size)
            hvg = set(a.var_names[a.var["highly_variable"].values])
        except Exception as e:
            print(f"    {name} HVG failed: {e}")
            hvg = set()
        hvg_sets[name] = hvg
        print(f"    {name}: HVG={len(hvg):,}")
    intersection = set.intersection(*hvg_sets.values()) if all(hvg_sets.values()) else set()
    print(f"    INTERSECTION: {len(intersection):,}")
    en_present = [s for s, e in EN_ENSG.items() if e in intersection]
    in_present = [s for s, e in IN_ENSG.items() if e in intersection]
    print(f"    EN markers in intersection: {en_present}")
    print(f"    IN markers in intersection: {in_present}")
    hvg_results[size] = {"intersection": intersection, "en_present": en_present, "in_present": in_present}


# ─── 6. Markers (CPM within intersection) ─────────────────────────
print("\n--- 6. Marker EN/IN per group ---")
results = []

print("\n  [Reference] FULL common-gene CPM:")
for name, a in [("PSYCHAD", psy_c), ("WANG", wang_c), ("VEL", vel_c)]:
    X = a.X
    lib = np.asarray(X.sum(axis=1)).flatten().astype(float)
    lib[lib == 0] = 1.0
    en_idx = [a.var_names.get_loc(EN_ENSG[s]) for s in en_in_common]
    in_idx = [a.var_names.get_loc(IN_ENSG[s]) for s in in_in_common]
    en_block = X[:, en_idx]; in_block = X[:, in_idx]
    if sp.issparse(en_block): en_block = en_block.toarray()
    if sp.issparse(in_block): in_block = in_block.toarray()
    en_norm = float((en_block / lib[:, None] * 1e4).mean())
    in_norm = float((in_block / lib[:, None] * 1e4).mean())
    ratio = en_norm / (in_norm + 1e-9)
    results.append({"hvg_size": "FULL_COMMON", "intersection_size": len(common),
                    "group": name, "n_cells": int(a.n_obs),
                    "EN_mean": en_norm, "IN_mean": in_norm, "EN_over_IN": ratio,
                    "lib_size_mean": float(lib.mean())})
    print(f"    {name}: n={a.n_obs:,}  lib_mean={lib.mean():,.0f}  EN={en_norm:.4f}  IN={in_norm:.4f}  EN/IN={ratio:.3f}")

for size in HVG_SIZES:
    intersection = hvg_results[size]["intersection"]
    if not intersection:
        continue
    en_present = hvg_results[size]["en_present"]
    in_present = hvg_results[size]["in_present"]
    inter_list = sorted(intersection)
    print(f"\n  HVG={size}, intersection={len(intersection)}, EN markers={len(en_present)}, IN markers={len(in_present)}")
    for name, a in [("PSYCHAD", psy_c), ("WANG", wang_c), ("VEL", vel_c)]:
        var_pos = {g: i for i, g in enumerate(a.var_names.values)}
        inter_idx = np.array([var_pos[g] for g in inter_list if g in var_pos])
        en_idx = [var_pos[EN_ENSG[s]] for s in en_present if EN_ENSG[s] in var_pos]
        in_idx = [var_pos[IN_ENSG[s]] for s in in_present if IN_ENSG[s] in var_pos]
        X = a.X
        inter_block = X[:, inter_idx]
        if sp.issparse(inter_block):
            lib = np.asarray(inter_block.sum(axis=1)).flatten().astype(float)
        else:
            lib = inter_block.sum(axis=1).astype(float)
        lib[lib == 0] = 1.0
        en_block = X[:, en_idx] if en_idx else None
        in_block = X[:, in_idx] if in_idx else None
        if en_block is not None and sp.issparse(en_block): en_block = en_block.toarray()
        if in_block is not None and sp.issparse(in_block): in_block = in_block.toarray()
        en_norm = float((en_block / lib[:, None] * 1e4).mean()) if en_block is not None else 0.0
        in_norm = float((in_block / lib[:, None] * 1e4).mean()) if in_block is not None else 0.0
        ratio = en_norm / (in_norm + 1e-9)
        results.append({"hvg_size": size, "intersection_size": len(intersection),
                        "group": name, "n_cells": int(a.n_obs),
                        "EN_mean": en_norm, "IN_mean": in_norm, "EN_over_IN": ratio,
                        "lib_size_mean": float(lib.mean())})
        print(f"    {name}: n={a.n_obs:,}  lib_mean={lib.mean():,.0f}  EN={en_norm:.4f}  IN={in_norm:.4f}  EN/IN={ratio:.3f}")


# ─── 7. Save & summary ─────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUT / "d10b_markers_by_hvg_size.csv", index=False)

print("\n--- Summary: EN/IN ratio by HVG size ---")
pivot_ratio = df.pivot_table(index="group", columns="hvg_size", values="EN_over_IN")
print(pivot_ratio.to_string())

print("\n--- Library size (within intersection) by HVG size ---")
pivot_lib = df.pivot_table(index="group", columns="hvg_size", values="lib_size_mean")
print(pivot_lib.to_string())

print("\n--- Key test: PsychAD EN/IN relative to Wang/Vel ---")
for size in ["FULL_COMMON"] + HVG_SIZES:
    sub = df[df["hvg_size"] == size].set_index("group")
    if "PSYCHAD" not in sub.index or "WANG" not in sub.index or "VEL" not in sub.index:
        continue
    p_r = sub.loc["PSYCHAD", "EN_over_IN"]
    w_r = sub.loc["WANG", "EN_over_IN"]
    v_r = sub.loc["VEL", "EN_over_IN"]
    print(f"  size={size:<12}  PsychAD={p_r:.3f}  Wang={w_r:.3f}  Vel={v_r:.3f}  PsychAD/Wang={p_r/(w_r+1e-9):.3f}  PsychAD/Vel={p_r/(v_r+1e-9):.3f}")

print(f"\nWritten:\n  {OUT / 'd10b_per_cell_qc.csv'}\n  {OUT / 'd10b_markers_by_hvg_size.csv'}")
print("\nDone.")
