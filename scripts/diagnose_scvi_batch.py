"""
Diagnostic script to investigate the scVI batch correction issue for ahbaC3 GRN.

Questions to answer:
1. What is adata.X in integrated.h5ad? Raw counts or already log-normalized?
2. What is the scvi_normalized layer? Is batch effect actually removed?
3. Why do Velmeshev/Wang appear separated from Aging/HBCC in the scVI notebook?
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

# ── Path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = "/home/rajd2/rds/hpc-work/snRNAseq_2026"
sys.path.insert(0, os.path.join(REPO_ROOT, 'code'))

from environment import get_environment
env = get_environment()
rds_dir = env['rds_dir']

INTEGRATED = (rds_dir + "/Cam_snRNAseq/combined/"
              "VelWangPsychad_100k_PFC_lessOld/scvi_output/integrated.h5ad")
ORIGINAL   = (rds_dir + "/Cam_snRNAseq/combined/"
              "VelWangPsychad_100k_PFC_lessOld_normal.h5ad")

# ── 1. Inspect integrated.h5ad ───────────────────────────────────────────────
print("=" * 70)
print("1. Loading integrated.h5ad")
print("=" * 70)
adata_int = ad.read_h5ad(INTEGRATED)
print(f"Shape: {adata_int.shape}")
print(f"Layers: {list(adata_int.layers.keys())}")

def sample_stats(mat, name, n=500):
    if sp.issparse(mat):
        arr = mat[:n].toarray()
    else:
        arr = np.array(mat[:n])
    print(f"\n  [{name}]")
    print(f"    min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    print(f"    Is integer? {np.allclose(arr, arr.astype(int))}")
    cell_sums = arr.sum(axis=1)
    print(f"    Per-cell sums: min={cell_sums.min():.1f}, max={cell_sums.max():.1f}, "
          f"mean={cell_sums.mean():.1f}")

sample_stats(adata_int.X, "adata.X")
if 'counts' in adata_int.layers:
    sample_stats(adata_int.layers['counts'], "layers['counts']")
if 'scvi_normalized' in adata_int.layers:
    sample_stats(adata_int.layers['scvi_normalized'], "layers['scvi_normalized']")

# ── 2. Simulate what the scVI notebook does ───────────────────────────────────
print("\n" + "=" * 70)
print("2. Simulating the scVI notebook's preprocessing (on 500 cells)")
print("=" * 70)
import scanpy as sc
adata_sim = adata_int[:500].copy()
print(f"  Before: X[0,0]={adata_sim.X[0,0] if not sp.issparse(adata_sim.X) else adata_sim.X[0,0]:.4f}")
adata_sim.layers['counts'] = adata_sim.X.copy()   # OVERWRITES any raw counts
sc.pp.normalize_total(adata_sim, target_sum=1e6)
x_after = adata_sim.X
if sp.issparse(x_after):
    x_after = x_after.toarray()
print(f"  After double-normalize: cell sums min={x_after.sum(1).min():.1f}, "
      f"max={x_after.sum(1).max():.1f}")
print(f"  (Should all be 1e6 if properly normalized — any deviation = artifact)")

# ── 3. GRN score distribution by source: scvi_normalized vs X ─────────────
print("\n" + "=" * 70)
print("3. GRN score by source — comparing X vs scvi_normalized vs counts")
print("=" * 70)

from regulons import get_ahba_GRN, project_GRN
from gene_mapping import map_grn_symbols_to_ensembl

grn_file = os.path.join(env['ref_dir'], "ahba_dme_hcp_top8kgenes_weights.csv")
ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file, use_weights=True)
ahba_GRN = map_grn_symbols_to_ensembl(ahba_GRN, adata_int)

# Use all excitatory cells for speed
mask_exc = adata_int.obs['cell_class'] == 'Excitatory'
adata_exc = adata_int[mask_exc].copy()
print(f"Excitatory cells: {adata_exc.n_obs} across sources:")
print(adata_exc.obs['source'].value_counts().to_string())

def score_by_source(adata_work, layer=None, label=""):
    """Score GRN and return per-source C3+ mean."""
    a = adata_work.copy()
    if layer is not None:
        a.X = a.layers[layer].copy()
    sc.pp.normalize_total(a, target_sum=1e6)
    project_GRN(a, ahba_GRN, 'X_ahba', use_highly_variable=False, log_transform=False)
    scores = pd.DataFrame(a.obsm['X_ahba'], index=a.obs_names,
                          columns=a.uns['X_ahba_names'])
    scores['source'] = a.obs['source'].values
    result = scores.groupby('source')['C3+'].mean()
    print(f"\n  [{label}] Mean C3+ score by source:")
    print(result.to_string())
    return result

# What the scVI notebook actually does (uses adata.X which is log-normalized)
print("\n--- What the scVI notebook actually computes (normalize_total on X) ---")
score_by_source(adata_exc, layer=None, label="notebook behavior: normalize_total(X)")

# What it SHOULD do if using raw counts
if 'counts' in adata_exc.layers:
    print("\n--- What it would look like using raw counts layer ---")
    score_by_source(adata_exc, layer='counts', label="raw counts (correct normalization)")

# What the scvi_normalized layer gives (no extra normalization)
if 'scvi_normalized' in adata_exc.layers:
    a_scvi = adata_exc.copy()
    a_scvi.X = a_scvi.layers['scvi_normalized'].copy()
    project_GRN(a_scvi, ahba_GRN, 'X_ahba', use_highly_variable=False, log_transform=False)
    scores_scvi = pd.DataFrame(a_scvi.obsm['X_ahba'], index=a_scvi.obs_names,
                               columns=a_scvi.uns['X_ahba_names'])
    scores_scvi['source'] = a_scvi.obs['source'].values
    print(f"\n  [scvi_normalized raw (no extra normalize)] Mean C3+ score by source:")
    print(scores_scvi.groupby('source')['C3+'].mean().to_string())

# ── 4. Age distribution by source ────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. Age distribution by source")
print("=" * 70)
print(adata_int.obs.groupby('source')['age_years'].describe().to_string())

# Check if Velmeshev/Wang over-represent younger donors
print("\n  Young donors (<18y) by source:")
young = adata_int.obs[adata_int.obs['age_years'] < 18]
print(young['source'].value_counts().to_string())
print("\n  Adult donors (>=18y) by source:")
adult = adata_int.obs[adata_int.obs['age_years'] >= 18]
print(adult['source'].value_counts().to_string())

# ── 5. Check what was in adata.X before the seurat HVG normalization ─────────
print("\n" + "=" * 70)
print("5. Is adata.X in the original VelWangPsychad_100k_PFC_lessOld.h5ad raw?")
print("=" * 70)
import os
if os.path.exists(ORIGINAL):
    adata_orig = ad.read_h5ad(ORIGINAL)
    sample_stats(adata_orig.X, "original .X")
    print(f"  Layers: {list(adata_orig.layers.keys())}")
else:
    print(f"  File not found: {ORIGINAL}")

# ── 6. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key questions:
(a) Is adata.X in integrated.h5ad raw counts or log-normalized?
    -> See section 1 above: 'Is integer?' and per-cell sums.
       Raw counts: integers, sums vary widely.
       Log-normalized (norm_total 1e4 + log1p): floats, sums ~constant.

(b) Does the scVI notebook correctly use batch-corrected data?
    -> The notebook copies X to counts, then calls normalize_total(X).
       If X is already log-normalized, this is a double-normalization bug.

(c) Is source separation from age confound or batch effect?
    -> Compare scores_by_source across methods above.
       If 'raw counts' and 'scvi_normalized' give similar source separation,
       it is a real age-composition confound (Vel/Wang are younger).
       If only the double-normalized X shows separation, it is an artifact.

(d) Does scvi_normalized actually remove batch effects?
    -> transform_batch=null was used (confirmed in config.yaml).
       This means scvi_normalized = batch-CONDITIONAL expected expression,
       NOT batch-corrected. Batch effects are preserved in this layer.
       True batch correction requires transform_batch set to a reference batch.
""")
