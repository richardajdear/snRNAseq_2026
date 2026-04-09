"""Standalone diagnostic script for the scANVI pipeline.

Replicates notebook cells 1-7 of ahbaC3_*_combined_scANVI.qmd but writes
all output directly to stdout/stderr, so it is visible in SLURM logs even
when the process is OOM-killed mid-run.

Submit with:
    sbatch scripts/run_script.sh scripts/debug_scanvi_pipeline.py

Or with a custom memory limit:
    sbatch --mem=300G scripts/run_script.sh scripts/debug_scanvi_pipeline.py
"""

import os
import sys
import gc

# ── unbuffered output so every print lands in the log before any OOM kill ─────
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ── memory helper ──────────────────────────────────────────────────────────────

def rss_gb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return float('nan')


def log_mem(label):
    flush_print(f"[MEM] {label}: {rss_gb():.1f} GB RSS")


# ── environment setup (mirrors notebook cells 1-2) ────────────────────────────

log_mem("startup")

def _find_repo_root(marker='.git'):
    current = os.path.abspath(os.getcwd())
    while True:
        if os.path.exists(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(f"Could not find '{marker}' above {os.getcwd()}")
        current = parent

_repo_root = _find_repo_root()
if os.path.join(_repo_root, 'code') not in sys.path:
    sys.path.insert(0, os.path.join(_repo_root, 'code'))

from environment import get_environment
_env = get_environment()
rds_dir  = _env['rds_dir']
code_dir = _env['code_dir']
ref_dir  = _env['ref_dir']

flush_print(f"Environment : {_env['name']}")
flush_print(f"  rds_dir  : {rds_dir}")
flush_print(f"  code_dir : {code_dir}")
flush_print(f"  ref_dir  : {ref_dir}")

# ── imports (mirrors notebook cell 3) ─────────────────────────────────────────

flush_print("\n[STEP] Importing libraries...")
log_mem("before imports")

import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import h5py

log_mem("after imports")

from hvg_investigation import (load_single_scvi, setup_grn,
                                run_projection_pipeline, load_cache,
                                _rss_gb, _log_mem)

# ── config ─────────────────────────────────────────────────────────────────────

DATA_FILE  = rds_dir + "/Cam_snRNAseq/integrated/VelWangPsychAD_100k_pearson/scvi_output/integrated.h5ad"
SCVI_LAYER = 'scanvi_normalized'
N_VALUES   = [1000, 2000, 4000, 6000, 8000, 10000]
CACHE_DIR  = os.path.join(_repo_root, 'notebooks', 'ahbaC3_hvg_investigation_combined_scANVI', '_cache')

# ── step 1: inspect h5ad structure without loading data ───────────────────────

flush_print(f"\n[STEP] Inspecting h5ad structure (backed mode, no data loaded)...")
log_mem("before backed inspect")

file_size_gb = os.path.getsize(DATA_FILE) / 1e9
flush_print(f"  File size on disk: {file_size_gb:.1f} GB")

# Use h5py directly to read metadata only — accessing backed AnnData arrays
# (e.g. via .nbytes or indexing) forces them into RAM and leaves a large
# residual even after gc.collect(), inflating all subsequent measurements.
with h5py.File(DATA_FILE, 'r') as _f:
    # Shape
    if 'X' in _f:
        _x = _f['X']
        if isinstance(_x, h5py.Dataset):
            _shape = _x.shape
        else:  # sparse stored as group
            _shape = (_f['obs']['_index'].shape[0], _f['var']['_index'].shape[0])
    else:
        _shape = (_f['obs']['_index'].shape[0], _f['var']['_index'].shape[0])
    flush_print(f"  Shape            : {_shape}")

    # Layers
    _layer_names = list(_f.get('layers', {}).keys())
    flush_print(f"  Layers           : {_layer_names}")

    # obsm / obs columns (read from h5py, no AnnData overhead)
    _obsm_keys = list(_f.get('obsm', {}).keys())
    flush_print(f"  obsm keys        : {_obsm_keys}")
    _obs_cols = list(_f.get('obs', {}).keys())
    flush_print(f"  obs columns      : {_obs_cols}")

    # Layer sizes from HDF5 metadata only — no data loaded
    flush_print("\n  Layer sizes (from HDF5 metadata, nothing loaded into RAM):")
    for lname in _layer_names:
        grp = _f['layers'][lname]
        if isinstance(grp, h5py.Dataset):
            gb = np.prod(grp.shape) * grp.dtype.itemsize / 1e9
            flush_print(f"    {lname}: {gb:.1f} GB (dense {grp.dtype}, shape {grp.shape})")
        else:
            # Sparse matrix stored as group with 'data', 'indices', 'indptr'
            data_ds = grp['data']
            nnz = data_ds.shape[0]
            gb = nnz * data_ds.dtype.itemsize / 1e9
            flush_print(f"    {lname}: ~{gb:.1f} GB sparse data ({data_ds.dtype}, nnz={nnz:,})")

    # X
    if 'X' in _f:
        _x = _f['X']
        if isinstance(_x, h5py.Dataset):
            gb = np.prod(_x.shape) * _x.dtype.itemsize / 1e9
            flush_print(f"    X (main): {gb:.1f} GB (dense {_x.dtype})")
        else:
            data_ds = _x['data']
            gb = data_ds.shape[0] * data_ds.dtype.itemsize / 1e9
            flush_print(f"    X (main): ~{gb:.1f} GB sparse (nnz={data_ds.shape[0]:,})")

log_mem("after h5py metadata inspect (no data loaded)")

# ── step 2: check cache ────────────────────────────────────────────────────────

flush_print(f"\n[STEP] Checking cache at {CACHE_DIR}...")
cached = load_cache(CACHE_DIR)

# ── step 3: run pipeline ───────────────────────────────────────────────────────

if cached is not None:
    scores_df, stats_df, final_df, hvg_df = cached
    flush_print(f"[INFO] Cache hit — skipping pipeline.")
    flush_print(f"  scores: {len(scores_df)}, stats: {len(stats_df)}, "
                f"final: {len(final_df)}, hvg: {len(hvg_df)}")
else:
    flush_print(f"\n[STEP] Loading data with load_single_scvi (backed mode)...")
    adata, adata_log = load_single_scvi(DATA_FILE, scvi_layer=SCVI_LAYER)

    flush_print(f"\n[STEP] Setting up GRN...")
    log_mem("before setup_grn")
    ahba_GRN, total_grn_genes = setup_grn(ref_dir, adata)
    log_mem("after setup_grn")

    flush_print(f"\n[STEP] Running projection pipeline...")
    scores_df, stats_df, final_df, hvg_df = run_projection_pipeline(
        adata, adata_log, ahba_GRN, total_grn_genes, N_VALUES, CACHE_DIR)

    del adata, adata_log
    gc.collect()
    log_mem("after pipeline + del adata")

flush_print(f"\n[DONE] Pipeline complete.")
flush_print(f"  scores: {len(scores_df):,}, stats: {len(stats_df):,}, "
            f"final: {len(final_df):,}, hvg: {len(hvg_df):,}")
log_mem("exit")
