"""
Smoke test for the pseudobulk_gap_analysis template pipeline.

Runs the full Python side of the pipeline (data loading → HVG selection →
GRN projection → prepare_for_r) on the real pseudobulk data without Quarto
or R.  Validates that final_df has the columns and value ranges expected by
the R sensitivity functions, then saves the cache so a subsequent full
notebook render can skip the projection step.

Usage (from repo root):
    PYTHONPATH=code python scripts/test_pseudobulk_gap.py [--cache-dir PATH]
"""

import argparse
import gc
import os
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings('ignore')

# ── Path setup ────────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent
if str(_REPO_ROOT / 'code') not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / 'code'))

from environment import get_environment
from hvg_investigation import (
    setup_grn, save_cache, load_cache,
    run_hvg_conditions, prepare_for_r,
)

# ── Config ────────────────────────────────────────────────────────────────────

_env = get_environment()
RDS_DIR  = _env['rds_dir']
REF_DIR  = _env['ref_dir']

PB_FILE = os.path.join(
    RDS_DIR,
    'Cam_snRNAseq/integrated/'
    'VelWangPsychAD_100k_dataset/pseudobulk_output/by_cell_class.h5ad')

CELL_CLASS  = 'Excitatory'
N_TOP_GENES = 8000

_CONDITIONS = [
    {'label': 'all_genes',            'flavor': None,                 'n_top_genes': None},
    {'label': f'pearson_{N_TOP_GENES}', 'flavor': 'pearson_residuals', 'n_top_genes': N_TOP_GENES},
]

# Required columns for the R gap-model functions
_REQUIRED_COLS = {'individual', 'age_years', 'source', 'C', 'value', 'condition'}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  OK  {msg}")


def _section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for pseudobulk_gap_analysis pipeline")
    parser.add_argument(
        '--cache-dir',
        default=str(_REPO_ROOT / 'notebooks/results/pseudobulk_gap_excitatory/_cache'),
        help='Where to read/write the projection cache.')
    parser.add_argument(
        '--force', action='store_true',
        help='Ignore existing cache and re-run projection.')
    args = parser.parse_args()

    cache_dir = args.cache_dir

    # ── 1. Cache check ─────────────────────────────────────────────────────────
    _section("1. Cache check")
    cached = None if args.force else load_cache(cache_dir)

    if cached is not None:
        scores_df, stats_df, final_df, hvg_df = cached
        print(f"  Loaded from cache: {cache_dir}")
    else:
        # ── 2. Load pseudobulk data ────────────────────────────────────────────
        _section("2. Load pseudobulk data")
        _check(os.path.exists(PB_FILE), f"Pseudobulk file exists: {PB_FILE}")

        adata_full = sc.read_h5ad(PB_FILE, backed='r')
        print(f"  Full shape    : {adata_full.shape}")
        print(f"  Cell classes  : {dict(adata_full.obs['cell_class'].value_counts())}")
        print(f"  Layers        : {list(adata_full.layers.keys())}")

        mask = adata_full.obs['cell_class'] == CELL_CLASS
        adata = adata_full[mask].to_memory()
        adata_full.file.close()
        del adata_full
        gc.collect()

        print(f"\n  Subset '{CELL_CLASS}' : {adata.shape}")
        print(f"  Age range     : {adata.obs.age_years.min():.2f} – "
              f"{adata.obs.age_years.max():.2f} years")
        print(f"  Sources       : {dict(adata.obs.source.value_counts())}")

        _check(adata.n_obs > 0,   "At least one donor after cell-class filter")
        _check('counts' in adata.layers, "layers['counts'] present")
        _check('gene_symbol' in adata.var.columns, "var has gene_symbol column")
        _check(adata.obs.age_years.min() < 0,  "Prenatal donors present (age < 0)")
        _check(adata.obs.age_years.max() > 60, "Adult donors present (age > 60)")

        # ── 3. Normalise ───────────────────────────────────────────────────────
        _section("3. Normalise")
        adata.X = adata.layers['counts'].copy()
        sc.pp.normalize_total(adata, target_sum=1e6)
        adata_log = ad.AnnData(X=adata.X.copy(), obs=adata.obs, var=adata.var.copy())
        sc.pp.log1p(adata_log)
        print("  CPM normalisation done.")

        # ── 4. GRN setup ───────────────────────────────────────────────────────
        _section("4. GRN setup")
        grn_file = os.path.join(REF_DIR, 'ahba_dme_hcp_top8kgenes_weights.csv')
        _check(os.path.exists(grn_file), f"GRN file exists: {grn_file}")
        ahba_GRN, total_grn_genes = setup_grn(REF_DIR, adata)
        _check(total_grn_genes > 0, f"GRN genes in adata: {total_grn_genes}")

        # ── 5. HVG projection ──────────────────────────────────────────────────
        _section("5. HVG projection (pearson_8000 + all_genes)")
        scores_df, stats_df, hvg_df = run_hvg_conditions(
            adata, adata_log, ahba_GRN, _CONDITIONS, total_grn_genes)

        final_df = prepare_for_r(scores_df, adata, n_values=[N_TOP_GENES])

        save_cache(cache_dir, scores_df, stats_df, final_df, hvg_df)

        del adata_log
        gc.collect()

    # ── 6. Validate outputs ────────────────────────────────────────────────────
    _section("6. Validate outputs")

    print(f"  scores_df  : {scores_df.shape}")
    print(f"  stats_df   :\n{stats_df.to_string(index=False)}")
    print(f"  final_df   : {final_df.shape}")
    print(f"  hvg_df     : {hvg_df.shape}")

    # final_df columns for R
    missing = _REQUIRED_COLS - set(final_df.columns)
    _check(not missing, f"final_df has required R columns (missing: {missing})")

    # Only expected conditions
    expected_conds = {'all_genes', f'pearson_{N_TOP_GENES}'}
    actual_conds   = set(final_df['condition'].unique())
    _check(actual_conds == expected_conds,
           f"Conditions match: {actual_conds}")

    # C column values
    c_vals = set(final_df['C'].unique())
    _check({'C3+', 'C3-'} <= c_vals,
           f"C column contains C3+/C3- (found: {c_vals})")

    # Age range spans childhood and adolescence
    child_mask = (final_df.age_years >= 1) & (final_df.age_years < 10)
    adol_mask  = (final_df.age_years >= 10) & (final_df.age_years < 25)
    _check(child_mask.any(), "final_df has childhood donors (age 1–10)")
    _check(adol_mask.any(),  "final_df has adolescence donors (age 10–25)")

    # Scores are finite
    _check(np.isfinite(final_df['value']).all(),
           "All C3+/C3- scores are finite")

    # HVG gene list only for the pearson condition
    if len(hvg_df) > 0:
        hvg_conds = set(hvg_df['condition'].unique())
        _check(hvg_conds == {f'pearson_{N_TOP_GENES}'},
               f"hvg_df has only pearson condition: {hvg_conds}")
        _check(len(hvg_df) <= N_TOP_GENES,
               f"hvg_df row count ≤ {N_TOP_GENES} ({len(hvg_df)})")

    # Source breakdown in final_df
    print(f"\n  Source breakdown in final_df (C3+, all_genes):")
    subset = final_df[(final_df.condition == 'all_genes') & (final_df.C == 'C3+')]
    print(subset.groupby('source')['individual'].nunique()
              .rename('n_donors').to_string())

    _section("ALL CHECKS PASSED")
    print(f"\n  Cache saved to: {cache_dir}")
    print(f"  Run notebook with:\n"
          f"    bash notebooks/render_single.sh pseudobulk_gap_excitatory\n")


if __name__ == '__main__':
    main()
