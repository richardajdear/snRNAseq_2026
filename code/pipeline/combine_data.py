"""
Combine pre-processed per-dataset h5ad files into a single concatenated file.

Each input h5ad must already have these obs columns (from downsample.py / read_data.py):
    source, cell_class, cell_type_raw, age_years, sex, individual, region, chemistry

Concatenation uses inner join (genes present in all datasets).
Gene metadata (gene_symbol, feature_length) is restored from the first input file.

Usage:
    python combine_data.py wang.h5ad velmeshev.h5ad aging.h5ad hbcc.h5ad \\
        --output combined.h5ad

    # Diagnose only (no save):
    python combine_data.py *.h5ad --diagnose_only
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
import argparse
import gc
import psutil
import time

START_TIME = time.time()


def log_mem(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    elapsed = time.time() - START_TIME
    print(f"\n[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB "
          f"({process.memory_percent():.2f}%) - Elapsed: {elapsed/60:.2f} min")
    sys.stdout.flush()


def check_structure(input_files):
    """Print diagnostic crosstabs from backed files (no full matrix load)."""
    print("\n" + "="*50)
    print("           DIAGNOSTIC STRUCTURE CHECK")
    print("="*50)

    combined_meta = []
    col_sets = []

    for path in input_files:
        ad = sc.read_h5ad(path, backed='r')
        obs = ad.obs
        col_sets.append(set(obs.columns))

        source = obs['source'].iloc[0] if 'source' in obs.columns else os.path.basename(path)
        df = obs[['source', 'region', 'cell_class', 'cell_type_raw']].copy() if all(
            c in obs.columns for c in ['source', 'region', 'cell_class', 'cell_type_raw']
        ) else obs.copy()
        combined_meta.append(df)

        print(f"  {source}: {ad.shape[0]} cells x {ad.shape[1]} genes")
        print(f"    obs columns: {sorted(obs.columns.tolist())}")
        del ad

    common_cols = sorted(set.intersection(*col_sets)) if col_sets else []
    print(f"\n[Common obs columns ({len(common_cols)})]: {', '.join(common_cols)}")

    full_df = pd.concat(combined_meta)

    if 'region' in full_df.columns and 'source' in full_df.columns:
        print("\n[Crosstab: Dataset x Region]")
        print(pd.crosstab(full_df['region'], full_df['source']))

    if 'cell_class' in full_df.columns and 'source' in full_df.columns:
        print("\n[Crosstab: Dataset x Cell Class]")
        print(pd.crosstab(full_df['cell_class'], full_df['source']))

    print("\n" + "="*50 + "\n")
    del full_df, combined_meta
    gc.collect()
    return common_cols


def combine_on_disk(input_files, output_path):
    """
    Use anndata.experimental.concat_on_disk to combine datasets without loading into memory.
    The 'source' column is already set in each file's obs; no label injection needed.
    """
    from anndata.experimental import concat_on_disk

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Combining {len(input_files)} files on disk → {output_path}")
    log_mem("Before concat_on_disk")

    concat_on_disk(input_files, output_path, join='inner')

    log_mem("After concat_on_disk")

    # Restore gene metadata from first input file
    _restore_gene_metadata(input_files[0], output_path)

    log_mem("After metadata restore")
    print(f"Combined file saved to: {output_path}")


def _restore_gene_metadata(ref_path, output_path):
    """Copy gene_symbol and feature_length from ref h5ad into output h5ad."""
    import h5py
    try:
        from anndata.io import write_elem
    except ImportError:
        from anndata.experimental import write_elem

    print("Restoring gene metadata from reference file...")
    ref = sc.read_h5ad(ref_path, backed='r')
    out = sc.read_h5ad(output_path, backed='r+')

    common_vars = out.var_names
    new_var = out.var.copy()

    if 'feature_name' in ref.var.columns:
        new_var['gene_symbol'] = ref.var.loc[common_vars, 'feature_name'].values
        print(f"  Restored 'gene_symbol' for {len(new_var)} genes.")
    elif 'gene_name' in ref.var.columns:
        new_var['gene_symbol'] = ref.var.loc[common_vars, 'gene_name'].values
        print(f"  Restored 'gene_symbol' for {len(new_var)} genes.")

    if 'feature_length' in ref.var.columns:
        new_var['feature_length'] = ref.var.loc[common_vars, 'feature_length'].values
        print(f"  Restored 'feature_length' for {len(new_var)} genes.")

    out.file.close()

    with h5py.File(output_path, 'r+') as f:
        if 'var' in f:
            del f['var']
        write_elem(f, 'var', new_var)

    del ref, out
    gc.collect()
    print("  Gene metadata restored.")


def main():
    parser = argparse.ArgumentParser(
        description="Combine pre-processed h5ad files into one concatenated file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+",
                        help="Pre-processed h5ad files to combine (from downsample.py).")
    parser.add_argument("--output", required=True,
                        help="Path for the combined output h5ad.")
    parser.add_argument("--diagnose_only", action='store_true',
                        help="Run diagnostics only, do not save.")
    args = parser.parse_args()

    # Validate inputs
    missing = [p for p in args.inputs if not os.path.exists(p)]
    if missing:
        print(f"Error: Input files not found: {missing}")
        sys.exit(1)

    log_mem("Start")

    check_structure(args.inputs)
    log_mem("After diagnostics")

    if args.diagnose_only:
        print("Diagnostics complete. Exiting.")
        return

    combine_on_disk(args.inputs, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
