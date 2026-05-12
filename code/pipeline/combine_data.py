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


def select_hvgs(output_path, n_top_genes: int, hvg_flavor: str = "pearson_residuals",
                hvg_batch_key: str = None, counts_layer: str = "counts"):
    """Load combined.h5ad, select HVGs, and overwrite with the gene-filtered version.

    HVG selection is performed on the full combined dataset (post-concatenation) with
    an optional batch_key to prevent batch-effect genes from dominating the selection.
    For pearson_residuals, batch_key is not supported by this scanpy version and is ignored.
    """
    print(f"\n[HVG] Loading {output_path} for HVG selection...")
    log_mem("Before HVG load")
    adata = sc.read_h5ad(output_path)
    n_genes_before = adata.n_vars
    print(f"[HVG] Loaded: {adata.n_obs} cells × {n_genes_before} genes")
    log_mem("After HVG load")

    print(f"[HVG] Selecting {n_top_genes} HVGs (flavor={hvg_flavor!r}, "
          f"batch_key={hvg_batch_key!r})")

    if hvg_flavor == "pearson_residuals":
        if hvg_batch_key:
            print("[HVG] Warning: pearson_residuals does not support batch_key "
                  f"in this scanpy version — ignoring batch_key={hvg_batch_key!r}")
        kwargs = {"n_top_genes": n_top_genes}
        if counts_layer in adata.layers:
            kwargs["layer"] = counts_layer
        sc.experimental.pp.highly_variable_genes(adata, **kwargs)
    else:
        kwargs = {"n_top_genes": n_top_genes, "flavor": hvg_flavor}
        if hvg_flavor == "seurat_v3":
            if counts_layer in adata.layers:
                kwargs["layer"] = counts_layer
        elif hvg_flavor == "seurat":
            print("[HVG] Log-normalizing .X for seurat HVG selection")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        if hvg_batch_key:
            kwargs["batch_key"] = hvg_batch_key
        sc.pp.highly_variable_genes(adata, **kwargs)

    n_hvg = int(adata.var["highly_variable"].sum())
    print(f"[HVG] {n_hvg} HVGs selected from {n_genes_before} genes "
          f"({n_genes_before - n_hvg} removed)")

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    # Restore raw counts to .X if seurat flavor modified it in-place
    if hvg_flavor == "seurat" and counts_layer in adata_hvg.layers:
        adata_hvg.X = adata_hvg.layers[counts_layer].copy()

    del adata
    gc.collect()
    log_mem("After HVG subset (before write)")

    print(f"[HVG] Writing HVG-filtered file ({adata_hvg.n_obs} cells × "
          f"{adata_hvg.n_vars} genes) → {output_path}")
    adata_hvg.write_h5ad(output_path)
    del adata_hvg
    gc.collect()
    log_mem("After HVG write")
    print(f"[HVG] Done. combined.h5ad reduced from {n_genes_before} → {n_hvg} genes.")


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
    parser.add_argument("--n_top_genes", type=int, default=0,
                        help="Number of HVGs to select after combining (0 = skip).")
    parser.add_argument("--hvg_flavor", type=str, default="pearson_residuals",
                        help="HVG selection method: pearson_residuals, seurat_v3, seurat.")
    parser.add_argument("--hvg_batch_key", type=str, default=None,
                        help="obs column to use as batch key for HVG selection "
                             "(supported by seurat_v3/seurat; ignored for pearson_residuals).")
    parser.add_argument("--counts_layer", type=str, default="counts",
                        help="Layer containing raw counts (default: counts).")
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

    if args.n_top_genes > 0:
        select_hvgs(
            output_path=args.output,
            n_top_genes=args.n_top_genes,
            hvg_flavor=args.hvg_flavor,
            hvg_batch_key=args.hvg_batch_key,
            counts_layer=args.counts_layer,
        )

    print("Done.")


if __name__ == "__main__":
    main()
