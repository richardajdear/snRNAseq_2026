"""
Process and Project AHBA C3 gene regulatory network onto snRNA-seq data.

Usage as library (via reticulate):
    source_python("../../code/process_and_project.py")
    process_and_project(input_file, output_csv, grn_file, region, log_transform=False)

Usage as CLI:
    python process_and_project.py --input data.h5ad --output results.csv --grn weights.csv --region "prefrontal cortex" --no-log
"""

import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add code directory to path
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(code_dir)

try:
    from regulons import get_ahba_GRN, project_GRN
    from process_data import process_adata
except ImportError as e:
    logging.error(f"Error importing modules from {code_dir}: {e}")


def process_and_project(input_file, output_csv, grn_file, region='prefrontal cortex', log_transform=True):
    """
    Loads raw data, processes it (filtering, Normalization, HVG, PCA),
    projects GRN, and saves results CSV.

    Parameters:
        input_file: path to raw .h5ad file
        output_csv: path to save projection results
        grn_file: path to AHBA GRN weights CSV
        region: region to filter for (default 'prefrontal cortex', set to None/'all' to skip)
        log_transform: bool, whether to apply log1p transformation (default True)
    """
    logging.info(f"Starting process_and_project for {input_file}...")
    logging.info(f"  region={region}, log_transform={log_transform}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # 1. Load Data
    adata = sc.read_h5ad(input_file)
    logging.info(f"Original shape: {adata.shape}")

    # 2. Filter by Region
    if region and region.lower() != 'all':
        if 'region' in adata.obs.columns:
            logging.info(f"Filtering for region == '{region}'...")
            adata = adata[adata.obs['region'] == region].copy()
            logging.info(f"Filtered shape after region: {adata.shape}")
        else:
            logging.warning("'region' column not found, skipping filter.")

    # 3. Map gene symbols from .var column if needed
    if 'gene_symbol' in adata.var.columns:
        logging.info("Mapping var_names to gene_symbol from .var column...")
        adata.var_names = adata.var['gene_symbol'].values
        adata.var_names_make_unique()
        logging.info(f"Gene symbol mapping complete. First 5 var_names: {list(adata.var_names[:5])}")
    elif 'gene_name' in adata.var.columns:
        logging.info("Mapping var_names to gene_name from .var column...")
        adata.var_names = adata.var['gene_name'].values
        adata.var_names_make_unique()
        logging.info(f"Gene name mapping complete. First 5 var_names: {list(adata.var_names[:5])}")
    else:
        logging.warning("No gene_symbol or gene_name column found in .var — using existing var_names.")

    # 4. Standard Processing (Norm, optional Log1p, HVG, PCA) via library
    logging.info("Calling process_data.py::process_adata...")
    process_adata(adata, n_top_genes=10000, n_pcs=50, log_transform=log_transform)

    # 5. Load GRN
    logging.info(f"Loading GRN from {grn_file}...")
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file)

    # 6. Project
    logging.info("Projecting GRN...")
    project_GRN(adata, ahba_GRN, 'X_ahba', use_highly_variable=True, log_transform=False)

    # 7. Export
    logging.info("Exporting results...")

    if 'X_ahba' not in adata.obsm:
        raise ValueError("Projection failed, X_ahba not found in obsm.")

    cols = adata.uns.get('X_ahba_names', [f"Network_{i}" for i in range(adata.obsm['X_ahba'].shape[1])])
    projection_df = pd.DataFrame(adata.obsm['X_ahba'], index=adata.obs_names, columns=cols)

    # Metadata
    cols_to_keep = ['age_years', 'sex', 'donor_id', 'dataset_source', 'cell_class', 'cell_type', 'region', 'source', 'age_category', 'age_log2']
    valid_cols = [c for c in cols_to_keep if c in adata.obs.columns]

    result_df = pd.concat([adata.obs[valid_cols], projection_df], axis=1)

    # Ensure directory exists
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    result_df.to_csv(output_csv)
    logging.info(f"Results saved to {output_csv}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process snRNA-seq data and project AHBA C3 gene regulatory network."
    )
    parser.add_argument("--input", required=True, help="Path to input .h5ad file")
    parser.add_argument("--output", required=True, help="Path to output projection results .csv")
    parser.add_argument("--grn", required=True, help="Path to AHBA GRN weights CSV")
    parser.add_argument("--region", default="prefrontal cortex", help="Region to filter for (default: 'prefrontal cortex', use 'all' to skip)")
    parser.add_argument("--no-log", action="store_true", help="Skip log1p transformation")

    args = parser.parse_args()

    process_and_project(
        input_file=args.input,
        output_csv=args.output,
        grn_file=args.grn,
        region=args.region,
        log_transform=not args.no_log
    )


if __name__ == "__main__" and len(sys.argv) > 1:
    main()
