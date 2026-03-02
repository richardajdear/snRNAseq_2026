import scanpy as sc
import pandas as pd
import numpy as np
import sys
import os
import argparse
import logging

sys.path.append('/home/rajd2/rds/hpc-work/snRNAseq_2026/code')
import process_data
from regulons import get_ahba_GRN, project_GRN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def project_raw_4(input_raw_h5ad, input_simple_h5ad, output_csv, grn_file):
    logging.info(f"Loading raw structure from {input_raw_h5ad}...")
    adata = sc.read_h5ad(input_raw_h5ad)
    
    logging.info(f"Loading true raw counts from {input_simple_h5ad}...")
    adata_simple = sc.read_h5ad(input_simple_h5ad, backed='r')
    
    # 1. Swap X with true unnormalized counts
    logging.info("Intersecting cells and swapping X matrix...")
    common_obs = adata.obs_names.intersection(adata_simple.obs_names)
    logging.info(f"Retaining {len(common_obs)} / {adata.shape[0]} cells with true raw counts available.")
    
    # Subset in-memory adata
    adata = adata[common_obs].copy()
    
    # Extract raw counts and replace .X
    adata.X = adata_simple[common_obs].X.toarray() if hasattr(adata_simple.X, 'toarray') else adata_simple[common_obs].X.copy()
    
    # Grab metadata
    logging.info("Extracting pure age and cell type metadata from simple...")
    for col in ['age_years', 'cell_type', 'cell_class', 'region', 'individual', 'dataset', 'chemistry']:
        if col in adata_simple.obs.columns:
            adata.obs[col] = adata_simple.obs.loc[common_obs, col].values

    # 2. Gene Map
    if 'feature_name' in adata.var.columns:
        logging.info("Mapping var_names to feature_name...")
        adata.var_names = adata.var['feature_name'].values
        adata.var_names_make_unique()
        
    # 3. Process
    logging.info("Processing data (Norm, HVG, PCA)...")
    adata = process_data.process_adata(adata, n_top_genes=10000, n_pcs=50, log_transform=False)
    
    # 4. Project GRN
    logging.info("Loading GRN...")
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file)
    logging.info("Projecting...")
    project_GRN(adata, ahba_GRN, 'X_ahba', use_highly_variable=True, log_transform=False)
    
    # 5. Export
    logging.info("Exporting CSV...")
    cols = adata.uns.get('X_ahba_names', [f"Network_{i}" for i in range(adata.obsm['X_ahba'].shape[1])])
    projection_df = pd.DataFrame(adata.obsm['X_ahba'], index=adata.obs_names, columns=cols)
    cols_to_keep = ['age_years', 'sex', 'donor_id', 'individual', 'dataset_source', 'dataset', 'cell_class', 'cell_type', 'lineage', 'batch', 'chemistry', 'region', 'source', 'age_category', 'age_log2']
    valid_cols = [c for c in cols_to_keep if c in adata.obs.columns]
    result_df = pd.concat([adata.obs[valid_cols], projection_df], axis=1)
    result_df.to_csv(output_csv)
    logging.info(f"Done! Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--simple", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grn", required=True)
    args = parser.parse_args()
    project_raw_4(args.raw, args.simple, args.output, args.grn)
