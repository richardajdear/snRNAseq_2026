import scanpy as sc
import pandas as pd
import numpy as np
import sys
import logging
import os

sys.path.append('/home/rajd2/rds/hpc-work/snRNAseq_2026/code')
import read_data
import process_data
from regulons import get_ahba_GRN, project_GRN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_L23(filter_fc=False, batch_key=None, thesis_bug=False, all_genes=False):
    h5ad_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/ad_L23.h5ad'
    grn_file = '/home/rajd2/rds/hpc-work/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv'
    
    if filter_fc and batch_key is not None and all_genes:
        output_csv = f'/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_L23_FC_batch_{batch_key}_allgenes.csv'
    elif filter_fc and batch_key is not None and thesis_bug:
        output_csv = f'/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_L23_FC_batch_{batch_key}_bug.csv'
    elif filter_fc and batch_key is not None:
        output_csv = f'/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_L23_FC_batch_{batch_key}.csv'
    elif filter_fc and all_genes:
        output_csv = '/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_L23_FC_allgenes.csv'
    elif filter_fc:
        output_csv = '/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_L23_FC.csv'
    else:
        output_csv = '/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_L23.csv'
    
    logging.info("Loading L23 and extracting true counts & metadata...")
    # read_velmeshev handles loading .raw counts and applying the TSV metadata
    adata = read_data.read_velmeshev(h5ad_path=h5ad_path)
    
    if filter_fc:
        logging.info("Filtering for Region_Broad == 'FC'")
        if 'Region_Broad' in adata.obs.columns:
            adata = adata[adata.obs['Region_Broad'] == 'FC'].copy()
        else:
            logging.warning("Region_Broad not found in metadata, cannot filter!")
            
    logging.info(f"Loaded {adata.shape[0]} cells and {adata.shape[1]} genes.")
    
    # Map feature_name to var_names if available
    if 'feature_name' in adata.var.columns:
        logging.info("Mapping var_names to feature_name...")
        adata.var_names = adata.var['feature_name'].values
        adata.var_names_make_unique()
        
    logging.info(f"Processing data (Norm, HVG, PCA) with batch_key={batch_key}, thesis_bug={thesis_bug}...")
    adata = process_data.process_adata(adata, n_top_genes=10000, n_pcs=50, log_transform=False, batch_key=batch_key, thesis_bug=thesis_bug)
    
    logging.info("Loading GRN...")
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file)
    logging.info("Projecting...")
    project_GRN(adata, ahba_GRN, 'X_ahba', use_highly_variable=not all_genes, log_transform=False)
    
    logging.info("Exporting CSV...")
    cols = adata.uns.get('X_ahba_names', [f"Network_{i}" for i in range(adata.obsm['X_ahba'].shape[1])])
    projection_df = pd.DataFrame(adata.obsm['X_ahba'], index=adata.obs_names, columns=cols)
    cols_to_keep = ['age_years', 'sex', 'donor_id', 'individual', 'dataset_source', 'dataset', 'cell_class', 'cell_type', 'lineage', 'batch', 'chemistry', 'region', 'source', 'age_category', 'age_log2']
    valid_cols = [c for c in cols_to_keep if c in adata.obs.columns]
    result_df = pd.concat([adata.obs[valid_cols], projection_df], axis=1)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    result_df.to_csv(output_csv)
    logging.info(f"Done! Saved to {output_csv}")

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter-fc', action='store_true', help='Filter for Region_Broad == FC')
    parser.add_argument('--batch-key', type=str, default=None, help='Batch key for HVG selection')
    parser.add_argument('--thesis-bug', action='store_true', help='Replicate thesis HVG bug')
    parser.add_argument('--all-genes', action='store_true', help='Project across all genes, ignoring HVG selection')
    args = parser.parse_args()
    process_L23(filter_fc=args.filter_fc, batch_key=args.batch_key, thesis_bug=args.thesis_bug, all_genes=args.all_genes)
