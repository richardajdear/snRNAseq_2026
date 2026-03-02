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

def check_hvgs():
    print("Loading ad_L23...")
    ad_l23 = sc.read_h5ad('/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/ad_L23.h5ad')
    meta = pd.read_csv('/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_L23_thesis.csv', index_col=0)[['Region_Broad', 'Chemistry']]
    
    ad_l23.X = ad_l23.raw.X.copy()
    ad_l23.obs = ad_l23.obs.join(meta)
    
    ad_l23 = ad_l23[ad_l23.obs['Region_Broad']=='FC'].copy()
    
    # Pathway 1: Thesis
    print("Running Thesis Pathway...")
    ad_thesis = ad_l23.copy()
    sc.pp.normalize_total(ad_thesis, target_sum=1e6)
    # flavor=seurat_v3 on normalized data!
    sc.pp.highly_variable_genes(ad_thesis, n_top_genes=10000, batch_key='Chemistry', flavor='seurat_v3')
    
    # Pathway 2: Pipeline
    print("Running Pipeline Pathway...")
    ad_pipe = ad_l23.copy()
    ad_pipe.layers['counts'] = ad_pipe.X.copy()
    sc.pp.normalize_total(ad_pipe, target_sum=1e6)
    sc.pp.highly_variable_genes(ad_pipe, layer='counts', n_top_genes=10000, batch_key='Chemistry', flavor='seurat_v3')
    
    thesis_genes = set(ad_thesis.var_names[ad_thesis.var['highly_variable']])
    pipe_genes = set(ad_pipe.var_names[ad_pipe.var['highly_variable']])
    
    print("\nThesis top genes:", len(thesis_genes))
    print("Pipeline top genes:", len(pipe_genes))
    print("Overlap:", len(thesis_genes.intersection(pipe_genes)))
    print("Jaccard Index:", len(thesis_genes.intersection(pipe_genes)) / len(thesis_genes.union(pipe_genes)))

if __name__ == '__main__':
    check_hvgs()
