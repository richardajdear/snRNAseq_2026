import sys
import scanpy as sc
import os

sys.path.insert(0, "/home/rajd2/rds/hpc-work/snRNAseq_2026/code")
from hvg_investigation import load_single_scvi, setup_grn, build_conditions, run_hvg_conditions

print("Testing refactored code pipeline...")
DATA_FILE = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld/scvi_output/integrated.h5ad"
ref_dir = "/home/rajd2/rds/hpc-work/snRNAseq_2026/reference"

print("1. Loading single scvi...")
adata, adata_log = load_single_scvi(DATA_FILE, source_label='combined')

print("2. Setup GRN...")
ahba_GRN, total_grn_genes = setup_grn(ref_dir, adata)

print("3. Downsample for fast testing...")
adata = adata[:2000].copy()
adata_log = adata_log[:2000].copy()

print("4. Testing HVG conditions...")
conds = build_conditions([1000])
scores_df, stats_df, hvg_df = run_hvg_conditions(adata, adata_log, ahba_GRN, conds, total_grn_genes)

print("Success!")
