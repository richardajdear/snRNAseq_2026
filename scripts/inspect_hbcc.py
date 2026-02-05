import scanpy as sc
import pandas as pd

try:
    adata = sc.read_h5ad("/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort_10k.h5ad")
    print("Columns in obs:")
    print(adata.obs.columns.tolist())
    
    cols_to_check = ['class', 'subclass', 'cell_type', 'Cell_Type', 'Class', 'Subclass']
    for c in cols_to_check:
        if c in adata.obs.columns:
            print(f"\n--- Unique values in '{c}' ---")
            print(adata.obs[c].unique().tolist()[:20]) # First 20
except Exception as e:
    print(e)
