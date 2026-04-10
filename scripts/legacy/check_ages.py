
import scanpy as sc
import pandas as pd
import os

# Paths to 10k datasets
BASE_DIR = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
PATHS = {
    "Velmeshev 10k": f"{BASE_DIR}/Cam_snRNAseq/velmeshev/velmeshev_10k_PFC_lessOld.h5ad",
    "Wang 10k": f"{BASE_DIR}/Cam_snRNAseq/wang/wang_10k_PFC_lessOld.h5ad",
    "Aging 10k": f"{BASE_DIR}/Cam_PsychAD/RNAseq/Aging_Cohort_10k_PFC_lessOld.h5ad",
    "HBCC 10k": f"{BASE_DIR}/Cam_PsychAD/RNAseq/HBCC_Cohort_10k_PFC_lessOld.h5ad",
    "Combined": f"{BASE_DIR}/Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld.h5ad"
}

def check_age(name, path):
    print(f"\n--- {name} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    
    try:
        adata = sc.read_h5ad(path, backed='r')
        if 'age_years' in adata.obs.columns:
            ages = adata.obs['age_years']
            print(f"Age Range: {ages.min():.2f} - {ages.max():.2f}")
            print(f"Count >= 40: {sum(ages >= 40)}")
            print(f"Count < 40: {sum(ages < 40)}")
        else:
            print("No 'age_years' column found.")
    except Exception as e:
        print(f"Error reading {name}: {e}")

for name, path in PATHS.items():
    check_age(name, path)
