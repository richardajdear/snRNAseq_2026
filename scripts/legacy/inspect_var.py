
import scanpy as sc
import pandas as pd
import os

# Paths to 10k datasets
BASE_DIR = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
PATHS = {
    "Velmeshev": f"{BASE_DIR}/Cam_snRNAseq/velmeshev/velmeshev_10k_PFC_lessOld.h5ad",
    "Wang": f"{BASE_DIR}/Cam_snRNAseq/wang/wang_10k_PFC_lessOld.h5ad",
    "Aging": f"{BASE_DIR}/Cam_PsychAD/RNAseq/Aging_Cohort_10k_PFC_lessOld.h5ad",
    "HBCC": f"{BASE_DIR}/Cam_PsychAD/RNAseq/HBCC_Cohort_10k_PFC_lessOld.h5ad",
    "Combined": f"{BASE_DIR}/Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld.h5ad"
}

def inspect_var(name, path):
    print(f"\n--- {name} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    
    try:
        adata = sc.read_h5ad(path, backed='r')
        print(f"Shape: {adata.shape}")
        print(f"Var Columns: {list(adata.var.columns)}")
        print("Var Head:")
        print(adata.var.head(3))
        return adata.var.head(5)
    except Exception as e:
        print(f"Error reading {name}: {e}")
        return None

results = {}
for name, path in PATHS.items():
    results[name] = inspect_var(name, path)

print("\n--- Summary of Gene Symbol Columns ---")
for name, res in results.items():
    if res is not None:
        cols = list(res.columns)
        # Guesses for gene symbol
        candidates = [c for c in cols if 'gene' in c.lower() or 'symbol' in c.lower() or 'feature' in c.lower() or 'name' in c.lower()]
        print(f"{name}: {candidates}")
