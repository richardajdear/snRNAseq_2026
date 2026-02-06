import scanpy as sc
import sys
import os
import pandas as pd

# Path setup
sys.path.append("/home/rajd2/rds/hpc-work/snRNAseq_2026/code")
from metadata_utils import get_original_metadata

base_dir = "/home/rajd2/rds/"
input_file = os.path.join(base_dir, "rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_postnatal_full_harmony_10k.h5ad")

print(f"Loading {input_file}...")
adata = sc.read_h5ad(input_file)
obs = adata.obs.copy()

print(f"Total cells: {len(obs)}")
print("Dataset counts:")
print(obs['dataset'].value_counts())

# Test Extraction Individually
datasets = ['VELMESHEV', 'WANG', 'HBCC', 'AGING']
expected_cols = ['cell_class', 'cell_type', 'region', 'age_years', 'sex', 'donor_id']

for ds in datasets:
    print(f"\n--- Testing {ds} ---")
    try:
        meta = get_original_metadata(obs, base_dir, datasets_to_load=[ds])
        print(f"  Result shape: {meta.shape}")
        
        # Check columns
        missing_cols = [c for c in expected_cols if c not in meta.columns]
        if missing_cols:
            print(f"  MISSING COLUMNS: {missing_cols}")
        else:
            print("  All expected columns present.")
            
        # Check Nulls
        for c in expected_cols:
            if c in meta.columns:
                n_na = meta[c].isna().sum()
                print(f"  {c}: {n_na} missing")
                if c == 'cell_type' and len(meta) > 0:
                    print(f"    Values: {meta[c].dropna().unique()[:10]}")

    except Exception as e:
        print(f"  CRASHED: {e}")

print("\n--- Testing ALL ---")
meta = get_original_metadata(obs, base_dir)
print(f"Final shape: {meta.shape}")
print("Final Cell Type Counts:")
print(meta['cell_type'].value_counts())
