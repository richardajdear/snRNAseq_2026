import scanpy as sc
import numpy as np
import pandas as pd
import sys

# Paths
VELMESHEV = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev10k.h5ad"
WANG = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/wang/processed/Wang_10k.h5ad"
ROUSSOS = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/processed/Roussos_10k.h5ad"

def check_counts(adata, name):
    print(f"\n--- Checking {name} ---")
    if adata.raw is not None:
        print("  Has .raw attribute.")
    if 'counts' in adata.layers:
        print("  Has layers['counts'].")
        X = adata.layers['counts']
    else:
        print("  No layers['counts']. Using .X")
        X = adata.X
        
    try:
        is_int = np.all(np.equal(np.mod(X.data if hasattr(X, 'data') else X, 1), 0))
        print(f"  Is integer? {is_int}")
        if not is_int:
            print(f"  Min/Max: {X.min()}/{X.max()}")
            # Check if looks log-transformed
            if X.max() < 20:
                print("  Likely log-transformed.")
    except Exception as e:
        print(f"  Error checking integer: {e}")

def check_glia(adata, name):
    print(f"\n--- Checking Glia in {name} ---")
    # Identify what mapped to Glia or generic labels
    # Look for original cell type column
    cands = ['cell_type', 'Cell_Type', 'subclass', 'class', 'cluster']
    found = [c for c in cands if c in adata.obs.columns]
    
    for col in found:
        print(f"  Column: {col}")
        # Print counts of categories that might map to glia
        vals = adata.obs[col].value_counts()
        print(vals.head(10))
        
def check_unknown_age(adata, name):
    print(f"\n--- Checking Unknown Age in {name} ---")
    # Look for age col
    cands = ['age', 'Age', 'development_stage', 'Estimated_postconceptional_age_in_days']
    for c in cands:
        if c in adata.obs.columns:
            unknowns = adata.obs[adata.obs[c].astype(str).str.lower().str.contains('unknown', na=False)]
            if len(unknowns) > 0:
                print(f"  Found {len(unknowns)} unknowns in column {c}")
                print(unknowns[c].unique())

print("Loading Velmeshev...")
ad_v = sc.read_h5ad(VELMESHEV)
check_counts(ad_v, "Velmeshev")
check_glia(ad_v, "Velmeshev")
check_unknown_age(ad_v, "Velmeshev")

print("\nLoading Wang...")
ad_w = sc.read_h5ad(WANG)
check_counts(ad_w, "Wang")
check_glia(ad_w, "Wang")
check_unknown_age(ad_w, "Wang")

print("\nLoading Roussos...")
ad_r = sc.read_h5ad(ROUSSOS)
check_counts(ad_r, "Roussos")
check_glia(ad_r, "Roussos")
check_unknown_age(ad_r, "Roussos")
