import sys
import os
import scanpy as sc
import psutil
import gc
import numpy as np
import pandas as pd
import argparse

# Add current dir to path to allow import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from read_and_combine import read_psychad, get_raw_counts

def print_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB")

def read_psychad_inplace(h5ad_path, source_name, dataset_name):
    print(f"Reading {dataset_name} from {h5ad_path} (In-Place Mode)...")
    print_memory_usage("Before Read")
    adata = sc.read_h5ad(h5ad_path)
    print_memory_usage("After Read")
    
    # Ensure raw counts
    adata.X = get_raw_counts(adata)
    print_memory_usage("After Raw Counts")
    
    # Age - Vectorized Extraction
    if 'development_stage' in adata.obs.columns:
        # Avoid apply if possible, but regex might need apply. 
        # But development_stage is categorical probably.
        def extract_age_psychad(val):
            import re
            if pd.isna(val): return np.nan
            # e.g. "82-year-old human stage"
            match = re.search(r'(\d+)-year-old', str(val))
            if match:
                return float(match.group(1))
            return np.nan

        adata.obs['age_years'] = adata.obs['development_stage'].apply(extract_age_psychad)
    print_memory_usage("After Age")

    # Filter Unknown Ages IN-PLACE
    if 'age_years' in adata.obs.columns:
        pre_n = adata.n_obs
        mask = ~adata.obs['age_years'].isna()
        if not mask.all():
            print(f"Filtering {pre_n - mask.sum()} cells with unknown age...")
            # Use inplace subsetting
            adata._inplace_subset_obs(mask)
            print(f"Filtered to {adata.n_obs} cells.")
            gc.collect()
            print_memory_usage("After Age Filter (Inplace)")

    # Lineage Mapping (Vectorized)
    # ... (Same logic as existing, but simplified for test)
    # ...
    
    return adata

def filter_age_thresholds_inplace(adata, dataset_name, min_age=None, max_age=40.0):
    print(f"[{dataset_name}] Filtering Age < {max_age} (In-Place)...")
    if 'age_years' in adata.obs.columns:
        mask = adata.obs['age_years'] < max_age
        if min_age is not None:
            mask &= (adata.obs['age_years'] >= min_age)
            
        print(f"[{dataset_name}] Age Filter: {adata.n_obs} -> {mask.sum()} cells")
        
        # IN-PLACE
        adata._inplace_subset_obs(mask)
        gc.collect()
        print_memory_usage("After Threshold Filter (Inplace)")
    return adata

def main():
    parser = argparse.ArgumentParser(description="Test reading HBCC dataset IN-PLACE")
    parser.add_argument("path", help="Path to h5ad file")
    args = parser.parse_args()
    
    file_path = args.path

    print(f"Starting IN-PLACE test loading of HBCC from {file_path}...")
    print_memory_usage("Start")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)

    try:
        ad = read_psychad_inplace(file_path, "psychAD", "HBCC")
        
        # Test filtering
        ad = filter_age_thresholds_inplace(ad, "HBCC", min_age=0, max_age=40)
        
        print(f"Final shape: {ad.shape}")
        if 'age_years' in ad.obs.columns:
            print(f"Age range: {ad.obs['age_years'].min()} - {ad.obs['age_years'].max()}")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
