
import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
import argparse
import warnings
import psutil
import time
import gc

# Import our new module
try:
    import read_data
except ImportError:
    # If running from outside code/ directory, add it to path
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import read_data

# Use constants from read_data as defaults
OUTPUT_DIR = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/"

START_TIME = time.time()

def log_mem(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    elapsed = time.time() - START_TIME
    print(f"\n[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB ({process.memory_percent():.2f}%) - Elapsed: {elapsed/60:.2f} min")
    sys.stdout.flush()

def check_structure(adatas):
    """
    Print diagnostic crosstabs and column intersections.
    """
    print("\n" + "="*40)
    print("      DIAGNOSTIC STRUCTURE CHECK")
    print("="*40)
    
    # 1. Identify shared columns
    col_sets = [set(ad.obs.columns) for ad in adatas.values()]
    common_cols = sorted(list(set.intersection(*col_sets)))
    
    print(f"\n[Common .obs Columns] Count: {len(common_cols)}")
    print(", ".join(common_cols))
    
    # 2. Crosstabs — use only obs metadata, not full data
    combined_meta = []
    for name, ad in adatas.items():
        df = ad.obs[['region', 'cell_class']].copy() if 'region' in ad.obs.columns and 'cell_class' in ad.obs.columns else ad.obs.copy()
        df['Dataset_Key'] = name
        if 'region' not in df.columns: df['region'] = 'MISSING'
        if 'cell_class' not in df.columns: df['cell_class'] = 'MISSING'
        combined_meta.append(df[['Dataset_Key', 'region', 'cell_class']])
        
    full_df = pd.concat(combined_meta)
    
    print("\n[Crosstab: Dataset x Region]")
    print(pd.crosstab(full_df['region'], full_df['Dataset_Key']))
    
    print("\n[Crosstab: Dataset x Cell Class]")
    print(pd.crosstab(full_df['cell_class'], full_df['Dataset_Key']))
    
    print("\n" + "="*40 + "\n")
    del full_df, combined_meta
    gc.collect()
    return common_cols

def main():
    parser = argparse.ArgumentParser(description="Combine snRNAseq datasets.")
    parser.add_argument("--postnatal", action='store_true', help="(Deprecated) Filter for age >= 0")
    parser.add_argument("--diagnose_only", action='store_true', help="Only run diagnostics, do not save")
    parser.add_argument("--output", default=f"{OUTPUT_DIR}/combined_postnatal_full.h5ad")
    
    # Allow overriding paths
    parser.add_argument("--aging_path", default=read_data.AGING_PATH)
    parser.add_argument("--hbcc_path", default=read_data.HBCC_PATH)
    parser.add_argument("--velmeshev_path", default=read_data.VELMESHEV_PATH)
    parser.add_argument("--wang_path", default=read_data.WANG_PATH)
    
    # Loading mode
    parser.add_argument("--direct_load", action='store_true', 
                        help="Load h5ad files directly (skip read_data processing). Use for pre-processed/downsampled inputs.")
    
    args = parser.parse_args()
    
    log_mem("Start")
    
    adatas = {}
    
    def load_dataset(path, name, reader_func=None, **reader_kwargs):
        """Load a dataset, optionally using a specific reader function."""
        if not path or not os.path.exists(path):
            if path: print(f"Warning: {name} path {path} does not exist.")
            return
        
        if args.direct_load:
            print(f"Loading {name} directly from {path}...")
            ad = sc.read_h5ad(path)
        elif reader_func:
            ad = reader_func(path, **reader_kwargs)
        else:
            ad = sc.read_h5ad(path)
        
        if ad is not None and ad.n_obs > 0:
            adatas[name] = ad
            print(f"  {name}: {ad.shape}")
        else:
            print(f"  Warning: {name} loaded empty or None.")
        log_mem(f"After loading {name}")
    
    # 1. Load Data
    load_dataset(args.velmeshev_path, 'VELMESHEV', 
                 reader_func=read_data.read_velmeshev, h5ad_path=args.velmeshev_path)
    load_dataset(args.wang_path, 'WANG',
                 reader_func=read_data.read_wang, h5ad_path=args.wang_path)
    load_dataset(args.aging_path, 'AGING',
                 reader_func=read_data.read_psychad, h5ad_path=args.aging_path, 
                 dataset_name='AGING', min_age=None, max_age=None)
    load_dataset(args.hbcc_path, 'HBCC',
                 reader_func=read_data.read_psychad, h5ad_path=args.hbcc_path,
                 dataset_name='HBCC', min_age=None, max_age=None)
        
    if not adatas:
        print("No datasets loaded.")
        sys.exit(1)
        
    # 2. Filter Postnatal (deprecated)
    if args.postnatal:
        print("Warning: --postnatal flag passed but inline filtering is deprecated in combine_data.py.")
        print("Assuming inputs are already filtered.")
    
    # 3. Diagnostics
    common_cols = check_structure(adatas)
    log_mem("After diagnostics")
    
    if args.diagnose_only:
        print("Diagnostics complete. Exiting.")
        return

    # 4. Combine
    print("Combining datasets...")
    
    for name in adatas:
        adData = adatas[name]
        
        # Standardize Gene Symbol Column
        if 'feature_name' in adData.var.columns:
            adData.var['gene_symbol'] = adData.var['feature_name']
        elif 'gene_name' in adData.var.columns:
            adData.var['gene_symbol'] = adData.var['gene_name']
        else:
            adData.var['gene_symbol'] = adData.var.index
            
        # Subset obs to common columns only
        valid_cols = [c for c in common_cols if c in adData.obs.columns]
        adData.obs = adData.obs[valid_cols]
        adatas[name] = adData
    
    log_mem("Before concat")
    
    combined = sc.concat(adatas, label='source', index_unique='-')
    print(f"Combined Shape: {combined.shape}")
    log_mem("After concat")
    
    # Free individual datasets to reclaim memory
    del adatas
    gc.collect()
    log_mem("After freeing individual datasets")
    
    # Restore Gene Metadata (Symbols and feature_length)
    # We need a reference — reload just .var from first dataset
    ref_path = args.velmeshev_path
    if ref_path and os.path.exists(ref_path):
        ref_adata = sc.read_h5ad(ref_path, backed='r')
        common_vars = combined.var_names
        
        # Standardize gene_symbol in reference
        if 'feature_name' in ref_adata.var.columns:
            ref_var = ref_adata.var.loc[common_vars]
            combined.var['gene_symbol'] = ref_var['feature_name'].values
            print(f"Restored 'gene_symbol' for {len(combined.var)} genes.")
        
        if 'feature_length' in ref_adata.var.columns:
            combined.var['feature_length'] = ref_var['feature_length'].values
            print(f"Restored 'feature_length' for {len(combined.var)} genes.")
        
        del ref_adata, ref_var
        gc.collect()
    
    log_mem("Before save")
    
    # Write
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
    
    print(f"Saving to {args.output}...")
    combined.write_h5ad(args.output, compression='gzip')
    log_mem("After save")
    print("Done.")

if __name__ == "__main__":
    main()
