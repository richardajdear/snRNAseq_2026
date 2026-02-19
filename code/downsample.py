
import scanpy as sc
import argparse
import os
import sys
import numpy as np
import psutil
import time
import pandas as pd

# Import our read_data module
try:
    import read_data
except ImportError:
    # If running from outside code/ directory, add it to path
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import read_data

START_TIME = time.time()

def log_mem(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    elapsed = time.time() - START_TIME
    print(f"\n[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB ({process.memory_percent():.2f}%) - Elapsed: {elapsed/60:.2f} min")

def normalize_region_filter(adata):
    """
    Ensure region mapping is applied if not already done by read_data.
    Since we use read_data to load, it should be done.
    But for safety, we check or re-apply if columns are raw.
    """
    # read_data functions ALREADY map region to 'prefrontal cortex', etc.
    # So we just filter on 'region' column.
    if 'region' not in adata.obs.columns:
        print("Warning: 'region' column not found in adata.obs")
        return adata
    return adata

def main():
    parser = argparse.ArgumentParser(description="Downsample and filter AnnData with advanced logic.")
    parser.add_argument("--input", required=True, help="Path to input .h5ad file")
    parser.add_argument("--output", required=True, help="Path to output .h5ad file")
    
    # Dataset Type for correct reading
    parser.add_argument("--dataset_type", choices=['Velmeshev', 'Wang', 'Aging', 'HBCC', 'Generic'], 
                        default='Generic', help="Dataset type to determine reading logic.")

    # Filtering Options
    parser.add_argument("--pfc_only", action='store_true', help="Keep only 'prefrontal cortex' regions.")
    
    # Downsampling Options
    parser.add_argument("--age_downsample", action='store_true', help="Keep all <40, 10% >=40.")
    parser.add_argument("--n_cells", type=int, help="Target number of cells (simple random downsample, alternative to age_downsample)")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    log_mem("Start")
    
    # 1. Load Data using read_data logic
    # We must use the specific readers to get the metadata columns RIGHT (region, age_years)
    # The readers take a path.
    print(f"Loading {args.dataset_type} data from {args.input}...")
    
    adata = None
    if args.dataset_type == 'Velmeshev':
        # Velmeshev reader needs main file + meta dir. 
        # Assuming standard meta dir location relative to file or default?
        # The default implementation in read_data uses constants. 
        # We should pass the input path. 
        # And we might need to assume the meta dir is in standard place if not passed.
        # But read_velmeshev signature is (h5ad_path, meta_dir). 
        # Let's try to infer meta_dir or use default if input matches default.
        
        # If input is custom, consistent metadata might be an issue if meta files aren't nearby.
        # For this task, we are likely using the standard files or copies of them.
        # Let's trust the default meta_dir if it exists, or infer from input path.
        
        # Try default load first
        adata = read_data.read_velmeshev(h5ad_path=args.input)
        
    elif args.dataset_type == 'Wang':
        adata = read_data.read_wang(h5ad_path=args.input)
        
    elif args.dataset_type == 'Aging':
        # Load all ages, no filtering. Downsampling logic below handles <40 vs >=40.
        adata = read_data.read_psychad(h5ad_path=args.input, dataset_name='AGING', min_age=None, max_age=None)
        
    elif args.dataset_type == 'HBCC':
        adata = read_data.read_psychad(h5ad_path=args.input, dataset_name='HBCC', min_age=None, max_age=None)
        
    else:
        # Generic load
        adata = sc.read_h5ad(args.input)
        
    if adata is None:
        print(f"Error: Failed to load data from {args.input}")
        sys.exit(1)
        
    log_mem("Data Loaded")
    print(f"Initial Shape: {adata.shape}")
    
    # 2. Region Filter
    if args.pfc_only:
        if 'region' not in adata.obs.columns:
            print("Error: --pfc_only requested but 'region' column missing.")
            sys.exit(1)
            
        print("Filtering for Prefrontal Cortex...")
        # Check what values exist
        # print(adata.obs['region'].value_counts())
        
        # Filter
        adata = adata[adata.obs['region'] == 'prefrontal cortex']
        print(f"Post-PFC Filter Shape: {adata.shape}")
        
        if adata.n_obs == 0:
            print("Warning: No PFC cells found.")
            sys.exit(1)
            
    log_mem("Post-Region Filter")

    # 3. Age Downsampling
    if args.age_downsample:
        if 'age_years' not in adata.obs.columns:
             print("Error: --age_downsample requested but 'age_years' column missing.")
             sys.exit(1)
        
        # Check for donor column
        donor_col = 'individual'
        if donor_col not in adata.obs.columns:
            # Fallback or check alternatives
            if 'individualID' in adata.obs.columns: donor_col = 'individualID'
            elif 'donor_id' in adata.obs.columns: donor_col = 'donor_id'
            else:
                print("Error: --age_downsample (donor-based) requires 'individual' or 'donor_id' column.")
                print(f"Columns: {adata.obs.columns}")
                sys.exit(1)
             
        print(f"Performing Age-Based Downsampling (Donor-Based)...")
        print(f"  Policy: Keep all donors < 40. Keep 20% of donors >= 40 (all cells).")
        
        # Identify Donors and their Ages
        # We assume one age per donor. If mixed, we take the mean or max? 
        # Usually consistent. We'll take the first value found.
        donor_ages = adata.obs.groupby(donor_col)['age_years'].mean()
        
        donors_young = donor_ages[donor_ages < 40].index.tolist()
        donors_old = donor_ages[donor_ages >= 40].index.tolist()
        
        n_donors_young = len(donors_young)
        n_donors_old = len(donors_old)
        
        print(f"  Young Donors (<40): {n_donors_young}")
        print(f"  Old Donors (>=40): {n_donors_old}")
        
        # Sample Old Donors
        n_donors_old_keep = int(np.ceil(n_donors_old * 0.20)) # 20%, ceil to keep at least 1 if few
        if n_donors_old_keep == 0 and n_donors_old > 0: n_donors_old_keep = 1
        
        np.random.seed(args.seed)
        if n_donors_old > 0:
            donors_old_keep = np.random.choice(donors_old, size=n_donors_old_keep, replace=False).tolist()
        else:
            donors_old_keep = []
            
        print(f"  Keeping {n_donors_young} young donors + {len(donors_old_keep)} old donors (Target 20% of {n_donors_old}).")
        
        donors_keep = set(donors_young + donors_old_keep)
        
        # Filter Cells
        # We need indices of cells belonging to these donors
        # Isin is efficient
        mask_keep = adata.obs[donor_col].isin(donors_keep).values
        keep_indices = np.where(mask_keep)[0]
        
        adata = adata[keep_indices]
        print(f"Post-Age Downsample Shape: {adata.shape}")
        
    elif args.n_cells:
        # Simple random downsample if age logic not used
        if adata.n_obs > args.n_cells:
             print(f"Downsampling to {args.n_cells} cells (random)...")
             sc.pp.subsample(adata, n_obs=args.n_cells, random_state=args.seed)
             print(f"Post-Downsample Shape: {adata.shape}")
             
    log_mem("Post-Downsample")

    # 4. Save
    print(f"Saving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    adata.write_h5ad(args.output, compression='gzip')
    log_mem("Saved")
    print("Done.")

if __name__ == "__main__":
    main()
