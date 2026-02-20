
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
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import read_data

START_TIME = time.time()

def log_mem(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    elapsed = time.time() - START_TIME
    print(f"\n[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB ({process.memory_percent():.2f}%) - Elapsed: {elapsed/60:.2f} min")
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description="Downsample and filter AnnData (fully backed mode).")
    parser.add_argument("--input", required=True, help="Path to input .h5ad file")
    parser.add_argument("--output", required=True, help="Path to output .h5ad file")
    
    parser.add_argument("--dataset_type", choices=['Velmeshev', 'Wang', 'Aging', 'HBCC', 'Generic'], 
                        default='Generic', help="Dataset type to determine reading logic.")

    parser.add_argument("--pfc_only", action='store_true', help="Keep only 'prefrontal cortex' regions.")
    parser.add_argument("--age_downsample", action='store_true', help="Keep all <40, 20% of donors >=40.")
    parser.add_argument("--n_cells", type=int, help="Target number of cells (random downsample)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    log_mem("Start")
    
    # =========================================================================
    # Step 1: Load data in BACKED mode (expression matrix stays on disk)
    # =========================================================================
    print(f"Loading {args.dataset_type} data from {args.input} (backed mode)...")
    
    adata_backed = None
    meta_df = None
    
    if args.dataset_type == 'Velmeshev':
        adata_backed, meta_df = read_data.read_velmeshev_backed(h5ad_path=args.input)
    elif args.dataset_type == 'Wang':
        adata_backed, meta_df = read_data.read_wang_backed(h5ad_path=args.input)
    elif args.dataset_type == 'Aging':
        adata_backed, meta_df = read_data.read_psychad_backed(h5ad_path=args.input, dataset_name='AGING')
    elif args.dataset_type == 'HBCC':
        adata_backed, meta_df = read_data.read_psychad_backed(h5ad_path=args.input, dataset_name='HBCC')
    else:
        # Generic: load backed, use obs as metadata
        adata_backed = sc.read_h5ad(args.input, backed='r')
        meta_df = adata_backed.obs.copy()
    
    if adata_backed is None or meta_df is None:
        print(f"Error: Failed to load data from {args.input}")
        sys.exit(1)
    
    log_mem("Data Loaded (backed)")
    print(f"Total cells in file: {adata_backed.shape[0]}")
    print(f"Cells with metadata: {len(meta_df)}")
    
    # =========================================================================
    # Step 2: Compute ALL filter masks on the lightweight metadata DataFrame
    #         (no expression matrix access needed)
    # =========================================================================
    
    # Start with all cells that have metadata
    mask = pd.Series(True, index=meta_df.index)
    
    # --- Region filter ---
    if args.pfc_only:
        if 'region' not in meta_df.columns:
            print("Error: --pfc_only requested but 'region' column missing.")
            sys.exit(1)
        
        pfc_mask = meta_df['region'] == 'prefrontal cortex'
        n_before = mask.sum()
        mask = mask & pfc_mask
        print(f"PFC filter: {n_before} -> {mask.sum()} cells")
    
    # --- Age-based donor downsampling ---
    if args.age_downsample:
        if 'age_years' not in meta_df.columns:
            print("Error: --age_downsample requested but 'age_years' column missing.")
            sys.exit(1)
        
        # Find donor column
        donor_col = None
        for col in ['individual', 'individualID', 'donor_id']:
            if col in meta_df.columns:
                donor_col = col
                break
        
        if donor_col is None:
            print("Error: --age_downsample requires 'individual', 'individualID', or 'donor_id' column.")
            print(f"Available columns: {list(meta_df.columns)}")
            sys.exit(1)
        
        print(f"Donor-based downsampling (using '{donor_col}')...")
        print(f"  Policy: Keep all donors < 40. Keep 20% of donors >= 40.")
        
        # Work with currently-masked cells only
        active_meta = meta_df[mask]
        donor_ages = active_meta.groupby(donor_col)['age_years'].mean()
        
        donors_young = donor_ages[donor_ages < 40].index.tolist()
        donors_old = donor_ages[donor_ages >= 40].index.tolist()
        
        print(f"  Young Donors (<40): {len(donors_young)}")
        print(f"  Old Donors (>=40): {len(donors_old)}")
        
        # Sample old donors
        n_keep = max(1, int(np.ceil(len(donors_old) * 0.20))) if donors_old else 0
        np.random.seed(args.seed)
        donors_old_keep = np.random.choice(donors_old, size=n_keep, replace=False).tolist() if donors_old else []
        
        print(f"  Keeping {len(donors_young)} young + {len(donors_old_keep)} old donors")
        
        donors_keep = set(donors_young + donors_old_keep)
        donor_mask = meta_df[donor_col].isin(donors_keep)
        
        n_before = mask.sum()
        mask = mask & donor_mask
        print(f"  Donor filter: {n_before} -> {mask.sum()} cells")
    
    # --- Random cell downsampling ---
    if args.n_cells and mask.sum() > args.n_cells:
        print(f"Random downsampling: {mask.sum()} -> {args.n_cells} cells...")
        np.random.seed(args.seed)
        keep_idx = np.random.choice(np.where(mask.values)[0], size=args.n_cells, replace=False)
        new_mask = pd.Series(False, index=meta_df.index)
        new_mask.iloc[keep_idx] = True
        mask = new_mask
        print(f"  After random downsample: {mask.sum()} cells")
    
    log_mem("All filters computed")
    print(f"\nFinal cell count: {mask.sum()} / {len(meta_df)}")
    
    # =========================================================================
    # Step 3: Materialize ONLY the final subset into memory
    # =========================================================================
    print(f"\nLoading {mask.sum()} cells into memory...")
    adata = read_data.materialize_subset(adata_backed, meta_df, mask)
    
    log_mem("Subset materialized")
    print(f"In-memory shape: {adata.shape}")
    
    # Close backed file
    del adata_backed
    
    # =========================================================================
    # Step 4: Save
    # =========================================================================
    print(f"Saving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    adata.write_h5ad(args.output, compression='gzip')
    log_mem("Saved")
    print("Done.")

if __name__ == "__main__":
    main()
