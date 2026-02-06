import scanpy as sc
import argparse
import os
import sys
import numpy as np
import psutil
import time

START_TIME = time.time()

def log_mem(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    elapsed = time.time() - START_TIME
    print(f"\n[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB ({process.memory_percent():.2f}%) - Elapsed: {elapsed/60:.2f} min")

def main():
    parser = argparse.ArgumentParser(description="Downsample and filter AnnData.")
    parser.add_argument("--input", required=True, help="Path to input .h5ad file")
    parser.add_argument("--output", required=True, help="Path to output .h5ad file")
    parser.add_argument("--n_cells", type=int, help="Target number of cells to downsample to")
    parser.add_argument("--filter_column", help="Metadata column to filter on (e.g., 'class')")
    parser.add_argument("--filter_value", help="Value to keep in filter_column (e.g., 'EN')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = parser.parse_args()

    log_mem("Start")
    print(f"Loading data from {args.input} (backed mode)...")
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        sys.exit(1)

    adata_backed = sc.read_h5ad(args.input, backed='r')
    log_mem("Metadata Loaded (Backed)")
    print(f"Original shape: {adata_backed.shape}")

    # Determine indices to keep
    import numpy as np
    
    n_total = adata_backed.n_obs
    keep_indices = np.arange(n_total)

    # 1. Filter
    if args.filter_column and args.filter_value:
        if args.filter_column not in adata_backed.obs.columns:
            print(f"Error: Column '{args.filter_column}' not found in obs.")
            sys.exit(1)
            
        print(f"Filtering for {args.filter_column} == '{args.filter_value}'...")
        # Boolean mask
        mask = (adata_backed.obs[args.filter_column] == args.filter_value).values
        keep_indices = keep_indices[mask]
        print(f"Cells matching filter: {len(keep_indices)}")
        
        if len(keep_indices) == 0:
            print(f"Error: No cells found matching filter.")
            sys.exit(1)
            
    log_mem("Post-Filtering")

    # 2. Downsample (on indices)
    if args.n_cells:
        if len(keep_indices) > args.n_cells:
            print(f"Downsampling from {len(keep_indices)} to {args.n_cells} cells...")
            np.random.seed(args.seed)
            keep_indices = np.random.choice(keep_indices, size=args.n_cells, replace=False)
            keep_indices.sort() # Good practice for HDF5 access
        else:
            print(f"Requested {args.n_cells} cells but only have {len(keep_indices)}. Keeping all.")

    log_mem("Post-Downsampling")

    # 3. Load subset into memory
    print(f"Loading {len(keep_indices)} cells into memory...")
    # Accessing by integer array works in backed mode to pull into memory
    adata_subset = adata_backed[keep_indices].to_memory()
    log_mem("Subset Loaded into Memory")

    # Save
    print(f"Saving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    adata_subset.write_h5ad(args.output)
    log_mem("Saved")
    print("Done.")

if __name__ == "__main__":
    main()
