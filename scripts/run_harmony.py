import scanpy as sc
import scanpy.external as sce
import argparse
import os
import sys
import psutil
import time

START_TIME = time.time()

def log_mem(label):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    elapsed = time.time() - START_TIME
    print(f"\n[Memory] {label}: {mem_gb:.2f} GB ({mem_gb/total_gb*100:.1f}%) - Elapsed: {elapsed/60:.2f} min")

def run_harmony(input_path, output_path, batch_key='source'):
    print(f"Reading {input_path}...")
    adata = sc.read_h5ad(input_path)

    if batch_key not in adata.obs.columns:
        print(f"Error: Batch key '{batch_key}' not found in obs.")
        sys.exit(1)
        
    if 'X_pca' not in adata.obsm.keys():
        print("Error: 'X_pca' not found. Please run PCA first.")
        sys.exit(1)

    print(f"Running Harmony integration on '{batch_key}'...")
    log_mem("Before Harmony")
    try:
        # Harmony expects PCA coordinates
        sce.pp.harmony_integrate(adata, key=batch_key, basis='X_pca', adjusted_basis='X_pca_harmony')
        print("Harmony integration complete. stored in 'X_pca_harmony'.")
        log_mem("After Harmony")
    except Exception as e:
        print(f"Failed to run Harmony: {e}")
        # Check if harmony-pytorch or harmony is installed
        print("Ensure 'harmony-pytorch' or 'harmony' is installed in the environment.")
        sys.exit(1)

    print("Computing neighbors using Harmony representation...")
    sc.pp.neighbors(adata, use_rep='X_pca_harmony', n_neighbors=30) 
    log_mem("Neighbors complete")

    print("Computing UMAP...")
    sc.tl.umap(adata, min_dist=0.3) 
    log_mem("UMAP complete")

    print(f"Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    adata.write_h5ad(output_path)
    
    elapsed = time.time() - START_TIME
    print(f"\nDone. Total time: {elapsed/60:.2f} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to processed .h5ad file (with PCA)")
    parser.add_argument("--output", required=True, help="Path to save output .h5ad")
    parser.add_argument("--batch_key", default="source", help="Column to use for batch correction (default: source)")
    
    args = parser.parse_args()
    
    run_harmony(args.input, args.output, args.batch_key)
