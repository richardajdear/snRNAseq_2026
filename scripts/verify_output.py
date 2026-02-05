import scanpy as sc
import pandas as pd
import argparse
import os

def verify(h5ad_path):
    print(f"Verifying {h5ad_path}...")
    if not os.path.exists(h5ad_path):
        print("File not found!")
        return

    adata = sc.read_h5ad(h5ad_path)
    print(f"Shape: {adata.shape}")
    print(f"Obs columns: {adata.obs.columns.tolist()}")
    
    if 'source' in adata.obs.columns:
        print("\nCounts by Source:")
        print(adata.obs['source'].value_counts())
    
    if 'dataset' in adata.obs.columns:
        print("\nCounts by Dataset:")
        print(adata.obs['dataset'].value_counts())
        
    print("\nMemory Size (Estimates):")
    print(f"X (RAM): {adata.X.nbytes / 1024**3:.2f} GB" if hasattr(adata.X, 'nbytes') else "X: Sparse/Unknown")
    
    print("\nSuccess.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to h5ad to verify")
    args = parser.parse_args()
    verify(args.path)
