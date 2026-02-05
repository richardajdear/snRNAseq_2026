import scanpy as sc
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

def process_data(input_file, output_file, n_top_genes=10000, n_pcs=50, hvg_subset=None, flavor='seurat_v3'):
    log_mem("Start")
    print(f"Processing {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        sys.exit(1)

    adata = sc.read_h5ad(input_file)
    log_mem("Data Loaded")
    print(f"Original shape: {adata.shape}")

    # 0. Add Derived Age Columns (Moved from read_and_combine to save memory there)
    print("Computing derived age columns (category, log2)...")
    import pandas as pd
    import numpy as np

    def categorize_age(age_years):
        if pd.isna(age_years): return "Unknown"
        if age_years < 0: return "Prenatal"
        if age_years <= 1: return "Infant"
        if age_years <= 12: return "Childhood"
        if age_years <= 19: return "Adolescence"
        return "Adulthood"

    if 'age_years' in adata.obs.columns:
        # Age Category
        adata.obs['age_category'] = adata.obs['age_years'].apply(categorize_age)
        
        # Log2 Age: using log2(age_years + 1)
        # Note: No filtering of negative prenatal ages here as per user request to avoid memory spike from masking.
        # Assuming age_years >= 0 (postnatal filter) or small negative (prenatal) which +1 handles safely.
        # If age < -1, log will be NaN/Error, but earliest prenatal is ~-0.75 (GW 12).
        adata.obs['age_log2'] = np.log2(adata.obs['age_years'] + 1)
    else:
        print("Warning: 'age_years' not found, skipping derived columns.")

    # 1. Backup raw counts
    if 'counts' not in adata.layers:
        print("Backing up raw counts to layers['counts']...")
        adata.layers['counts'] = adata.X.copy()
        log_mem("Raw counts backed up")
    else:
        print("layers['counts'] already exists.")

    # 2. Normalize
    print("Normalizing total to 1e6...")
    sc.pp.normalize_total(adata, target_sum=1e6)
    log_mem("Normalization complete")

    # 3. Log Transform
    print("Log1p transforming...")
    sc.pp.log1p(adata)
    log_mem("Log1p complete")
    
    # Remove stale embeddings if present
    for key in ['X_umap', 'X_pca']:
        if key in adata.obsm:
            del adata.obsm[key]

    # 4. Highly Variable Genes
    print(f"Selecting {n_top_genes} highly variable genes ({flavor})...")
    try:
        kwargs = {'n_top_genes': n_top_genes, 'flavor': flavor}
        if flavor == 'seurat_v3':
            kwargs['layer'] = 'counts'
        
        if hvg_subset and hvg_subset < adata.n_obs:
            print(f"HVG: Subsetting to {hvg_subset} cells for calculation...")
            # Use random subset for HVG calculation
            adata_sub = sc.pp.subsample(adata, n_obs=hvg_subset, copy=True)
            log_mem("HVG Subset Created")
            sc.pp.highly_variable_genes(adata_sub, **kwargs)
            adata.var['highly_variable'] = adata_sub.var['highly_variable']
            # Also transfer other HVG related stats if needed (means, variances, dispersions)
            for col in ['means', 'variances', 'variances_norm', 'highly_variable_rank']:
                if col in adata_sub.var.columns:
                    adata.var[col] = adata_sub.var[col]
            
            del adata_sub
            import gc
            gc.collect()
            log_mem("HVG Subset Complete & Cleared")
            
        else:
            sc.pp.highly_variable_genes(adata, **kwargs)
        
        log_mem("HVG complete")
        print(f"HVG selection complete. {sum(adata.var['highly_variable'])} genes selected.")
    except Exception as e:
        print(f"Error in HVG selection: {e}")
        # Proceeding might be dangerous but let's exit to be safe
        sys.exit(1)

    # 5. PCA
    print(f"Running PCA with {n_pcs} components...")
    sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True)
    log_mem("PCA complete")

    # Save
    print(f"Saving processed data to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    adata.write_h5ad(output_file)
    
    elapsed = time.time() - START_TIME
    print(f"\nDone. Total time: {elapsed/60:.2f} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize, HVG, and PCA processing.")
    parser.add_argument("--input", required=True, help="Input h5ad file")
    parser.add_argument("--output", required=True, help="Output processed h5ad file")
    parser.add_argument("--n_top_genes", type=int, default=10000, help="Number of HVGs")
    parser.add_argument("--n_pcs", type=int, default=50, help="Number of PCs")
    parser.add_argument("--hvg_subset", type=int, default=None, help="Number of cells to subset for HVG calculation (save memory)")
    parser.add_argument("--flavor", type=str, default='seurat_v3', help="HVG selection flavor (seurat_v3 or seurat)")
    args = parser.parse_args()

    process_data(args.input, args.output, args.n_top_genes, args.n_pcs, args.hvg_subset, args.flavor)
