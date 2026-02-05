import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import pandas as pd
import os
import sys
import re
import argparse
import psutil
import time

START_TIME = time.time()

def log_mem(label):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    elapsed = time.time() - START_TIME
    print(f"\n[Memory] {label}: {mem_gb:.2f} GB ({mem_gb/total_gb*100:.1f}%) - Elapsed: {elapsed/60:.2f} min")

def extract_age(age_str):
    if pd.isna(age_str) or str(age_str).lower() == 'unknown':
        return np.nan
    try:
        match = re.search(r'(\d+)', str(age_str))
        if match:
            age = float(match.group(1))
            if 'month' in str(age_str).lower(): age /= 12.0
            elif 'week' in str(age_str).lower(): age /= 52.0
            return age
    except: return np.nan
    return np.nan

def plot_umap(input_file, output_file, colors=None, recompute=False, use_rep=None):
    print(f"Processing {input_file}...")
    log_mem("Start")
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
        return

    adata = sc.read_h5ad(input_file)
    log_mem("Data Loaded")
    
    # Age Extraction (Fallback)
    if 'numeric_age' not in adata.obs.columns:
        if 'development_stage' in adata.obs.columns:
            adata.obs['numeric_age'] = adata.obs['development_stage'].apply(extract_age)
        elif 'age_years' in adata.obs.columns:
            adata.obs['numeric_age'] = adata.obs['age_years']
            
    if recompute or 'X_umap' not in adata.obsm.keys():
        print("Computing Neighbors and UMAP...")
        
        # Determine representation
        rep = use_rep
        if rep is None:
            if 'X_pca_harmony' in adata.obsm.keys():
                print("Auto-detected X_pca_harmony. Using it...")
                rep = 'X_pca_harmony'
            elif 'X_pca' in adata.obsm.keys():
                print("Using existing X_pca for neighbors...")
                rep = 'X_pca'
            else:
                rep = None # Use .X
        
        if rep:
            print(f"Calculating neighbors using representation: {rep}")
            sc.pp.neighbors(adata, use_rep=rep)
        else:
            print("Computing neighbors on .X...")
            sc.pp.neighbors(adata)
            
        log_mem("Neighbors complete")
        sc.tl.umap(adata)
        log_mem("UMAP complete")

    # Point size heuristic
    n_cells = adata.n_obs
    pt_size =  30000 / n_cells
    pt_size = min(10, max(0.02, pt_size))
    print(f"Point size: {pt_size} for {n_cells} cells.")

    # Output setup
    if not output_file.endswith('.png'):
        output_file = os.path.splitext(output_file)[0] + '.png'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Plotting to {output_file}...")
    
    if colors:
        # Filter colors to those present in obs
        valid_colors = [c for c in colors if c in adata.obs.columns]
        missing = [c for c in colors if c not in adata.obs.columns]
        if missing:
            print(f"Warning: Columns not found in obs: {missing}")
        
        if not valid_colors:
            print("No valid colors found to plot.")
            return

        print(f"Plotting facets: {valid_colors}")
        
        # Grid setup: 2 rows x 3 columns for 6 facets
        fig = plt.figure(figsize=(18, 12)) 
        
        for i, color_key in enumerate(valid_colors):
            ax = fig.add_subplot(2, 3, i+1)
            
            if color_key == 'age_log2':
                # Custom Ticks for Age
                # Logic: Plot log2 values, but label them as original years
                # Ticks: 0, 1, 9, 25, 60 -> log2(0+1)=0, log2(1+1)=1, log2(9+1)=3.32, log2(25+1)=4.7, log2(60+1)=5.93
                tick_vals = [0, 1, 9, 25, 60]
                tick_locs = [np.log2(v + 1) for v in tick_vals]
                tick_labels = [str(v) for v in tick_vals]
                
                # Plot
                sc.pl.umap(adata, color=color_key, size=pt_size, ax=ax, show=False, 
                           title="Age (Years)", color_map='magma') # magma/viridis
                           
                # Find colorbar for this axis
                # Scanpy usually adds a cbar axis next to the plot axis
                # We can try to modify it. 
                # Alternative: pass 'colorbar_loc=None' and add our own?
                # Scanpy's UMAP is rigid.
                # Hack: Access the figure's axes, find the one that corresponds to the cbar of 'ax'
                # Recent axis created is likely the cbar
                
                cbar_ax = fig.axes[-1] 
                # Check if this axis is indeed a colorbar (usually narrow)
                # Apply ticks
                cbar_ax.set_yticks(tick_locs)
                cbar_ax.set_yticklabels(tick_labels)
                cbar_ax.set_ylabel("Age (Years)")
                
            else:
                 sc.pl.umap(adata, color=color_key, size=pt_size, ax=ax, show=False)
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        log_mem("Plot saved")
        plt.close()
        elapsed = time.time() - START_TIME
        print(f"\nDone. Total time: {elapsed/60:.2f} min")

    else:
        # Default single plot code... (omitted for brevity as user asked for grid)
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input h5ad file")
    parser.add_argument("--output", required=True, help="Output PNG file")
    parser.add_argument("--colors", nargs="+", help="List of columns to color by (plots grid)")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of neighbors and UMAP")
    parser.add_argument("--use_rep", help="Representation to use for neighbors (e.g., X_pca, X_pca_harmony)")
    args = parser.parse_args()
    
    plot_umap(args.input, args.output, colors=args.colors, recompute=args.recompute, use_rep=args.use_rep)
