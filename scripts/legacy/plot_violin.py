import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import re
import argparse

# Reuse extraction logic (simplest to duplicate for standalone script, or import if modules allowed)
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

def plot_violin(input_file, output_file):
    print(f"Processing {input_file}...")
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
        return

    adata = sc.read_h5ad(input_file)
    
    # Age Extraction
    age_col = 'development_stage'
    if age_col not in adata.obs.columns:
        print(f"Column {age_col} not found.")
        return
        
    print("Calculating log2_age...")
    adata.obs['numeric_age'] = adata.obs[age_col].apply(extract_age)
    adata.obs['log2_age'] = np.log2(adata.obs['numeric_age'].astype(float) + 1e-6)
    
    # Subclass
    subclass_col = 'subclass'
    if subclass_col not in adata.obs.columns:
        print(f"Column {subclass_col} not found.")
        return

    # Plot
    print(f"Plotting Violin to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Violin plot
    # Rotation 90 for x-axis labels is crucial for many subclasses
    sc.pl.violin(adata, keys='log2_age', groupby=subclass_col, rotation=90, save=False, show=False)
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    plot_violin(args.file, args.output)
