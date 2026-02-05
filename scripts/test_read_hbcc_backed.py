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
from read_and_combine import get_raw_counts

def print_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB")

def read_psychad_backed_filtered(h5ad_path, dataset_name, min_age=None, max_age=40.0):
    print(f"Reading {dataset_name} from {h5ad_path} (BACKED Mode)...")
    print_memory_usage("Start Backed Read")
    
    # Read in backed mode
    adata_backed = sc.read_h5ad(h5ad_path, backed='r')
    print_memory_usage("Backed Object Created")
    
    # Age - Vectorized Extraction (On Backed Object's OBS)
    # Obs is loaded into memory even in backed mode
    if 'development_stage' in adata_backed.obs.columns:
        def extract_age_psychad(val):
            import re
            if pd.isna(val): return np.nan
            match = re.search(r'(\d+)-year-old', str(val))
            if match:
                return float(match.group(1))
            return np.nan

        print("Extracting age from metadata...")
        adata_backed.obs['age_years'] = adata_backed.obs['development_stage'].apply(extract_age_psychad)
    
    # Create Filter Mask
    print(f"[{dataset_name}] Creating Age Mask (< {max_age})...")
    mask = np.ones(adata_backed.n_obs, dtype=bool)
    
    if 'age_years' in adata_backed.obs.columns:
        mask = ~adata_backed.obs['age_years'].isna()
        mask &= (adata_backed.obs['age_years'] < max_age)
        if min_age is not None:
            mask &= (adata_backed.obs['age_years'] >= min_age)
            
    n_keep = mask.sum()
    print(f"[{dataset_name}] filtering {adata_backed.n_obs} -> {n_keep} cells")
    
    if n_keep == 0:
        print("No cells passed filter!")
        return None

    # Load SUBSET into memory
    print("Loading subset into memory...")
    print_memory_usage("Before To Memory")
    
    # Efficiently load subset
    adata = adata_backed[mask].to_memory()
    
    # Close backed file
    if hasattr(adata_backed.file, 'close'):
        adata_backed.file.close()
    del adata_backed
    gc.collect()
    
    print_memory_usage("After To Memory")
    
    # Now proceed with normal processing on the memory object
    
    # Ensure raw counts
    adata.X = get_raw_counts(adata)
    print_memory_usage("After Raw Counts")

    # Lineage Mapping (Vectorized)
    # Must ensure dtype='U' (unicode string) for np.char operations
    if 'class' in adata.obs.columns:
        cls = adata.obs['class'].astype(str).str.lower().values.astype('U')
    else:
        cls = np.full(adata.n_obs, '', dtype='U')
        
    if 'subclass' in adata.obs.columns:
        sub = adata.obs['subclass'].astype(str).str.lower().values.astype('U')
    else:
        sub = np.full(adata.n_obs, '', dtype='U')
    
    conditions = [
        (np.char.find(cls, 'en') != -1) | (np.char.find(cls, 'excitatory') != -1),
        (np.char.find(cls, 'in') != -1) | (np.char.find(cls, 'inhibitory') != -1),
        (np.char.find(cls, 'astro') != -1) | (np.char.find(sub, 'astro') != -1),
        (np.char.find(cls, 'oligo') != -1) | (np.char.find(sub, 'oligo') != -1),
        (np.char.find(cls, 'opc') != -1) | (np.char.find(sub, 'opc') != -1),
        (np.char.find(cls, 'micro') != -1) | (np.char.find(sub, 'micro') != -1) | (np.char.find(cls, 'immune') != -1),
        (np.char.find(cls, 'endo') != -1) | (np.char.find(sub, 'endo') != -1)
    ]
    
    choices = ['Excitatory', 'Inhibitory', 'Astrocytes', 'Oligos', 'OPC', 'Microglia', 'Endothelial']
    adata.obs['lineage'] = np.select(conditions, choices, default='Other')
         
    return adata

def main():
    parser = argparse.ArgumentParser(description="Test reading HBCC dataset BACKED")
    parser.add_argument("path", help="Path to h5ad file")
    args = parser.parse_args()
    
    file_path = args.path

    print(f"Starting BACKED test loading of HBCC from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)

    try:
        ad = read_psychad_backed_filtered(file_path, "HBCC", min_age=0, max_age=40)
        
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
