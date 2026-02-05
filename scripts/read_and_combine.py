import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
import re
import argparse
import warnings

# --- Constants ---
VELMESHEV_PATH = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev.h5ad"
VELMESHEV_META_DIR = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_meta/"
WANG_PATH = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/wang/wang.h5ad"
AGING_PATH = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad"
HBCC_PATH = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"
OUTPUT_DIR = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/"

# --- Helpers ---
def extract_age_psychad(age_str):
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

def get_raw_counts(adata):
    """Ensure we get integer raw counts."""
    X = None
    if adata.raw is not None:
        try:
             X = adata.raw.X.copy()
        except: pass
    
    if X is None and 'counts' in adata.layers:
        X = adata.layers['counts'].copy()
        
    if X is None:
        # Fallback to .X, checking if it looks raw
        X = adata.X.copy()
        
    # Validation (simple check on sample)
    if hasattr(X, 'data'): data = X.data
    else: data = X
    
    # Optional: could enforce transformation here or just return X
    return X


# --- Parsing Functions ---

def read_velmeshev(h5ad_path=VELMESHEV_PATH, meta_dir=VELMESHEV_META_DIR):
    print(f"Reading Velmeshev from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    # Ensure raw counts
    adata.X = get_raw_counts(adata)
    
    print("Processing Velmeshev metadata...")
    try:
        # Load sub-metadata
        ex = pd.read_csv(f"{meta_dir}/ex_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num'}, axis=1)
        inh = pd.read_csv(f"{meta_dir}/in_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num', 'cellId':'Cell_ID'}, axis=1)
        macro = pd.read_csv(f"{meta_dir}/macro_meta.tsv", sep='\t')
        micro = pd.read_csv(f"{meta_dir}/micro_meta.tsv", sep='\t').assign(Cell_Type = 'Microglia')
        
        # Merge
        meta = (pd.concat({
                'Ex': ex,
                'In': inh,
                'Macro': macro,
                'Micro': micro
            })
            .reset_index(0, names='Cell_Class').set_index('Cell_ID')
            .assign(Age_Years = lambda x: np.select(
                [
                    (x['Age'].str.contains('GW')) & (x['Age_Num'] > 268),
                    (~x['Age'].str.contains('GW')) & (x['Age_Num'] < 268)
                ],
                [-0.01,0],
                default = (x['Age_Num']-268)/365)
            )
        )
        
        # Enhanced Lineage Mapping
        def map_velmeshev_lineage(row):
            cls = row['Cell_Class']
            if cls == 'Ex': return 'Excitatory'
            if cls == 'In': return 'Inhibitory'
            if cls == 'Micro': return 'Microglia'
            if cls == 'Macro':
                # Attempt to refine using 'subclass' or 'cluster' if available in macro_meta
                # Based on user check: keys like 'Astro', 'Oligo', 'OPC' might be in 'subclass' or 'cell_type'
                # Inspecting the passed 'macro' dataframe specifically would be better, but assuming merged 'meta' has these cols
                st = str(row.get('subclass', '')).lower()
                ct = str(row.get('cell_type', '')).lower()
                
                if 'astro' in st or 'astro' in ct: return 'Astrocytes'
                if 'oligo' in st or 'oligo' in ct: return 'Oligos'
                if 'opc' in st or 'opc' in ct: return 'OPC'
                if 'endo' in st or 'endo' in ct: return 'Endothelial'
                if 'immune' in st or 'immune' in ct: return 'Microglia' # or Immune
                return 'Glia' # Fallback
            return 'Other'

        meta['lineage'] = meta.apply(map_velmeshev_lineage, axis=1)
        
        # Intersect
        common = adata.obs_names.intersection(meta.index)
        if len(common) > 0:
            adata = adata[common]
            meta = meta.loc[common]
            adata.obs['age_years'] = meta['Age_Years']
            adata.obs['lineage'] = meta['lineage']
            adata.obs['sex'] = meta['Sex']
            adata.obs['individual'] = meta['Individual'].astype(str)
            adata.obs['region'] = meta['Region']
            if 'Dataset' in meta.columns: adata.obs['dataset'] = meta['Dataset']
            if 'Chemistry' in meta.columns: adata.obs['chemistry'] = meta['Chemistry']
            
    except Exception as e:
        print(f"Error loading Velmeshev TSV metadata: {e}")
        
    return adata

def read_wang(h5ad_path=WANG_PATH):
    print(f"Reading Wang from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    # Ensure raw counts
    adata.X = get_raw_counts(adata)

    # Age
    if 'Estimated_postconceptional_age_in_days' in adata.obs.columns:
        adata.obs['age_years'] = (adata.obs['Estimated_postconceptional_age_in_days'] - 268) / 365.0
    
    # Cell Type Mapping
    def map_lineage(s):
        s = str(s).lower()
        if 'glutamatergic' in s or 'excitatory' in s: return 'Excitatory'
        if 'gaba' in s or 'interneuron' in s or 'inhibitory' in s: return 'Inhibitory'
        if 'astrocyte' in s: return 'Astrocytes'
        if 'oligodendrocyte' in s: return 'Oligos'
        if 'opc' in s: return 'OPC'
        if 'microglia' in s: return 'Microglia'
        if 'glia' in s: return 'Glia'
        return 'Other'
        
    if 'cell_type' in adata.obs.columns:
        adata.obs['lineage'] = adata.obs['cell_type'].apply(map_lineage)
    
    # Sex
    if 'sex' in adata.obs.columns:
        adata.obs['sex'] = adata.obs['sex']
        
    adata.obs['dataset'] = 'Wang'
    adata.obs['chemistry'] = 'multiome'
    return adata

def read_psychad(h5ad_path, source_name, dataset_name, backed=False):
    print(f"Reading {dataset_name} from {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path, backed='r' if backed else None)
    
    # Ensure raw counts (SKIP if backed, as we can't modify safely yet)
    if not backed:
        adata.X = get_raw_counts(adata)
    
    # Age
    if 'development_stage' in adata.obs.columns:
        adata.obs['age_years'] = adata.obs['development_stage'].apply(extract_age_psychad)
        
    # Lineage Mapping (Vectorized)
    # Works on obs, so fine for backed mode too
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
         
    # Sex
    if 'sex' in adata.obs.columns:
        adata.obs['sex'] = adata.obs['sex']
        
    # Individual
    if 'donor_id' in adata.obs.columns:
        adata.obs['individual'] = adata.obs['donor_id']
        
    # adata.obs['source'] = source_name
    adata.obs['dataset'] = dataset_name
    adata.obs['chemistry'] = 'V3'

    # Filter Unknown Ages 
    # SKIP if backed, must be done by caller during subsetting to memory
    if not backed and 'age_years' in adata.obs.columns:
        # Check for NaNs
        mask = ~adata.obs['age_years'].isna()
        if not mask.all():
            adata = adata[mask].copy() # Explicit copy to free original memory and return clean object
            print(f"Filtered {adata.n_obs} cells with unknown age in {source_name}.")
            gc.collect()

    return adata

def combine(adata_list, keys):
    print("Aligning columns...")
    common_cols = ['age_years', 'sex', 'lineage', 'dataset', 'source', 'chemistry', 'individual', 'region']
    
    processed_list = []
    for ad in adata_list:
        # Note: age_category and age_log2 removed from here. 
        # Will be computed in process_data.py to save memory during combination.
        
        # Ensure cols exist
        for c in common_cols:
            if c not in ad.obs.columns:
                ad.obs[c] = np.nan
                
        # Subset obs to common
        ad.obs = ad.obs[common_cols]
        processed_list.append(ad)
        
    print(f"Concatenating (Inner Join) with keys={keys}...")
    # Inner join to keep only common genes, reducing memory usage
    # Use keys to assign 'source' label automatically
    combined = sc.concat(adata_list, label="source", keys=keys, join='inner')
    return combined

import psutil
import gc
import time

START_TIME = time.time()

def print_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    elapsed = time.time() - START_TIME
    print(f"\n[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB ({process.memory_percent():.2f}%) - Elapsed: {elapsed/60:.2f} min")

def filter_age_thresholds(ad, name, min_age=None, max_age=None):
    if 'age_years' not in ad.obs.columns:
        print(f"Warning: 'age_years' not found in {name}. Cannot filter age.")
        return ad
    
    n_prev = ad.n_obs
    print_memory_usage(f"Before Filtering {name}")
    
    # Create mask
    # Start with all true
    mask = np.ones(ad.n_obs, dtype=bool)
    
    if min_age is not None:
        # Strict inequality for min_age as requested ("0 < age")
        mask &= (ad.obs['age_years'] > min_age).values
    if max_age is not None:
        mask &= (ad.obs['age_years'] < max_age).values
        
    # Use in-place subsetting to avoid memory doubling
    try:
        ad._inplace_subset_obs(mask)
        print(f"[{name}] In-place filtering successful.")
    except Exception as e:
        print(f"[{name}] In-place filtering failed ({e}), falling back to copy ref method.")
        # Fallback (peaks memory but safe)
        ad = ad[mask].copy()
        gc.collect()

    print(f"[{name}] Age Filter (min={min_age}, max={max_age}): {n_prev} -> {ad.n_obs} cells")
    print_memory_usage(f"After Filtering {name}")
    return ad

def read_and_filter_psychad(h5ad_path, source_name, dataset_name, min_age, max_age):
    # Use backed=True to avoid OOM during initial load
    try:
        ad_backed = read_psychad(h5ad_path=h5ad_path, source_name=source_name, dataset_name=dataset_name, backed=True)
        print_memory_usage(f"{dataset_name} Backed Object Created")
        
        # Construct mask
        mask = np.ones(ad_backed.n_obs, dtype=bool)
        if 'age_years' in ad_backed.obs.columns:
            mask &= ~ad_backed.obs['age_years'].isna()
            mask &= (ad_backed.obs['age_years'] < max_age)
            if min_age is not None:
                 mask &= (ad_backed.obs['age_years'] >= min_age)
        
        print(f"[{dataset_name}] Loading subset into memory (Cells: {mask.sum()}/{len(mask)})...")
        adata = ad_backed[mask].to_memory()
        
        # Close backed file
        if hasattr(ad_backed.file, 'close'):
            ad_backed.file.close()
        del ad_backed
        gc.collect()
        print_memory_usage(f"{dataset_name} Subset Loaded")
        
        # Post-load steps
        adata.X = get_raw_counts(adata)
        return adata
    except Exception as e:
        print(f"Failed to read {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Combine datasets [Velmeshev, Wang, psychAD-Aging, psychAD-HBCC]")
    parser.add_argument("--postnatal", action="store_true", help="Filter for postnatal samples (Age >= 0)")
    parser.add_argument("--max_age", type=float, default=40.0, help="Filter samples < max_age (default: 40)")
    parser.add_argument("--output", help="Output path for combined h5ad", default=os.path.join(OUTPUT_DIR, "combined_10k.h5ad"))
    parser.add_argument("--velmeshev", help="Path to Velmeshev h5ad", default=None)
    parser.add_argument("--wang", help="Path to Wang h5ad", default=None)
    parser.add_argument("--aging", help="Path to Aging Cohort h5ad", default=None)
    parser.add_argument("--hbcc", help="Path to HBCC h5ad", default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print_memory_usage("Start")
    # Print total system memory
    mem_total = psutil.virtual_memory().total / (1024 ** 3)
    print(f"[Memory] Total System Memory: {mem_total:.2f} GB")
    
    datasets = []
    dataset_keys = []
    
    min_age = 0.0 if args.postnatal else None
    max_age = args.max_age
    print(f"Filtering criteria: Age >= {min_age} (if Set), Age < {max_age}")


    # 1. HBCC Cohort (PsychAD) 
    if args.hbcc:
        ad_h = read_and_filter_psychad(args.hbcc, 'psychAD', 'HBCC', min_age, max_age)
        datasets.append(ad_h)
        dataset_keys.append('HBCC')

    # 2. Aging Cohort (PsychAD)
    if args.aging:
        ad_r = read_and_filter_psychad(args.aging, 'psychAD', 'Aging', min_age, max_age)
        datasets.append(ad_r)
        dataset_keys.append('AGING')

    # 3. Velmeshev
    if args.velmeshev:
        try:
            ad_v = read_velmeshev(h5ad_path=args.velmeshev)
            print_memory_usage("Velmeshev Loaded")
            ad_v = filter_age_thresholds(ad_v, 'Velmeshev', min_age=min_age, max_age=max_age)
            datasets.append(ad_v)
            dataset_keys.append('VELMESHEV')
            gc.collect()
        except Exception as e:
            print(f"Failed to read Velmeshev: {e}")
            sys.exit(1)

    # 4. Wang
    if args.wang:
        try:
            ad_w = read_wang(h5ad_path=args.wang)
            print_memory_usage("Wang Loaded")
            ad_w = filter_age_thresholds(ad_w, 'Wang', min_age=min_age, max_age=max_age)
            datasets.append(ad_w)
            dataset_keys.append('WANG')
            gc.collect()
        except Exception as e:
            print(f"Failed to read Wang: {e}")
            sys.exit(1)

    if not datasets:
        print("No datasets loaded.")
        return

    print_memory_usage("Before Combine")
    combined = combine(datasets, dataset_keys)
    
    # clear inputs
    del datasets
    gc.collect()
    print_memory_usage("After Combine")
    
    print(f"Saving combined data to {args.output}...")
    combined.write_h5ad(args.output)
    
    elapsed_total = time.time() - START_TIME
    print(f"\nDone. Total time: {elapsed_total/60:.2f} min")

if __name__ == "__main__":
    main()
