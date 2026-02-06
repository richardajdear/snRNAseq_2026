
import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys

# Paths to 10k subsets
DATASETS = {
    'Velmeshev': '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev10k.h5ad',
    'Wang': '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/wang/wang_10k.h5ad',
    'HBCC': '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort_10k.h5ad',
    'Aging': '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort_10k.h5ad'
}

def check_structure():
    print("=== Checking Dataset Structures ===")
    
    adatas = {}
    obs_cols = {}
    var_cols = {}
    
    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"ERROR: File not found for {name}: {path}")
            continue
            
        print(f"\nLoading {name} from {path}...")
        try:
            adata = sc.read_h5ad(path, backed='r')
            adatas[name] = adata
            obs_cols[name] = set(adata.obs.columns)
            var_cols[name] = set(adata.var.columns)
            print(f"  Shape: {adata.shape}")
            print(f"  Obs columns: {len(obs_cols[name])}")
            print(f"  Var columns: {len(var_cols[name])}")
        except Exception as e:
            print(f"  Error loading: {e}")

    # Find common columns
    if not obs_cols:
        return

    common_obs = set.intersection(*obs_cols.values())
    print("\n=== Common .obs Columns ===")
    print(sorted(list(common_obs)))

    common_var = set.intersection(*var_cols.values())
    print("\n=== Common .var Columns ===")
    print(sorted(list(common_var)))
    
    # Detailed Column Report
    print("\n=== Unique Columns per Dataset (Top 5) ===")
    for name in obs_cols:
        unique = obs_cols[name] - common_obs
        print(f"{name}: {list(unique)[:5]} ... ({len(unique)} total unique)")

    return adatas

def diagnose_alignment(adatas):
    print("\n=== Diagnosing Region/Cell Type Alignment ===")
    
    results = []
    
    for name, adata in adatas.items():
        # Load obs to memory for manipulation
        df = adata.obs.copy()
        
        # --- Region Alignment Logic ---
        df['aligned_region'] = 'Unknown'
        
        # 1. Velmeshev
        if name == 'Velmeshev':
            if 'region' in df.columns:
                df['aligned_region'] = df['region']
            elif 'tissue' in df.columns:
                df['aligned_region'] = df['tissue']
                
        # 2. Wang
        elif name == 'Wang':
            # Map tissue to region if available
            if 'tissue' in df.columns:
                 df['aligned_region'] = df['tissue'].str.replace('Brodmann (1909) area ', 'BA')
                 # Map forebrain -> neocortex logic from read_data.py
                 df['aligned_region'] = df['aligned_region'].replace({'forebrain': 'neocortex'})

        # 3. HBCC / Aging
        elif name in ['HBCC', 'Aging']:
             if 'tissue' in df.columns:
                 df['aligned_region'] = df['tissue']

        # --- Cell Class Alignment Logic ---
        df['aligned_class'] = 'Unknown'
        
        # 1. HBCC / Aging: Use 'class' directly
        if name in ['HBCC', 'Aging'] and 'class' in df.columns:
            df['aligned_class'] = df['class']
            
        # 2. Wang: Map from cell_type using regex (simplified from read_data.py)
        elif name == 'Wang' and 'cell_type' in df.columns:
            import re
            def map_wang(s):
                s = str(s).lower()
                if re.search(r'glutamatergic|corticothalamic|intratelencephalic|extratelencephalic|near-projecting', s): return 'Excitatory'
                if re.search(r'gaba|interneuron', s): return 'Inhibitory'
                if re.search(r'astrocyte', s): return 'Astrocytes'
                if re.search(r'oligodendrocyte', s) and 'precursor' not in s: return 'Oligos'
                if 'precursor' in s or 'opc' in s: return 'OPC'
                if 'microglia' in s: return 'Microglia'
                return 'Other'
            df['aligned_class'] = df['cell_type'].apply(map_wang)
            
        # 3. Velmeshev: Use 'lineage' or 'cell_type'
        elif name == 'Velmeshev':
            # Check cell_type if lineage is missing
            if 'cell_type' in df.columns:
                # Velmeshev cell types are usually short codes like AST-FB, IN-SST etc, or mapped names.
                # But read_data.py assigns 'lineage' from metadata. Let's see if 'lineage' is in the file.
                # Based on previous output, 'lineage' wasn't in top 5 unique but might be there.
                # Let's map typical Velmeshev labels just in case.
                def map_velm(s):
                    s = str(s).upper()
                    if 'AST' in s: return 'Astrocytes'
                    if 'OLI' in s: return 'Oligos'
                    if 'OPC' in s: return 'OPC'
                    if 'MIC' in s: return 'Microglia'
                    if 'IN-' in s: return 'Inhibitory'
                    if 'EX-' in s or 'L2' in s or 'L4' in s or 'L5' in s: return 'Excitatory'
                    return 'Other'
                
                # Check if we have a better column
                if 'lineage' in df.columns:
                    df['aligned_class'] = df['lineage'] # Should be Excitatory/Inhibitory/etc
                else:
                    df['aligned_class'] = df['cell_type'].apply(map_velm)

        # Standardize Class Names
        # HBCC/Aging use 'EN', 'IN', 'Astro', 'Oligo'
        # Wang/Velmeshev use 'Excitatory', 'Inhibitory', 'Astrocytes', 'Oligos'
        # Let's align to full names
        mapper = {
            'EN': 'Excitatory', 
            'IN': 'Inhibitory', 
            'Astro': 'Astrocytes', 
            'Oligo': 'Oligos',
            'Mural': 'Endothelial', # Close enough for summary? Or keep separate.
            'Endo': 'Endothelial',
            'Immune': 'Microglia' # Often overlaps
        }
        df['aligned_class'] = df['aligned_class'].replace(mapper)

        # Store
        df['Dataset'] = name
        results.append(df)

    # Combine
    full_df = pd.concat(results)
    
    # Calculate Shared Columns Intersection
    # Get columns from each df in results
    cols_sets = [set(df.columns) for df in results]
    common_final = set.intersection(*cols_sets)
    print("\n=== Proposed Common Columns (Count: {}) ===".format(len(common_final)))
    print(sorted(list(common_final)))

    # Crosstabs
    print("\n--- Crosstab: Dataset x Aligned Region ---")
    print(pd.crosstab(full_df['Dataset'], full_df['aligned_region']))
    
    print("\n--- Crosstab: Dataset x Aligned Cell Class ---")
    print(pd.crosstab(full_df['Dataset'], full_df['aligned_class']))

if __name__ == "__main__":
    adatas = check_structure()
    if adatas:
        diagnose_alignment(adatas)
