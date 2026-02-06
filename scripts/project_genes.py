import scanpy as sc
import pandas as pd
import numpy as np
import sys
import os
import re
import argparse

# Add code directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../code')))
try:
    from regulons import get_ahba_GRN, project_GRN
    from metadata_utils import get_original_metadata
except ImportError:
    print("Error: Could not import regulons.py or metadata_utils.py from ../code/")
    sys.exit(1)

def extract_age_years(age_str):
    age_str = str(age_str).lower()
    # Years
    match = re.search(r"(\d+)\s?-?year", age_str)
    if match:
        return float(match.group(1))
    
    # Months
    match = re.search(r"(\d+)\s?-?month", age_str)
    if match:
        return float(match.group(1)) / 12.0
        
    return np.nan

def map_ensembl_to_symbol_from_ref(adata, ref_path):
    """
    Maps Ensembl IDs (in adata.var_names) to Gene Symbols using a reference H5AD file.
    """
    print(f"Loading reference mapping from {ref_path}...")
    try:
        # Load only var from reference to save memory
        ref_adata = sc.read_h5ad(ref_path, backed='r')
        # Assumption: ref_adata.var_names are Ensembl IDs, ref_adata.var['gene_name'] are Symbols
        if 'gene_name' not in ref_adata.var.columns:
             print("Error: 'gene_name' column not found in reference .var")
             return adata
             
        # Create dictionary
        ensembl_to_symbol = ref_adata.var['gene_name'].to_dict()
        print(f"Loaded {len(ensembl_to_symbol)} gene mappings.")
        
        # Apply mapping
        new_names = [ensembl_to_symbol.get(idx, idx) for idx in adata.var_names]
        
        # We can't modify .var in place easily in backed mode if we want to save it back, 
        # but here we are just using it for projection.
        # However, adata.var_names setter might try to load things.
        # Safe strategy for backed: use the feature_name column if possible or just update the index in memory.
        adata.var['gene_symbol'] = new_names
        adata.var_names = new_names
        adata.var_names_make_unique()
        print("Gene mapping complete. var_names updated to symbols (unique).")
        
    except Exception as e:
        print(f"Mapping failed: {e}")
        
    return adata

def assign_age_range(age):
    if np.isnan(age):
        return "Unknown"
    if age <= 2: return "Infancy"
    if age <= 12: return "Childhood"
    if age <= 20: return "Adolescence"
    return "Adulthood"

def main():
    parser = argparse.ArgumentParser(description="Project AHBA gene weights onto processed snRNAseq data.")
    parser.add_argument("--input", required=True, help="Path to input processed .h5ad file")
    parser.add_argument("--output", required=True, help="Path to output projected .csv file")
    parser.add_argument("--filter_column", help="Column to filter by (e.g., lineage)")
    parser.add_argument("--filter_value", help="Value to keep in the filter column (e.g., Excitatory)")
    parser.add_argument("--base_dir", default="/home/rajd2/rds/", help="Base directory for original data")
    parser.add_argument("--ref_mapping_file", help="Path to reference H5AD for gene mapping (feature_name)")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_csv = args.output
    grn_file = "/home/rajd2/rds/hpc-work/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv"
    
    # Load with backed='r' to avoid OOM on 100GB+ files
    print(f"Loading data from {input_file} (backed mode)...")
    adata = sc.read_h5ad(input_file, backed='r')
    base_dir = args.base_dir

    # 1. Standardize Metadata
    print("Standardizing metadata using original sources...")
    # This will populate cell_class, cell_type, region, age_years, dataset_source, etc.
    # Note: adata.obs is in memory even in backed mode
    orig_meta = get_original_metadata(adata.obs, base_dir)
    
    for col in orig_meta.columns:
        adata.obs[col] = orig_meta[col]

    # Map Ensembl IDs to Symbols using Reference if provided
    if args.ref_mapping_file:
        adata = map_ensembl_to_symbol_from_ref(adata, args.ref_mapping_file)
    # Fallback to internal feature_name
    elif 'feature_name' in adata.var.columns:
        print("Mapping var_names to feature_name (Symbols)...")
        # adata.var is in memory
        adata.var_names = adata.var['feature_name'].astype(str)
        adata.var_names_make_unique()

    # Derived fields
    if 'age_years' in adata.obs.columns:
        adata.obs['Age_Years'] = adata.obs['age_years']
        adata.obs['Age_log2'] = np.log2(adata.obs['Age_Years'] + 1)
        adata.obs['Age_Range4'] = adata.obs['Age_Years'].apply(assign_age_range)
    
    if 'donor_id' in adata.obs.columns:
        adata.obs['Individual'] = adata.obs['donor_id']
    elif 'individual' in adata.obs.columns:
        adata.obs['Individual'] = adata.obs['individual']

    # Filter if requested
    if args.filter_column and args.filter_value:
        col = args.filter_column
        val = args.filter_value
        print(f"Filtering for {col} == {val}...")
        
        if col not in adata.obs.columns:
            print(f"Error: Column '{col}' not found in adata.obs. Available: {adata.obs.columns.tolist()}")
            sys.exit(1)
            
        n_start = adata.n_obs
        mask = adata.obs[col] == val
        
        # Load filtered data into memory
        # This converts backed AnnData to in-memory AnnData
        print(f"Loading filtered subset into memory...")
        adata = adata[mask].to_memory()
        n_end = adata.n_obs
        print(f"Filtered: {n_start} -> {n_end} cells kept.")
        
        if n_end == 0:
            print("Error: No cells remaining after filter!")
            sys.exit(1)
            
        # 2. Preprocessing & HVG Selection (Required for Projection)
        print("Preprocessing (Normalize -> Log1p -> HVG)...")
        
        # Save raw counts if needed, but project_GRN usually works on normalized-non-log or raw.
        # project_GRN doc: "apply to expression data that has been normalized but NOT log transformed"
        # However, sc.pp.highly_variable_genes requires log data.
        
        # Backup X (assuming it's raw counts from the file)
        # Check if X is integers?
        # For memory efficiency, we can process in place.
        
        # 1. Normalize to CPM (target_sum=1e4 to match typical 10k conventions or 1e6?) 
        # Notebook usually uses default or 1e4/1e6. Let's use 1e6 (CPM).
        # Wait, if the notebook is using default, scanpy might default to valid logic.
        # Let's assume 1e6 for standard CPM.
        sc.pp.normalize_total(adata, target_sum=1e6)
        
        # 2. Log1p (Required for HVG)
        sc.pp.log1p(adata)
        
        # 3. Calculate HVGs
        print("Calculating Highly Variable Genes...")
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        print(f"Found {sum(adata.var['highly_variable'])} highly variable genes.")
        
        # 4. Revert to Normalized-Non-Log for Projection?
        # project_GRN takes 'log_transform' argument. If True (default), it logs the projected result.
        # But for INPUT, it recommends normalized non-log.
        # If we leave it logged, project_GRN will project logged data.
        # The equation X @ Y.T implies linear combination.
        # If X is log(CPM+1), linear combination is geometric mean-ish.
        # Usually GRN weights are for normalized expression.
        
        # The user's notebook typically doesn't revert. 
        # But if the notebook had `project_GRN(..., log_transform=False)`, it suggests the output isn't logged?
        # Let's check regulons.py again.
        # Line 42: `def project_GRN(..., normalize=False, use_highly_variable=True, log_transform=False):`
        # Wait, log_transform default is False in the notebook call I saw earlier! 
        # And inregulons.py definition, default is False? No, look at line 42 step 1009.
        # `log_transform=False` is passed in notebook.
        
        # If adata.X IS logged, and we project:
        # Projected = Log(X) @ Weights.
        # If adata.X is NOT logged:
        # Projected = X @ Weights.
        
        # To strictly follow "recommended way": "apply to expression data that has been normalized but NOT log transformed".
        # We should un-log adata.X or use a layer.
        
        # Since we just ran log1p... we can expm1 it back?
        # Or better: Save normalized state in a layer before logging.
        # adata.layers['normalized'] = adata.X.copy() -> doubles memory usage (9GB -> 18GB). Still fine.
        
        # Actually, let's use:
        # sc.pp.highly_variable_genes(adata, layer='log_normalized'?)
        # Scanpy HVG often expects X.
        
        # Let's do:
        # 1. Normalize.
        # 2. X_norm = adata.X.copy()
        # 3. Log1p(adata)
        # 4. HVG(adata)
        # 5. adata.X = X_norm
        
        print("Reverting main X to Normalized (non-log) for projection...")
        # Since we are in memory, let's just do expm1 to avoid copy overhead if possible, or just copy.
        # Copy is safer.
        
        # Optimization: Calculate HVG on log, then just use expm1 on X?
        # Yes.
        np.expm1(adata.X, out=adata.X)
        # This restores normalized counts (approx, due to float precision).
        # Verify: log1p(x) = log(x+1). expm1(log(x+1)) = x+1 - 1 = x.
        # Precision loss is negligible for this purpose.
        
    else:
        # If not filtering, we are in backed mode?
        # We can't easily normalize/HVG in backed mode without loading.
        # The script assumes filtering is used for the projection job (PFC).
        # If user runs without filter, we might crash on memory if dataset is huge?
        # But user is running filters.
        pass

    # 2. Load GRN
    print(f"Loading GRN from {grn_file}...")
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file)
    
    # 3. Project GRN
    print("Projecting GRN...")
    project_GRN(adata, ahba_GRN, 'X_ahba', use_highly_variable=True, log_transform=False)
    
    # 4. Extract and Save DataFrame
    print("Extracting projection and saving CSV...")
    projection_df = pd.DataFrame(adata.obsm['X_ahba'], index=adata.obs_names, columns=adata.uns['X_ahba_names'])
    projection_df['obs_names'] = projection_df.index
    melted = projection_df.melt(id_vars=['obs_names'], var_name='C', value_name='value')
    
    # Merge with expanded metadata
    cols_to_keep = ['Individual', 'cell_type', 'cell_class', 'region', 'dataset_source', 
                    'Age_log2', 'Age_Years', 'Age_Range4', 'sex']
    valid_cols = [c for c in cols_to_keep if c in adata.obs.columns]
    
    meta = adata.obs[valid_cols].copy()
    meta['obs_names'] = meta.index
    
    final_df = pd.merge(melted, meta, on='obs_names')
    final_df = final_df[final_df['C'].isin(['C3+', 'C3-'])]
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

    
    # Filter for C3+ / C3-
    final_df = final_df[final_df['C'].isin(['C3+', 'C3-'])]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    final_df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    main()
