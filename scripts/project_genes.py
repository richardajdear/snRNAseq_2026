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
except ImportError:
    print("Error: Could not import regulons.py from ../code/")
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

def assign_age_range(age):
    if np.isnan(age):
        return "Unknown"
    # User requested: Infancy, Childhood, Adolescence, Adulthood
    if age <= 2: return "Infancy"
    if age <= 12: return "Childhood"
    if age <= 20: return "Adolescence"
    return "Adulthood"

def abbreviate_cell_type(ctype):
    ctype = str(ctype)
    if "intratelencephalic" in ctype:
        return ctype.replace("intratelencephalic projecting glutamatergic neuron", "IT").replace("intratelencephalic projecting glutamatergic cortical neuron", "IT")
    if "extratelencephalic" in ctype:
        return ctype.replace("extratelencephalic projecting glutamatergic cortical neuron", "ET").replace("extratelencephalic projecting glutamatergic neuron", "ET")
    if "corticothalamic" in ctype:
        return ctype.replace("corticothalamic-projecting glutamatergic cortical neuron", "CT").replace("corticothalamic-projecting glutamatergic neuron", "CT")
    if "near-projecting" in ctype:
        return ctype.replace("near-projecting glutamatergic neuron", "NP").replace("near-projecting glutamatergic cortical neuron", "NP")
    if "L6b" in ctype:
        return "L6b"
    return ctype

def main():
    parser = argparse.ArgumentParser(description="Project AHBA gene weights onto processed snRNAseq data.")
    parser.add_argument("--input", required=True, help="Path to input processed .h5ad file")
    parser.add_argument("--output", required=True, help="Path to output projected .csv file")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_csv = args.output
    grn_file = "/home/rajd2/rds/hpc-work/snRNAseq_2026/data/ahba_dme_hcp_top8kgenes_weights.csv"
    
    # Load with backed='r' to avoid OOM on 100GB+ files
    print(f"Loading data from {input_file} (backed mode)...")
    adata = sc.read_h5ad(input_file, backed='r')
    
    # Map Ensembl IDs to Symbols if needed
    if 'feature_name' in adata.var.columns:
        print("Mapping var_names to feature_name (Symbols)...")
        adata.var['feature_name'] = adata.var['feature_name'].astype(str)
        adata.var_names = adata.var['feature_name']
        adata.var_names_make_unique()
    
    # 1. Prepare Metadata
    print("Processing metadata...")
    
    # Age Handling
    if 'age_years' in adata.obs.columns:
        print("Mapping 'age_years' to 'Age_Years'...")
        adata.obs['Age_Years'] = adata.obs['age_years']
    elif 'development_stage' in adata.obs.columns:
        print("Calculating 'Age_Years' from 'development_stage'...")
        adata.obs['Age_Years'] = adata.obs['development_stage'].apply(str).apply(extract_age_years)
    
    if 'Age_Years' in adata.obs.columns:
        adata.obs['Age_log2'] = np.log2(adata.obs['Age_Years'] + 1)
        adata.obs['Age_Range4'] = adata.obs['Age_Years'].apply(assign_age_range)
    else:
        print("WARNING: No age column found! Using placeholders.")
        adata.obs['Age_Years'] = np.nan
        adata.obs['Age_log2'] = np.nan
        adata.obs['Age_Range4'] = "Unknown"

    # Map Individual
    if 'individual' in adata.obs.columns:
        adata.obs['Individual'] = adata.obs['individual']
    elif 'donor_id' in adata.obs.columns:
        adata.obs['Individual'] = adata.obs['donor_id']
    else:
        adata.obs['Individual'] = "Unknown"
        
    # Map Cell Type
    if 'lineage' in adata.obs.columns:
        adata.obs['Cell_Type'] = adata.obs['lineage']
    elif 'cell_type' in adata.obs.columns:
        adata.obs['Cell_Type'] = adata.obs['cell_type'].apply(abbreviate_cell_type)
    else:
        adata.obs['Cell_Type'] = "Unknown"

    # Debug Unknown Ages for User (Safe check)
    if 'Age_Range4' in adata.obs.columns:
        unknown_mask = adata.obs['Age_Range4'] == 'Unknown'
        if unknown_mask.any():
            source_col = 'development_stage' if 'development_stage' in adata.obs.columns else ('age_years' if 'age_years' in adata.obs.columns else None)
            if source_col:
                unknowns = adata.obs[unknown_mask][source_col].unique()
                print(f"Report: 'Unknown' age categories found for {source_col}: {list(unknowns)}")

    # 2. Load GRN
    print(f"Loading GRN from {grn_file}...")
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file)
    
    # 3. Project GRN
    print("Projecting GRN...")
    
    project_GRN(adata, ahba_GRN, 'X_ahba', use_highly_variable=True, log_transform=False)
    print(f"Projected shape: {adata.obsm['X_ahba'].shape}")
    print(f"Non-zero elements in projection: {np.count_nonzero(adata.obsm['X_ahba'])}")
    if np.count_nonzero(adata.obsm['X_ahba']) == 0:
        print("WARNING: ALL VALUES ARE ZERO.")
    
    # 4. Extract and Save DataFrame
    print("Extracting projection and saving CSV...")
    # Extract projection
    projection_df = pd.DataFrame(adata.obsm['X_ahba'], index=adata.obs_names, columns=adata.uns['X_ahba_names'])
    
    projection_df['obs_names'] = projection_df.index
    melted = projection_df.melt(id_vars=['obs_names'], var_name='C', value_name='value')
    
    # Merge with metadata
    meta = adata.obs[['Individual', 'Cell_Type', 'Age_log2', 'Age_Years', 'Age_Range4']].copy()
    meta['obs_names'] = meta.index
    
    final_df = pd.merge(melted, meta, on='obs_names')
    
    # Filter for C3+ / C3-
    final_df = final_df[final_df['C'].isin(['C3+', 'C3-'])]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    final_df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

if __name__ == "__main__":
    main()
