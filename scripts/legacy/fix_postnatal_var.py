import scanpy as sc
import os

print("Loading Postnatal 10k dataset...")
postnatal_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_postnatal_10k.h5ad'
adata = sc.read_h5ad(postnatal_path)

print("Loading reference var_names for mapping...")
# Using the corrected 10k dataset as a reference for Ensembl -> Symbol mapping
ref_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld.h5ad'
ref_adata = sc.read_h5ad(ref_path, backed='r')

print("Mapping Ensembl IDs to Gene Symbols...")
# Build a mapping from ref_adata (which has Ensembl IDs as index and gene_symbol column)
# Actually, in VelWangPsychad_10k_PFC_lessOld, Ensembl IDs are likely the index.
# Let's confirm if they overlap.

# Intersection of Ensembl IDs
common_ensembl = adata.var_names.intersection(ref_adata.var_names)
print(f"Overlap: {len(common_ensembl)} / {len(adata.var_names)} genes.")

if len(common_ensembl) > 0:
    # Get mapping from reference
    mapping = ref_adata.var.loc[common_ensembl, 'gene_symbol'].to_dict()
    
    # Apply mapping
    new_names = [mapping.get(idx, idx) for idx in adata.var_names]
    adata.var['gene_symbol'] = new_names
    
    # We also need feature_length for the projection script if it uses it for TPM-like normalization
    if 'feature_length' in ref_adata.var.columns:
        adata.var['feature_length'] = [ref_adata.var.loc[idx, 'feature_length'] if idx in mapping else 0 
                                      for idx in adata.var_names]

print("Saving updated dataset to a new test file...")
output_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_postnatal_10k_fixed.h5ad'
adata.write_h5ad(output_path, compression='gzip')
print(f"Fixed dataset saved to: {output_path}")
