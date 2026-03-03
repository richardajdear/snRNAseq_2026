import scanpy as sc

# Define paths
lifespan_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/lifespan_10k_processed.h5ad'
ref_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld.h5ad'
output_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/lifespan_10k_fixed.h5ad'

print(f"Loading {lifespan_path}...")
adata = sc.read_h5ad(lifespan_path)

print(f"Loading reference {ref_path}...")
ref_adata = sc.read_h5ad(ref_path, backed='r')

print("Mapping Ensembl IDs to gene symbols...")
# var_names in lifespan are Ensembl IDs. var_names in ref_adata are Ensembl IDs.
# we map them.
common_ensembl = adata.var_names.intersection(ref_adata.var_names)

if len(common_ensembl) > 0:
    # get mapping from ensembl to symbol
    mapping = ref_adata.var.loc[common_ensembl, 'gene_symbol'].to_dict()
    new_names = [mapping.get(idx, idx) for idx in adata.var_names]
    adata.var['gene_symbol'] = new_names
    print(f"Mapped {len(common_ensembl)} genes.")
else:
    print("Warning: No matching Ensembl IDs found.")

print(f"Saving fixed dataset to {output_path}...")
adata.write_h5ad(output_path, compression='gzip')
print("Done.")
