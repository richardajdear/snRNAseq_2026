import scanpy as sc
import os

print("Loading 100k unified dataset...")
out_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld.h5ad'
adata = sc.read_h5ad(out_path)

print("Loading reference var_names...")
ref_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad'
ref_adata = sc.read_h5ad(ref_path, backed='r')

print("Fixing .var metadata...")
if 'feature_name' in ref_adata.var.columns:
    adata.var['gene_symbol'] = ref_adata.var.loc[adata.var_names, 'feature_name'].values
elif 'gene_name' in ref_adata.var.columns:
    adata.var['gene_symbol'] = ref_adata.var.loc[adata.var_names, 'gene_name'].values

if 'feature_length' in ref_adata.var.columns:
    adata.var['feature_length'] = ref_adata.var.loc[adata.var_names, 'feature_length'].values

print("Saving fixed 100k dataset...")
adata.write_h5ad(out_path, compression='gzip')
print("Done!")
