import scanpy as sc
import pandas as pd
import numpy as np
import os
import glob
import sys
sys.path.append('/home/rajd2/rds/hpc-work/snRNAseq_2026/code')
from regulons import get_ahba_GRN, project_GRN

def project_combined_postnatal_EN_10k_thesis():
    
    ad = sc.read_h5ad('/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_postnatal_full_harmony_EN_10k.h5ad')
    
    # Load var from the other dataset just to get mapping from ENSG to gene_symbol
    try:
        ad_ref = sc.read_h5ad('/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld.h5ad', backed='r')
        ensg_to_symbol = ad_ref.var['gene_symbol'].to_dict()
        ad.var_names = pd.Series(ad.var_names).map(ensg_to_symbol).fillna(pd.Series(ad.var_names)).values
        ad.var_names_make_unique()
    except Exception as e:
        print("Warning: could not map gene symbols:", e)
    
    sc.pp.normalize_total(ad, target_sum=1e6)
    sc.pp.highly_variable_genes(ad, n_top_genes=10000, batch_key='chemistry', flavor='seurat_v3')
    
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights='/home/rajd2/rds/hpc-work/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv')
    project_GRN(ad, ahba_GRN, 'X_ahba', use_highly_variable=True, log_transform=False)
    
    out_df = pd.concat([ad.obs, pd.DataFrame(ad.obsm['X_ahba'], index=ad.obs_names, columns=ad.uns.get('X_ahba_names', [f"Network_{i}" for i in range(ad.obsm['X_ahba'].shape[1])]))], axis=1)

    # Re-normalize for Rmd Compatibility (columns from current dataset)
    # Target RMD format expects columns like age_years, age_category, age_log2, cell_class, donor_id
    if 'individual' in out_df.columns:
        out_df['donor_id'] = out_df['individual']
    if 'lineage' in out_df.columns:
        out_df['cell_class'] = out_df['lineage']
        
    age_years = out_df['age_years']
    
    if 'age_category' not in out_df.columns:
        out_df['age_category'] = pd.Categorical(np.select(
            [
                age_years < 0,
                age_years < 1,
                age_years < 9,
                age_years < 25,
                age_years >= 25
            ],
            [
                'Prenatal',
                'Infant',
                'Childhood',
                'Adolescence',
                'Adulthood'
            ],
            default = 'Unknown'
        ),
        ordered=True,
        categories = ['Prenatal', 'Infant', 'Childhood', 'Adolescence', 'Adulthood']
        )
    
    if 'age_log2' not in out_df.columns:
        out_df['age_log2'] = np.log2(1 + out_df['age_years'])
    
    out_path = '/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_combined_postnatal_EN_10k_thesis.csv'
    out_df.to_csv(out_path)
    print("Saved exact thesis replication to", out_path)

if __name__ == '__main__':
    project_combined_postnatal_EN_10k_thesis()
