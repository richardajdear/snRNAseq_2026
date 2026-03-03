import scanpy as sc
import pandas as pd
import numpy as np
import os
import glob
import sys
sys.path.append('/home/rajd2/rds/hpc-work/snRNAseq_2026/code')
from regulons import get_ahba_GRN, project_GRN

def project_VelWangPsychad_10k_thesis():
    
    ad = sc.read_h5ad('/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld.h5ad')
    ad.var_names = ad.var['gene_symbol'].values
    ad.var_names_make_unique()
    
    sc.pp.normalize_total(ad, target_sum=1e6)
    sc.pp.highly_variable_genes(ad, n_top_genes=10000, batch_key='chemistry', flavor='seurat_v3')
    
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights='/home/rajd2/rds/hpc-work/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv')
    project_GRN(ad, ahba_GRN, 'X_ahba', use_highly_variable=True, log_transform=False)
    
    out_df = pd.concat([ad.obs, pd.DataFrame(ad.obsm['X_ahba'], index=ad.obs_names, columns=ad.uns.get('X_ahba_names', [f"Network_{i}" for i in range(ad.obsm['X_ahba'].shape[1])]))], axis=1)

    # Re-normalize for Rmd Compatibility
    age_years = out_df['age_years']
    
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
    
    out_df['age_log2'] = np.log2(1 + out_df['age_years'])
    
    out_path = '/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_VelWangPsychad_10k_thesis.csv'
    out_df.to_csv(out_path)
    print("Saved exact thesis replication to", out_path)

if __name__ == '__main__':
    project_VelWangPsychad_10k_thesis()
