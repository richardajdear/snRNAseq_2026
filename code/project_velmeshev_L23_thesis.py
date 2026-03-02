import scanpy as sc
import pandas as pd
import numpy as np
import os
import glob
import sys
sys.path.append('/home/rajd2/rds/hpc-work/snRNAseq_2026/code')
from regulons import get_ahba_GRN, project_GRN

def read_velmeshev_meta(
        base_path="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_meta",
):
    # Read metadata on velmeshev cells
    meta = (pd.concat({
                'Ex': pd.read_csv(f"{base_path}/ex_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num'}, axis=1),
                'In': pd.read_csv(f"{base_path}/in_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num', 'cellId':'Cell_ID'}, axis=1),
                'Macro': pd.read_csv(f"{base_path}/macro_meta.tsv", sep='\t'),
                'Micro': pd.read_csv(f"{base_path}/micro_meta.tsv", sep='\t').assign(Cell_Type = 'Microglia')
            })
            .reset_index(0, names='Cell_Class').set_index('Cell_ID')
            # Logic to assign 0 as birth
            .assign(Age_Years = lambda x: np.select(
                [
                    (x['Age'].str.contains('GW', na=False)) & (x['Age_Num'] > 268),
                    (~x['Age'].str.contains('GW', na=False)) & (x['Age_Num'] < 268)
                ],
                [-0.01,0],
                default = (x['Age_Num']-268)/365)
            )
            .assign(Cell_Class = lambda x: x['Cell_Class'].replace({'Macro':'Glia', 'Micro':'Glia'})) 
            .assign(Cell_Class = lambda x: pd.Categorical(x['Cell_Class'], ordered=True, categories=['Ex','In','Glia']))
            .assign(Cell_Type = lambda x: x['Cell_Type'].replace({'PV_MP':'PV', 'SST_RELN':'SST'}))
            .assign(Cell_Type = lambda x: pd.Categorical(x['Cell_Type'], ordered=True, categories=x['Cell_Type'].dropna().unique()))
            .assign(Cell_Lineage = lambda x: np.select(
                [
                    x['Cell_Class'] == 'Ex', 
                    x['Cell_Class'] == 'In',
                    x['Cell_Class'] == 'Glia'
                ],
                ['Excitatory', 'Inhibitory', x['Cell_Type']],
                default='Other'
            ))
            .assign(Cell_Lineage = lambda x: x['Cell_Lineage'].replace({'Fibrous_astrocytes':'Astrocytes', 'Protoplasmic_astrocytes':'Astrocytes'}))
            .assign(Cell_Lineage = lambda x: pd.Categorical(x['Cell_Lineage'], ordered=True, categories=['Excitatory', 'Inhibitory', 'Astrocytes', 'Oligos', 'OPC', 'Microglia', 'Glial_progenitors']))
            .assign(Age_Range2 = lambda x: pd.Categorical(np.where(
                    np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']), 
                    x['Age_Range'],
                    pd.cut((x['Age_Num']-273)/365, 
                            bins=[-np.inf,1,2,9,18,25,np.inf],
                            labels=['0-1','1-2','2-9','9-18', '18-25','25+'])
                    ),
                ordered=True, 
                categories=['2nd trimester', '3rd trimester']+['0-1','1-2','2-9','9-18','18-25','25+'])
            )
            .assign(Age_Range3 = lambda x: pd.Categorical(np.select(
                    [
                        np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']), 
                        np.isin(x['Age_Range2'], ['0-1', '1-3'])
                    ],
                    [
                        x['Age_Range'],
                        pd.cut(x['Age_Num']-273, 
                                bins=[-np.inf,91,182,365,2*365,3*365],
                                labels=['0-3m','3m-6m','6m-1y', '1-2y', '2-3y']
                        )
                    ],
                    default = x['Age_Range2']
                ),
                ordered=True, 
                categories=['2nd trimester', '3rd trimester']+['0-3m','3m-6m','6m-1y', '1-2y', '2-3y']+['3-9','9-18','18-25','25+'])
            )
            .assign(Age_Range4 = lambda x: pd.Categorical(np.select(
                    [
                        np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']), 
                        x['Age_Years'] < 1,
                        x['Age_Years'] < 9,
                        x['Age_Years'] < 25,
                        x['Age_Years'] >= 25
                    ],
                    [
                        'Prenatal',
                        'Infancy',
                        'Childhood',
                        'Adolescence',
                        'Adulthood'
                    ],
                    default = x['Age_Range']
                ),
                ordered=True,
                categories = ['Prenatal', 'Infancy', 'Childhood', 'Adolescence', 'Adulthood']
            ))
            .assign(Age_log2 = lambda x: np.log2(1 + x['Age_Years'] ) )
            .assign(Age_log10 = lambda x: np.log10(1 + x['Age_Years'] ) )
            .assign(Age_Postnatal = lambda x: ~np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']))
            .assign(Age_Postinfant = lambda x: x['Age_Years'].fillna(0)>=2)
            .assign(Individual = lambda x: x['Individual'].astype('str'))
            .assign(Pseudotime_pct = lambda x: x.groupby('Cell_Class')['Pseudotime'].apply(lambda y: y*100/y.max()).reset_index(0, drop=True))
            .drop('PMI', axis=1, errors='ignore')
        )

    return meta



def project_L23_thesis():
    meta = read_velmeshev_meta()
    
    ad_l23 = sc.read_h5ad('/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/ad_L23.h5ad')
    ad_l23.X = ad_l23.raw.X.copy()
    ad_l23.obs = ad_l23.obs.join(meta)
    
    ad_l23 = ad_l23[ad_l23.obs['Region_Broad']=='FC'].copy()
    
    sc.pp.normalize_total(ad_l23, target_sum=1e6)
    sc.pp.highly_variable_genes(ad_l23, n_top_genes=10000, batch_key='Chemistry', flavor='seurat_v3')
    
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights='/home/rajd2/rds/hpc-work/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv')
    project_GRN(ad_l23, ahba_GRN, 'X_ahba', use_highly_variable=True, log_transform=False)
    
    ahba_l23 = pd.DataFrame(ad_l23.obsm['X_ahba'], index=ad_l23.obs_names, columns=ad_l23.uns.get('X_ahba_names', [f"Network_{i}" for i in range(ad_l23.obsm['X_ahba'].shape[1])]))
    
    # Thesis-style output filtering
    ahba_l23 = (ahba_l23
               .melt(ignore_index=False, var_name='C')
               # .loc[lambda x: np.isin(x['C'], ['C3+', 'C3-'])]  <-- We keep all networks right now or just output the standard dataframe
    )
    # Actually, we should just save the standard CSV with networks and metadata so it matches the other pipeline format for analysis.Rmd to consume
    out_df = pd.concat([ad_l23.obs, pd.DataFrame(ad_l23.obsm['X_ahba'], index=ad_l23.obs_names, columns=ad_l23.uns.get('X_ahba_names', [f"Network_{i}" for i in range(ad_l23.obsm['X_ahba'].shape[1])]))], axis=1)

    # Re-normalize for Rmd Compatibility
    out_df['age_years'] = out_df['Age_Years']
    out_df['age_category'] = out_df['Age_Range4'].astype(str)
    out_df['age_log2'] = out_df['Age_log2']
    out_df['cell_class'] = out_df['Cell_Class'].astype(str)
    out_df['donor_id'] = out_df['Individual']
    
    out_path = '/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_L23_thesis.csv'
    out_df.to_csv(out_path)
    print("Saved exact thesis replication to", out_path)

if __name__ == '__main__':
    project_L23_thesis()
