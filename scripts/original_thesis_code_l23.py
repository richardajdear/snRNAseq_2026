
def read_velmeshev_meta(
        base_path="../velmeshev2023/cell_meta/",
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
                    (x['Age'].str.contains('GW')) & (x['Age_Num'] > 268),
                    (~x['Age'].str.contains('GW')) & (x['Age_Num'] < 268)
                ],
                [-0.01,0],
                default = (x['Age_Num']-268)/365)
            )
            .assign(Cell_Class = lambda x: x['Cell_Class'].replace({'Macro':'Glia', 'Micro':'Glia'})) 
            .assign(Cell_Class = lambda x: pd.Categorical(x['Cell_Class'], ordered=True, categories=['Ex','In','Glia']))
            .assign(Cell_Type = lambda x: x['Cell_Type'].replace({'PV_MP':'PV', 'SST_RELN':'SST'}))
            .assign(Cell_Type = lambda x: pd.Categorical(x['Cell_Type'], ordered=True, categories=x['Cell_Type'].unique()))
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
                            # bins=[-np.inf,1,2,5,9,16,25,np.inf],
                            # labels=['0-1','1-2','2-5','5-9','9-16','16-25','25+'])
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
            # .assign(Age_Years = lambda x: (x['Age_Num']-273)/365)
            .assign(Age_log2 = lambda x: np.log2(1 + x['Age_Years'] ) )
            .assign(Age_log10 = lambda x: np.log10(1 + x['Age_Years'] ) )
            .assign(Age_Postnatal = lambda x: ~np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']))
            .assign(Age_Postinfant = lambda x: x['Age_Years'].fillna(0)>=2)
            .assign(Individual = lambda x: x['Individual'].astype('str'))
            .assign(Pseudotime_pct = lambda x: x.groupby('Cell_Class')['Pseudotime'].apply(lambda y: y*100/y.max()).reset_index(0, drop=True)) # 
            .drop('PMI', axis=1)
        )

    return(meta)



ad_l23 = sc.read_h5ad('../velmeshev2023/expression/ad_L23.h5ad')
# Set X to raw counts
ad_l23.X = ad_l23.raw.X
ad_l23.obs = ad_l23.obs.join(meta)
# meta_l23 = ad_l23.obs
ad_l23 = ad_l23[ad_l23.obs['Region_Broad']=='FC']

sc.pp.normalize_total(ad_l23, target_sum=1e6)
sc.pp.highly_variable_genes(ad_l23, n_top_genes=10000, batch_key='Chemistry', flavor='seurat_v3')

ahba_GRN = get_ahba_GRN()
project_GRN(ad_l23, ahba_GRN, 'X_ahba', use_highly_variable=True, log_transform=False)

ahba_l23 = pd.DataFrame(ad_l23.obsm['X_ahba'], index=ad_l23.obs_names, columns=ad_l23.uns['X_ahba_names'])
# ahba_l23 = pd.DataFrame(ad_ahba.X, index=ad_ahba.obs_names, columns=ad_l23.uns['X_ahba_names'])

meta = read_velmeshev_meta()
ahba_l23 = (ahba_l23
           .melt(ignore_index=False, var_name='C')
           .loc[lambda x: np.isin(x['C'], ['C3+', 'C3-'])]
           .assign(C = lambda x: pd.Categorical(x['C'], ordered=True, categories=['C3+', 'C3-']))
           .join(meta[['Individual', 'Cell_Type', 'Dataset', 
                       'Age_log2', 'Age_Years', 'Age_Range2', 'Age_Range4', 'Pseudotime_pct']])
           .assign(age_range=lambda x: x['Age_Range4'])
)

# The below is run in R
source("../code/thesis_plots.r")

p_age <- ahba_l23 %>% plot_age()

comparisons = list(
    c('Adolescence', 'Adulthood'),
    c('Adolescence', 'Childhood')
)
p_boxes <- ahba_l23 %>% plot_boxes() + stat_compare_means(comparisons = comparisons, color='blue', label='p.signif')

(p_age | p_boxes) + 
    plot_annotation(tag_levels='a', 
    title='Developmental expression of AHBA C3 in frontal cortex Layer 2-3 Lineage cells')