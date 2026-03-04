import scanpy as sc
import pandas as pd
import numpy as np
import os
import re

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metadata_utils import simplify_cell_type, extract_age_psychad

# redefine get_original_metadata inside script to match logic but run without query_obs
def get_all_metadata(rds_dir='/home/rajd2/rds', datasets_to_load=['VELMESHEV', 'WANG', 'HBCC', 'AGING']):
    
    print(f"Loading metadata from sources: {datasets_to_load}")
    
    results = []
    
    # Common Output Columns
    final_cols = ['cell_class', 'cell_subclass', 'cell_type', 'region', 'age_years', 'sex', 'donor_id', 'dataset_source']
    
    # --- VELMESHEV ---
    if 'VELMESHEV' in datasets_to_load:
        print(f"Processing VELMESHEV...")
        meta_dir = os.path.join(rds_dir, "Cam_snRNAseq/velmeshev/velmeshev_meta/")
        try:
            ex = pd.read_csv(os.path.join(meta_dir, "ex_meta.tsv"), sep='\t', low_memory=False).rename({'cellId':'Cell_ID', 'Age_(days)':'Age_Num'}, axis=1)
            inh = pd.read_csv(os.path.join(meta_dir, "in_meta.tsv"), sep='\t', low_memory=False).rename({'cellId':'Cell_ID', 'Age_(days)':'Age_Num'}, axis=1)
            macro = pd.read_csv(os.path.join(meta_dir, "macro_meta.tsv"), sep='\t')
            micro = pd.read_csv(os.path.join(meta_dir, "micro_meta.tsv"), sep='\t').assign(Cell_Type = 'Microglia')
            
            for df in [ex, inh, macro, micro]:
                 if 'Cell_ID' in df.columns: 
                     df.set_index('Cell_ID', inplace=True)
            
            meta = pd.concat({'Ex': ex, 'In': inh, 'Macro': macro, 'Micro': micro})
            meta = meta.reset_index(0, names='Cell_Class_Broad')
            
            def map_velmeshev_lineage(row):
                cls = row['Cell_Class_Broad']
                if cls == 'Ex': return 'Excitatory'
                if cls == 'In': return 'Inhibitory'
                if cls == 'Micro': return 'Microglia'
                if cls == 'Macro':
                    st = str(row.get('subclass', '')).lower()
                    ct = str(row.get('Cell_Type', '')).lower()
                    if 'astro' in st or 'astro' in ct: return 'Astrocytes'
                    if 'oligo' in st or 'oligo' in ct: return 'Oligos'
                    if 'opc' in st or 'opc' in ct: return 'OPC'
                    if 'endo' in st or 'endo' in ct: return 'Endothelial'
                    if 'immune' in st or 'immune' in ct: return 'Microglia'
                    return 'Glia'
                return 'Other'

            meta['cell_class'] = meta.apply(map_velmeshev_lineage, axis=1)
            
            if 'Cell_Type' in meta.columns: meta['cell_type_raw'] = meta['Cell_Type']
            if 'subclass' in meta.columns: meta['cell_subclass'] = meta['subclass']
            if 'cluster' in meta.columns: meta['cluster'] = meta['cluster']
            
            meta['cell_type'] = meta.rename(columns={'cell_type_raw': 'cell_type'}).apply(simplify_cell_type, axis=1)

            if 'Age_Num' in meta.columns:
                 meta['age_years'] = (meta['Age_Num'] - 268) / 365.0
                 mask_prenatal = meta['Age'].astype(str).str.contains('GW')
                 meta.loc[mask_prenatal, 'age_years'] = -0.01
            else:
                 meta['age_years'] = np.nan

            region_map = {
                'BA10': 'prefrontal cortex', 'BA11': 'prefrontal cortex', 'BA9': 'prefrontal cortex',
                'BA46': 'prefrontal cortex', 'BA9/46': 'prefrontal cortex', 'BA8': 'prefrontal cortex', 
                'PFC': 'prefrontal cortex', 'FC': 'prefrontal cortex', 'FIC': 'prefrontal cortex',
                'dorsolateral prefrontal cortex': 'prefrontal cortex', 'DLPFC': 'prefrontal cortex',
                'V1': 'visual cortex', 'primary visual cortex': 'visual cortex',
                'GE': 'telencephalon', 'CGE': 'telencephalon', 'MGE': 'telencephalon', 'LGE': 'telencephalon',
                'ganglionic eminence': 'telencephalon',
                'BA24': 'cingulate cortex', 'ACC': 'cingulate cortex', 'Cing': 'cingulate cortex', 'cing': 'cingulate cortex',
                'Primary motor cortex': 'motor cortex', 'BA4': 'motor cortex', 
                'BA22': 'temporal cortex', 'STG': 'temporal cortex', 'temp': 'temporal cortex', 'temporal lobe': 'temporal cortex',
                'S1': 'neocortex', 'BA13': 'neocortex', 'INS': 'neocortex',
                'Frontoparietal cortex': 'prefrontal cortex', 'cortex': 'neocortex', 
                'cerebral cortex': 'neocortex', 'visual cortex': 'visual cortex',
            }
            if 'Region' in meta.columns:
                meta['region'] = meta['Region'].replace(region_map)
            else:
                meta['region'] = np.nan
            
            if 'Sex' in meta.columns: meta['sex'] = meta['Sex']
            if 'Individual' in meta.columns: meta['donor_id'] = meta['Individual']
            meta['dataset_source'] = 'Velmeshev'

            valid_cols = [c for c in final_cols if c in meta.columns]
            results.append(meta[valid_cols])
            
        except Exception as e:
            print(f"Error processing Velmeshev: {e}")

    # --- WANG ---
    if 'WANG' in datasets_to_load:
        path = os.path.join(rds_dir, "Cam_snRNAseq/wang/wang.h5ad")
        print(f"Processing WANG...")
        try:
            ad = sc.read_h5ad(path, backed='r')
            meta_loaded = ad.obs.copy()
            
            if not meta_loaded.empty:
                def map_wang_class(s):
                    s = str(s).lower()
                    if re.search(r'glutamatergic|corticothalamic|intratelencephalic|extratelencephalic|near-projecting', s): return 'Excitatory'
                    if re.search(r'gaba|interneuron', s): return 'Inhibitory'
                    if re.search(r'astrocyte', s): return 'Astrocytes'
                    if re.search(r'oligodendrocyte', s) and 'precursor' not in s: return 'Oligos'
                    if 'precursor' in s or 'opc' in s: return 'OPC'
                    if 'microglia' in s: return 'Microglia'
                    if 'endothelial' in s or 'vascular' in s: return 'Endothelial'
                    return 'Other'

                if 'cell_type' in meta_loaded.columns:
                    meta_loaded['cell_class'] = meta_loaded['cell_type'].apply(map_wang_class)
                    meta_loaded['cell_subclass'] = meta_loaded['cell_type'] 
                
                meta_loaded['cell_type'] = meta_loaded.apply(simplify_cell_type, axis=1)
                
                if 'Estimated_postconceptional_age_in_days' in meta_loaded.columns:
                    meta_loaded['age_years'] = (meta_loaded['Estimated_postconceptional_age_in_days'] - 268) / 365.0
                else: meta_loaded['age_years'] = np.nan
                
                if 'tissue' in meta_loaded.columns:
                     r = meta_loaded['tissue'].astype(str)
                     r = r.replace({
                        'Brodmann (1909) area 17': 'visual cortex',
                        'Brodmann (1909) area 10': 'prefrontal cortex',
                        'Brodmann (1909) area 9': 'prefrontal cortex',
                        'forebrain': 'neocortex'
                     })
                     meta_loaded['region'] = r
                
                if 'Sex' in meta_loaded.columns: meta_loaded['sex'] = meta_loaded['Sex']
                
                if 'donor_id' in meta_loaded.columns: pass
                elif 'Donor' in meta_loaded.columns: meta_loaded['donor_id'] = meta_loaded['Donor']
                else: meta_loaded['donor_id'] = np.nan
                
                meta_loaded['dataset_source'] = 'Wang'

                valid_cols = [c for c in final_cols if c in meta_loaded.columns]
                results.append(meta_loaded[valid_cols])
        except Exception as e:
            print(f"Error processing Wang: {e}")

    # --- PSYCHAD (HBCC & AGING) ---
    for key, subpath in [('HBCC', 'Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad'),
                         ('AGING', 'Cam_PsychAD/RNAseq/Aging_Cohort.h5ad')]:
        if key in datasets_to_load:
            path = os.path.join(rds_dir, subpath)
            print(f"Processing {key}...")
            
            try:
                ad = sc.read_h5ad(path, backed='r')
                m_df = ad.obs.copy()
                
                mapper = {
                    'EN': 'Excitatory', 'IN': 'Inhibitory', 'Astro': 'Astrocytes', 
                    'Oligo': 'Oligos', 'Mural': 'Endothelial', 'Endo': 'Endothelial',
                    'Immune': 'Microglia', 'OPC': 'OPC'
                }
                
                if not m_df.empty:
                    if 'class' in m_df.columns:
                        m_df['cell_class'] = m_df['class'].map(mapper).fillna(m_df['class'])
                    if 'subclass' in m_df.columns:
                        m_df['cell_subclass'] = m_df['subclass']
                    
                    m_df['cell_type'] = m_df.apply(simplify_cell_type, axis=1)

                    if 'development_stage' in m_df.columns:
                        m_df['age_years'] = m_df['development_stage'].apply(extract_age_psychad)
                    else: m_df['age_years'] = np.nan
                    
                    if 'tissue' in m_df.columns:
                        m_df['region'] = m_df['tissue'].replace({
                            'dorsolateral prefrontal cortex': 'prefrontal cortex',
                            'DLPFC': 'prefrontal cortex'
                        })
                    else: m_df['region'] = np.nan
                    
                    if 'sex' in m_df.columns: pass
                    elif 'Sex' in m_df.columns: m_df['sex'] = m_df['Sex']
                    
                    if 'donor_id' in m_df.columns: pass
                    elif 'col_id' in m_df.columns: m_df['donor_id'] = m_df['col_id'] # PsychAD often uses col_id or individualID?
                    # Check 'individualID'
                    if 'individualID' in m_df.columns: m_df['donor_id'] = m_df['individualID']

                    m_df['dataset_source'] = key
                
                    valid_cols = [c for c in final_cols if c in m_df.columns]
                    results.append(m_df[valid_cols])

            except Exception as e:
                print(f"Error processing {key}: {e}")
    
    # --- Combine ---
    if not results:
        print("No results")
        return pd.DataFrame()
        
    combined_meta = pd.concat(results)
    
    return combined_meta
    
if __name__ == "__main__":
    meta = get_all_metadata()
    out_path = '/home/rajd2/rds/hpc-work/snRNAseq_2026/results/metadata_utils_meta.csv'
    meta.to_csv(out_path)
    print(f"Saved metadata to {out_path}")

