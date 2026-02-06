import scanpy as sc
import pandas as pd
import numpy as np
import os
import re

def simplify_cell_type(row):
    """
    Simplifies cell type information into short categories (<10 chars).
    Prioritizes granular excitatory/inhibitory types if available.
    Falls back to broad class if specific type is missing.
    """
    # Combine relevant fields to search
    # We look at 'cell_type', 'subclass', 'cluster', 'Cell_Type', 'subclass_label' etc.
    # But usually passed as a single string or we look at the row.
    
    # We'll use a hierarchy of inspection:
    # 1. Specific 'cell_type' or 'subclass' or 'cluster'
    # 2. 'cell_class'
    
    s = ""
    if pd.notna(row.get('cell_type')): s += str(row['cell_type']) + " "
    if pd.notna(row.get('cell_subclass')): s += str(row['cell_subclass']) + " "
    if pd.notna(row.get('subclass')): s += str(row['subclass']) + " "
    if pd.notna(row.get('cluster')): s += str(row['cluster']) + " "
    
    s = s.lower()
    
    # --- Excitatory ---
    if 'l2/3' in s or 'l2-3' in s: return 'L2/3 IT'
    if 'l4' in s: return 'L4 IT'
    if 'l5' in s and 'et' in s: return 'L5 ET'
    if 'l5' in s and 'it' in s: return 'L5 IT'
    if 'l6' in s and 'ct' in s: return 'L6 CT'
    if 'l6' in s and 'it' in s: return 'L6 IT'
    if 'l6b' in s: return 'L6b'
    if 'np' in s or 'near-projecting' in s: return 'L5/6 NP'
    if 'it' in s: return 'IT' # Generic IT if layer unknown
    
    # --- Inhibitory ---
    if 'pvalb' in s: return 'Pvalb'
    if 'sst' in s: return 'Sst'
    if 'vip' in s: return 'Vip'
    if 'lamp5' in s: return 'Lamp5'
    if 'sncg' in s: return 'Sncg'
    if 'pvm' in s: return 'PVM' # Perivascular Macrophage? No, usually immune.
    
    # --- Non-Neuronal ---
    if 'astro' in s: return 'Astro'
    if 'oligo' in s and 'precursor' not in s: return 'Oligo'
    if 'opc' in s or 'precursor' in s: return 'OPC'
    if 'micro' in s: return 'Micro'
    if 'endo' in s or 'mural' in s: return 'Endo'
    if 'vlmc' in s: return 'VLMC'
    if 'pericyte' in s: return 'Pericyte'
    if 'immune' in s or 'macrophage' in s: return 'Micro' # Broadly map immune to Micro/Immune
    
    # --- Fallback to Class/Broad if valid ---
    # Retrieve standardized cell class from row if available
    cls = str(row.get('cell_class', '')).lower()
    if cls == 'excitatory': return 'Excitatory'
    if cls == 'inhibitory': return 'Inhibitory'
    if cls == 'astrocytes': return 'Astro'
    if cls == 'oligos': return 'Oligo'
    if cls == 'opc': return 'OPC'
    if cls == 'microglia': return 'Micro'
    if cls == 'endothelial': return 'Endo'
    
    return np.nan # Unknown

def extract_age_psychad(age_str):
    if pd.isna(age_str) or str(age_str).lower() == 'unknown':
        return np.nan
    try:
        match = re.search(r'(\d+)', str(age_str))
        if match:
            age = float(match.group(1))
            if 'month' in str(age_str).lower(): age /= 12.0
            elif 'week' in str(age_str).lower(): age /= 52.0
            return age
    except: return np.nan
    return np.nan

def get_original_metadata(query_obs, base_dir, datasets_to_load=None):
    """
    Extracts and filters metadata from original datasets for the given obs (DataFrame).
    Requires 'dataset' column in query_obs to identify source.
    Standardizes cell_class, cell_subclass, cell_type (simplified), region, age, sex, donor_id.
    """
    print("Extracting original metadata...")
    
    if 'dataset' not in query_obs.columns:
        print("Error: 'dataset' column missing from query_obs.")
        return pd.DataFrame(index=query_obs.index)

    # 1. Prepare Query with Order Tracking
    query_base = query_obs[['dataset']].copy()
    query_base['_order'] = np.arange(len(query_base))
    
    def map_source(d):
        d = str(d).upper()
        if d in ['U01', 'RAMOS', 'HERRING', 'VELMESHEV']: return 'VELMESHEV'
        if 'WANG' in d: return 'WANG'
        if 'HBCC' in d: return 'HBCC'
        if 'AGING' in d: return 'AGING'
        return 'OTHER'
        
    query_base['source_key'] = query_base['dataset'].apply(map_source)
    unique_sources = query_base['source_key'].unique()
    
    if datasets_to_load:
        unique_sources = [s for s in unique_sources if s in datasets_to_load]
        
    print(f"Loading metadata from sources: {unique_sources}")
    
    results = []
    
    # Common Output Columns
    final_cols = ['cell_class', 'cell_subclass', 'cell_type', 'region', 'age_years', 'sex', 'donor_id', 'dataset_source']
    
    # --- VELMESHEV ---
    if 'VELMESHEV' in unique_sources:
        subset = query_base[query_base['source_key'] == 'VELMESHEV']
        print(f"Processing VELMESHEV ({len(subset)} cells)...")
        
        meta_dir = os.path.join(base_dir, "rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_meta/")
        
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
            
            # --- Field Extraction ---
            
            # Cell Class/Type
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
            
            # Subclass/Type info for simplification
            if 'Cell_Type' in meta.columns: meta['cell_type_raw'] = meta['Cell_Type']
            if 'subclass' in meta.columns: meta['cell_subclass'] = meta['subclass']
            if 'cluster' in meta.columns: meta['cluster'] = meta['cluster']
            
            # Simplification
            # We rename 'cell_type_raw' to 'cell_type' temporarily for the simplifier to find it, 
            # OR just update simplifier. Simplifier looks at row keys.
            meta['cell_type'] = meta.rename(columns={'cell_type_raw': 'cell_type'}).apply(simplify_cell_type, axis=1)

            # Age
            # (Age_Num - 268) / 365
            if 'Age_Num' in meta.columns:
                 meta['age_years'] = (meta['Age_Num'] - 268) / 365.0
                 # Fix prenatal?
                 # If Age contains "GW", it's prenatal. Age_Num > 268 is postnatal.
                 # existing logic:
                 mask_prenatal = meta['Age'].astype(str).str.contains('GW')
                 meta.loc[mask_prenatal, 'age_years'] = -0.01
            else:
                 meta['age_years'] = np.nan

            # Region
            # Use 'Region' column mapping from read_and_combine
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
            
            # Sex & Donor
            if 'Sex' in meta.columns: meta['sex'] = meta['Sex']
            if 'Individual' in meta.columns: meta['donor_id'] = meta['Individual']
            meta['dataset_source'] = 'Velmeshev'

            # Join
            # Using suffix strip fallback logic just in case
            merged = subset[['_order']].join(meta, how='left')
            
            if merged['cell_class'].isna().mean() > 0.9 and len(subset) > 0:
                 print("  High failure rate in direct join. Trying suffix strip...")
                 subset_reset = subset[['_order']].copy()
                 subset_reset['key'] = subset_reset.index.astype(str).str.rsplit('-', n=1).str[0]
                 merged = subset_reset.merge(meta, left_on='key', right_index=True, how='left').set_index(subset.index)

            results.append(merged)
            
        except Exception as e:
            print(f"Error processing Velmeshev: {e}")
            results.append(subset[['_order']]) 

    # --- WANG ---
    if 'WANG' in unique_sources:
        path = os.path.join(base_dir, "rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/wang/wang.h5ad")
        subset = query_base[query_base['source_key'] == 'WANG']
        print(f"Processing WANG ({len(subset)} cells)...")
        
        try:
            ad = sc.read_h5ad(path, backed='r')
            meta = ad.obs
            
            # Load subset
            common = meta.index.intersection(subset.index)
            meta_loaded = pd.DataFrame()
            if len(common) > 0: meta_loaded = meta.loc[common].copy()
            
            if len(common) < len(subset) * 0.1:
                 stripped = [i.rsplit('-', 1)[0] for i in subset.index]
                 common_stripped = meta.index.intersection(stripped)
                 if len(common_stripped) > 0:
                      print(f"  Matched {len(common_stripped)} cells via stripped suffix.")
                      meta_loaded = meta.loc[common_stripped].copy()

            if not meta_loaded.empty:
                # Class mapping
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
                
                # Simplify
                meta_loaded['cell_type'] = meta_loaded.apply(simplify_cell_type, axis=1)
                
                # Age
                if 'Estimated_postconceptional_age_in_days' in meta_loaded.columns:
                    meta_loaded['age_years'] = (meta_loaded['Estimated_postconceptional_age_in_days'] - 268) / 365.0
                else: meta_loaded['age_years'] = np.nan
                
                # Region
                if 'tissue' in meta_loaded.columns:
                     # Wang tissue column has "Brodmann (1909) area X"
                     r = meta_loaded['tissue'].astype(str)
                     r = r.replace({
                        'Brodmann (1909) area 17': 'visual cortex',
                        'Brodmann (1909) area 10': 'prefrontal cortex',
                        'Brodmann (1909) area 9': 'prefrontal cortex',
                        'forebrain': 'neocortex'
                     })
                     meta_loaded['region'] = r
                
                # Sex & Donor
                if 'Sex' in meta_loaded.columns: meta_loaded['sex'] = meta_loaded['Sex']
                
                # Donor
                if 'donor_id' in meta_loaded.columns: pass
                elif 'Donor' in meta_loaded.columns: meta_loaded['donor_id'] = meta_loaded['Donor']
                else: meta_loaded['donor_id'] = np.nan
                
                meta_loaded['dataset_source'] = 'Wang'

                # Join
                subset_reset = subset[['_order']].copy()
                subset_reset['key_orig'] = subset_reset.index
                subset_reset['key_strip'] = subset_reset.index.astype(str).str.rsplit('-', n=1).str[0]
                
                m1 = subset_reset.merge(meta_loaded, left_on='key_orig', right_index=True, how='left')
                m2 = subset_reset.merge(meta_loaded, left_on='key_strip', right_index=True, how='left', suffixes=('', '_strip'))
                
                for c in final_cols:
                    if c not in m1.columns: m1[c] = np.nan
                    if c+'_strip' in m2.columns:
                        m1[c] = m1[c].fillna(m2[c+'_strip'])
                        
                merged = m1.set_index('key_orig')
                results.append(merged[['_order'] + final_cols])
            else:
                results.append(subset[['_order']])
        except Exception as e:
            print(f"Error processing Wang: {e}")
            results.append(subset[['_order']])

    # --- PSYCHAD (HBCC & AGING) ---
    for key, subpath in [('HBCC', 'rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad'), 
                         ('AGING', 'rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad')]:
        if key in unique_sources:
            path = os.path.join(base_dir, subpath)
            subset = query_base[query_base['source_key'] == key]
            print(f"Processing {key} ({len(subset)} cells)...")
            
            try:
                ad = sc.read_h5ad(path, backed='r')
                meta = ad.obs
                
                # Load subset keys
                common = meta.index.intersection(subset.index)
                m_exact = pd.DataFrame()
                if len(common) > 0: m_exact = meta.loc[common].copy()
                    
                m_strip = pd.DataFrame()
                stripped_unique = list(set([i.rsplit('-', 1)[0] for i in subset.index if i not in common]))
                common_strip = meta.index.intersection(stripped_unique)
                if len(common_strip) > 0: m_strip = meta.loc[common_strip].copy()
                
                print(f"  {key}: Loaded {len(m_exact)} exact matches, {len(m_strip)} stripped matches.")
                
                # Standardize
                mapper = {
                    'EN': 'Excitatory', 'IN': 'Inhibitory', 'Astro': 'Astrocytes', 
                    'Oligo': 'Oligos', 'Mural': 'Endothelial', 'Endo': 'Endothelial',
                    'Immune': 'Microglia', 'OPC': 'OPC'
                }
                
                for m_df in [m_exact, m_strip]:
                    if m_df.empty: continue
                    if 'class' in m_df.columns:
                        m_df['cell_class'] = m_df['class'].map(mapper).fillna(m_df['class'])
                    if 'subclass' in m_df.columns:
                        m_df['cell_subclass'] = m_df['subclass']
                    
                    # Simplify Cell Type using 'subclass' (granular)
                    m_df['cell_type'] = m_df.apply(simplify_cell_type, axis=1)

                    # Age
                    if 'development_stage' in m_df.columns:
                        m_df['age_years'] = m_df['development_stage'].apply(extract_age_psychad)
                    else: m_df['age_years'] = np.nan
                    
                    # Region
                    if 'tissue' in m_df.columns:
                        m_df['region'] = m_df['tissue'].replace({
                            'dorsolateral prefrontal cortex': 'prefrontal cortex',
                            'DLPFC': 'prefrontal cortex'
                        })
                    else: m_df['region'] = np.nan
                    
                    # Sex
                    if 'sex' in m_df.columns: pass
                    elif 'Sex' in m_df.columns: m_df['sex'] = m_df['Sex']
                    
                    # Donor
                    if 'donor_id' in m_df.columns: pass
                    elif 'col_id' in m_df.columns: m_df['donor_id'] = m_df['col_id'] # PsychAD often uses col_id or individualID?
                    # Check 'individualID'
                    if 'individualID' in m_df.columns: m_df['donor_id'] = m_df['individualID']

                    m_df['dataset_source'] = key
                
                # Join
                subset_reset = subset[['_order']].copy()
                subset_reset['key_orig'] = subset_reset.index
                subset_reset['key_strip'] = subset_reset.index.astype(str).str.rsplit('-', n=1).str[0]
                
                m1 = subset_reset.merge(m_exact, left_on='key_orig', right_index=True, how='left')
                m2 = subset_reset.merge(m_strip, left_on='key_strip', right_index=True, how='left', suffixes=('', '_strip'))
                
                for c in final_cols:
                    if c not in m1.columns: m1[c] = np.nan
                    if c+'_strip' in m2.columns:
                        m1[c] = m1[c].fillna(m2[c+'_strip'])
                        
                merged = m1.set_index('key_orig')
                results.append(merged[['_order'] + final_cols])
                print(f"  {key} processed successfully.")

            except Exception as e:
                print(f"Error processing {key}: {e}")
                results.append(subset[['_order']])
    
    # --- Combine ---
    if not results:
        return pd.DataFrame(index=query_obs.index)
        
    combined_meta = pd.concat(results)
    combined_meta = combined_meta.sort_values('_order')
    
    valid_cols = [c for c in final_cols if c in combined_meta.columns]
    
    return combined_meta[valid_cols]
