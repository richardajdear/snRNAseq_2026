
import scanpy as sc
import pandas as pd
import numpy as np
import os
import re

# Paths
PATHS = {
    'Velmeshev': '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev10k.h5ad',
    'Wang': '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/wang/wang_10k.h5ad',
    'HBCC': '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort_10k.h5ad',
    'Aging': '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort_10k.h5ad'
}

VELMESHEV_META_DIR = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_meta/"

dfs = []

print("Loading and aligning metadata...")

# --- Helper for Wang Classes ---
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

# 1. Velmeshev
if os.path.exists(PATHS['Velmeshev']):
    try:
        ad = sc.read_h5ad(PATHS['Velmeshev'], backed='r')
        obs = ad.obs.copy()
        
        # Merge TSV Metadata
        ex = pd.read_csv(f"{VELMESHEV_META_DIR}/ex_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num'}, axis=1)
        inh = pd.read_csv(f"{VELMESHEV_META_DIR}/in_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num', 'cellId':'Cell_ID'}, axis=1)
        macro = pd.read_csv(f"{VELMESHEV_META_DIR}/macro_meta.tsv", sep='\t')
        micro = pd.read_csv(f"{VELMESHEV_META_DIR}/micro_meta.tsv", sep='\t').assign(Cell_Type = 'Microglia')
        
        meta = (pd.concat({
                'Ex': ex, 'In': inh, 'Macro': macro, 'Micro': micro
            })
            .reset_index(0, names='Cell_Class_Broad').set_index('Cell_ID')
        )
        
        # Filter meta to matching cells
        common = obs.index.intersection(meta.index)
        meta = meta.loc[common]
        
        # Create temp DF for aligned counts
        df_v = pd.DataFrame(index=common)
        df_v['aligned_region'] = meta['Region']
        
        # Inspect Velmeshev Regions (Debugging)
        # Ensure string type and strip whitespace
        meta['Region'] = meta['Region'].astype(str).str.strip()
        
        if 'Region_Broad' in meta.columns:
            meta['Region_Broad'] = meta['Region_Broad'].astype(str).str.strip()
            print("\n=== Velmeshev: Region x Region_Broad ===")
            print(pd.crosstab(meta['Region'], meta['Region_Broad']).to_string())
            
        unique_regions = sorted(meta['Region'].unique())
        print(f"\n[Velmeshev Unique Regions (Stripped)]: {unique_regions}")
        
        # Test replace keys check
        test_keys = ['BA10', 'V1', 'ACC']
        for k in test_keys:
            if k in unique_regions:
                print(f"Key '{k}' found in regions.")
            else:
                print(f"Key '{k}' NOT found in regions.")
            
        # Tissue Alignment
            
        # Tissue Alignment
        # 'cerebral cortex' -> 'neocortex'
        # Check source column. Usually 'Region' or 'Region_Broad' or 'Tissue'?
        # In Velmeshev metadata we have 'Region' and 'Region_Broad'. 
        # Read_data.py mapped 'tissue' from... wait, where does 'tissue' come from in Velmeshev?
        # In the provided read_data.py snippet:
        # .assign(tissue = lambda x: x['tissue'].replace({...})) (Line 79)
        # But where is 'tissue' assigned? It wasn't shown in the snippet I saw earlier (lines 61-90).
        # Ah, looking at lines 1-100 of read_data.py... I didn't see 'tissue' created.
        # But 'Region' and 'Region_Broad' exist.
        # Let's assume 'Region' is the source for tissue-like info or check columns.
        # Diagnosed columns included: Region, Region_Broad.
        # Often 'Region' contains 'PFC', 'V1', etc.
        # 'Region_Broad' contains 'primary visual cortex', 'ganglionic eminence'?
        
        # Tissue Alignment
        # Detailed mapping based on User Request + Region inspection
        # Source is 'Region' for granular BAs
        
        # 1. Base map from Region
        df_v['aligned_tissue'] = meta['Region'].replace({
            'BA10': 'prefrontal cortex',
            'BA11': 'prefrontal cortex',
            'BA9': 'prefrontal cortex',
            'BA46': 'prefrontal cortex',
            'dorsolateral prefrontal cortex': 'prefrontal cortex',
            'DLPFC': 'prefrontal cortex',
            
            'V1': 'visual cortex',
            'primary visual cortex': 'visual cortex',
            
            'S1': 'neocortex',
            'BA4': 'neocortex',
            'temporal lobe': 'neocortex',
            'cerebral cortex': 'neocortex',
             
            'ganglionic eminence': 'telencephalon',
            'medial ganglionic eminence': 'telencephalon',
            'lateral ganglionic eminence': 'telencephalon',
            'caudal ganglionic eminence': 'telencephalon',
        })
        
        # Tissue Alignment (Comprehensive)
        # Observed Uniques: ['ACC', 'BA10', 'BA13', 'BA22', 'BA24', 'BA46', 'BA8', 'BA9', 'BA9/46', 
        # 'CGE', 'Cing', 'FC', 'FIC', 'Frontoparietal cortex', 'GE', 'INS', 'LGE', 'MGE', 'PFC', 
        # 'Primary motor cortex', 'STG', 'cing', 'cortex', 'temp']
        
        df_v['aligned_tissue'] = meta['Region'].replace({
            # Prefrontal / Frontal
            'BA10': 'prefrontal cortex',
            'BA11': 'prefrontal cortex',
            'BA9': 'prefrontal cortex',
            'BA46': 'prefrontal cortex',
            'BA9/46': 'prefrontal cortex',
            'BA8': 'prefrontal cortex', # Frontal Eye Fields, often grouped
            'PFC': 'prefrontal cortex',
            'FC': 'prefrontal cortex', # Frontal Cortex -> Broad Region FC
            'FIC': 'prefrontal cortex', 
            'dorsolateral prefrontal cortex': 'prefrontal cortex',
            'DLPFC': 'prefrontal cortex',

            # Visual
            'V1': 'visual cortex',
            'primary visual cortex': 'visual cortex',
            
            # Telencephalon / GE
            'GE': 'telencephalon',
            'CGE': 'telencephalon',
            'MGE': 'telencephalon',
            'LGE': 'telencephalon',
            'ganglionic eminence': 'telencephalon', 
            
            # Cingulate Cortex (User Request)
            'BA24': 'cingulate cortex', 
            'ACC': 'cingulate cortex',
            'Cing': 'cingulate cortex',
            'cing': 'cingulate cortex',
            
            # Motor Cortex (User Request)
            'Primary motor cortex': 'motor cortex',
            'BA4': 'motor cortex',
            
            # Temporal Cortex (User Request)
            'BA22': 'temporal cortex',
            'STG': 'temporal cortex',
            'temp': 'temporal cortex',
            'temporal lobe': 'temporal cortex',

            # Neocortex / Other Cortex
            'S1': 'neocortex',
            'BA13': 'neocortex', 
            'INS': 'neocortex',
            'Frontoparietal cortex': 'prefrontal cortex', # Maps to FC
            'Frontoparietal\xa0cortex': 'prefrontal cortex',
            'cortex': 'neocortex',
            'cerebral cortex': 'neocortex',
            'visual cortex': 'visual cortex', # Self-map
        })
        
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

        df_v['aligned_class'] = meta.apply(map_velmeshev_lineage, axis=1)
        df_v['Dataset'] = 'Velmeshev'
        dfs.append(df_v)
        
    except Exception as e:
        print(f"Error processing Velmeshev: {e}")

# 2. Wang
if os.path.exists(PATHS['Wang']):
    ad = sc.read_h5ad(PATHS['Wang'], backed='r')
    obs = ad.obs.copy()
    
    # Region (Existing)
    if 'tissue' in obs.columns:
        obs['aligned_region'] = obs['tissue'].str.replace('Brodmann (1909) area ', 'BA')
        obs['aligned_region'] = obs['aligned_region'].replace({'forebrain': 'neocortex'})
        
        # Tissue Alignment (New Request)
        obs['aligned_tissue'] = obs['tissue'].replace({
            'Brodmann (1909) area 17': 'visual cortex',
            'Brodmann (1909) area 10': 'prefrontal cortex',
            'Brodmann (1909) area 9': 'prefrontal cortex',
            'forebrain': 'neocortex'
        })
    else:
        obs['aligned_region'] = 'Unknown'
        obs['aligned_tissue'] = 'Unknown'

    # Class
    if 'cell_type' in obs.columns:
        obs['aligned_class'] = obs['cell_type'].apply(map_wang_class)
    else:
        obs['aligned_class'] = 'Unknown'
        
    obs['Dataset'] = 'Wang'
    dfs.append(obs[['Dataset', 'aligned_region', 'aligned_tissue', 'aligned_class']])

# 3. Aging
if os.path.exists(PATHS['Aging']):
    ad = sc.read_h5ad(PATHS['Aging'], backed='r')
    obs = ad.obs.copy()
    
    # Region
    if 'tissue' in obs.columns:
        obs['aligned_region'] = obs['tissue']
        # Normalize to 'prefrontal cortex' to match Wang
        obs['aligned_tissue'] = obs['tissue'].replace({'dorsolateral prefrontal cortex': 'prefrontal cortex'})
    elif 'region' in obs.columns:
        obs['aligned_region'] = obs['region']
        obs['aligned_tissue'] = obs['region'].replace({'dorsolateral prefrontal cortex': 'prefrontal cortex'})
    else:
        obs['aligned_region'] = 'Unknown'
        obs['aligned_tissue'] = 'Unknown'

    # Class
    mapper = {
        'EN': 'Excitatory', 'IN': 'Inhibitory', 'Astro': 'Astrocytes', 
        'Oligo': 'Oligos', 'Mural': 'Endothelial', 'Endo': 'Endothelial',
        'Immune': 'Microglia', 'OPC': 'OPC'
    }
    if 'class' in obs.columns:
        obs['aligned_class'] = obs['class'].map(mapper).fillna(obs['class'])
    else:
        obs['aligned_class'] = 'Unknown'

    obs['Dataset'] = 'Aging'
    dfs.append(obs[['Dataset', 'aligned_region', 'aligned_tissue', 'aligned_class']])

# 4. HBCC
if os.path.exists(PATHS['HBCC']):
    ad = sc.read_h5ad(PATHS['HBCC'], backed='r')
    obs = ad.obs.copy()
    
    # Region
    if 'tissue' in obs.columns:
        obs['aligned_region'] = obs['tissue']
        obs['aligned_tissue'] = obs['tissue'].replace({'dorsolateral prefrontal cortex': 'prefrontal cortex'})
    else:
        obs['aligned_region'] = 'Unknown'
        obs['aligned_tissue'] = 'Unknown'

        
    # Class
    mapper = {
        'EN': 'Excitatory', 'IN': 'Inhibitory', 'Astro': 'Astrocytes', 
        'Oligo': 'Oligos', 'Mural': 'Endothelial', 'Endo': 'Endothelial',
        'Immune': 'Microglia', 'OPC': 'OPC'
    }
    if 'class' in obs.columns:
        obs['aligned_class'] = obs['class'].map(mapper).fillna(obs['class'])
    else:
        obs['aligned_class'] = 'Unknown'
        
    obs['Dataset'] = 'HBCC'
    dfs.append(obs[['Dataset', 'aligned_region', 'aligned_tissue', 'aligned_class']])

# Combine
full_df = pd.concat(dfs)

# Crosstab Region
ct_region = pd.crosstab(full_df['Dataset'], full_df['aligned_region'])
print("\n=== Crosstab: Dataset x Aligned Region (Original) ===")
print(ct_region.to_string())

# Crosstab Tissue (New)
ct_tissue = pd.crosstab(full_df['Dataset'], full_df['aligned_tissue'])
print("\n=== Crosstab: Dataset x Aligned Tissue (New) ===")
print(ct_tissue.to_string())

# Crosstab Class
ct_class = pd.crosstab(full_df['Dataset'], full_df['aligned_class'])
print("\n=== Crosstab: Dataset x Cell Class ===")
print(ct_class.to_string())
