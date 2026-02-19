
import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
import re
import warnings

# --- Constants ---
VELMESHEV_PATH = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev.h5ad"
VELMESHEV_META_DIR = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_meta/"
WANG_PATH = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/wang/wang.h5ad"
AGING_PATH = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad"
HBCC_PATH = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"

# --- Helpers ---
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

def get_raw_counts(adata):
    """Ensure we get integer raw counts."""
    X = None
    if adata.raw is not None:
        try:
             X = adata.raw.X.copy()
        except: pass
    
    if X is None and 'counts' in adata.layers:
        X = adata.layers['counts'].copy()
        
    if X is None:
        # Fallback to .X, checking if it looks raw
        X = adata.X.copy()
        
    return X

# --- Parsing Functions ---

def read_velmeshev(h5ad_path=VELMESHEV_PATH, meta_dir=VELMESHEV_META_DIR):
    print(f"Reading Velmeshev from {h5ad_path}...")
    if not os.path.exists(h5ad_path):
        print(f"Error: {h5ad_path} not found.")
        return None
        
    adata = sc.read_h5ad(h5ad_path)
    adata.X = get_raw_counts(adata)
    
    print("Processing Velmeshev metadata...")
    try:
        # Load sub-metadata
        ex = pd.read_csv(f"{meta_dir}/ex_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num'}, axis=1)
        inh = pd.read_csv(f"{meta_dir}/in_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num', 'cellId':'Cell_ID'}, axis=1)
        macro = pd.read_csv(f"{meta_dir}/macro_meta.tsv", sep='\t')
        micro = pd.read_csv(f"{meta_dir}/micro_meta.tsv", sep='\t').assign(Cell_Type = 'Microglia')
        
        # Merge
        meta = (pd.concat({
                'Ex': ex,
                'In': inh,
                'Macro': macro,
                'Micro': micro
            })
            .reset_index(level=0).rename(columns={'level_0': 'Cell_Class_Broad'}).set_index('Cell_ID')
            .assign(Age_Years = lambda x: np.select(
                [
                    (x['Age'].str.contains('GW')) & (x['Age_Num'] > 268),
                    (~x['Age'].str.contains('GW')) & (x['Age_Num'] < 268)
                ],
                [-0.01,0],
                default = (x['Age_Num']-268)/365)
            )
        )
        
        # Enhanced Lineage Mapping
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
        
        # Intersect
        common = adata.obs_names.intersection(meta.index)
        if len(common) > 0:
            adata = adata[common]
            meta = meta.loc[common]
            
            # Key Variables
            adata.obs['age_years'] = meta['Age_Years']
            adata.obs['cell_class'] = meta['cell_class']
            adata.obs['sex'] = meta['Sex']
            adata.obs['individual'] = meta['Individual'].astype(str)
            adata.obs['region'] = meta['Region']
            
            # Aligned region
            # Use 'Region' column for granular mapping (BA10, etc.)
            # Do NOT use Region_Broad as it obscures the BAs
            combined_mapping = {
                # Prefrontal / Frontal
                'BA10': 'prefrontal cortex',
                'BA11': 'prefrontal cortex',
                'BA9': 'prefrontal cortex',
                'BA46': 'prefrontal cortex',
                'BA9/46': 'prefrontal cortex',
                'BA8': 'prefrontal cortex', 
                'PFC': 'prefrontal cortex',
                'FC': 'prefrontal cortex',
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
                'Frontoparietal\xa0cortex': 'prefrontal cortex', # Maps to FC
                'cortex': 'neocortex',
                'cerebral cortex': 'neocortex',
                'visual cortex': 'visual cortex',
            }
            adata.obs['region'] = meta['Region'].replace(combined_mapping)
            
            # Keep original cell types/subclasses if available
            # Velmeshev TSV uses 'Cell_Type' (capitalized) for fine-grained types like 'L2-3'
            if 'Cell_Type' in meta.columns:
                adata.obs['cell_type'] = meta['Cell_Type']
            elif 'cell_type' in meta.columns:
                adata.obs['cell_type'] = meta['cell_type']
            if 'subclass' in meta.columns: adata.obs['cell_subclass'] = meta['subclass']
            
            if 'Dataset' in meta.columns: adata.obs['dataset'] = meta['Dataset']
            if 'Chemistry' in meta.columns: adata.obs['chemistry'] = meta['Chemistry']
            
    except Exception as e:
        print(f"Error loading Velmeshev TSV metadata: {e}")
        return None
        
    return adata

def read_wang(h5ad_path=WANG_PATH):
    print(f"Reading Wang from {h5ad_path}...")
    if not os.path.exists(h5ad_path):
         print(f"Error: {h5ad_path} not found.")
         return None

    adata = sc.read_h5ad(h5ad_path)
    adata.X = get_raw_counts(adata)

    # Age: (days - 268) / 365
    if 'Estimated_postconceptional_age_in_days' in adata.obs.columns:
        adata.obs['age_years'] = (adata.obs['Estimated_postconceptional_age_in_days'] - 268) / 365.0
        
    # Sex: 'Sex' -> 'sex'
    if 'Sex' in adata.obs.columns:
         adata.obs['sex'] = adata.obs['Sex']

    # Region: 'tissue' -> 'region' (Map Brodmann)
    if 'tissue' in adata.obs.columns:
        adata.obs['region'] = adata.obs['tissue'].str.replace('Brodmann (1909) area ', 'BA')
        adata.obs['region'] = adata.obs['region'].replace({
            'BA17': 'visual cortex',
            'BA10': 'prefrontal cortex',
            'BA9': 'prefrontal cortex',
            'forebrain': 'neocortex'
        })
        
    # Cell Class Mapping
    # Wang has 'cell_type' with specific types
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

    if 'cell_type' in adata.obs.columns:
        adata.obs['cell_class'] = adata.obs['cell_type'].apply(map_wang_class)
        adata.obs['cell_subclass'] = adata.obs['cell_type'] # Wang uses fine types
        
    adata.obs['chemistry'] = 'multiome'
    adata.obs['dataset'] = 'Wang' # Will be overwritten by concat key but good to have
    
    return adata

def read_psychad(h5ad_path, dataset_name, min_age=0, max_age=40):
    # Generic for HBCC and Aging
    print(f"Reading {dataset_name} from {h5ad_path}...")
    if not os.path.exists(h5ad_path):
         print(f"Error: {h5ad_path} not found.")
         return None

    # Use backed mode to handle metadata without loading full X
    try:
        adata_backed = sc.read_h5ad(h5ad_path, backed='r')
        
        # Identify columns
        obs = adata_backed.obs
        
        # Calculate Age
        if 'development_stage' in obs.columns:
             # We need to compute age to filter (if filtering is requested - here we just load)
             # But 'extract_age_psychad' works on strings.
             # We can't easily modify backed obs.
             pass
        
        # For actual loading:
        # We need to filter for < 40y HERE to save memory if strict mode is ON.
        # But this script is 'read_and_combine'.
        # Let's assume we load the filtered subset.
        
        # Calculate age first to create mask
        # Ensure we work with strings/factors, not categories
        if 'development_stage' in obs.columns:
            dev_stage = obs['development_stage'].astype(str)
            ages = dev_stage.apply(extract_age_psychad)
        else:
            ages = pd.Series(np.nan, index=obs.index)
        
        # Filter Mask: Age >= 0 (and < 40 if desired, but user said 'Postnatal' usually implies 0+)
        # Existing logic filtered < 40. We will keep that.
        if min_age is not None and max_age is not None:
             mask = (ages >= min_age) & (ages < max_age)
             print(f"  Filtering {dataset_name}: {mask.sum()} / {len(mask)} cells kept (Age {min_age}-{max_age}).")
        elif min_age is not None:
             mask = (ages >= min_age)
             print(f"  Filtering {dataset_name}: {mask.sum()} / {len(mask)} cells kept (Age >={min_age}).")
        elif max_age is not None:
             mask = (ages < max_age)
             print(f"  Filtering {dataset_name}: {mask.sum()} / {len(mask)} cells kept (Age <{max_age}).")
        else:
             mask = pd.Series(True, index=ages.index)
             print(f"  No age filtering for {dataset_name}. {len(mask)} cells.")
        
        if mask.sum() == 0:
            print(f"  Warning: No cells left in {dataset_name} after filter.")
            return None
            
        # Load filtered subset to memory
        adata = adata_backed[mask].to_memory()
        
        # Compute proper age column
        adata.obs['age_years'] = ages[mask].values
        
        # Metadata
        if 'sex' in adata.obs.columns: pass # already 'sex'
        
        # Region
        if 'tissue' in adata.obs.columns:
            adata.obs['region'] = adata.obs['tissue']
            # Normalize to 'prefrontal cortex'
            adata.obs['region'] = adata.obs['region'].replace({'dorsolateral prefrontal cortex': 'prefrontal cortex'})
            
        # Cell Class
        # PsychAD uses 'class' (EN, IN, Astro, Oligo, OPC, Micro, Endo, Mural, Immune)
        mapper = {
            'EN': 'Excitatory', 
            'IN': 'Inhibitory', 
            'Astro': 'Astrocytes', 
            'Oligo': 'Oligos',
            'Mural': 'Endothelial', 
            'Endo': 'Endothelial',
            'Immune': 'Microglia',
            'OPC': 'OPC'
        }
        if 'class' in adata.obs.columns:
            adata.obs['cell_class'] = adata.obs['class'].map(mapper).fillna(adata.obs['class'])
            
        if 'subclass' in adata.obs.columns:
            adata.obs['cell_subclass'] = adata.obs['subclass']
            
        if 'cell_type' in adata.obs.columns:
             pass # keep existing
             
        adata.obs['dataset'] = dataset_name
        
        # Chemistry is usually 10xv3 for these newer datasets, but verify?
        # PsychAD is 10X v3.
        adata.obs['chemistry'] = 'V3'

        return adata
        
    except Exception as e:
        print(f"Error reading {dataset_name}: {e}")
        return None
