
import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
import re
import warnings

# --- Constants ---
from environment import get_environment as _get_env
_rds = _get_env()['rds_dir']

VELMESHEV_PATH     = os.path.join(_rds, 'Cam_snRNAseq/velmeshev/velmeshev.h5ad')
VELMESHEV_META_DIR = os.path.join(_rds, 'Cam_snRNAseq/velmeshev/velmeshev_meta')
WANG_PATH          = os.path.join(_rds, 'Cam_snRNAseq/wang/wang.h5ad')
AGING_PATH         = os.path.join(_rds, 'Cam_PsychAD/RNAseq/Aging_Cohort.h5ad')
HBCC_PATH          = os.path.join(_rds, 'Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad')

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
        X = adata.X.copy()
        
    return X

# --- Cell subclass mappings ---
# Maps source-specific granular cell type labels to a canonical cross-dataset
# cell_subclass vocabulary. PsychAD naming conventions are used as the standard.

_VELMESHEV_SUBCLASS = {
    'L2-3': 'EN_L2_3_IT',  'L4': 'EN_L4_IT',       'L5': 'EN_L5_IT',
    'L5-6-IT': 'EN_L5_6_IT', 'L6': 'EN_L6_IT',     'SP': 'EN_L6B',
    'PV': 'IN_PVALB',       'PV_MP': 'IN_PVALB',
    'SST': 'IN_SST',        'SST_RELN': 'IN_SST',
    'VIP': 'IN_VIP',
    'RELN': 'IN_LAMP5_RELN',
    'CALB2': 'IN_ADARB2',   'CCK': 'IN_ADARB2',    'SV2C': 'IN_ADARB2',
    'NOS': 'IN_LAMP5',
    'INT': 'Inhibitory',    'Interneurons': 'Inhibitory',
    'Fibrous_astrocytes': 'Astro', 'Protoplasmic_astrocytes': 'Astro',
    'Glial_progenitors': 'Glial_progenitors', 'Progenitors': 'Progenitors',
    'Microglia': 'Micro',   'Oligos': 'Oligo',      'OPC': 'OPC',
}

def map_velmeshev_subclass(cell_type):
    return _VELMESHEV_SUBCLASS.get(cell_type, cell_type)


_WANG_SUBCLASS = {
    'EN-L2_3-IT': 'EN_L2_3_IT',    'EN-L4-IT': 'EN_L4_IT',
    'EN-L5-IT': 'EN_L5_IT',        'EN-L5-ET': 'EN_L5_ET',
    'EN-L5_6-NP': 'EN_L5_6_NP',   'EN-L6-CT': 'EN_L6_CT',
    'EN-L6-IT': 'EN_L6_IT',        'EN-L6b': 'EN_L6B',
    'EN-Newborn': 'EN_Immature',   'EN-IT-Immature': 'EN_Immature',
    'EN-Non-IT-Immature': 'EN_Immature',
    'IN-MGE-PV': 'IN_PVALB',       'IN-MGE-SST': 'IN_SST',
    'IN-CGE-VIP': 'IN_VIP',        'IN-CGE-SNCG': 'IN_ADARB2',
    'IN-Mix-LAMP5': 'IN_LAMP5',
    'IN-MGE-Immature': 'IN_Immature', 'IN-CGE-Immature': 'IN_Immature',
    'IN-dLGE-Immature': 'IN_Immature',
    'Astrocyte-Fibrous': 'Astro',  'Astrocyte-Protoplasmic': 'Astro',
    'Astrocyte-Immature': 'Astro_Immature',
    'Oligodendrocyte': 'Oligo',    'Oligodendrocyte-Immature': 'Oligo',
    'OPC': 'OPC',                  'Microglia': 'Micro',
    'Vascular': 'Endo',
    'IPC-EN': 'IPC_EN',            'IPC-Glia': 'IPC_Glia',
    'RG-vRG': 'Radial_glia',       'RG-oRG': 'Radial_glia',
    'RG-tRG': 'Radial_glia',
    'Cajal-Retzius cell': 'CR_cell', 'Unknown': 'Unknown',
}

def map_wang_subclass(cell_type):
    return _WANG_SUBCLASS.get(cell_type, cell_type)


# Maps CellxGene ontology cell_type labels (present in pre-combined h5ad files)
# to the canonical subclass vocabulary.  Used when patching existing combined files
# where source-specific Type/subclass columns are no longer available.
_CELLXGENE_SUBCLASS = {
    # Excitatory
    'L2/3 intratelencephalic projecting glutamatergic neuron':        'EN_L2_3_IT',
    'L2/3-6 intratelencephalic projecting glutamatergic neuron':      'EN_L2_3_IT',
    'L4 intratelencephalic projecting glutamatergic neuron':          'EN_L4_IT',
    'L5 intratelencephalic projecting glutamatergic neuron':          'EN_L5_IT',
    'L5 extratelencephalic projecting glutamatergic cortical neuron': 'EN_L5_ET',
    'L5/6 near-projecting glutamatergic neuron':                      'EN_L5_6_NP',
    'near-projecting glutamatergic cortical neuron':                  'EN_L5_6_NP',
    'L6 intratelencephalic projecting glutamatergic neuron':          'EN_L6_IT',
    'L6 corticothalamic-projecting glutamatergic cortical neuron':    'EN_L6_CT',
    'corticothalamic-projecting glutamatergic cortical neuron':       'EN_L6_CT',
    'L6b glutamatergic cortical neuron':                              'EN_L6B',
    'L6b glutamatergic neuron of the primary motor cortex':           'EN_L6B',
    'intratelencephalic-projecting glutamatergic cortical neuron':    'Excitatory',
    'glutamatergic neuron':                                           'Excitatory',
    # Inhibitory
    'GABAergic neuron':                                                      'Inhibitory',
    'VIP GABAergic cortical interneuron':                                    'IN_VIP',
    'pvalb GABAergic cortical interneuron':                                  'IN_PVALB',
    'sst GABAergic cortical interneuron':                                    'IN_SST',
    'lamp5 GABAergic cortical interneuron':                                  'IN_LAMP5',
    'sncg GABAergic cortical interneuron':                                   'IN_ADARB2',
    'caudal ganglionic eminence derived cortical interneuron':               'Inhibitory',
    'medial ganglionic eminence derived GABAergic cortical interneuron':     'Inhibitory',
    # Glia
    'astrocyte':                                    'Astro',
    'immature astrocyte':                           'Astro_Immature',
    'oligodendrocyte':                              'Oligo',
    'oligodendrocyte precursor cell':               'OPC',
    'differentiation-committed oligodendrocyte precursor': 'OPC',
    'microglial cell':                              'Micro',
    # Vascular
    'endothelial cell':           'Endo',
    'brain vascular cell':        'Endo',
    'pericyte':                   'PC',
    'perivascular macrophage':    'PVM',
    'smooth muscle cell':         'SMC',
    'vascular leptomeningeal cell': 'VLMC',
    # Developmental
    'forebrain radial glial cell': 'Radial_glia',
    'neural progenitor cell':      'Progenitors',
    'Cajal-Retzius cell':          'CR_cell',
    # Immune / adaptive
    'natural killer cell': 'Adaptive',
    'T cell':              'Adaptive',
    'B cell':              'Adaptive',
    'plasma cell':         'Adaptive',
    # Unknown
    'unknown': 'Unknown',
}

def map_cellxgene_subclass(cell_type):
    return _CELLXGENE_SUBCLASS.get(cell_type, cell_type)


# PsychAD uses EN_L3_5_IT_1/2/3 and EN_L6_IT_1/2; collapse to broad labels.
_EN_COLLAPSE = {
    'EN_L3_5_IT_1': 'EN_L3_5_IT', 'EN_L3_5_IT_2': 'EN_L3_5_IT',
    'EN_L3_5_IT_3': 'EN_L3_5_IT',
    'EN_L6_IT_1': 'EN_L6_IT',     'EN_L6_IT_2': 'EN_L6_IT',
}

def collapse_en_subclass(label):
    return _EN_COLLAPSE.get(label, label)


# ============================================================================
# BACKED-MODE READERS
# These functions return (adata_backed, meta_df) where:
#   - adata_backed: AnnData in backed='r' mode (expression matrix stays on disk)
#   - meta_df: DataFrame with computed metadata columns (age_years, region, 
#              cell_class, individual, etc.) indexed by cell barcode
#
# The caller is responsible for:
#   1. Computing filter masks from meta_df
#   2. Loading only the needed subset via adata_backed[mask].to_memory()
#   3. Applying meta_df columns to the in-memory subset
# ============================================================================

def read_velmeshev_backed(h5ad_path=VELMESHEV_PATH, meta_dir=VELMESHEV_META_DIR):
    """Load Velmeshev in backed mode and return computed metadata."""
    print(f"Reading Velmeshev (backed) from {h5ad_path}...")
    if not os.path.exists(h5ad_path):
        print(f"Error: {h5ad_path} not found.")
        return None, None
    
    adata_backed = sc.read_h5ad(h5ad_path, backed='r')
    
    print("Processing Velmeshev metadata...")
    try:
        # Load sub-metadata TSVs
        ex = pd.read_csv(f"{meta_dir}/ex_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num'}, axis=1)
        inh = pd.read_csv(f"{meta_dir}/in_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num', 'cellId':'Cell_ID'}, axis=1)
        macro = pd.read_csv(f"{meta_dir}/macro_meta.tsv", sep='\t')
        micro = pd.read_csv(f"{meta_dir}/micro_meta.tsv", sep='\t').assign(Cell_Type = 'Microglia')
        
        meta = (pd.concat({
                'Ex': ex, 'In': inh, 'Macro': macro, 'Micro': micro
            })
            .reset_index(level=0).rename(columns={'level_0': 'Cell_Class_Broad'}).set_index('Cell_ID')
            .assign(Age_Years = lambda x: np.select(
                [
                    (x['Age'].str.contains('GW')) & (x['Age_Num'] > 268),
                    (~x['Age'].str.contains('GW')) & (x['Age_Num'] < 268)
                ],
                [-0.01, 0],
                default = (x['Age_Num'] - 268) / 365)
            )
        )
        
        # Lineage mapping
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
        
        # Region mapping
        combined_mapping = {
            'BA10': 'prefrontal cortex', 'BA11': 'prefrontal cortex',
            'BA9': 'prefrontal cortex', 'BA46': 'prefrontal cortex',
            'BA9/46': 'prefrontal cortex', 'BA8': 'prefrontal cortex',
            'PFC': 'prefrontal cortex', 'FC': 'prefrontal cortex',
            'FIC': 'prefrontal cortex',
            'dorsolateral prefrontal cortex': 'prefrontal cortex',
            'DLPFC': 'prefrontal cortex',
            'V1': 'visual cortex', 'primary visual cortex': 'visual cortex',
            'GE': 'telencephalon', 'CGE': 'telencephalon',
            'MGE': 'telencephalon', 'LGE': 'telencephalon',
            'ganglionic eminence': 'telencephalon',
            'BA24': 'cingulate cortex', 'ACC': 'cingulate cortex',
            'Cing': 'cingulate cortex', 'cing': 'cingulate cortex',
            'Primary motor cortex': 'motor cortex', 'BA4': 'motor cortex',
            'BA22': 'temporal cortex', 'STG': 'temporal cortex',
            'temp': 'temporal cortex', 'temporal lobe': 'temporal cortex',
            'S1': 'neocortex', 'BA13': 'neocortex', 'INS': 'neocortex',
            'Frontoparietal cortex': 'prefrontal cortex',
            'Frontoparietal\xa0cortex': 'prefrontal cortex',
            'cortex': 'neocortex', 'cerebral cortex': 'neocortex',
            'visual cortex': 'visual cortex',
        }
        
        # Intersect with h5ad cell barcodes
        common = adata_backed.obs_names.intersection(meta.index)
        if len(common) == 0:
            print("  Warning: No common cell IDs between h5ad and metadata.")
            return None, None
        
        meta = meta.loc[common]
        
        # Build output metadata DataFrame
        meta_df = pd.DataFrame(index=common)
        meta_df['age_years'] = meta['Age_Years']
        meta_df['cell_class'] = meta['cell_class']
        meta_df['sex'] = meta['Sex']
        meta_df['individual'] = meta['Individual'].astype(str)
        meta_df['region'] = meta['Region'].replace(combined_mapping)
        if 'Region_Broad' in meta.columns:
            meta_df['Region_Broad'] = meta['Region_Broad']

        if 'Cell_Type' in meta.columns:
            meta_df['cell_type'] = meta['Cell_Type']
        elif 'cell_type' in meta.columns:
            meta_df['cell_type'] = meta['cell_type']
        if 'cell_type' in adata_backed.obs.columns:
            meta_df['cell_subclass'] = adata_backed.obs.loc[common, 'cell_type'].map(map_velmeshev_subclass)
        if 'Dataset' in meta.columns:
            meta_df['dataset'] = meta['Dataset']
        if 'Chemistry' in meta.columns:
            meta_df['chemistry'] = meta['Chemistry']
            
        print(f"  Velmeshev backed: {adata_backed.shape[0]} cells, {len(meta_df)} have metadata")
        return adata_backed, meta_df
        
    except Exception as e:
        print(f"Error loading Velmeshev TSV metadata: {e}")
        import traceback; traceback.print_exc()
        return None, None


def read_wang_backed(h5ad_path=WANG_PATH):
    """Load Wang in backed mode and return computed metadata."""
    print(f"Reading Wang (backed) from {h5ad_path}...")
    if not os.path.exists(h5ad_path):
         print(f"Error: {h5ad_path} not found.")
         return None, None

    adata_backed = sc.read_h5ad(h5ad_path, backed='r')
    obs = adata_backed.obs
    
    # Build metadata from obs columns (no matrix access needed)
    meta_df = pd.DataFrame(index=obs.index)
    
    # Age
    if 'Estimated_postconceptional_age_in_days' in obs.columns:
        meta_df['age_years'] = (obs['Estimated_postconceptional_age_in_days'] - 268) / 365.0
    
    # Sex
    if 'Sex' in obs.columns:
        meta_df['sex'] = obs['Sex']
    
    # Region
    if 'tissue' in obs.columns:
        region = obs['tissue'].astype(str).str.replace('Brodmann (1909) area ', 'BA')
        region = region.replace({
            'BA17': 'visual cortex',
            'BA10': 'prefrontal cortex',
            'BA9': 'prefrontal cortex',
            'forebrain': 'neocortex'
        })
        meta_df['region'] = region
    
    # Cell Class
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

    if 'cell_type' in obs.columns:
        meta_df['cell_class'] = obs['cell_type'].apply(map_wang_class)
    if 'Type' in obs.columns:
        meta_df['cell_subclass'] = obs['Type'].map(map_wang_subclass)
    
    meta_df['chemistry'] = 'multiome'
    meta_df['dataset'] = 'Wang'
    
    # Individual / donor
    if 'donor_id' in obs.columns:
        meta_df['individual'] = obs['donor_id'].astype(str)
    
    print(f"  Wang backed: {adata_backed.shape[0]} cells")
    return adata_backed, meta_df


def read_psychad_backed(h5ad_path, dataset_name):
    """Load PsychAD (HBCC or Aging) in backed mode and return computed metadata."""
    print(f"Reading {dataset_name} (backed) from {h5ad_path}...")
    if not os.path.exists(h5ad_path):
         print(f"Error: {h5ad_path} not found.")
         return None, None

    adata_backed = sc.read_h5ad(h5ad_path, backed='r')
    obs = adata_backed.obs
    
    # Build metadata from obs columns
    meta_df = pd.DataFrame(index=obs.index)
    
    # Age
    if 'development_stage' in obs.columns:
        dev_stage = obs['development_stage'].astype(str)
        meta_df['age_years'] = dev_stage.apply(extract_age_psychad)
    else:
        meta_df['age_years'] = np.nan
    
    # Sex
    if 'sex' in obs.columns:
        meta_df['sex'] = obs['sex']
    
    # Region
    if 'tissue' in obs.columns:
        meta_df['region'] = obs['tissue'].replace({'dorsolateral prefrontal cortex': 'prefrontal cortex'})
    
    # Cell Class
    mapper = {
        'EN': 'Excitatory', 'IN': 'Inhibitory',
        'Astro': 'Astrocytes', 'Oligo': 'Oligos',
        'Mural': 'Endothelial', 'Endo': 'Endothelial',
        'Immune': 'Microglia', 'OPC': 'OPC'
    }
    if 'class' in obs.columns:
        meta_df['cell_class'] = obs['class'].map(mapper).fillna(obs['class'])
    
    if 'subclass' in obs.columns:
        meta_df['cell_subclass'] = obs['subclass'].map(collapse_en_subclass)
    
    if 'cell_type' in obs.columns:
        meta_df['cell_type'] = obs['cell_type']
    
    # Individual / donor
    if 'individualID' in obs.columns:
        meta_df['individual'] = obs['individualID'].astype(str)
    elif 'donor_id' in obs.columns:
        meta_df['individual'] = obs['donor_id'].astype(str)
    
    meta_df['dataset'] = dataset_name
    meta_df['chemistry'] = 'V3'
    
    print(f"  {dataset_name} backed: {adata_backed.shape[0]} cells")
    return adata_backed, meta_df


# ============================================================================
# MATERIALIZE HELPER
# Apply metadata and load a subset into memory in one step.
# ============================================================================

def materialize_subset(adata_backed, meta_df, mask=None, apply_raw_counts=True):
    """
    Load a subset of a backed AnnData into memory and apply metadata.
    
    Args:
        adata_backed: AnnData in backed='r' mode
        meta_df: DataFrame with computed metadata
        mask: Boolean Series/array for subsetting (None = all cells in meta_df)
        apply_raw_counts: Whether to extract raw counts
    
    Returns:
        AnnData in memory with metadata applied
    """
    if mask is not None:
        indices = meta_df.index[mask]
    else:
        indices = meta_df.index
    
    # Subset backed to only the cells we want, then load into memory
    bool_mask = adata_backed.obs_names.isin(indices)
    print(f"  Loading {bool_mask.sum()} / {adata_backed.shape[0]} cells into memory...")
    adata = adata_backed[bool_mask].to_memory()
    
    if apply_raw_counts:
        adata.X = get_raw_counts(adata)
    
    # Apply metadata columns
    sub_meta = meta_df.loc[adata.obs_names]
    for col in sub_meta.columns:
        adata.obs[col] = sub_meta[col].values
    
    return adata


# ============================================================================
# LEGACY WRAPPERS (for backward compatibility)
# These load data into memory. They call the backed functions internally.
# ============================================================================

def read_velmeshev(h5ad_path=VELMESHEV_PATH, meta_dir=VELMESHEV_META_DIR):
    """Load Velmeshev fully into memory (legacy wrapper)."""
    adata_backed, meta_df = read_velmeshev_backed(h5ad_path, meta_dir)
    if adata_backed is None: return None
    adata = materialize_subset(adata_backed, meta_df)
    del adata_backed
    return adata

def read_wang(h5ad_path=WANG_PATH):
    """Load Wang fully into memory (legacy wrapper)."""
    adata_backed, meta_df = read_wang_backed(h5ad_path)
    if adata_backed is None: return None
    adata = materialize_subset(adata_backed, meta_df)
    del adata_backed
    return adata

def read_psychad(h5ad_path, dataset_name, min_age=0, max_age=40):
    """Load PsychAD fully into memory with optional age filter (legacy wrapper)."""
    adata_backed, meta_df = read_psychad_backed(h5ad_path, dataset_name)
    if adata_backed is None: return None
    
    # Apply age filter mask
    ages = meta_df['age_years']
    if min_age is not None and max_age is not None:
        mask = (ages >= min_age) & (ages < max_age)
        print(f"  Filtering {dataset_name}: {mask.sum()} / {len(mask)} cells (Age {min_age}-{max_age}).")
    elif min_age is not None:
        mask = (ages >= min_age)
        print(f"  Filtering {dataset_name}: {mask.sum()} / {len(mask)} cells (Age >={min_age}).")
    elif max_age is not None:
        mask = (ages < max_age)
        print(f"  Filtering {dataset_name}: {mask.sum()} / {len(mask)} cells (Age <{max_age}).")
    else:
        mask = None
        print(f"  No age filtering for {dataset_name}. {len(meta_df)} cells.")
    
    adata = materialize_subset(adata_backed, meta_df, mask)
    del adata_backed
    return adata
