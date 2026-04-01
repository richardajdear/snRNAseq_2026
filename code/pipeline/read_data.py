
import scanpy as sc
import pandas as pd
import numpy as np
import os
import re

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


# ============================================================================
# BACKED-MODE READERS
# These functions return (adata_backed, meta_df) where:
#   - adata_backed: AnnData in backed='r' mode (expression matrix stays on disk)
#   - meta_df: DataFrame with computed metadata columns indexed by cell barcode
#
# Output columns: age_years, cell_class, cell_type_raw, sex, individual,
#                 region, source, chemistry
#
# cell_class  — broad lineage aligned across datasets (Excitatory, Inhibitory,
#               Astrocytes, Oligos, OPC, Microglia, Endothelial, Glia, Other)
# cell_type_raw — raw source-specific fine-grained label; used as input to
#               label transfer to produce cell_type_aligned
# ============================================================================

def read_velmeshev_backed(h5ad_path=VELMESHEV_PATH, meta_dir=VELMESHEV_META_DIR,
                          cell_type_field='Cell_Type'):
    """Load Velmeshev in backed mode and return computed metadata.

    Args:
        cell_type_field: Column in the TSV metadata to use as cell_type_raw.
                         Defaults to 'Cell_Type' (the Velmeshev-specific label).
    """
    print(f"Reading Velmeshev (backed) from {h5ad_path}...")
    if not os.path.exists(h5ad_path):
        print(f"Error: {h5ad_path} not found.")
        return None, None

    adata_backed = sc.read_h5ad(h5ad_path, backed='r')

    print("Processing Velmeshev metadata...")
    try:
        # Load sub-metadata TSVs
        ex    = pd.read_csv(f"{meta_dir}/ex_meta.tsv",    sep='\t', low_memory=False).rename({'Age_(days)': 'Age_Num'}, axis=1)
        inh   = pd.read_csv(f"{meta_dir}/in_meta.tsv",    sep='\t', low_memory=False).rename({'Age_(days)': 'Age_Num', 'cellId': 'Cell_ID'}, axis=1)
        macro = pd.read_csv(f"{meta_dir}/macro_meta.tsv", sep='\t')
        micro = pd.read_csv(f"{meta_dir}/micro_meta.tsv", sep='\t').assign(Cell_Type='Microglia')

        meta = (pd.concat({'Ex': ex, 'In': inh, 'Macro': macro, 'Micro': micro})
                .reset_index(level=0).rename(columns={'level_0': 'Cell_Class_Broad'})
                .set_index('Cell_ID')
                .assign(Age_Years=lambda x: np.select(
                    [
                        (x['Age'].str.contains('GW')) & (x['Age_Num'] > 268),
                        (~x['Age'].str.contains('GW')) & (x['Age_Num'] < 268),
                    ],
                    [-0.01, 0],
                    default=(x['Age_Num'] - 268) / 365))
                )

        # Lineage mapping → cell_class
        def _map_velmeshev_lineage(row):
            cls = row['Cell_Class_Broad']
            if cls == 'Ex':   return 'Excitatory'
            if cls == 'In':   return 'Inhibitory'
            if cls == 'Micro': return 'Microglia'
            if cls == 'Macro':
                st = str(row.get('subclass', '')).lower()
                ct = str(row.get('Cell_Type', '')).lower()
                if 'astro'  in st or 'astro'  in ct: return 'Astrocytes'
                if 'oligo'  in st or 'oligo'  in ct: return 'Oligos'
                if 'opc'    in st or 'opc'    in ct: return 'OPC'
                if 'endo'   in st or 'endo'   in ct: return 'Endothelial'
                if 'immune' in st or 'immune' in ct: return 'Microglia'
                return 'Glia'
            return 'Other'

        meta['cell_class'] = meta.apply(_map_velmeshev_lineage, axis=1)

        # Region mapping
        _region_map = {
            'BA10': 'prefrontal cortex',  'BA11': 'prefrontal cortex',
            'BA9': 'prefrontal cortex',   'BA46': 'prefrontal cortex',
            'BA9/46': 'prefrontal cortex','BA8': 'prefrontal cortex',
            'PFC': 'prefrontal cortex',   'FC': 'prefrontal cortex',
            'FIC': 'prefrontal cortex',
            'dorsolateral prefrontal cortex': 'prefrontal cortex',
            'DLPFC': 'prefrontal cortex',
            'V1': 'visual cortex', 'primary visual cortex': 'visual cortex',
            'BA17': 'visual cortex',
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
        meta_df['age_years']  = meta['Age_Years']
        meta_df['cell_class'] = meta['cell_class']
        meta_df['sex']        = meta['Sex']
        meta_df['individual'] = meta['Individual'].astype(str)
        meta_df['region']     = meta['Region'].replace(_region_map)
        meta_df['source']     = 'VELMESHEV'

        # cell_type_raw: source-specific fine-grained label
        raw_col = cell_type_field if cell_type_field in meta.columns else 'Cell_Type'
        if raw_col in meta.columns:
            meta_df['cell_type_raw'] = meta[raw_col].astype(str)
        else:
            meta_df['cell_type_raw'] = 'Unknown'

        if 'Chemistry' in meta.columns:
            meta_df['chemistry'] = meta['Chemistry']
        else:
            meta_df['chemistry'] = 'unknown'

        print(f"  Velmeshev backed: {adata_backed.shape[0]} cells, {len(meta_df)} have metadata")
        return adata_backed, meta_df

    except Exception as e:
        print(f"Error loading Velmeshev TSV metadata: {e}")
        import traceback; traceback.print_exc()
        return None, None


def read_wang_backed(h5ad_path=WANG_PATH, cell_type_field='Type-updated'):
    """Load Wang in backed mode and return computed metadata.

    Args:
        cell_type_field: obs column to use as cell_type_raw.
                         Defaults to 'Type-updated' (Wang's curated cell type labels).
    """
    print(f"Reading Wang (backed) from {h5ad_path}...")
    if not os.path.exists(h5ad_path):
        print(f"Error: {h5ad_path} not found.")
        return None, None

    adata_backed = sc.read_h5ad(h5ad_path, backed='r')
    obs = adata_backed.obs

    meta_df = pd.DataFrame(index=obs.index)

    # Age
    if 'Estimated_postconceptional_age_in_days' in obs.columns:
        meta_df['age_years'] = (obs['Estimated_postconceptional_age_in_days'] - 268) / 365.0

    # Sex
    if 'Sex' in obs.columns:
        meta_df['sex'] = obs['Sex']

    # Region
    if 'tissue' in obs.columns:
        region = obs['tissue'].astype(str).str.replace('Brodmann (1909) area ', 'BA', regex=False)
        region = region.replace({
            'BA17': 'visual cortex',
            'BA10': 'prefrontal cortex',
            'BA9':  'prefrontal cortex',
            'forebrain': 'neocortex',
        })
        meta_df['region'] = region

    # Cell class: derived from Type-updated (same field as cell_type_raw) using
    # prefix-based logic consistent with Wang's own label vocabulary.
    def _map_wang_class(s):
        s = str(s)
        if s.startswith('EN-'):          return 'Excitatory'
        if s.startswith('IN-'):          return 'Inhibitory'
        if s.startswith('Astrocyte'):    return 'Astrocytes'
        if s.startswith('Oligodendro'): return 'Oligos'
        if s == 'OPC':                   return 'OPC'
        if s == 'Microglia':             return 'Microglia'
        if s == 'Vascular':              return 'Endothelial'
        if s.startswith('IPC-') or s.startswith('RG-'): return 'Glia'
        return 'Other'

    if cell_type_field in obs.columns:
        meta_df['cell_class'] = obs[cell_type_field].apply(_map_wang_class)

    # cell_type_raw: Wang's curated label (Type-updated)
    if cell_type_field in obs.columns:
        meta_df['cell_type_raw'] = obs[cell_type_field].astype(str)
    else:
        print(f"  Warning: '{cell_type_field}' not found in Wang obs. Available: {list(obs.columns)}")
        meta_df['cell_type_raw'] = 'Unknown'

    meta_df['source']    = 'WANG'
    meta_df['chemistry'] = 'multiome'

    if 'donor_id' in obs.columns:
        meta_df['individual'] = obs['donor_id'].astype(str)

    print(f"  Wang backed: {adata_backed.shape[0]} cells")
    return adata_backed, meta_df


def read_psychad_backed(h5ad_path, dataset_name, cell_type_field='subclass'):
    """Load PsychAD (HBCC or Aging) in backed mode and return computed metadata.

    Args:
        dataset_name: 'AGING' or 'HBCC'
        cell_type_field: obs column to use as cell_type_raw. Defaults to 'subclass'.
    """
    print(f"Reading {dataset_name} (backed) from {h5ad_path}...")
    if not os.path.exists(h5ad_path):
        print(f"Error: {h5ad_path} not found.")
        return None, None

    adata_backed = sc.read_h5ad(h5ad_path, backed='r')
    obs = adata_backed.obs

    meta_df = pd.DataFrame(index=obs.index)

    # Age
    if 'development_stage' in obs.columns:
        meta_df['age_years'] = obs['development_stage'].astype(str).apply(extract_age_psychad)
    else:
        meta_df['age_years'] = np.nan

    # Sex
    if 'sex' in obs.columns:
        meta_df['sex'] = obs['sex']

    # Region
    if 'tissue' in obs.columns:
        meta_df['region'] = obs['tissue'].replace({'dorsolateral prefrontal cortex': 'prefrontal cortex'})

    # Cell class
    _class_map = {
        'EN': 'Excitatory', 'IN': 'Inhibitory',
        'Astro': 'Astrocytes', 'Oligo': 'Oligos',
        'Mural': 'Endothelial', 'Endo': 'Endothelial',
        'Immune': 'Microglia', 'OPC': 'OPC',
    }
    if 'class' in obs.columns:
        meta_df['cell_class'] = obs['class'].map(_class_map).fillna(obs['class'])

    # cell_type_raw: source-specific fine-grained label
    if cell_type_field in obs.columns:
        meta_df['cell_type_raw'] = obs[cell_type_field].astype(str)
    else:
        print(f"  Warning: '{cell_type_field}' not found in {dataset_name} obs. Available: {list(obs.columns)}")
        meta_df['cell_type_raw'] = 'Unknown'

    # Individual / donor
    if 'individualID' in obs.columns:
        meta_df['individual'] = obs['individualID'].astype(str)
    elif 'donor_id' in obs.columns:
        meta_df['individual'] = obs['donor_id'].astype(str)

    meta_df['source']    = dataset_name
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

    bool_mask = adata_backed.obs_names.isin(indices)
    print(f"  Loading {bool_mask.sum()} / {adata_backed.shape[0]} cells into memory...")
    adata = adata_backed[bool_mask].to_memory()

    if apply_raw_counts:
        adata.X = get_raw_counts(adata)

    sub_meta = meta_df.loc[adata.obs_names]
    for col in sub_meta.columns:
        adata.obs[col] = sub_meta[col].values

    return adata
