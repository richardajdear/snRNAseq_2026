
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
PSYCHAD_AGING_PATH = os.path.join(_rds, 'Cam_PsychAD/RNAseq/Aging_Cohort.h5ad')
PSYCHAD_HBCC_PATH  = os.path.join(_rds, 'Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad')
# Legacy aliases (kept for any external scripts that import these names)
AGING_PATH = PSYCHAD_AGING_PATH
HBCC_PATH  = PSYCHAD_HBCC_PATH

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
        meta_df['dataset']    = meta['Dataset'].astype(str) if 'Dataset' in meta.columns else 'VELMESHEV'

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
    meta_df['dataset']   = 'WANG'
    meta_df['chemistry'] = 'multiome'

    if 'donor_id' in obs.columns:
        meta_df['individual'] = obs['donor_id'].astype(str)

    print(f"  Wang backed: {adata_backed.shape[0]} cells")
    return adata_backed, meta_df


def _extract_psychad_meta(obs, cell_type_field='subclass'):
    """Build standardised metadata from a PsychAD obs DataFrame (in memory).

    Used internally by read_psychad_backed. Takes the obs of one h5ad and
    returns a metadata DataFrame with the pipeline's canonical column names.
    source and dataset are both set to 'PSYCHAD'.
    """
    meta = pd.DataFrame(index=obs.index)

    if 'development_stage' in obs.columns:
        meta['age_years'] = obs['development_stage'].astype(str).apply(extract_age_psychad)
    else:
        meta['age_years'] = np.nan

    if 'sex' in obs.columns:
        meta['sex'] = obs['sex']

    if 'tissue' in obs.columns:
        meta['region'] = obs['tissue'].replace(
            {'dorsolateral prefrontal cortex': 'prefrontal cortex'})

    _class_map = {
        'EN': 'Excitatory', 'IN': 'Inhibitory',
        'Astro': 'Astrocytes', 'Oligo': 'Oligos',
        'Mural': 'Endothelial', 'Endo': 'Endothelial',
        'Immune': 'Microglia', 'OPC': 'OPC',
    }
    if 'class' in obs.columns:
        meta['cell_class'] = obs['class'].map(_class_map).fillna(obs['class'])

    if cell_type_field in obs.columns:
        meta['cell_type_raw'] = obs[cell_type_field].astype(str)
    else:
        print(f"  Warning: '{cell_type_field}' not found in PsychAD obs. "
              f"Available: {list(obs.columns)}")
        meta['cell_type_raw'] = 'Unknown'

    for col in ('individualID', 'donor_id', 'individual'):
        if col in obs.columns:
            meta['individual'] = obs[col].astype(str)
            break

    meta['source']    = 'PSYCHAD'
    meta['dataset']   = 'PSYCHAD'
    meta['chemistry'] = 'V3'

    return meta


def read_psychad_backed(aging_path, hbcc_path, cell_type_field='subclass'):
    """Load both PsychAD h5ads in backed mode and deduplicate at the barcode level.

    AGING and HBCC were processed in the same batch and share 899,123 bit-for-bit
    identical cells (same barcodes, same UMI counts). AGING cells are kept as
    primary; any HBCC cell whose barcode already appears in AGING is excluded.
    All retained cells receive source='PSYCHAD' and dataset='PSYCHAD'.

    Args:
        aging_path:       Path to Aging_Cohort.h5ad
        hbcc_path:        Path to HBCC_Cohort.h5ad
        cell_type_field:  obs column for cell_type_raw (default: 'subclass')

    Returns:
        aging_backed     : AnnData (backed='r')
        hbcc_backed      : AnnData (backed='r', full HBCC file — caller must use
                           hbcc_unique_mask to select non-duplicate cells)
        hbcc_unique_mask : bool ndarray, length = len(hbcc_backed),
                           True for HBCC cells whose barcode is NOT in AGING
        meta_df          : combined metadata DataFrame indexed by barcode;
                           AGING rows first, then HBCC-unique rows
    """
    print(f"Reading AGING (backed) from {aging_path}...")
    if not os.path.exists(aging_path):
        print(f"Error: {aging_path} not found.")
        return None, None, None, None

    aging_backed = sc.read_h5ad(aging_path, backed='r')
    aging_meta   = _extract_psychad_meta(aging_backed.obs, cell_type_field)
    print(f"  AGING: {len(aging_meta):,} cells, "
          f"{aging_meta['individual'].nunique()} donors")

    print(f"Reading HBCC (backed) from {hbcc_path}...")
    if not os.path.exists(hbcc_path):
        print(f"Error: {hbcc_path} not found.")
        return None, None, None, None

    hbcc_backed      = sc.read_h5ad(hbcc_path, backed='r')
    aging_barcodes   = set(aging_backed.obs_names)
    hbcc_unique_mask = ~hbcc_backed.obs_names.isin(aging_barcodes)
    n_removed        = int((~hbcc_unique_mask).sum())
    n_kept           = int(hbcc_unique_mask.sum())
    print(f"  HBCC: {len(hbcc_backed):,} total cells")
    print(f"  Dedup: removing {n_removed:,} cells whose barcodes are already in AGING")
    print(f"  HBCC-unique cells retained: {n_kept:,}")

    hbcc_meta = _extract_psychad_meta(
        hbcc_backed.obs[hbcc_unique_mask], cell_type_field)

    meta_df   = pd.concat([aging_meta, hbcc_meta])
    print(f"  Combined PSYCHAD: {len(meta_df):,} cells, "
          f"{meta_df['individual'].nunique()} unique donors")

    return aging_backed, hbcc_backed, hbcc_unique_mask, meta_df


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
