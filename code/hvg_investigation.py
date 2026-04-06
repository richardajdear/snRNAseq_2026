import os
import gc
import scanpy as sc
import pandas as pd
import numpy as np


def _rss_gb():
    """Current process RSS in GB (reads /proc/self/status on Linux)."""
    try:
        with open('/proc/self/status') as _f:
            for _line in _f:
                if _line.startswith('VmRSS:'):
                    return int(_line.split()[1]) / 1e6  # KB → GB
    except Exception:
        pass
    try:
        import resource, sys
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return ru.ru_maxrss / (1e6 if sys.platform == 'linux' else 1e9)
    except Exception:
        return float('nan')


def _log_mem(label):
    print(f"[MEM] {label}: {_rss_gb():.1f} GB RSS", flush=True)


def save_cache(cache_dir, scores_df, stats_df, final_df, hvg_df=None):
    """Save intermediate results to parquet files."""
    os.makedirs(cache_dir, exist_ok=True)
    scores_df.to_parquet(os.path.join(cache_dir, 'scores_df.parquet'))
    stats_df.to_parquet(os.path.join(cache_dir, 'stats_df.parquet'))
    final_df.to_parquet(os.path.join(cache_dir, 'final_df.parquet'))
    if hvg_df is not None:
        hvg_df.to_parquet(os.path.join(cache_dir, 'hvg_df.parquet'))
    print(f"Cache saved to {cache_dir}")


def load_cache(cache_dir, use_cache=True):
    """Load cached results. Returns (scores_df, stats_df, final_df, hvg_df) or None.

    Parameters
    ----------
    cache_dir : str
        Directory containing cached parquet files.
    use_cache : bool, default True
        If False, skip cache loading and return None immediately.
    """
    if not use_cache:
        return None

    files = ['scores_df.parquet', 'stats_df.parquet', 'final_df.parquet', 'hvg_df.parquet']
    paths = [os.path.join(cache_dir, f) for f in files]
    if all(os.path.exists(p) for p in paths):
        scores_df = pd.read_parquet(paths[0])
        stats_df = pd.read_parquet(paths[1])
        final_df = pd.read_parquet(paths[2])
        hvg_df = pd.read_parquet(paths[3])
        print(f"Loaded cache from {cache_dir}")
        return scores_df, stats_df, final_df, hvg_df
    return None


def build_conditions(n_values):
    """Build list of HVG conditions to test."""
    conditions = [{'label': 'all_genes', 'flavor': None, 'n_top_genes': None}]
    for n in n_values:
        conditions.append({'label': f'seurat_v3_{n}', 'flavor': 'seurat_v3', 'n_top_genes': n})
        conditions.append({'label': f'seurat_{n}', 'flavor': 'seurat', 'n_top_genes': n})
        conditions.append({'label': f'pearson_{n}', 'flavor': 'pearson_residuals', 'n_top_genes': n})
    return conditions


def run_hvg_conditions(adata, adata_log, ahba_GRN, conditions, total_grn_genes):
    """Run HVG selection + GRN projection for each condition.

    Parameters
    ----------
    adata : AnnData
        CPM-normalized, with layers['counts'] containing raw counts.
    adata_log : AnnData
        Log-normalized copy of adata (for seurat flavor HVG selection).
    ahba_GRN : DataFrame
        GRN with columns Network, Gene, Importance.
    conditions : list of dict
        From build_conditions().
    total_grn_genes : int
        Number of GRN genes present in adata (baseline reference).

    Returns
    -------
    scores_df : DataFrame
        Melted C3+/C3- scores for all conditions.
    stats_df : DataFrame
        Gene overlap statistics per condition.
    """
    from regulons import project_GRN

    grn_pivot = ahba_GRN.pivot_table(
        index='Network', columns='Gene', values='Importance', fill_value=0)

    all_scores = []
    gene_stats = []
    all_hvg_genes = []

    for cond in conditions:
        label = cond['label']
        _log_mem(f"run_hvg_conditions: start '{label}'")
        print(f"\n{'='*60}\nCondition: {label}")

        if cond['flavor'] is None:
            if 'highly_variable' in adata.var.columns:
                del adata.var['highly_variable']
            project_GRN(adata, ahba_GRN, 'X_ahba',
                        use_highly_variable=False, log_transform=False)
            n_hvg = adata.shape[1]
            n_grn_used = total_grn_genes
        else:
            if cond['flavor'] == 'seurat_v3':
                sc.pp.highly_variable_genes(
                    adata, flavor='seurat_v3',
                    n_top_genes=cond['n_top_genes'], layer='counts')
            elif cond['flavor'] == 'seurat':
                sc.pp.highly_variable_genes(
                    adata_log, flavor='seurat',
                    n_top_genes=cond['n_top_genes'])
                adata.var['highly_variable'] = adata_log.var['highly_variable']
            elif cond['flavor'] == 'pearson_residuals':
                sc.experimental.pp.highly_variable_genes(
                    adata, n_top_genes=cond['n_top_genes'], layer='counts')

            n_hvg = int(adata.var['highly_variable'].sum())
            hvg_genes = adata.var_names[adata.var['highly_variable']]
            n_grn_used = len(np.intersect1d(grn_pivot.columns, hvg_genes))

            # Record HVG gene list for Euler overlap analysis
            for g in hvg_genes:
                all_hvg_genes.append({'condition': label, 'gene': g})

            project_GRN(adata, ahba_GRN, 'X_ahba',
                        use_highly_variable=True, log_transform=False)

        gene_stats.append({
            'condition': label,
            'n_hvg': n_hvg,
            'n_grn_genes_used': n_grn_used,
            'pct_grn_retained': round(100 * n_grn_used / total_grn_genes, 1)
        })

        proj = pd.DataFrame(
            adata.obsm['X_ahba'], index=adata.obs_names,
            columns=adata.uns['X_ahba_names'])
        proj = proj[['C3+', 'C3-']].copy()
        proj['obs_names'] = proj.index
        proj['condition'] = label
        melted = proj.melt(
            id_vars=['obs_names', 'condition'], var_name='C', value_name='value')
        all_scores.append(melted)

        print(f"  HVGs: {n_hvg}, GRN genes used: {n_grn_used}/{total_grn_genes} "
              f"({gene_stats[-1]['pct_grn_retained']}%)")

    scores_df = pd.concat(all_scores, ignore_index=True)
    stats_df = pd.DataFrame(gene_stats)
    hvg_df = (pd.DataFrame(all_hvg_genes) if all_hvg_genes
              else pd.DataFrame(columns=['condition', 'gene']))
    return scores_df, stats_df, hvg_df


def prepare_for_r(scores_df, adata, n_values):
    """Merge scores with metadata, filter to excitatory, set condition ordering."""
    cols_to_keep = ['individual', 'age_years', 'cell_class', 'cell_subclass',
                    'cell_type', 'cell_type_aligned', 'source']
    # Map donor_id → individual if individual is absent
    if 'individual' not in adata.obs.columns and 'donor_id' in adata.obs.columns:
        adata.obs['individual'] = adata.obs['donor_id']
    cols_to_keep = [c for c in cols_to_keep if c in adata.obs.columns]

    meta = adata.obs[cols_to_keep].copy()
    meta['obs_names'] = meta.index

    final_df = pd.merge(scores_df, meta, on='obs_names')
    final_df = final_df[final_df['cell_class'] == 'Excitatory']

    condition_order = (['all_genes'] +
                       [f'seurat_v3_{n}' for n in n_values] +
                       [f'seurat_{n}' for n in n_values] +
                       [f'pearson_{n}' for n in n_values])
    final_df['condition'] = pd.Categorical(
        final_df['condition'], categories=condition_order, ordered=True)

    return final_df


# ── High-level helpers for notebook data loading and pipeline execution ───────

def setup_grn(ref_dir, adata):
    """Load the AHBA GRN and remap gene symbols to Ensembl IDs in adata.

    Returns
    -------
    ahba_GRN : DataFrame
        Gene-level GRN with columns Network, Gene (Ensembl), Importance.
    total_grn_genes : int
        Number of GRN genes present in adata.
    """
    from regulons import get_ahba_GRN
    from gene_mapping import map_grn_symbols_to_ensembl
    grn_file = os.path.join(ref_dir, 'ahba_dme_hcp_top8kgenes_weights.csv')
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file, use_weights=True)
    ahba_GRN = map_grn_symbols_to_ensembl(ahba_GRN, adata)
    grn_pivot = ahba_GRN.pivot_table(
        index='Network', columns='Gene', values='Importance', fill_value=0)
    total_grn_genes = len(np.intersect1d(grn_pivot.columns, adata.var_names))
    print(f'GRN genes in adata: {total_grn_genes} / {grn_pivot.shape[1]}')
    return ahba_GRN, total_grn_genes


def load_single_raw(data_file, source_label='combined'):
    """Load a single h5ad file with raw-count normalization.

    Stores raw counts in layers['counts'], CPM-normalizes adata.X, and
    returns a log-normalized copy for seurat-flavor HVG selection.

    Returns
    -------
    adata : AnnData  (CPM-normalized raw counts)
    adata_log : AnnData  (log1p of CPM)
    """
    adata = sc.read_h5ad(data_file)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6)
    if 'source' not in adata.obs.columns:
        adata.obs['source'] = source_label
    print(f'Shape: {adata.shape}')
    adata_log = adata.copy()
    sc.pp.log1p(adata_log)
    return adata, adata_log


def load_combo_raw(aging_file, hbcc_file):
    """Load and concatenate Aging + HBCC PsychAD cohorts with raw-count normalization.

    Returns
    -------
    adata : AnnData  (CPM-normalized raw counts)
    adata_log : AnnData  (log1p of CPM)
    """
    import gc
    aging = sc.read_h5ad(aging_file)
    hbcc  = sc.read_h5ad(hbcc_file)
    aging.obs['source'] = 'Aging'
    hbcc.obs['source']  = 'HBCC'
    print(f'Aging shape: {aging.shape}, HBCC shape: {hbcc.shape}')
    # sc.concat drops var columns, so save and restore them
    var_df = aging.var.copy()
    adata = sc.concat([aging, hbcc], keys=['Aging', 'HBCC'], index_unique='-',
                      join='outer', fill_value=0)
    adata.var = var_df.loc[adata.var_names]
    del aging, hbcc
    gc.collect()
    print(f'Combined shape: {adata.shape}')
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6)
    adata_log = adata.copy()
    sc.pp.log1p(adata_log)
    return adata, adata_log


def load_single_scvi(data_file, source_label='combined',
                     scvi_layer='scvi_normalized'):
    """Load a single h5ad that contains scVI-normalized expression.

    Moves scvi_layer into adata.X *before* normalize_total so that both
    HVG selection (via layers['counts']) and the GRN projection (which reads
    adata.X directly) operate on the same batch-corrected expression.

    Returns
    -------
    adata : AnnData  (CPM-scaled scvi_layer)
    adata_log : AnnData  (log1p of CPM-scaled scvi_layer)
    """
    _log_mem("load_single_scvi: start")
    adata = sc.read_h5ad(data_file)
    _log_mem(f"load_single_scvi: after read_h5ad (shape {adata.shape}, "
             f"layers: {list(adata.layers.keys())})")

    # Drop every layer except the one we need, and clear obsm to save memory.
    for layer in list(adata.layers.keys()):
        if layer != scvi_layer:
            del adata.layers[layer]
    for key in list(adata.obsm.keys()):
        del adata.obsm[key]
    gc.collect()
    _log_mem("load_single_scvi: after dropping extra layers/obsm")

    # Set X from the layer then immediately remove the layer reference so we
    # do not hold two copies of the same array.
    adata.X = adata.layers[scvi_layer]
    del adata.layers[scvi_layer]
    gc.collect()
    _log_mem("load_single_scvi: after setting X and deleting source layer")

    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6)
    _log_mem("load_single_scvi: after normalize_total")

    if 'source' not in adata.obs.columns:
        adata.obs['source'] = source_label
    print(f'Shape: {adata.shape}')
    adata_log = adata.copy()
    sc.pp.log1p(adata_log)
    _log_mem("load_single_scvi: after adata_log copy")
    return adata, adata_log


def run_projection_pipeline(adata, adata_log, ahba_GRN, total_grn_genes,
                             N_VALUES, CACHE_DIR):
    """Run HVG conditions, prepare R data, cache, and return all four dataframes.

    Combines build_conditions → run_hvg_conditions → prepare_for_r →
    save_cache into a single call for use in notebooks.

    Returns
    -------
    scores_df, stats_df, final_df, hvg_df
    """
    _log_mem("run_projection_pipeline: start")
    conditions = build_conditions(N_VALUES)
    scores_df, stats_df, hvg_df = run_hvg_conditions(
        adata, adata_log, ahba_GRN, conditions, total_grn_genes)
    _log_mem("run_projection_pipeline: after run_hvg_conditions")
    final_df = prepare_for_r(scores_df, adata, N_VALUES)
    save_cache(CACHE_DIR, scores_df, stats_df, final_df, hvg_df)
    _log_mem("run_projection_pipeline: done")
    return scores_df, stats_df, final_df, hvg_df
