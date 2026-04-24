"""Diagnostics for scANVI-based label transfer.

Reads integrated.h5ad (which contains cell_type_aligned and
cell_type_aligned_confidence from model.predict()), constructs DataFrames
compatible with the existing label_transfer/diagnostics.py infrastructure,
and adds scANVI-specific validation plots.

Usage
-----
    PYTHONPATH=code python3 -m pipeline.scanvi_diagnostics \
        --input  path/to/scvi_output/integrated.h5ad \
        --output_dir  path/to/scanvi_diagnostics/ \
        --confidence_threshold 0.5

Outputs (per target source, in output_dir/{SOURCE}/)
-------
    remapping_crosstab.csv
    confidence_summary.csv
    class_remapping_table.csv
    confidence_histogram.png
    umap_target.png
    sankey_remapping.png
    marker_validation*.png

Plus in output_dir/wang_selfcheck/
    wang_confusion_matrix.png   — cell_type_for_scanvi vs cell_type_aligned (WANG cells)
    subtype_distribution.png    — cell_type_aligned proportions per dataset (EN + IN)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pipeline.label_transfer.transfer import aligned_to_class
from pipeline.label_transfer import diagnostics as diag


def _compute_inferred_umap(adata, all_df, n_pcs=50):
    """Add inferred_umap_1/2 to all_df via PCA on scanvi_normalized layer.

    PCA is computed on highly-variable genes (adata.var['highly_variable'] if
    available, otherwise all genes).  Falls back gracefully if the layer is
    absent or X_umap_inferred is already stored in obsm.

    Returns all_df with columns added in-place (or unchanged on failure).
    """
    if 'X_umap_inferred' in adata.obsm:
        print("  Using pre-computed X_umap_inferred from obsm …")
        umap = np.array(adata.obsm['X_umap_inferred'])
        all_df['inferred_umap_1'] = umap[:, 0]
        all_df['inferred_umap_2'] = umap[:, 1]
        return all_df

    if 'scanvi_normalized' not in adata.layers:
        print("  scanvi_normalized layer not found — skipping inferred UMAP.")
        return all_df

    try:
        from sklearn.decomposition import PCA
        from umap import UMAP as UMAPReducer
    except ImportError as e:
        print(f"  sklearn / umap-learn not available ({e}) — skipping inferred UMAP.")
        return all_df

    try:
        # Determine gene subset
        if 'highly_variable' in adata.var.columns:
            hvg_mask = adata.var['highly_variable'].values
            print(f"  Inferred UMAP: loading scanvi_normalized for "
                  f"{hvg_mask.sum():,} HVGs × {adata.n_obs:,} cells …")
        else:
            hvg_mask = np.ones(adata.n_vars, dtype=bool)
            print(f"  Inferred UMAP: loading scanvi_normalized "
                  f"({adata.n_obs:,} cells × all {adata.n_vars:,} genes) …")

        hvg_idx = np.where(hvg_mask)[0]
        import scipy.sparse as sp
        X = adata.layers['scanvi_normalized'][:, hvg_idx]
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        print(f"  Running PCA (n_components={n_pcs}) …")
        pcs = PCA(n_components=min(n_pcs, X.shape[1] - 1)).fit_transform(X)

        print(f"  Running UMAP on PCA …")
        coords = UMAPReducer(n_neighbors=30, min_dist=0.3,
                             random_state=42).fit_transform(pcs)

        all_df['inferred_umap_1'] = coords[:, 0]
        all_df['inferred_umap_2'] = coords[:, 1]
        print("  Inferred UMAP computed and stored in all_df.")

    except Exception as e:
        print(f"  Warning: inferred UMAP computation failed ({e}) — skipping.")

    return all_df


def _build_dataframes(adata, umap_key='X_umap_scanvi', confidence_threshold=0.5,
                      include_wang_target=False):
    """Build all_df and tf_df compatible with diagnostics.py from integrated.h5ad.

    all_df : all cells with UMAP coordinates and both old/new labels
    tf_df  : target (non-WANG) cells only, with kNN-compatible column names
    """
    obs = adata.obs.copy()

    # UMAP coordinates — fall back to scVI UMAP if scanvi not present
    if umap_key in adata.obsm:
        umap = np.array(adata.obsm[umap_key])
    elif 'X_umap_scvi' in adata.obsm:
        umap = np.array(adata.obsm['X_umap_scvi'])
        print(f"  Warning: {umap_key} not found, using X_umap_scvi")
    else:
        raise ValueError(
            f"Neither {umap_key} nor X_umap_scvi found in adata.obsm. "
            "Run UMAP step first."
        )

    obs['umap_1'] = umap[:, 0]
    obs['umap_2'] = umap[:, 1]
    obs['h5ad_pos'] = np.arange(len(obs))

    # Derive old/new class from cell_type_aligned
    obs['new_cell_class'] = obs['cell_type_aligned'].map(aligned_to_class)
    # Use cell_class_original (pre-transfer) when available; fall back to cell_class
    # for backwards compatibility with h5ad files created before this column was added.
    if 'cell_class_original' in obs.columns:
        obs['old_cell_class'] = obs['cell_class_original']
    else:
        obs['old_cell_class'] = obs['cell_class']  # broad class from original data

    obs['is_class_remapped'] = obs['old_cell_class'] != obs['new_cell_class']
    obs['is_low_confidence'] = obs['cell_type_aligned_confidence'] < confidence_threshold

    # Rename for compatibility with diagnostics.py
    obs = obs.rename(columns={
        'cell_type_aligned_confidence': 'transfer_confidence',
    })
    obs['mean_knn_distance'] = np.nan  # not applicable for scANVI

    all_df = obs.copy()

    # tf_df: target cells that received predictions.
    # Default behavior excludes WANG since it is the reference dataset.
    if include_wang_target:
        tf_df = obs.copy()
    else:
        tf_df = obs[obs['source'] != 'WANG'].copy()

    return all_df, tf_df


# ══════════════════════════════════════════════════════════════════════════════
# NEW: WANG SELF-PREDICTION CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def _ct_sort_key(label):
    """Biological sort key: groups cell types by broad class for a readable matrix."""
    s = str(label)
    prefixes = [
        ('RG-',    0), ('RG_',    0),
        ('IPC-',   1), ('IPC_',   1), ('Tri-', 1),
        ('EN-',    2), ('EN_',    2),
        ('Cajal',  3), ('CR',     3),
        ('IN-',    4), ('IN_',    4),
        ('Astro',  5),
        ('Oligo',  6),
        ('OPC',    7),
        ('Micro',  8),
        ('Endo',   9), ('Vascular', 9), ('PC', 9), ('SMC', 9), ('VLMC', 9),
    ]
    for prefix, order in prefixes:
        if s.startswith(prefix):
            return (order, s)
    return (99, s)


def make_wang_confusion_matrix(adata, out):
    """Confusion matrix: cell_type_for_scanvi vs cell_type_aligned for WANG cells.

    This is the primary model quality check — if scANVI cannot recover the
    labels it was trained on, predictions for other datasets are unreliable.

    Rows   = cell_type_for_scanvi (the training label given to scANVI).
    Cols   = cell_type_aligned    (scANVI's prediction after self-prediction).
    Values = fraction of cells with that training label that received each
             predicted label (row-normalised, so each row sums to 1).

    Diagonal cells (correct prediction) are greyed out so the colour scale
    can focus on the off-diagonal errors.  Off-diagonal cells with non-zero
    misassignment fractions are annotated with their value.

    Rows and columns share the same biological ordering so the diagonal is
    visually unambiguous.
    """
    obs = adata.obs
    wang = obs[obs['source'] == 'WANG'].copy()

    if 'cell_type_for_scanvi' not in wang.columns:
        print("  cell_type_for_scanvi not found in WANG obs — skipping confusion matrix.")
        return

    labeled = wang[wang['cell_type_for_scanvi'] != 'Unknown'].copy()
    if labeled.empty:
        print("  No labeled WANG cells found — skipping confusion matrix.")
        return

    ct = pd.crosstab(
        labeled['cell_type_for_scanvi'],
        labeled['cell_type_aligned'],
        normalize='index',
    )
    ct.to_csv(os.path.join(out, 'wang_confusion_matrix.csv'))

    # Build a common label set with the same biological ordering for both axes.
    all_labels = sorted(
        set(labeled['cell_type_for_scanvi'].unique()) |
        set(labeled['cell_type_aligned'].unique()),
        key=_ct_sort_key,
    )
    ct = ct.reindex(index=all_labels, columns=all_labels, fill_value=0)

    mat = ct.values.astype(float)
    n_types = len(all_labels)

    # Compute diagonal accuracy (using the square matrix where row==col labels).
    diag_vals = np.diag(mat)
    diag_acc = diag_vals.mean()

    # Off-diagonal matrix: set diagonal to NaN so the colormap's "bad" colour
    # (light grey) is used there, leaving the colour scale for errors only.
    off_diag = mat.copy()
    np.fill_diagonal(off_diag, np.nan)

    # Cap the colour scale at the 99th percentile of non-zero off-diagonal
    # values to prevent a single large outlier from compressing the scale.
    nonzero_off = off_diag[~np.isnan(off_diag) & (off_diag > 0)]
    vmax = float(np.percentile(nonzero_off, 99)) if nonzero_off.size else 0.1
    vmax = max(vmax, 0.05)   # at least 5 % so single-cell rounding errors show

    cmap = plt.cm.Oranges.copy()
    cmap.set_bad('#D8D8D8')   # diagonal shown in light grey

    # Compact figure: smaller overall size but larger fonts for readability.
    # ~0.28" per cell gives enough room for 8pt tick labels.
    fig_size = max(8, min(12, n_types * 0.28))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.92))

    im = ax.imshow(off_diag, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, label='Fraction of row misassigned to this column',
                        shrink=0.55, pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    # Annotate off-diagonal cells where fraction > 1 % (shown as %)
    annotation_threshold = 0.01
    fontsize_annot = 8
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            val = mat[i, j]
            if val >= annotation_threshold:
                normed = val / vmax
                text_color = 'white' if normed > 0.6 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=fontsize_annot, color=text_color, fontweight='bold')

    ax.set_xticks(range(n_types))
    ax.set_yticks(range(n_types))
    ax.set_xticklabels(all_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.set_xlabel('cell_type_aligned (scANVI prediction)', fontsize=11)
    ax.set_ylabel('cell_type_for_scanvi (training label)', fontsize=11)

    ax.set_title(
        f'WANG self-prediction: training label vs scANVI prediction\n'
        f'n={len(labeled):,} WANG cells   '
        f'mean diagonal accuracy = {diag_acc:.3f}\n'
        f'Diagonal greyed out; colour scale shows off-diagonal misassignment fractions',
        fontsize=10,
    )
    plt.tight_layout()
    path = os.path.join(out, 'wang_confusion_matrix.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  wang_confusion_matrix.png  (mean diagonal={diag_acc:.3f}, "
          f"n={len(labeled):,} WANG cells)")


# ══════════════════════════════════════════════════════════════════════════════
# NEW: CROSS-DATASET SUBTYPE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def make_subtype_distribution(all_df, out):
    """Single 4-panel figure: EN and IN neuron subtype proportions per dataset.

    Layout (rows = period, cols = neuron class):
      Row 0 (top)   : postnatal (age ≥ 0) — all four datasets
      Row 1 (bottom): prenatal  (age < 0) — WANG + VELMESHEV only

    Colours match the fixed EN/IN palettes used in umap_excitatory and
    umap_inhibitory (diag.get_en_palette() / diag.get_in_palette()), so the
    same subtype always has the same colour across every diagnostic plot.

    Each stacked bar shows what fraction of cells in a dataset are assigned
    to each subtype by scANVI.  If the model transfers labels correctly the
    proportions should be broadly consistent with the WANG reference.
    """
    en_pal = diag.get_en_palette()
    in_pal = diag.get_in_palette()

    periods = [
        ('postnatal (age ≥ 0y)',
         all_df[all_df['age_years'] >= 0].copy(),
         'all datasets'),
        ('prenatal (age < 0y)',
         all_df[(all_df['age_years'] < 0) &
                (all_df['source'].isin(['WANG', 'VELMESHEV']))].copy(),
         'WANG + VELMESHEV'),
    ]
    broad_classes = [
        ('Excitatory', 'EN', en_pal),
        ('Inhibitory',  'IN',  in_pal),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    for row, (period_label, df_period, src_note) in enumerate(periods):
        for col, (broad_class, class_label, pal) in enumerate(broad_classes):
            ax = axes[row, col]
            sub = df_period[df_period['cell_class'] == broad_class].copy()

            if sub.empty:
                ax.set_title(f'{class_label} — {period_label}\n(no cells)')
                ax.axis('off')
                continue

            props = (sub.groupby(['source', 'cell_type_aligned'])
                     .size().unstack(fill_value=0))
            props = props.div(props.sum(axis=1), axis=0)

            sources = ['WANG'] + sorted([s for s in props.index if s != 'WANG'])
            props = props.reindex([s for s in sources if s in props.index])

            # Sort columns in biological order using the appropriate sort key
            sort_fn = diag._en_sort_key if broad_class == 'Excitatory' \
                else diag._in_sort_key
            col_order = sorted(props.columns.tolist(), key=sort_fn)
            props = props[col_order]

            src_counts = sub.groupby('source').size()

            bottom = np.zeros(len(props))
            for ct in props.columns:
                color = pal.get(ct, diag._FALLBACK)
                vals = props[ct].values
                ax.bar(range(len(props)), vals, bottom=bottom,
                       color=color, edgecolor='white', linewidth=0.4,
                       label=ct, width=0.7)
                bottom += vals

            ax.set_xticks(range(len(props)))
            ax.set_xticklabels(
                [f'{s}\n(n={src_counts.get(s, 0):,})' for s in props.index],
                fontsize=9)
            ax.set_ylabel('Proportion of cells', fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title(
                f'{broad_class} ({class_label}) — {period_label}\n({src_note})',
                fontsize=10)

            # Single-column legend; skip subtypes <1 % everywhere
            max_prop = props.max(axis=0)
            legend_types = max_prop[max_prop >= 0.01].index.tolist()
            legend_types = sorted(legend_types, key=sort_fn)
            handles = [plt.Rectangle((0, 0), 1, 1,
                                     color=pal.get(c, diag._FALLBACK), label=c)
                       for c in legend_types]
            n_hidden = len(props.columns) - len(legend_types)
            if n_hidden > 0:
                handles.append(plt.Rectangle((0, 0), 1, 1, color='#cccccc',
                                             label=f'({n_hidden} subtypes <1%)'))
            ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left',
                      fontsize=7.5, ncol=1, framealpha=0.9,
                      handlelength=1.2, handleheight=0.9)

    fig.suptitle(
        'Neuron subtype proportions per dataset (scANVI cell_type_aligned)\n'
        'Top: postnatal | Bottom: prenatal    —    '
        'Colours match umap_excitatory / umap_inhibitory',
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    fname = 'subtype_distribution.png'
    plt.savefig(os.path.join(out, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='scANVI label transfer diagnostics')
    p.add_argument('--input', required=True,
                   help='integrated.h5ad from scVI/scANVI pipeline')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--confidence_threshold', type=float, default=0.5)
    p.add_argument('--umap_key', default='X_umap_scanvi')
    p.add_argument('--embedding_key', default='X_scANVI',
                   help='Embedding key for per-class UMAP recomputation')
    p.add_argument('--include_wang_target', action='store_true',
                   help='Also generate full per-source diagnostics for WANG')
    p.add_argument('--skip_inferred_umap', action='store_true',
                   help='Skip PCA+UMAP on scanvi_normalized (saves memory)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Validate that required obs columns are present
    required_cols = ['cell_type_aligned', 'cell_type_aligned_confidence',
                     'source', 'cell_class', 'cell_type_raw', 'age_years']
    print(f"Loading {args.input} (backed) …")
    adata = sc.read_h5ad(args.input, backed='r')
    print(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    missing = [c for c in required_cols if c not in adata.obs.columns]
    if missing:
        print(f"ERROR: Missing obs columns: {missing}")
        print(f"  Available: {sorted(adata.obs.columns.tolist())}")
        sys.exit(1)

    print("\nBuilding diagnostic DataFrames …")
    all_df, tf_df = _build_dataframes(
        adata, umap_key=args.umap_key,
        confidence_threshold=args.confidence_threshold,
        include_wang_target=args.include_wang_target,
    )

    target_sources = sorted(tf_df['source'].unique().tolist())
    print(f"  Target sources: {target_sources}")
    print(f"  all_df: {len(all_df):,} cells")
    print(f"  tf_df:  {len(tf_df):,} target cells")

    # ── Inferred UMAP from PCA on scanvi_normalized ───────────────────────────
    if not args.skip_inferred_umap:
        print("\nComputing inferred UMAP (PCA on scanvi_normalized) …")
        all_df = _compute_inferred_umap(adata, all_df)
    _has_inferred = 'inferred_umap_1' in all_df.columns

    # ── WANG self-check ───────────────────────────────────────────────────────
    wang_out = os.path.join(args.output_dir, 'WANG_REFERENCE')
    os.makedirs(wang_out, exist_ok=True)
    print(f"\n{'='*60}\nWANG self-prediction check\n{'='*60}")
    make_wang_confusion_matrix(adata, wang_out)

    print("\n── Cross-dataset subtype distribution ──")
    make_subtype_distribution(all_df, wang_out)

    # ── Per-source diagnostics ────────────────────────────────────────────────
    deprecated_outputs = [
        'umap_global.png', 'umap_perclass.png',
        'umap_all.png', 'umap_excitatory.png', 'umap_inhibitory.png',
    ]
    for src in target_sources:
        tf_src = tf_df[tf_df['source'] == src].copy()
        if tf_src.empty:
            print(f"\nNo cells for source {src} — skipping.")
            continue

        src_out = os.path.join(args.output_dir, src)
        os.makedirs(src_out, exist_ok=True)

        # Remove stale files from previous runs now that these plots are disabled.
        for fname in deprecated_outputs:
            stale_path = os.path.join(src_out, fname)
            if os.path.exists(stale_path):
                os.remove(stale_path)

        sep = '=' * 60
        print(f"\n{sep}\nDiagnostics for {src}  ({len(tf_src):,} cells)\n{sep}")

        print("\n── Tables ──")
        diag.make_tables(tf_src, src_out)

        print("\n── Class remapping analysis ──")
        diag.make_class_remapping_tables(tf_src, src_out)

        print("\n── Confidence histogram ──")
        diag.make_confidence_histogram(tf_src, src_out)

        print("\n── Class-remapped confidence histogram ──")
        diag.make_remapped_confidence_histogram(tf_src, src_out)

        print("\n── Age histogram (class-remapped) ──")
        diag.make_age_histogram_remapped(tf_src, src_out)

        print("\n── Age vs confidence density ──")
        diag.make_age_confidence_density(tf_src, src_out)

        print("\n── Latent-space UMAPs (scANVI embedding) ──")
        diag.make_umap_all(all_df, src_out, target_source=src,
                           name='umap_latent_all')
        diag.make_umap_excitatory(all_df, src_out, target_source=src,
                                  name='umap_latent_excitatory')
        diag.make_umap_inhibitory(all_df, src_out, target_source=src,
                                  name='umap_latent_inhibitory')

        if _has_inferred:
            print("\n── Inferred UMAPs (PCA on scanvi_normalized) ──")
            diag.make_umap_all(all_df, src_out, target_source=src,
                               umap_cols=('inferred_umap_1', 'inferred_umap_2'),
                               name='umap_inferred_all')
            diag.make_umap_excitatory(all_df, src_out, target_source=src,
                                      umap_cols=('inferred_umap_1', 'inferred_umap_2'),
                                      name='umap_inferred_excitatory')
            diag.make_umap_inhibitory(all_df, src_out, target_source=src,
                                      umap_cols=('inferred_umap_1', 'inferred_umap_2'),
                                      name='umap_inferred_inhibitory')

        print("\n── Sankey diagram ──")
        diag.make_sankey(tf_src, src_out, source_label=src)

        has_prenatal  = (tf_src['age_years'] < 0).any()
        has_postnatal = (tf_src['age_years'] >= 0).any()

        if has_prenatal:
            print("\n── Marker validation (prenatal, ref=WANG) ──")
            diag.make_marker_validation(tf_src, adata, src_out,
                                        ref_sources=['WANG'],
                                        age_lo=-np.inf, age_hi=0,
                                        suffix='_prenatal',
                                        period_label=f'{src} prenatal (age < 0y)')
            print("\n── Marker scatter validation (prenatal, ref=WANG) ──")
            diag.make_marker_scatter_validation(tf_src, adata, src_out,
                                               ref_sources=['WANG'],
                                               age_lo=-np.inf, age_hi=0,
                                               suffix='_prenatal',
                                               period_label=f'{src} prenatal (age < 0y)')
        if has_postnatal:
            print("\n── Marker validation (postnatal, ref=WANG) ──")
            diag.make_marker_validation(tf_src, adata, src_out,
                                        ref_sources=['WANG'],
                                        age_lo=0, age_hi=np.inf,
                                        suffix='_postnatal',
                                        period_label=f'{src} postnatal (age ≥ 0y)')
            print("\n── Marker scatter validation (postnatal, ref=WANG) ──")
            diag.make_marker_scatter_validation(tf_src, adata, src_out,
                                               ref_sources=['WANG'],
                                               age_lo=0, age_hi=np.inf,
                                               suffix='_postnatal',
                                               period_label=f'{src} postnatal (age ≥ 0y)')

    print("\nDone.")


if __name__ == '__main__':
    main()
