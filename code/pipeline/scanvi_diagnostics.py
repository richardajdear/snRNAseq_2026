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

def make_wang_confusion_matrix(adata, out):
    """Confusion matrix: cell_type_for_scanvi vs cell_type_aligned for WANG cells.

    This is the primary model quality check — if scANVI cannot recover the
    labels it was trained on, predictions for other datasets are unreliable.
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

    # Sort rows/cols by frequency for readability
    row_order = labeled['cell_type_for_scanvi'].value_counts().index
    col_order = labeled['cell_type_aligned'].value_counts().index
    ct = ct.reindex(index=row_order, columns=col_order, fill_value=0)

    n_types = len(ct)
    fig_size = max(8, n_types * 0.35)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(ct.values, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Fraction of row (input label)')

    ax.set_xticks(range(len(ct.columns)))
    ax.set_yticks(range(len(ct.index)))
    ax.set_xticklabels(ct.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(ct.index, fontsize=7)
    ax.set_xlabel('cell_type_aligned (scANVI prediction)', fontsize=10)
    ax.set_ylabel('cell_type_for_scanvi (training label)', fontsize=10)

    diag_acc = np.diag(ct.reindex(index=ct.index,
                                   columns=ct.index, fill_value=0).values).mean()
    ax.set_title(
        f'WANG self-prediction: cell_type_for_scanvi vs cell_type_aligned\n'
        f'n={len(labeled):,} WANG cells   mean diagonal = {diag_acc:.3f}',
        fontsize=11,
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
    """Stacked bar chart of cell_type_aligned proportions per dataset.

    Shown separately for Excitatory and Inhibitory neurons.
    Quickly reveals whether EN-L2_3-IT appears in expected proportions
    across datasets (the original kNN failure mode collapsed these to EN-L5).
    """
    for broad_class, prefix in [('Excitatory', 'EN'), ('Inhibitory', 'IN')]:
        sub = all_df[all_df['cell_class'] == broad_class].copy()
        if sub.empty:
            continue

        # Cell-type proportions per source
        props = (sub.groupby(['source', 'cell_type_aligned'])
                 .size()
                 .unstack(fill_value=0))
        props = props.div(props.sum(axis=1), axis=0)

        # Order sources: WANG first (reference), then alphabetical
        sources = ['WANG'] + sorted([s for s in props.index if s != 'WANG'])
        props = props.reindex(sources).dropna()

        # Sort columns by WANG proportion (most common first)
        if 'WANG' in props.index:
            col_order = props.loc['WANG'].sort_values(ascending=False).index
        else:
            col_order = props.sum().sort_values(ascending=False).index
        props = props[col_order]

        # Assign colors from diagnostics palette
        colors = [diag.PALETTE.get(c, diag._FALLBACK) for c in props.columns]

        fig, ax = plt.subplots(figsize=(max(6, len(props.columns) * 0.35 + 2), 5))
        bottom = np.zeros(len(props))
        for i, (col, color) in enumerate(zip(props.columns, colors)):
            vals = props[col].values
            bars = ax.bar(range(len(props)), vals, bottom=bottom,
                          color=color, edgecolor='white', linewidth=0.3,
                          label=col, width=0.7)
            bottom += vals

        ax.set_xticks(range(len(props)))
        ax.set_xticklabels(props.index, fontsize=10)
        ax.set_ylabel('Proportion of cells', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(
            f'{broad_class} neuron subtype distribution per dataset\n'
            f'(cell_type_aligned from scANVI predictions)',
            fontsize=11,
        )

        # Legend outside (can be many types)
        ncol = max(1, (len(props.columns) + 4) // 5)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                  fontsize=6.5, ncol=ncol, framealpha=0.9,
                  handlelength=1.2, handleheight=0.9)

        plt.tight_layout()
        fname = f'subtype_distribution_{prefix}.png'
        path = os.path.join(out, fname)
        plt.savefig(path, dpi=200, bbox_inches='tight')
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

    # ── WANG self-check ───────────────────────────────────────────────────────
    wang_out = os.path.join(args.output_dir, 'wang_selfcheck')
    os.makedirs(wang_out, exist_ok=True)
    print(f"\n{'='*60}\nWANG self-prediction check\n{'='*60}")
    make_wang_confusion_matrix(adata, wang_out)

    print("\n── Cross-dataset subtype distribution ──")
    make_subtype_distribution(all_df, wang_out)

    # ── Per-source diagnostics ────────────────────────────────────────────────
    deprecated_outputs = ['umap_global.png', 'umap_perclass.png']
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

        print("\n── All-cells UMAP ──")
        diag.make_umap_all(all_df, src_out, target_source=src)

        print("\n── Excitatory-cells UMAP ──")
        diag.make_umap_excitatory(all_df, src_out, target_source=src)

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
        if has_postnatal:
            print("\n── Marker validation (postnatal, ref=WANG) ──")
            diag.make_marker_validation(tf_src, adata, src_out,
                                        ref_sources=['WANG'],
                                        age_lo=0, age_hi=np.inf,
                                        suffix='_postnatal',
                                        period_label=f'{src} postnatal (age ≥ 0y)')

    print("\nDone.")


if __name__ == '__main__':
    main()
