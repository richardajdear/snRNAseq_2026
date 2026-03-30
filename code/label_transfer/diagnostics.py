"""Diagnostic comparison of old vs transferred cell subclass labels.

Usage
-----
    PYTHONPATH=code python3 -m label_transfer.diagnostics \
        --all_labels  path/to/label_transfer/all_cell_labels.csv \
        --transfer    path/to/label_transfer/transferred_labels.csv \
        --input       path/to/scvi_output/integrated.h5ad \
        --output_dir  path/to/label_transfer/diagnostics/

Outputs
-------
    remapping_crosstab.csv      — cell_type × transferred_subclass (Velmeshev)
    interneurons_detail.csv     — how 'Interneurons' / 'INT' cells were remapped
    confidence_summary.csv      — confidence by cell_class and age bin
    class_remapping_table.csv   — proportions of cells remapped between classes
    class_remapping_by_age.csv  — age breakdown of class-remapped cells
    confidence_histogram.png    — histogram of transfer confidence
    confidence_histogram_remapped.png — histogram for class-remapped cells only
    umap_global.png             — 2×3 UMAP grid (global embedding, in-panel legends)
    umap_perclass.png           — 2×3 UMAP grid (per-class recomputed UMAP)
    sankey_remapping.png        — Sankey diagram of old → new labels (3 panels)
    marker_validation.png       — Marker gene expression for class-remapped cells
"""

import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    # Excitatory (warm)
    'EN_L2_3_IT':   '#E41A1C',
    'EN_L3_5_IT':   '#FF4500',
    'EN_L4_IT':     '#FF7F00',
    'EN_L5_IT':     '#FFA500',
    'EN_L5_6_IT':   '#FFD700',
    'EN_L5_6_NP':   '#BDB76B',
    'EN_L5_ET':     '#DAA520',
    'EN_L6B':       '#D2691E',
    'EN_L6_CT':     '#CD853F',
    'EN_L6_IT':     '#DEB887',
    'EN_Immature':  '#FFB6C1',
    # Inhibitory (cool)
    'IN_PVALB':     '#377EB8',
    'IN_SST':       '#1F4E79',
    'IN_VIP':       '#984EA3',
    'IN_LAMP5':     '#6A3D9A',
    'IN_LAMP5_RELN':'#9370DB',
    'IN_ADARB2':    '#4B0082',
    'IN_Immature':  '#ADD8E6',
    # Glia
    'Astro':        '#4DAF4A',
    'Astro_Immature':'#90EE90',
    'Oligo':        '#8B4513',
    'OPC':          '#6B8E23',
    'Micro':        '#008080',
    'Endo':         '#778899',
    'PC':           '#A9A9A9',
    'PVM':          '#696969',
    'SMC':          '#C0C0C0',
    'VLMC':         '#B0C4DE',
    # Developmental
    'Radial_glia':  '#FF7F50',
    'Progenitors':  '#FA8072',
    'Glial_progenitors': '#2E8B57',
    'CR_cell':      '#E9967A',
    'IPC_EN':       '#FFDAB9',
    'IPC_Glia':     '#FFE4E1',
    # Broad / catch-all (muted)
    'Excitatory':   '#FFD0D0',
    'Inhibitory':   '#D0D0FF',
    'Glia':         '#D0FFD0',
    'Other':        '#E0E0E0',
    'Unknown':      '#808080',
    'Adaptive':     '#333333',
}

_FALLBACK = '#BFBFBF'

def _col(label):
    return PALETTE.get(label, _FALLBACK)


# ── cell class grouping ──────────────────────────────────────────────────────
CELL_CLASS_GROUPS = {
    'Excitatory': ['Excitatory'],
    'Inhibitory': ['Inhibitory'],
    'Glia':       ['Astrocytes', 'Oligos', 'OPC', 'Microglia',
                   'Glia', 'Endothelial', 'Other'],
}


# ── marker genes for validation ──────────────────────────────────────────────
MARKERS = {
    'Excitatory': ['SLC17A7', 'SATB2', 'NEUROD6', 'TBR1', 'RBFOX3', 'CUX2', 'RORB'],
    'Inhibitory': ['GAD1', 'GAD2', 'SLC32A1'],
    'Oligos':     ['MBP', 'PLP1', 'MAG'],
    'OPC':        ['PDGFRA', 'CSPG4'],
    'Astrocytes': ['AQP4', 'GFAP', 'SLC1A3', 'ALDH1L1'],
    'Microglia':  ['CSF1R', 'P2RY12', 'CX3CR1', 'TMEM119'],
    'Endothelial':['CLDN5', 'FLT1'],
    'Developmental': ['PAX6', 'VIM', 'EOMES', 'HES1'],
    'Layer':      ['FEZF2', 'TLE4', 'BCL11B'],
}


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════

def make_tables(tf, out):
    """Cross-tabulation and detail tables for Velmeshev transfer results."""

    # 1. Full cross-tab: raw cell_type → transferred subclass
    ct = pd.crosstab(tf['cell_type'], tf['transferred_subclass'], margins=True)
    ct.to_csv(os.path.join(out, 'remapping_crosstab.csv'))
    print(f"  remapping_crosstab.csv  ({ct.shape})")

    # 2. Interneurons / INT detail
    for label in ('Interneurons', 'INT'):
        sub = tf[tf['cell_type'] == label]
        if sub.empty:
            continue
        detail = (sub.groupby(['cell_class', 'transferred_subclass'])
                  .agg(n=('transfer_confidence', 'size'),
                       mean_conf=('transfer_confidence', 'mean'),
                       mean_dist=('mean_knn_distance', 'mean'))
                  .reset_index()
                  .sort_values(['cell_class', 'n'], ascending=[True, False]))
        fname = f'{label.lower()}_remapping.csv'
        detail.to_csv(os.path.join(out, fname), index=False)
        print(f"  {fname}  (n={len(sub)})")

    # 3. Confidence summary by cell_class × age bin
    tf2 = tf.copy()
    tf2['age_bin'] = pd.cut(tf2['age_years'],
                            bins=[-np.inf, 0, 18, np.inf],
                            labels=['Fetal', 'Postnatal', 'Adult'])
    summary = (tf2.groupby(['cell_class', 'age_bin'], observed=True)
               .agg(n=('transfer_confidence', 'size'),
                    mean_conf=('transfer_confidence', 'mean'),
                    median_conf=('transfer_confidence', 'median'),
                    low_conf_pct=('is_low_confidence', 'mean'))
               .reset_index())
    summary['low_conf_pct'] = (summary['low_conf_pct'] * 100).round(1)
    summary.to_csv(os.path.join(out, 'confidence_summary.csv'), index=False)
    print(f"  confidence_summary.csv")


def make_class_remapping_tables(tf, out):
    """Tables for cells remapped between major cell classes."""
    remap = tf[tf['is_class_remapped']].copy()
    total = len(tf)

    # 1. Proportions table
    if remap.empty:
        print("  No class-remapped cells found.")
        return

    props = (remap.groupby(['old_cell_class', 'new_cell_class'])
             .agg(n=('transfer_confidence', 'size'),
                  mean_conf=('transfer_confidence', 'mean'),
                  median_conf=('transfer_confidence', 'median'))
             .reset_index()
             .sort_values('n', ascending=False))
    props['pct_of_total'] = (100 * props['n'] / total).round(2)
    props['pct_of_old_class'] = 0.0
    for _, row in props.iterrows():
        n_old = (tf['old_cell_class'] == row['old_cell_class']).sum()
        props.loc[props.index == _, 'pct_of_old_class'] = round(
            100 * row['n'] / n_old, 2) if n_old > 0 else 0
    props.to_csv(os.path.join(out, 'class_remapping_table.csv'), index=False)
    print(f"  class_remapping_table.csv  ({len(props)} remapping types, "
          f"{remap.shape[0]} cells = {100*len(remap)/total:.1f}% of total)")

    # 2. Age breakdown of class-remapped cells
    remap['age_bin'] = pd.cut(remap['age_years'],
                              bins=[-np.inf, 0, 1, 5, np.inf],
                              labels=['Fetal (<0y)', 'Perinatal (0-1y)',
                                      'Childhood (1-5y)', 'Post-5y (>5y)'])
    age_breakdown = (remap.groupby(['old_cell_class', 'new_cell_class', 'age_bin'],
                                    observed=True)
                     .agg(n=('transfer_confidence', 'size'),
                          mean_conf=('transfer_confidence', 'mean'))
                     .reset_index()
                     .sort_values(['old_cell_class', 'new_cell_class', 'age_bin']))

    # Add within-group percentages
    for (oc, nc), grp in age_breakdown.groupby(['old_cell_class', 'new_cell_class']):
        total_in_remap = grp['n'].sum()
        age_breakdown.loc[grp.index, 'pct_of_remap'] = (
            100 * grp['n'] / total_in_remap).round(1)

    age_breakdown.to_csv(os.path.join(out, 'class_remapping_by_age.csv'), index=False)
    print(f"  class_remapping_by_age.csv")

    # 3. Print interpretation
    print(f"\n  CLASS REMAPPING INTERPRETATION:")
    print(f"  Total cells remapped: {len(remap)} / {total} "
          f"({100*len(remap)/total:.1f}%)")
    for _, row in props.head(5).iterrows():
        print(f"    {row['old_cell_class']:15s} → {row['new_cell_class']:15s}: "
              f"{row['n']:5d} ({row['pct_of_old_class']:.1f}% of old class, "
              f"mean conf {row['mean_conf']:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE HISTOGRAMS
# ══════════════════════════════════════════════════════════════════════════════

def make_confidence_histogram(tf, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall
    axes[0].hist(tf['transfer_confidence'], bins=50, color='steelblue',
                 edgecolor='white', linewidth=0.3)
    axes[0].axvline(0.5, color='red', ls='--', lw=1, label='threshold=0.5')
    axes[0].set(xlabel='Transfer confidence', ylabel='Cells',
                title='Overall confidence')
    axes[0].legend(fontsize=8)

    # Per cell_class
    classes = sorted(tf['cell_class'].unique())
    for cls in classes:
        axes[1].hist(tf.loc[tf['cell_class'] == cls, 'transfer_confidence'],
                     bins=50, alpha=0.5, label=cls)
    axes[1].axvline(0.5, color='red', ls='--', lw=1)
    axes[1].set(xlabel='Transfer confidence', ylabel='Cells',
                title='Confidence by cell_class')
    axes[1].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    path = os.path.join(out, 'confidence_histogram.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  confidence_histogram.png")


def make_remapped_confidence_histogram(tf, out):
    """Confidence histogram for class-remapped cells only."""
    remap = tf[tf['is_class_remapped']]
    if remap.empty:
        print("  No class-remapped cells — skipping histogram.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall remapped
    axes[0].hist(remap['transfer_confidence'], bins=50, color='#D62728',
                 edgecolor='white', linewidth=0.3)
    axes[0].axvline(0.5, color='black', ls='--', lw=1, label='threshold=0.5')
    axes[0].set(xlabel='Transfer confidence', ylabel='Cells',
                title=f'Class-remapped cells (n={len(remap)})')
    axes[0].legend(fontsize=8)

    # By remapping type (top 6)
    top_remaps = (remap.groupby(['old_cell_class', 'new_cell_class'])
                  .size().sort_values(ascending=False).head(6))
    for (oc, nc), _ in top_remaps.items():
        m = (remap['old_cell_class'] == oc) & (remap['new_cell_class'] == nc)
        axes[1].hist(remap.loc[m, 'transfer_confidence'],
                     bins=30, alpha=0.5, label=f'{oc}→{nc}')
    axes[1].axvline(0.5, color='black', ls='--', lw=1)
    axes[1].set(xlabel='Transfer confidence', ylabel='Cells',
                title='By remapping type')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(out, 'confidence_histogram_remapped.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  confidence_histogram_remapped.png")


# ══════════════════════════════════════════════════════════════════════════════
# UMAP GRIDS
# ══════════════════════════════════════════════════════════════════════════════

def _scatter_panel(ax, xy, labels, is_target,
                   bg_size=0.4, bg_alpha=0.15,
                   fg_size=1.0, fg_alpha=0.5):
    """Plot background (reference) cells in grey, then foreground (target) cells colored."""
    # Background: all non-target cells in uniform grey for structure context
    bg = ~is_target
    if bg.sum():
        ax.scatter(xy[bg.values, 0], xy[bg.values, 1],
                   c='#CCCCCC', s=bg_size, alpha=bg_alpha,
                   linewidths=0, rasterized=True)

    # Foreground: target cells colored by label; plot rare labels last (on top)
    freq = labels[is_target].value_counts()
    for lbl in reversed(freq.index.tolist()):
        fg = (labels == lbl) & is_target
        if fg.sum():
            ax.scatter(xy[fg.values, 0], xy[fg.values, 1],
                       c=_col(lbl), s=fg_size, alpha=fg_alpha,
                       linewidths=0, rasterized=True)

    ax.set_xticks([])
    ax.set_yticks([])


def _add_panel_legend(ax, labels):
    """Add in-panel legend for the labels present in this panel."""
    unique_labels = sorted(labels.unique())
    # Sort: EN → IN → Glia → Dev → Broad
    def _sort_key(l):
        if l.startswith('EN_'): return (0, l)
        if l.startswith('IN_'): return (1, l)
        if l in ('Astro', 'Astro_Immature', 'Oligo', 'OPC', 'Micro',
                 'Endo', 'PC', 'PVM', 'SMC', 'VLMC'):
            return (2, l)
        if l in ('Radial_glia', 'Progenitors', 'Glial_progenitors',
                 'CR_cell', 'IPC_EN', 'IPC_Glia'):
            return (3, l)
        return (4, l)
    unique_labels.sort(key=_sort_key)

    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=_col(l), markersize=5, label=l)
               for l in unique_labels]
    ncol = max(1, (len(handles) + 5) // 6)
    ax.legend(handles=handles, loc='lower right', fontsize=5.5, ncol=ncol,
              framealpha=0.85, edgecolor='0.8', handletextpad=0.2,
              columnspacing=0.5, borderpad=0.3)


def make_umap_global(all_df, out, target_source='VELMESHEV'):
    """2×3 UMAP grid using the global embedding, with in-panel legends."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    is_target = all_df['source'] == target_source
    xy_full = all_df[['umap_1', 'umap_2']].values

    for col, (group_name, classes) in enumerate(CELL_CLASS_GROUPS.items()):
        cls_mask = all_df['cell_class'].isin(classes)
        sub = all_df.loc[cls_mask]
        xy = xy_full[cls_mask.values]
        is_tgt = is_target[cls_mask]

        for row, (label_col, row_title) in enumerate([
            ('old_subclass', 'Original subclass'),
            ('new_subclass', 'Transferred subclass'),
        ]):
            ax = axes[row, col]
            _scatter_panel(ax, xy, sub[label_col], is_tgt)
            _add_panel_legend(ax, sub.loc[is_tgt, label_col])
            ax.set_title(f'{group_name}  —  {row_title}', fontsize=11)

    fig.suptitle(
        'Cell subclass labels: original vs kNN-transferred '
        f'({target_source} highlighted, reference cells faded)\n'
        'Global UMAP embedding',
        fontsize=13, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out, 'umap_global.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  umap_global.png")


def make_umap_perclass(all_df, emb, out, target_source='VELMESHEV'):
    """2×3 UMAP grid with per-class recomputed UMAP embedding."""
    from umap import UMAP

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    is_target = all_df['source'] == target_source

    for col, (group_name, classes) in enumerate(CELL_CLASS_GROUPS.items()):
        cls_mask = all_df['cell_class'].isin(classes).values
        sub = all_df.loc[cls_mask].copy()
        sub_emb = emb[cls_mask]
        is_tgt = is_target[cls_mask]

        print(f"    Computing UMAP for {group_name} ({cls_mask.sum()} cells) …")
        reducer = UMAP(n_neighbors=30, min_dist=0.3, random_state=42, n_jobs=-1)
        xy = reducer.fit_transform(sub_emb)

        for row, (label_col, row_title) in enumerate([
            ('old_subclass', 'Original subclass'),
            ('new_subclass', 'Transferred subclass'),
        ]):
            ax = axes[row, col]
            _scatter_panel(ax, xy, sub[label_col], is_tgt)
            _add_panel_legend(ax, sub.loc[is_tgt.values, label_col])
            ax.set_title(f'{group_name}  —  {row_title}', fontsize=11)

    fig.suptitle(
        'Cell subclass labels: original vs kNN-transferred '
        f'({target_source} highlighted, reference cells faded)\n'
        'Per-class recomputed UMAP from scVI embedding',
        fontsize=13, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out, 'umap_perclass.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  umap_perclass.png")


def make_umap_velmeshev(all_df, out, target_source='VELMESHEV'):
    """2×2 UMAP of Velmeshev cells only: pre/post-remapping × class / remap-status."""
    vel = all_df[all_df['source'] == target_source].copy()
    xy  = vel[['umap_1', 'umap_2']].values
    vel['is_class_remapped'] = vel['old_cell_class'] != vel['new_cell_class']

    CLASS_PALETTE = {
        'Excitatory':  '#E41A1C',
        'Inhibitory':  '#377EB8',
        'Astrocytes':  '#4DAF4A',
        'Oligos':      '#984EA3',
        'OPC':         '#FF7F00',
        'Microglia':   '#008080',
        'Endothelial': '#778899',
        'Glia':        '#2CA25F',
        'Other':       '#BDBDBD',
    }
    AGE_BINS   = [-np.inf, 0, 1, 5, np.inf]
    AGE_LABELS = ['Fetal (<0y)', 'Perinatal (0-1y)', 'Childhood (1-5y)', 'Post-5y (>5y)']
    AGE_COLORS = ['#2171B5', '#6BAED6', '#FD8D3C', '#D62728']

    vel['age_cat'] = pd.cut(vel['age_years'], bins=AGE_BINS, labels=AGE_LABELS)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    n_remap    = vel['is_class_remapped'].sum()
    pct_remap  = 100 * n_remap / len(vel)

    for row, (class_col, row_title) in enumerate([
        ('old_cell_class', 'Pre-remapping'),
        ('new_cell_class', 'Post-remapping'),
    ]):
        # ── col 0: broad cell class ──────────────────────────────────────────
        ax = axes[row, 0]
        classes = sorted(vel[class_col].unique())
        for cls in classes:
            m = (vel[class_col] == cls).values
            ax.scatter(xy[m, 0], xy[m, 1],
                       c=CLASS_PALETTE.get(cls, '#BFBFBF'),
                       s=0.5, alpha=0.4, linewidths=0, rasterized=True)
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=CLASS_PALETTE.get(c, '#BFBFBF'),
                          markersize=6, label=c)
                   for c in classes]
        ax.legend(handles=handles, loc='lower right', fontsize=7,
                  framealpha=0.85, edgecolor='0.8', handletextpad=0.3)
        ax.set_title(f'{row_title} — cell class', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        # ── col 1: top row = remap status, bottom row = age category ─────────
        ax = axes[row, 1]
        if row == 0:
            # Pre-remapping: show class-remapped binary
            not_remap = ~vel['is_class_remapped'].values
            ax.scatter(xy[not_remap, 0], xy[not_remap, 1],
                       c='#CCCCCC', s=0.3, alpha=0.3, linewidths=0, rasterized=True)
            ax.scatter(xy[vel['is_class_remapped'].values, 0],
                       xy[vel['is_class_remapped'].values, 1],
                       c='#D62728', s=0.8, alpha=0.5, linewidths=0, rasterized=True)
            handles = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#CCCCCC',
                       markersize=6, label=f'No class change  (n={len(vel)-n_remap:,})'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#D62728',
                       markersize=6, label=f'Class remapped  (n={n_remap:,}, {pct_remap:.1f}%)'),
            ]
            ax.legend(handles=handles, loc='lower right', fontsize=7,
                      framealpha=0.85, edgecolor='0.8', handletextpad=0.3)
            ax.set_title('Pre-remapping — class remapping status', fontsize=11)
        else:
            # Post-remapping: show age category
            for age_lbl, age_col in zip(AGE_LABELS, AGE_COLORS):
                m = (vel['age_cat'] == age_lbl).values
                n_age = m.sum()
                ax.scatter(xy[m, 0], xy[m, 1],
                           c=age_col, s=0.5, alpha=0.4, linewidths=0, rasterized=True,
                           label=f'{age_lbl}  (n={n_age:,})')
            handles = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=c, markersize=6,
                              label=f'{l}  (n={( vel["age_cat"]==l).sum():,})')
                       for l, c in zip(AGE_LABELS, AGE_COLORS)]
            ax.legend(handles=handles, loc='lower right', fontsize=7,
                      framealpha=0.85, edgecolor='0.8', handletextpad=0.3)
            ax.set_title('Post-remapping — donor age', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f'{target_source} cells only (n={len(vel):,}) — global UMAP embedding\n'
                 'Pre- vs post-remapping',
                 fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out, 'umap_target.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  umap_target.png")


# ══════════════════════════════════════════════════════════════════════════════
# SANKEY DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def _make_sankey_panel(ax, tf_sub, title, within_color='#4DBEEE',
                       cross_color='#D62728'):
    """Single sankey panel: old cell_type → transferred_subclass.

    Within-class flows in blue, cross-class flows in red.
    """
    from label_transfer.transfer import subclass_to_class

    # Build flow table: old_subclass → transferred_subclass
    flows = (tf_sub.groupby(['old_subclass', 'transferred_subclass'])
             .size().reset_index(name='n'))
    flows = flows[flows['n'] > 0].sort_values('n', ascending=False)

    if flows.empty:
        ax.set_title(title)
        ax.axis('off')
        return

    # Determine if each flow is within-class or cross-class
    flows['new_class'] = flows['transferred_subclass'].map(subclass_to_class)
    ct_class = tf_sub.groupby('old_subclass')['old_cell_class'].first().to_dict()
    flows['old_class'] = flows['old_subclass'].map(ct_class)
    flows['is_cross'] = flows['old_class'] != flows['new_class']

    # Filter to top flows for readability (keep flows with ≥ 0.5% of panel total)
    min_n = max(5, int(0.005 * flows['n'].sum()))
    flows = flows[flows['n'] >= min_n].copy()

    if flows.empty:
        ax.set_title(title)
        ax.axis('off')
        return

    # Left side: old_subclass, right side: transferred_subclass
    left_labels = flows.groupby('old_subclass')['n'].sum().sort_values(ascending=True)
    right_labels = flows.groupby('transferred_subclass')['n'].sum().sort_values(ascending=True)
    total_shown = left_labels.sum()

    gap = 0.02
    x_left, x_right = 0.0, 1.0

    # Calculate bar positions (bottom-up, smallest at bottom)
    def _bar_positions(label_sizes, gap_frac=0.02):
        """Return dict of {label: (y_bottom, y_top)} normalized to [0, 1]."""
        total_size = label_sizes.sum()
        n_gaps = len(label_sizes) - 1
        usable = 1.0 - n_gaps * gap_frac
        positions = {}
        cursor = 0.0
        for lbl, sz in label_sizes.items():
            h = usable * sz / total_size
            positions[lbl] = (cursor, cursor + h)
            cursor += h + gap_frac
        return positions

    left_pos = _bar_positions(left_labels, gap)
    right_pos = _bar_positions(right_labels, gap)

    # Track fill cursors for connecting flows
    left_cursor = {lbl: pos[0] for lbl, pos in left_pos.items()}
    right_cursor = {lbl: pos[0] for lbl, pos in right_pos.items()}

    bar_width = 0.12

    # Draw flows: within-class first (background), cross-class on top
    for is_cross_val in [False, True]:
        sub = flows[flows['is_cross'] == is_cross_val]
        color = cross_color if is_cross_val else within_color
        alpha = 0.6 if is_cross_val else 0.3

        for _, row in sub.iterrows():
            lbl_l, lbl_r, n = row['old_subclass'], row['transferred_subclass'], row['n']
            if lbl_l not in left_pos or lbl_r not in right_pos:
                continue

            # Flow height proportional to n
            h_left = (left_pos[lbl_l][1] - left_pos[lbl_l][0]) * n / left_labels[lbl_l]
            h_right = (right_pos[lbl_r][1] - right_pos[lbl_r][0]) * n / right_labels[lbl_r]

            y_l0 = left_cursor[lbl_l]
            y_l1 = y_l0 + h_left
            y_r0 = right_cursor[lbl_r]
            y_r1 = y_r0 + h_right

            left_cursor[lbl_l] = y_l1
            right_cursor[lbl_r] = y_r1

            # Bezier path
            xm = (x_left + bar_width + x_right - bar_width) / 2
            verts = [
                (x_left + bar_width, y_l0),
                (xm, y_l0),
                (xm, y_r0),
                (x_right - bar_width, y_r0),
                (x_right - bar_width, y_r1),
                (xm, y_r1),
                (xm, y_l1),
                (x_left + bar_width, y_l1),
                (x_left + bar_width, y_l0),
            ]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.CLOSEPOLY]
            patch = PathPatch(Path(verts, codes), facecolor=color,
                              alpha=alpha, edgecolor='none', zorder=2 if is_cross_val else 1)
            ax.add_patch(patch)

    # Draw bars — use bar center (y0+y1)/2 since barh default align='center'
    for lbl, (y0, y1) in left_pos.items():
        pct = 100 * left_labels[lbl] / total_shown
        ax.barh((y0 + y1) / 2, bar_width, height=y1 - y0, left=x_left,
                color='#6BAED6', edgecolor='white', linewidth=0.5, zorder=3)
        ax.text(x_left - 0.01, (y0 + y1) / 2, f'{lbl} ({pct:.1f}%)',
                ha='right', va='center', fontsize=6)

    for lbl, (y0, y1) in right_pos.items():
        ax.barh((y0 + y1) / 2, bar_width, height=y1 - y0,
                left=x_right - bar_width,
                color=_col(lbl), edgecolor='white', linewidth=0.5, zorder=3)
        ax.text(x_right + 0.01, (y0 + y1) / 2, lbl,
                ha='left', va='center', fontsize=6)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis('off')


def make_sankey(tf, out, source_label=''):
    """3-panel Sankey diagram: Excitatory / Inhibitory / Glia."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for i, (group_name, classes) in enumerate(CELL_CLASS_GROUPS.items()):
        sub = tf[tf['old_cell_class'].isin(classes)]
        _make_sankey_panel(axes[i], sub, group_name)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4DBEEE', alpha=0.5, label='Within-class transfer'),
        Patch(facecolor='#D62728', alpha=0.7, label='Cross-class remapping'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.01))

    title = f'Sankey: old subclass → kNN-transferred Wang subclass'
    if source_label:
        title = f'{source_label}  —  {title}'
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = os.path.join(out, 'sankey_remapping.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  sankey_remapping.png")


# ══════════════════════════════════════════════════════════════════════════════
# MARKER GENE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def make_marker_validation(tf, adata, out, max_ref=5000,
                            ref_sources=None, age_lo=-np.inf, age_hi=np.inf,
                            suffix='', period_label=''):
    """Violin plots of marker gene expression for class-remapped cells.

    ref_sources  : sources to use as reference (default AGING+HBCC)
    age_lo/hi    : restrict remapped cells to this age window (age_lo <= age < age_hi)
    suffix       : appended to output filename, e.g. '_prenatal'
    period_label : human-readable label for the plot title
    """
    import scipy.sparse as sp

    if ref_sources is None:
        ref_sources = ['WANG']

    remap = tf[tf['is_class_remapped'] &
               (tf['age_years'] >= age_lo) &
               (tf['age_years'] < age_hi)].copy()
    if remap.empty:
        print(f"  No class-remapped cells in [{age_lo}, {age_hi}) — skipping {suffix}.")
        return

    # Gene symbol → Ensembl ID
    var = adata.var
    sym2ens = (dict(zip(var['gene_symbol'], var.index))
               if 'gene_symbol' in var.columns
               else {g: g for g in var.index})

    top_remaps = (remap.groupby(['old_cell_class', 'new_cell_class'])
                  .size().sort_values(ascending=False).head(3))

    all_markers_needed = set()
    for (old_cls, new_cls), _ in top_remaps.items():
        for cls in [old_cls, new_cls]:
            if cls in MARKERS:
                all_markers_needed.update(MARKERS[cls])

    marker_ens = {sym: sym2ens[sym] for sym in all_markers_needed if sym in sym2ens}
    if not marker_ens:
        print(f"  No marker genes found — skipping {suffix}.")
        return

    # Collect positions: remapped cells + period-appropriate reference cells
    obs = adata.obs
    all_positions = set(remap['h5ad_pos'].values)
    ref_positions = {}
    for (old_cls, new_cls), _ in top_remaps.items():
        for cls in [old_cls, new_cls]:
            if cls not in ref_positions:
                mask = obs['source'].isin(ref_sources) & (obs['cell_class'] == cls)
                pos = np.where(mask.values)[0]
                if len(pos) > max_ref:
                    pos = np.random.RandomState(42).choice(pos, max_ref, replace=False)
                ref_positions[cls] = pos
                all_positions.update(pos)

    all_positions = sorted(all_positions)
    pos_to_local = {p: i for i, p in enumerate(all_positions)}

    ens_to_varidx = {eid: i for i, eid in enumerate(adata.var_names)}
    valid_ens = [e for e in marker_ens.values() if e in ens_to_varidx]
    var_indices = [ens_to_varidx[e] for e in valid_ens]
    ref_label = '+'.join(ref_sources)
    print(f"    [{period_label or 'all'}] {len(all_positions)} cells × "
          f"{len(valid_ens)} markers (ref: {ref_label}) …")

    chunk_size = 5000
    expr = np.zeros((len(all_positions), len(valid_ens)), dtype=np.float32)
    for start_i in range(0, len(all_positions), chunk_size):
        end_i = min(start_i + chunk_size, len(all_positions))
        chunk_pos = all_positions[start_i:end_i]
        if 'scvi_normalized' in adata.layers:
            chunk_data = adata.layers['scvi_normalized'][chunk_pos][:, var_indices]
        elif 'counts' in adata.layers:
            chunk_data = adata.layers['counts'][chunk_pos][:, var_indices]
        else:
            chunk_data = adata.X[chunk_pos][:, var_indices]
        if sp.issparse(chunk_data):
            chunk_data = chunk_data.toarray()
        expr[start_i:end_i] = np.asarray(chunk_data, dtype=np.float32)

    expr = np.log1p(expr)

    # Per-remapping marker gene lists (deduplicated, old then new class)
    markers_per_remap = {}
    for (old_cls, new_cls), _ in top_remaps.items():
        seen, unique_m = set(), []
        for cls in [old_cls, new_cls]:
            for g in MARKERS.get(cls, []):
                ens = marker_ens.get(g)
                if ens and g not in seen and ens in valid_ens:
                    seen.add(g); unique_m.append(g)
        markers_per_remap[(old_cls, new_cls)] = unique_m[:8]

    n_remaps = len(top_remaps)
    max_cols = max((len(m) for m in markers_per_remap.values()), default=1)
    fig, axes = plt.subplots(n_remaps, max_cols,
                             figsize=(1.6 * max_cols, 2 * n_remaps),
                             squeeze=False)
    colors_violin = {'ref_old': '#377EB8', 'remapped': '#D62728', 'ref_new': '#4DAF4A'}

    for row_i, ((old_cls, new_cls), n_cells) in enumerate(top_remaps.items()):
        gene_list = markers_per_remap[(old_cls, new_cls)]
        remap_sub = remap[(remap['old_cell_class'] == old_cls) &
                          (remap['new_cell_class'] == new_cls)]
        remap_local = [pos_to_local[p] for p in remap_sub['h5ad_pos'].values
                       if p in pos_to_local]
        ref_old_local = [pos_to_local[p] for p in ref_positions.get(old_cls, [])
                         if p in pos_to_local]
        ref_new_local = [pos_to_local[p] for p in ref_positions.get(new_cls, [])
                         if p in pos_to_local]

        for col_i, gene_sym in enumerate(gene_list):
            ax = axes[row_i, col_i]
            ens_id = marker_ens[gene_sym]
            gene_col = valid_ens.index(ens_id) if ens_id in valid_ens else None
            if gene_col is None:
                ax.axis('off'); continue

            groups, lbls, cols = [], [], []
            if ref_old_local:
                groups.append(expr[ref_old_local, gene_col])
                lbls.append(f'Ref\n{old_cls}'); cols.append(colors_violin['ref_old'])
            if remap_local:
                groups.append(expr[remap_local, gene_col])
                lbls.append('Remapped'); cols.append(colors_violin['remapped'])
            if ref_new_local:
                groups.append(expr[ref_new_local, gene_col])
                lbls.append(f'Ref\n{new_cls}'); cols.append(colors_violin['ref_new'])
            if not groups:
                ax.axis('off'); continue

            parts = ax.violinplot(groups, showextrema=False, showmedians=True)
            for pc_i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(cols[pc_i]); pc.set_alpha(0.7)
            ax.set_xticks(range(1, len(lbls) + 1))
            ax.set_xticklabels(lbls, fontsize=6, rotation=30, ha='right')
            ax.set_title(gene_sym, fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f'{old_cls}\u2192{new_cls}\n(n={n_cells})', fontsize=8)

        for col_i in range(len(gene_list), max_cols):
            axes[row_i, col_i].axis('off')

    title_period = f' — {period_label}' if period_label else ''
    fig.suptitle(f'Marker gene expression: class-remapped cells vs reference{title_period}\n'
                 f'(log1p scVI-normalized; reference: {ref_label})',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fname = f'marker_validation{suffix}.png'
    plt.savefig(os.path.join(out, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# AGE-BASED PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def make_age_histogram_remapped(tf, out):
    """Histogram of class-remapped cells by donor age (x-axis = age_years, capped at 30)."""
    remap = tf[tf['is_class_remapped']].copy()
    if remap.empty:
        print("  No class-remapped cells — skipping age histogram.")
        return

    age_min = max(-5, np.floor(remap['age_years'].min()))
    bins = np.linspace(age_min, 30, 36)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall
    age_clip = remap['age_years'].clip(upper=30)
    axes[0].hist(age_clip, bins=bins, color='#D62728', edgecolor='white', linewidth=0.3)
    axes[0].axvline(0, color='black', ls='--', lw=1, label='Birth')
    axes[0].set(xlabel='Age (years)', ylabel='Cells',
                title=f'Age distribution of class-remapped cells (n={len(remap)})')
    axes[0].legend(fontsize=8)

    # By remapping type (top 5)
    top_remaps = (remap.groupby(['old_cell_class', 'new_cell_class'])
                  .size().sort_values(ascending=False).head(5))
    for (oc, nc), _ in top_remaps.items():
        m = (remap['old_cell_class'] == oc) & (remap['new_cell_class'] == nc)
        axes[1].hist(remap.loc[m, 'age_years'].clip(upper=30),
                     bins=bins, alpha=0.5, label=f'{oc}→{nc}')
    axes[1].axvline(0, color='black', ls='--', lw=1)
    axes[1].set(xlabel='Age (years)', ylabel='Cells', title='By remapping type')
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(out, 'age_histogram_remapped.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  age_histogram_remapped.png")


def make_age_confidence_density(tf, out):
    """2×3 faceted density plot: x=age (up to 30), y=confidence, top 6 class transitions."""
    remap = tf[tf['is_class_remapped']].copy()
    if remap.empty:
        print("  No class-remapped cells — skipping density plot.")
        return

    top_remaps = (remap.groupby(['old_cell_class', 'new_cell_class'])
                  .size().sort_values(ascending=False).head(6))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for i, ((oc, nc), n_cells) in enumerate(top_remaps.items()):
        ax = axes_flat[i]
        m = (remap['old_cell_class'] == oc) & (remap['new_cell_class'] == nc)
        sub = remap[m].copy()
        age = sub['age_years'].clip(upper=30)
        conf = sub['transfer_confidence']

        hb = ax.hexbin(age, conf, gridsize=30, cmap='Reds', mincnt=1,
                       extent=[age.min() - 0.5, 30.5, 0, 1])
        plt.colorbar(hb, ax=ax, label='cells', pad=0.02)

        ax.axhline(0.5, color='black', ls='--', lw=0.8, alpha=0.6, label='conf=0.5')
        ax.axvline(0, color='steelblue', ls='--', lw=0.8, alpha=0.6, label='Birth')
        ax.set_title(f'{oc} → {nc}  (n={n_cells:,})', fontsize=9)
        ax.set_xlabel('Age (years)', fontsize=8)
        ax.set_ylabel('Transfer confidence', fontsize=8)
        ax.set_xlim(age.min() - 1, 30)
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for i in range(len(top_remaps), 6):
        axes_flat[i].axis('off')

    fig.suptitle('Age vs transfer confidence for top 6 class remappings\n'
                 '(hexbin density; dashed lines = birth / confidence threshold)',
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(out, 'age_confidence_density.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  age_confidence_density.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description='Label-transfer diagnostics')
    p.add_argument('--all_labels', required=True,
                   help='all_cell_labels.csv from run_transfer')
    p.add_argument('--transfer', required=True,
                   help='transferred_labels.csv from run_transfer')
    p.add_argument('--input', required=True,
                   help='integrated.h5ad (for per-class UMAP and marker validation)')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--embedding_key', default='X_scVI')
    p.add_argument('--target_sources', nargs='+', default=None,
                   help='Sources to generate diagnostics for. '
                        'Default: all sources present in transferred_labels.csv')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data …")
    all_df = pd.read_csv(args.all_labels)
    tf     = pd.read_csv(args.transfer)
    target_sources = args.target_sources or sorted(tf['source'].unique().tolist())
    print(f"  all_cell_labels: {len(all_df)} cells")
    print(f"  transferred:     {len(tf)} cells")
    print(f"  target sources:  {target_sources}")

    print(f"\nLoading {args.input} (backed) …")
    adata = sc.read_h5ad(args.input, backed='r')
    emb   = np.array(adata.obsm[args.embedding_key])
    print(f"  {adata.shape[0]} cells, embedding dim={emb.shape[1]}")

    # ── Per-source diagnostics ─────────────────────────────────────────────
    for src in target_sources:
        tf_src = tf[tf['source'] == src].copy()
        if tf_src.empty:
            print(f"\nNo cells for source {src} — skipping.")
            continue

        src_out = os.path.join(args.output_dir, src)
        os.makedirs(src_out, exist_ok=True)

        sep = '=' * 60
        print(f"\n{sep}\nDiagnostics for {src}  ({len(tf_src):,} cells)\n{sep}")

        # Tables
        print("\n── Tables ──")
        make_tables(tf_src, src_out)

        print("\n── Class remapping analysis ──")
        make_class_remapping_tables(tf_src, src_out)

        # Histograms
        print("\n── Confidence histogram ──")
        make_confidence_histogram(tf_src, src_out)

        print("\n── Class-remapped confidence histogram ──")
        make_remapped_confidence_histogram(tf_src, src_out)

        print("\n── Age histogram (class-remapped) ──")
        make_age_histogram_remapped(tf_src, src_out)

        print("\n── Age vs confidence density ──")
        make_age_confidence_density(tf_src, src_out)

        # UMAPs
        print("\n── Global UMAP grid ──")
        make_umap_global(all_df, src_out, target_source=src)

        print("\n── Target-cells UMAP ──")
        make_umap_velmeshev(all_df, src_out, target_source=src)

        print("\n── Per-class UMAP grid ──")
        make_umap_perclass(all_df, emb, src_out, target_source=src)

        # Sankey
        print("\n── Sankey diagram ──")
        make_sankey(tf_src, src_out, source_label=src)

        # Marker validation — split by prenatal / postnatal when both exist
        has_prenatal  = (tf_src['age_years'] < 0).any()
        has_postnatal = (tf_src['age_years'] >= 0).any()

        if has_prenatal:
            print("\n── Marker validation (prenatal, ref=WANG) ──")
            make_marker_validation(tf_src, adata, src_out,
                                   ref_sources=['WANG'],
                                   age_lo=-np.inf, age_hi=0,
                                   suffix='_prenatal',
                                   period_label=f'{src} prenatal (age < 0y)')
        if has_postnatal:
            print("\n── Marker validation (postnatal, ref=WANG) ──")
            make_marker_validation(tf_src, adata, src_out,
                                   ref_sources=['WANG'],
                                   age_lo=0, age_hi=np.inf,
                                   suffix='_postnatal',
                                   period_label=f'{src} postnatal (age ≥ 0y)')

    print("\nDone.")


if __name__ == '__main__':
    main()
