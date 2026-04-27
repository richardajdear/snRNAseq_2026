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
    remapping_crosstab.csv      — cell_type_raw × cell_type_aligned
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
import re
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
# Broad cell-class palette — used for Sankey right-hand bars and UMAP class plots
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


def _norm_key(label):
    """Normalise a label for PALETTE lookup: replace hyphens with underscores."""
    return str(label).replace('-', '_')


def _col(label):
    """Look up colour from PALETTE, trying normalised key if exact match fails."""
    c = PALETTE.get(label)
    if c is not None:
        return c
    c = PALETTE.get(_norm_key(label))
    if c is not None:
        return c
    return _FALLBACK


def _make_local_palette(labels):
    """Build a colour dict covering all *labels*.

    Uses PALETTE (with normalisation fallback) where possible, then cycles
    through tab20 for any remaining labels so nothing is left grey.
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', 20)
    result = {}
    fallback_idx = 0
    for lbl in sorted(set(str(l) for l in labels)):
        c = PALETTE.get(lbl) or PALETTE.get(_norm_key(lbl))
        if c is None:
            result[lbl] = matplotlib.colors.to_hex(cmap(fallback_idx % 20))
            fallback_idx += 1
        else:
            result[lbl] = c
    return result


# ── EN subtype biological sort key (module level for reuse) ──────────────────
def _en_sort_key(ct):
    """Biological sort order for EN subtypes used in legends and palettes.

    Order: Newborn → IT-Immature → Non-IT-Immature → layer types (L2→L6) → rest.
    """
    s = str(ct)
    if 'Newborn' in s:                         return (0, s)
    if 'IT-Immature' in s and 'Non' not in s:  return (1, s)
    if 'Non-IT-Immature' in s:                 return (2, s)
    m = re.search(r'L(\d+)', s)
    if m:                                      return (10 + int(m.group(1)), s)
    if s == 'Excitatory':                      return (90, s)
    return (50, s)


# ── IN subtype biological sort key ────────────────────────────────────────────
def _in_sort_key(ct):
    """Biological sort order for IN subtypes: immature first, then MGE, CGE, Mix."""
    s = str(ct)
    if 'Immature' in s:     return (0, s)
    if 'MGE' in s:          return (1, s)
    if 'CGE' in s:          return (2, s)
    if 'Mix' in s:          return (3, s)
    if s == 'Inhibitory':   return (90, s)
    return (50, s)


# ── Canonical type lists for cross-plot palette consistency ───────────────────
# All known Wang EN/IN subtypes in biological order.  The palettes are built
# from this fixed list so the same subtype always gets the same colour.
_EN_CANONICAL_TYPES = [
    'EN-Newborn', 'EN-IT-Immature', 'EN-Non-IT-Immature',
    'EN-L2_3-IT', 'EN-L3_5-IT', 'EN-L4-IT', 'EN-L5-IT',
    'EN-L5_6-IT', 'EN-L5_6-NP', 'EN-L5-ET',
    'EN-L6-CT', 'EN-L6-IT', 'EN-L6b',
    'Excitatory',
]
_IN_CANONICAL_TYPES = [
    'IN-CGE-Immature', 'IN-MGE-Immature', 'IN-NCx_dGE-Immature',
    'IN-MGE-PV', 'IN-MGE-SST',
    'IN-CGE-VIP', 'IN-Mix-LAMP5', 'IN-CGE-SNCG',
    'Inhibitory',
]

# Fixed palettes computed once (initialised below after the helper functions)
_EN_FIXED_PALETTE = None
_IN_FIXED_PALETTE = None


def _make_excitatory_umap_palette(labels=None):
    """Build a stable colour palette for Excitatory subtypes.

    Immature types (EN-Newborn / EN-IT-Immature / EN-Non-IT-Immature) share
    a blue-grey family so they read as related.  Layer-based types get
    distinct colours from a qualitative palette.  The broad 'Excitatory'
    label gets a warm amber.

    When called with labels=None (or from get_en_palette()) the canonical
    type list is used, guaranteeing the same colour for each subtype across
    all plots in a diagnostic run.
    """
    IMMATURE_COLORS = ['#9DC3E6', '#4472C4', '#1F3864']   # light→dark blue-grey
    LAYER_PALETTE   = [
        '#E6550D', '#31A354', '#E7298A', '#FFD700', '#7570B3',
        '#8B4513', '#00CED1', '#A0522D', '#2CA02C', '#D62728',
        '#17BECF', '#BCBD22',
    ]

    if labels is None:
        labels = _EN_CANONICAL_TYPES
    all_labels = sorted(set(str(l) for l in labels), key=_en_sort_key)
    en_labels = [l for l in all_labels
                 if _norm_key(l).startswith('EN_') or l == 'Excitatory']

    result = {}
    imm_idx = layer_idx = 0
    for lbl in en_labels:
        sk = _en_sort_key(lbl)[0]
        if sk < 3:       # immature
            result[lbl] = IMMATURE_COLORS[min(imm_idx, len(IMMATURE_COLORS) - 1)]
            imm_idx += 1
        elif sk >= 90:   # broad 'Excitatory' label
            result[lbl] = '#FF8C00'
        else:            # layer-based types
            result[lbl] = LAYER_PALETTE[layer_idx % len(LAYER_PALETTE)]
            layer_idx += 1
    for lbl in all_labels:
        if lbl not in result:
            c = PALETTE.get(lbl) or PALETTE.get(_norm_key(lbl))
            result[lbl] = c if c else _FALLBACK
    return result


def _make_inhibitory_umap_palette(labels=None):
    """Build a stable colour palette for Inhibitory subtypes.

    Immature types share a muted-green family.
    MGE-derived types (PV, SST) get olive/teal shades.
    CGE-derived types (VIP, LAMP5, SNCG) get purple/magenta shades.
    Mixed / other IN types get orange-brown shades.
    Broad 'Inhibitory' gets a slate blue.
    """
    IMMATURE_COLORS = ['#A8D5A2', '#4CAF50', '#1B6B1B']   # light→dark green
    MGE_PALETTE     = ['#8DB4B4', '#2E8B8B']
    CGE_PALETTE     = ['#C99BD4', '#9B59B6', '#6A0080']
    MIX_PALETTE     = ['#E59866', '#CA6F1E', '#784212']

    if labels is None:
        labels = _IN_CANONICAL_TYPES
    all_labels = sorted(set(str(l) for l in labels), key=_in_sort_key)
    in_labels = [l for l in all_labels
                 if _norm_key(l).startswith('IN_') or l == 'Inhibitory']

    result = {}
    imm_i = mge_i = cge_i = mix_i = 0
    for lbl in in_labels:
        sk = _in_sort_key(lbl)[0]
        if sk == 0:      # immature
            result[lbl] = IMMATURE_COLORS[min(imm_i, len(IMMATURE_COLORS) - 1)]
            imm_i += 1
        elif sk == 1:    # MGE
            result[lbl] = MGE_PALETTE[mge_i % len(MGE_PALETTE)]
            mge_i += 1
        elif sk == 2:    # CGE
            result[lbl] = CGE_PALETTE[cge_i % len(CGE_PALETTE)]
            cge_i += 1
        elif sk == 3:    # Mix
            result[lbl] = MIX_PALETTE[mix_i % len(MIX_PALETTE)]
            mix_i += 1
        elif sk >= 90:   # broad 'Inhibitory' label
            result[lbl] = '#5B7EB5'
    for lbl in all_labels:
        if lbl not in result:
            c = PALETTE.get(lbl) or PALETTE.get(_norm_key(lbl))
            result[lbl] = c if c else _FALLBACK
    return result


def get_en_palette():
    """Return the fixed canonical EN palette (computed once)."""
    global _EN_FIXED_PALETTE
    if _EN_FIXED_PALETTE is None:
        _EN_FIXED_PALETTE = _make_excitatory_umap_palette()
    return _EN_FIXED_PALETTE


def get_in_palette():
    """Return the fixed canonical IN palette (computed once)."""
    global _IN_FIXED_PALETTE
    if _IN_FIXED_PALETTE is None:
        _IN_FIXED_PALETTE = _make_inhibitory_umap_palette()
    return _IN_FIXED_PALETTE


def _is_L23(label):
    """Return True if *label* refers to Layer 2/3 in any naming convention.

    Matches: L2-3, L2_3, EN-L2_3-IT, EN_L2_3_IT, L2/3, L2.3 …
    """
    return bool(re.search(r'L2[_\-\/\.]?3', str(label), re.IGNORECASE))


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

# Expanded marker sets used for PC1-based scatter validation plots.
MARKERS_SCATTER = {
    'Excitatory': ['SLC17A7', 'SATB2', 'NEUROD6', 'TBR1', 'RBFOX3',
                   'CUX2', 'RORB', 'CAMK2A', 'SLC17A6', 'NRGN', 'SNAP25'],
    'Inhibitory': ['GAD1', 'GAD2', 'SLC32A1', 'ADARB2', 'RELN',
                   'PVALB', 'SST', 'VIP', 'LAMP5', 'CXCL14', 'LHX6'],
    'Oligos':     ['MBP', 'PLP1', 'MAG', 'MOG', 'OLIG1', 'OLIG2', 'CNP'],
    'OPC':        ['PDGFRA', 'CSPG4', 'SOX10'],
    'Astrocytes': ['AQP4', 'GFAP', 'SLC1A3', 'ALDH1L1', 'S100B', 'GLUL', 'CLU'],
    'Microglia':  ['CSF1R', 'P2RY12', 'CX3CR1', 'TMEM119', 'AIF1', 'HEXB'],
    'Endothelial':['CLDN5', 'FLT1', 'PECAM1', 'CD34'],
}


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════

def make_tables(tf, out):
    """Cross-tabulation and detail tables for Velmeshev transfer results."""

    # 1. Full cross-tab: raw cell_type → aligned cell type
    ct = pd.crosstab(tf['cell_type_raw'], tf['cell_type_aligned'], margins=True)
    ct.to_csv(os.path.join(out, 'remapping_crosstab.csv'))
    print(f"  remapping_crosstab.csv  ({ct.shape})")

    # 2. Interneurons / INT detail
    for label in ('Interneurons', 'INT'):
        sub = tf[tf['cell_type_raw'] == label]
        if sub.empty:
            continue
        detail = (sub.groupby(['cell_class', 'cell_type_aligned'])
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
    # Uniform 2×2 grid:
    #   TL: overall absolute counts stacked by cell class
    #   TR: fraction stacked by cell class
    #   BL: excitatory subtype absolute counts
    #   BR: excitatory subtype fraction stacked  (matches TR format)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_overall  = axes[0, 0]
    ax_class    = axes[0, 1]
    ax_exc_abs  = axes[1, 0]
    ax_exc_frac = axes[1, 1]

    bins = np.linspace(0, 1, 51)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]

    # ── TL: overall absolute counts stacked by cell class ────────────────────
    classes = sorted(tf['cell_class'].unique())
    class_colors = [CLASS_PALETTE.get(c, _FALLBACK) for c in classes]
    data_by_class = [tf.loc[tf['cell_class'] == c, 'transfer_confidence'].values
                     for c in classes]
    ax_overall.hist(data_by_class, bins=bins, stacked=True,
                    color=class_colors, edgecolor='none', label=classes)
    ax_overall.axvline(0.5, color='red', ls='--', lw=1, label='threshold=0.5')
    ax_overall.set(xlabel='Transfer confidence', ylabel='Cells',
                   title='Overall confidence (stacked by cell class)')
    ax_overall.legend(fontsize=7, ncol=2)

    # ── TR: fraction stacked by cell class ───────────────────────────────────
    counts_cls = np.zeros((len(classes), len(bins) - 1))
    for i, c in enumerate(classes):
        counts_cls[i], _ = np.histogram(
            tf.loc[tf['cell_class'] == c, 'transfer_confidence'].values, bins=bins)
    totals_cls = counts_cls.sum(axis=0)
    totals_cls[totals_cls == 0] = 1
    fracs_cls = counts_cls / totals_cls
    bottom = np.zeros(len(bins) - 1)
    for i, (c, col) in enumerate(zip(classes, class_colors)):
        ax_class.bar(bin_centres, fracs_cls[i], width=width, bottom=bottom,
                     color=col, edgecolor='none', label=c)
        bottom += fracs_cls[i]
    ax_class.axvline(0.5, color='red', ls='--', lw=1)
    ax_class.set(xlabel='Transfer confidence', ylabel='Fraction of cells',
                 ylim=(0, 1), title='Confidence by cell class (fraction stacked)')
    ax_class.legend(fontsize=7, ncol=2)

    # ── BL & BR: excitatory neurons by subtype ───────────────────────────────
    exc = tf[tf['cell_class'] == 'Excitatory'].copy()
    if not exc.empty:
        en_pal = get_en_palette()
        subtype_order = sorted(exc['cell_type_aligned'].unique(), key=_en_sort_key)
        colors_sub = [en_pal.get(ct, _FALLBACK) for ct in subtype_order]

        # BL: absolute counts
        data_by_sub = [exc.loc[exc['cell_type_aligned'] == ct,
                                'transfer_confidence'].values
                       for ct in subtype_order]
        ax_exc_abs.hist(data_by_sub, bins=bins, stacked=True,
                        color=colors_sub, edgecolor='none', label=subtype_order)
        ax_exc_abs.axvline(0.5, color='red', ls='--', lw=1, label='threshold=0.5')
        ax_exc_abs.set(xlabel='Transfer confidence', ylabel='Cells',
                       title=f'Excitatory by subtype — counts  (n={len(exc):,})')
        ax_exc_abs.legend(fontsize=6.5, ncol=3, loc='upper left',
                          framealpha=0.85, handlelength=1.2)

        # BR: fraction stacked (same format as TR)
        counts_sub = np.zeros((len(subtype_order), len(bins) - 1))
        for i, ct in enumerate(subtype_order):
            counts_sub[i], _ = np.histogram(
                exc.loc[exc['cell_type_aligned'] == ct,
                        'transfer_confidence'].values, bins=bins)
        totals_sub = counts_sub.sum(axis=0)
        totals_sub[totals_sub == 0] = 1
        fracs_sub = counts_sub / totals_sub
        bottom = np.zeros(len(bins) - 1)
        for i, (ct, col) in enumerate(zip(subtype_order, colors_sub)):
            ax_exc_frac.bar(bin_centres, fracs_sub[i], width=width, bottom=bottom,
                            color=col, edgecolor='none', label=ct)
            bottom += fracs_sub[i]
        ax_exc_frac.axvline(0.5, color='red', ls='--', lw=1)
        ax_exc_frac.set(xlabel='Transfer confidence', ylabel='Fraction of cells',
                        ylim=(0, 1),
                        title='Excitatory by subtype — fraction stacked')
        ax_exc_frac.legend(fontsize=6.5, ncol=3, loc='upper left',
                           framealpha=0.85, handlelength=1.2)
    else:
        for ax in (ax_exc_abs, ax_exc_frac):
            ax.set_title('Excitatory neurons (none in dataset)')
            ax.axis('off')

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

def _adaptive_umap_point_size(n_points, min_size=0.2, max_size=2.0, scale=220.0):
    """Compute point size that shrinks with dataset size for readability."""
    n = max(int(n_points), 1)
    size = scale / np.sqrt(n)
    return float(np.clip(size, min_size, max_size))


def _scatter_panel(ax, xy, labels, is_target,
                   bg_size=None, bg_alpha=0.15,
                   fg_size=None, fg_alpha=0.5):
    """Plot background (reference) cells in grey, then foreground (target) cells colored."""
    if bg_size is None:
        bg_size = _adaptive_umap_point_size(len(xy), min_size=0.15, max_size=1.0, scale=180.0)
    if fg_size is None:
        fg_size = _adaptive_umap_point_size(is_target.sum(), min_size=0.2, max_size=2.0, scale=260.0)

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
            ('cell_type_raw', 'Source label (cell_type_raw)'),
            ('cell_type_aligned', 'Transferred label (cell_type_aligned)'),
        ]):
            ax = axes[row, col]
            _scatter_panel(ax, xy, sub[label_col], is_tgt)
            _add_panel_legend(ax, sub.loc[is_tgt, label_col])
            ax.set_title(f'{group_name}  —  {row_title}', fontsize=11)

    fig.suptitle(
        'Cell type labels: cell_type_raw vs cell_type_aligned (kNN-transferred) '
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
            ('cell_type_raw', 'Source label (cell_type_raw)'),
            ('cell_type_aligned', 'Transferred label (cell_type_aligned)'),
        ]):
            ax = axes[row, col]
            _scatter_panel(ax, xy, sub[label_col], is_tgt)
            _add_panel_legend(ax, sub.loc[is_tgt.values, label_col])
            ax.set_title(f'{group_name}  —  {row_title}', fontsize=11)

    fig.suptitle(
        'Cell type labels: cell_type_raw vs cell_type_aligned (kNN-transferred) '
        f'({target_source} highlighted, reference cells faded)\n'
        'Per-class recomputed UMAP from scVI embedding',
        fontsize=13, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out, 'umap_perclass.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  umap_perclass.png")


def make_umap_all(all_df, out, target_source='VELMESHEV',
                  umap_cols=('umap_1', 'umap_2'), name='umap_all'):
    """2×2 UMAP of target-source cells only: pre/post-remapping × class / remap-status."""
    vel = all_df[all_df['source'] == target_source].copy()
    xy  = vel[list(umap_cols)].values
    vel['is_class_remapped'] = vel['old_cell_class'] != vel['new_cell_class']

    AGE_BINS   = [-np.inf, 0, 1, 10, np.inf]
    AGE_LABELS = ['Fetal (<0y)', 'Perinatal (0-1y)', 'Childhood (1-10y)', 'Post-10y (>10y)']
    AGE_COLORS = ['#1B9E77', '#E6AB02', '#E7298A', '#7570B3']

    vel['age_cat'] = pd.cut(vel['age_years'], bins=AGE_BINS, labels=AGE_LABELS)
    base_size = _adaptive_umap_point_size(len(vel), min_size=1.0, max_size=6.0, scale=400.0)

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
                       s=base_size, alpha=0.4, linewidths=0, rasterized=True)
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
                       c='#CCCCCC', s=base_size * 0.9, alpha=0.3, linewidths=0, rasterized=True)
            ax.scatter(xy[vel['is_class_remapped'].values, 0],
                       xy[vel['is_class_remapped'].values, 1],
                       c='#D62728', s=base_size * 1.25, alpha=0.5, linewidths=0, rasterized=True)
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
                           c=age_col, s=base_size, alpha=0.45, linewidths=0, rasterized=True,
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
    fname = f'{name}.png'
    plt.savefig(os.path.join(out, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# EXCITATORY-ONLY UMAP
# ══════════════════════════════════════════════════════════════════════════════

def make_umap_excitatory(all_df, out, target_source='VELMESHEV',
                         umap_cols=('umap_1', 'umap_2'), name='umap_excitatory'):
    """2×2 UMAP focusing on Excitatory cells (before OR after remapping).

    Panel layout:
      TL: colour by cell_type_raw
      BL: colour by cell_type_aligned
      TR: remapping-status highlight
            grey  = no remap at all
            red   = within-Excitatory subtype remap
            blue  = Excitatory → other class (lost excitatory)
            green = other class → Excitatory (gained excitatory)
      BR: age category (same bins as umap_all)
    """
    src = all_df[all_df['source'] == target_source].copy()

    # Include cells that were Excitatory before OR after remapping
    mask = (src['old_cell_class'] == 'Excitatory') | (src['new_cell_class'] == 'Excitatory')
    df = src[mask].copy()

    if len(df) < 5:
        print(f"  {name}.png  SKIPPED (too few Excitatory cells: {len(df)})")
        return

    xy = df[list(umap_cols)].values
    base_size = _adaptive_umap_point_size(len(df), min_size=1.5, max_size=8.0, scale=500.0)

    AGE_BINS   = [-np.inf, 0, 1, 10, np.inf]
    AGE_LABELS = ['Fetal (<0y)', 'Perinatal (0-1y)', 'Childhood (1-10y)', 'Post-10y (>10y)']
    AGE_COLORS = ['#1B9E77', '#E6AB02', '#E7298A', '#7570B3']
    df['age_cat'] = pd.cut(df['age_years'], bins=AGE_BINS, labels=AGE_LABELS)

    # ── L2/3 status for top-right panel ──────────────────────────────────────
    raw_is_L23    = df['cell_type_raw'].astype(str).apply(_is_L23)
    aligned_is_L23 = df['cell_type_aligned'].astype(str).apply(_is_L23)

    m_other  = ~raw_is_L23 & ~aligned_is_L23  # other Excitatory (or cross-class)
    m_stayed = raw_is_L23  &  aligned_is_L23  # L2/3 → L2/3
    m_gained = ~raw_is_L23 &  aligned_is_L23  # other → L2/3
    m_lost   =  raw_is_L23 & ~aligned_is_L23  # L2/3 → other

    # BL panel (cell_type_aligned) uses fixed canonical palette — WANG format.
    aligned_pal = get_en_palette()
    # TL panel (cell_type_raw) builds a dynamic palette over the actual raw
    # labels that appear in the Excitatory subset, so non-WANG naming
    # conventions get distinct colours via tab20 fallback.
    _exc_raw_types = df.loc[
        df['old_cell_class'].astype(str) == 'Excitatory', 'cell_type_raw'
    ].astype(str).unique()
    _raw_dyn = _make_local_palette(_exc_raw_types)
    _can_en  = get_en_palette()
    # Canonical palette wins for WANG-format keys already in the dynamic palette.
    raw_pal = {**_raw_dyn, **{k: v for k, v in _can_en.items() if k in _raw_dyn}}

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # ── helpers for consolidated legend ──────────────────────────────────────
    def _consolidated_handles(type_series, class_series, palette,
                               merge_inh_into_other=False):
        """Return scatter handles with Excitatory subtypes listed individually.

        merge_inh_into_other: if True, count Inhibitory cells as Other
        (used for the post-remapping / aligned panel).
        """
        types = type_series.astype(str)
        classes = class_series.astype(str)
        handles_out = []
        if merge_inh_into_other:
            other_n = (~classes.isin(['Excitatory'])).sum()
        else:
            inh_n   = (classes == 'Inhibitory').sum()
            other_n = (~classes.isin(['Excitatory', 'Inhibitory'])).sum()
        # Excitatory subtypes individually, in biological order
        exc_types = sorted(types[classes == 'Excitatory'].unique(),
                           key=_en_sort_key)
        for ct in exc_types:
            n = (types == ct).sum()
            handles_out.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=palette.get(ct, _FALLBACK),
                                      markersize=6, label=f'{ct}  (n={n:,})'))
        if not merge_inh_into_other and inh_n > 0:
            handles_out.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='#5B7EB5', markersize=6,
                                      label=f'Inhibitory  (n={inh_n:,})'))
        if other_n > 0:
            handles_out.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='#999999', markersize=6,
                                      label=f'Other  (n={other_n:,})'))
        return handles_out

    # ── TL: colour by cell_type_raw ───────────────────────────────────────────
    ax = axes[0, 0]
    # Sort raw types in biological EN order so rare-on-top scatter order is consistent
    raw_types = sorted(df['cell_type_raw'].astype(str).unique(), key=_en_sort_key)
    for ct in raw_types:
        m = (df['cell_type_raw'].astype(str) == ct).values
        ct_norm = _norm_key(ct)
        if ct_norm.startswith('IN_') or ct in ('Inhibitory', 'Interneurons', 'INT'):
            col = '#5B7EB5'
        else:
            # raw_pal covers all Excitatory raw types via dynamic palette;
            # anything else (cross-class stray cells) falls back to grey.
            col = raw_pal.get(ct, '#999999')
        ax.scatter(xy[m, 0], xy[m, 1],
                   c=col, s=base_size, alpha=0.45, linewidths=0, rasterized=True)
    handles = _consolidated_handles(df['cell_type_raw'], df['old_cell_class'], raw_pal)
    ax.legend(handles=handles, loc='lower right', fontsize=6,
              framealpha=0.85, edgecolor='0.8', handletextpad=0.3,
              ncol=max(1, len(handles) // 12))
    ax.set_title('Raw cell type (original label)', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # ── BL: colour by cell_type_aligned ──────────────────────────────────────
    # Inhibitory cells in this panel are merged into "Other" (grey) because
    # they are a small minority in an Excitatory-focused UMAP.
    ax = axes[1, 0]
    aligned_types = sorted(df['cell_type_aligned'].astype(str).unique(),
                           key=_en_sort_key)
    for ct in aligned_types:
        m = (df['cell_type_aligned'].astype(str) == ct).values
        ct_norm = _norm_key(ct)
        if ct_norm.startswith('EN_') or ct == 'Excitatory':
            col = aligned_pal.get(ct, _FALLBACK)
        else:
            col = '#999999'   # Inhibitory + other → grey "Other"
        ax.scatter(xy[m, 0], xy[m, 1],
                   c=col, s=base_size, alpha=0.45, linewidths=0, rasterized=True)
    handles = _consolidated_handles(df['cell_type_aligned'], df['new_cell_class'],
                                    aligned_pal, merge_inh_into_other=True)
    ax.legend(handles=handles, loc='lower right', fontsize=6,
              framealpha=0.85, edgecolor='0.8', handletextpad=0.3,
              ncol=max(1, len(handles) // 12))
    ax.set_title('Aligned cell type (after label transfer)', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # ── TR: Layer 2/3 status + cross-class overlay ───────────────────────────
    ax = axes[0, 1]
    old_exc = df['old_cell_class'].astype(str) == 'Excitatory'
    new_exc = df['new_cell_class'].astype(str) == 'Excitatory'
    cross_out = old_exc & ~new_exc   # Excitatory → other class (red)
    cross_in  = ~old_exc & new_exc   # other class → Excitatory (pink)
    cross_class = cross_out | cross_in
    layers = [
        (m_other & ~cross_class, '#CCCCCC', 0.25, base_size * 0.7,
         f'Other Excitatory  (n={(m_other & ~cross_class).sum():,})'),
        (m_stayed, '#1F77B4', 0.75, base_size * 1.3,
         f'L2/3 → L2/3 (stayed)  (n={m_stayed.sum():,})'),
        (m_gained & ~cross_class, '#2CA02C', 0.80, base_size * 1.4,
         f'Other → L2/3 (gained, same class)  (n={(m_gained & ~cross_class).sum():,})'),
        (m_lost & ~cross_class, '#FF7F0E', 0.80, base_size * 1.4,
         f'L2/3 → Other (lost, same class)  (n={(m_lost & ~cross_class).sum():,})'),
        (cross_out, '#D62728', 0.85, base_size * 1.5,
         f'Excitatory → other class  (n={cross_out.sum():,})'),
        (cross_in,  '#F48CB4', 0.85, base_size * 1.5,
         f'Other class → Excitatory  (n={cross_in.sum():,})'),
    ]
    for mask_arr, color, alpha, size, label in layers:
        idx = mask_arr.values if hasattr(mask_arr, 'values') else mask_arr
        if idx.any():
            ax.scatter(xy[idx, 0], xy[idx, 1],
                       c=color, s=size, alpha=alpha, linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color, markersize=6, label=label)
               for (mask_arr, color, alpha, size, label) in layers]
    ax.legend(handles=handles, loc='lower right', fontsize=7,
              framealpha=0.85, edgecolor='0.8', handletextpad=0.3)
    ax.set_title('Layer 2/3 remapping status', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # ── BR: age category ─────────────────────────────────────────────────────
    ax = axes[1, 1]
    for age_lbl, age_col in zip(AGE_LABELS, AGE_COLORS):
        m = (df['age_cat'] == age_lbl).values
        n_age = m.sum()
        if n_age:
            ax.scatter(xy[m, 0], xy[m, 1],
                       c=age_col, s=base_size, alpha=0.45, linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=c, markersize=6,
                      label=f'{l}  (n={( df["age_cat"]==l).sum():,})')
               for l, c in zip(AGE_LABELS, AGE_COLORS)]
    ax.legend(handles=handles, loc='lower right', fontsize=7,
              framealpha=0.85, edgecolor='0.8', handletextpad=0.3)
    ax.set_title('Donor age category', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f'{target_source} — Excitatory cells (n={len(df):,}; '
        f'pre- or post-remap class = Excitatory)\nGlobal UMAP embedding',
        fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f'{name}.png'
    plt.savefig(os.path.join(out, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# INHIBITORY-ONLY UMAP
# ══════════════════════════════════════════════════════════════════════════════

def make_umap_inhibitory(all_df, out, target_source='VELMESHEV',
                         umap_cols=('umap_1', 'umap_2'), name='umap_inhibitory'):
    """2×2 UMAP focusing on Inhibitory cells (before OR after remapping).

    Panel layout:
      TL: colour by cell_type_raw (original label)
      BL: colour by cell_type_aligned (scANVI prediction)
      TR: MGE / CGE / Immature lineage highlight + cross-class overlay
      BR: donor age category

    Colours are drawn from the canonical IN palette (get_in_palette()) so
    they match the subtype_distribution plots.
    """
    src = all_df[all_df['source'] == target_source].copy()
    mask = (src['old_cell_class'] == 'Inhibitory') | (src['new_cell_class'] == 'Inhibitory')
    df = src[mask].copy()

    if len(df) < 5:
        print(f"  {name}.png  SKIPPED (too few Inhibitory cells: {len(df)})")
        return

    xy = df[list(umap_cols)].values
    base_size = _adaptive_umap_point_size(len(df), min_size=1.5, max_size=8.0, scale=500.0)

    AGE_BINS   = [-np.inf, 0, 1, 10, np.inf]
    AGE_LABELS = ['Fetal (<0y)', 'Perinatal (0-1y)', 'Childhood (1-10y)', 'Post-10y (>10y)']
    AGE_COLORS = ['#1B9E77', '#E6AB02', '#E7298A', '#7570B3']
    df['age_cat'] = pd.cut(df['age_years'], bins=AGE_BINS, labels=AGE_LABELS)

    # BL panel (cell_type_aligned) uses the fixed canonical IN palette.
    in_pal = get_in_palette()
    # TL panel (cell_type_raw) builds a dynamic palette over the actual raw
    # labels in the Inhibitory subset so non-WANG naming gets distinct colours.
    _inh_raw_types = df.loc[
        df['old_cell_class'].astype(str) == 'Inhibitory', 'cell_type_raw'
    ].astype(str).unique()
    _raw_in_dyn = _make_local_palette(_inh_raw_types)
    _can_in     = get_in_palette()
    raw_in_pal  = {**_raw_in_dyn, **{k: v for k, v in _can_in.items() if k in _raw_in_dyn}}

    # ── lineage classification helpers ────────────────────────────────────────
    def _in_lineage(label):
        s = str(label)
        if 'Immature' in s:  return 'immature'
        if 'MGE' in s:       return 'mge'
        if 'CGE' in s or 'Mix' in s: return 'cge'
        return 'other'

    def _in_consolidated_handles(type_series, class_series, palette,
                                  merge_exc_into_other=True):
        types   = type_series.astype(str)
        classes = class_series.astype(str)
        handles_out = []
        other_n = (~classes.isin(['Inhibitory'])).sum() if merge_exc_into_other else 0
        in_types = sorted(
            types[classes == 'Inhibitory'].unique(), key=_in_sort_key)
        for ct in in_types:
            n = (types == ct).sum()
            handles_out.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=palette.get(ct, _FALLBACK),
                                      markersize=6, label=f'{ct}  (n={n:,})'))
        if other_n > 0:
            handles_out.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='#999999', markersize=6,
                                      label=f'Other  (n={other_n:,})'))
        return handles_out

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # ── TL: colour by cell_type_raw ───────────────────────────────────────────
    ax = axes[0, 0]
    raw_types = sorted(df['cell_type_raw'].astype(str).unique(), key=_in_sort_key)
    for ct in raw_types:
        m = (df['cell_type_raw'].astype(str) == ct).values
        ct_norm = _norm_key(ct)
        if ct_norm.startswith('EN_') or ct == 'Excitatory':
            col = '#E67E22'   # warm orange for stray excitatory raw labels
        else:
            # raw_in_pal covers all IN raw types via dynamic palette;
            # anything else falls back to grey.
            col = raw_in_pal.get(ct, '#999999')
        ax.scatter(xy[m, 0], xy[m, 1],
                   c=col, s=base_size, alpha=0.45, linewidths=0, rasterized=True)
    handles = _in_consolidated_handles(df['cell_type_raw'], df['old_cell_class'], raw_in_pal)
    ax.legend(handles=handles, loc='lower right', fontsize=6,
              framealpha=0.85, edgecolor='0.8', handletextpad=0.3,
              ncol=max(1, len(handles) // 12))
    ax.set_title('Raw cell type (original label)', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # ── BL: colour by cell_type_aligned (Excitatory → Other) ─────────────────
    ax = axes[1, 0]
    aligned_types = sorted(df['cell_type_aligned'].astype(str).unique(), key=_in_sort_key)
    for ct in aligned_types:
        m = (df['cell_type_aligned'].astype(str) == ct).values
        ct_norm = _norm_key(ct)
        if ct_norm.startswith('IN_') or ct == 'Inhibitory':
            col = in_pal.get(ct, _FALLBACK)
        else:
            col = '#999999'
        ax.scatter(xy[m, 0], xy[m, 1],
                   c=col, s=base_size, alpha=0.45, linewidths=0, rasterized=True)
    handles = _in_consolidated_handles(df['cell_type_aligned'], df['new_cell_class'],
                                       in_pal, merge_exc_into_other=True)
    ax.legend(handles=handles, loc='lower right', fontsize=6,
              framealpha=0.85, edgecolor='0.8', handletextpad=0.3,
              ncol=max(1, len(handles) // 12))
    ax.set_title('Aligned cell type (after label transfer)', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # ── TR: MGE / CGE / Immature lineage highlight ────────────────────────────
    ax = axes[0, 1]
    old_inh = df['old_cell_class'].astype(str) == 'Inhibitory'
    new_inh = df['new_cell_class'].astype(str) == 'Inhibitory'
    cross_out = old_inh & ~new_inh
    cross_in  = ~old_inh & new_inh
    cross_class = cross_out | cross_in

    raw_lineage = df['cell_type_raw'].astype(str).apply(_in_lineage)
    aln_lineage = df['cell_type_aligned'].astype(str).apply(_in_lineage)

    lineage_layers = [
        (cross_out,                  '#D62728', 0.85, base_size * 1.5,
         f'Inhibitory → other class  (n={cross_out.sum():,})'),
        (cross_in,                   '#F48CB4', 0.85, base_size * 1.5,
         f'Other class → Inhibitory  (n={cross_in.sum():,})'),
        ((raw_lineage == 'other') & ~cross_class, '#CCCCCC', 0.25, base_size * 0.7,
         f'Other IN (unclassified)  (n={((raw_lineage=="other")&~cross_class).sum():,})'),
        ((raw_lineage == 'immature') & ~cross_class, '#A8D5A2', 0.75, base_size * 1.2,
         f'Immature  (n={((raw_lineage=="immature")&~cross_class).sum():,})'),
        ((raw_lineage == 'mge') & ~cross_class,     '#2E8B8B', 0.80, base_size * 1.3,
         f'MGE-derived  (n={((raw_lineage=="mge")&~cross_class).sum():,})'),
        ((raw_lineage == 'cge') & ~cross_class,     '#9B59B6', 0.80, base_size * 1.3,
         f'CGE/Mix-derived  (n={((raw_lineage=="cge")&~cross_class).sum():,})'),
    ]
    for mask_arr, color, alpha, size, label in lineage_layers:
        idx = mask_arr.values if hasattr(mask_arr, 'values') else mask_arr
        if idx.any():
            ax.scatter(xy[idx, 0], xy[idx, 1],
                       c=color, s=size, alpha=alpha, linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=col, markersize=6, label=lbl)
               for (_, col, _, _, lbl) in lineage_layers]
    ax.legend(handles=handles, loc='lower right', fontsize=7,
              framealpha=0.85, edgecolor='0.8', handletextpad=0.3)
    ax.set_title('IN lineage (MGE / CGE / Immature)', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # ── BR: age category ─────────────────────────────────────────────────────
    ax = axes[1, 1]
    for age_lbl, age_col in zip(AGE_LABELS, AGE_COLORS):
        m = (df['age_cat'] == age_lbl).values
        if m.sum():
            ax.scatter(xy[m, 0], xy[m, 1],
                       c=age_col, s=base_size, alpha=0.45, linewidths=0, rasterized=True)
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=c, markersize=6,
                      label=f'{l}  (n={(df["age_cat"]==l).sum():,})')
               for l, c in zip(AGE_LABELS, AGE_COLORS)]
    ax.legend(handles=handles, loc='lower right', fontsize=7,
              framealpha=0.85, edgecolor='0.8', handletextpad=0.3)
    ax.set_title('Donor age category', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f'{target_source} — Inhibitory cells (n={len(df):,}; '
        f'pre- or post-remap class = Inhibitory)\nGlobal UMAP embedding',
        fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f'{name}.png'
    plt.savefig(os.path.join(out, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# SANKEY DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════

def _make_sankey_panel(ax, tf_sub, title, within_color='#4DBEEE',
                       cross_color='#D62728'):
    """Single sankey panel: cell_type_raw → cell_type_aligned.

    Within-class flows in blue, cross-class flows in red.
    """
    from pipeline.label_transfer.transfer import aligned_to_class

    # Build flow table: cell_type_raw → cell_type_aligned
    flows = (tf_sub.groupby(['cell_type_raw', 'cell_type_aligned'], observed=True)
             .size().reset_index(name='n'))
    flows = flows[flows['n'] > 0].sort_values('n', ascending=False)

    if flows.empty:
        ax.set_title(title)
        ax.axis('off')
        return

    # Determine if each flow is within-class or cross-class
    flows['new_class'] = flows['cell_type_aligned'].map(aligned_to_class)
    ct_class = tf_sub.groupby('cell_type_raw', observed=True)['old_cell_class'].first().to_dict()
    flows['old_class'] = flows['cell_type_raw'].map(ct_class)
    flows['is_cross'] = flows['old_class'] != flows['new_class']

    # Filter to informative flows for readability.
    total_flows = int(flows['n'].sum())
    flows['pct_panel'] = flows['n'] / total_flows

    # Restrict to dominant node labels on each side to avoid unreadable lists.
    left_totals = flows.groupby('cell_type_raw', observed=True)['n'].sum().sort_values(ascending=False)
    right_totals = flows.groupby('cell_type_aligned', observed=True)['n'].sum().sort_values(ascending=False)
    keep_left = set(left_totals.head(10).index.tolist())
    keep_right = set(right_totals.head(10).index.tolist())
    flows = flows[
        flows['cell_type_raw'].isin(keep_left)
        & flows['cell_type_aligned'].isin(keep_right)
    ].copy()

    min_n = max(20, int(0.01 * total_flows))
    flows = flows[flows['n'] >= min_n].copy()
    flows = flows[flows['pct_panel'] >= 0.01].copy()

    # Keep enough flows to cover most mass, while capping visual clutter.
    if not flows.empty:
        flows = flows.sort_values('n', ascending=False)
        flows['cum_pct'] = flows['pct_panel'].cumsum()
        keep = (flows['cum_pct'] <= 0.95)
        keep.iloc[:8] = True  # always retain at least a compact top set
        flows = flows[keep].copy()

    if flows.empty:
        ax.set_title(title)
        ax.axis('off')
        return

    # Left side: cell_type_raw, right side: cell_type_aligned
    left_labels = flows.groupby('cell_type_raw', observed=True)['n'].sum().sort_values(ascending=True)
    right_labels = flows.groupby('cell_type_aligned', observed=True)['n'].sum().sort_values(ascending=True)
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
            lbl_l, lbl_r, n = row['cell_type_raw'], row['cell_type_aligned'], row['n']
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
        cls_color = CLASS_PALETTE.get(aligned_to_class(lbl), _FALLBACK)
        ax.barh((y0 + y1) / 2, bar_width, height=y1 - y0,
                left=x_right - bar_width,
                color=cls_color, edgecolor='white', linewidth=0.5, zorder=3)
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

    title = f'Sankey: cell_type_raw → kNN-transferred cell_type_aligned (Wang reference)'
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

    # Gene symbol → Ensembl ID (var.index).
    # Some pipelines store gene_symbol as "SYMBOL_ENSGxxx" (symbol + Ensembl
    # suffix separated by "_ENSG").  Strip that suffix so lookups like
    # sym2ens['SLC17A7'] work even when stored as 'SLC17A7_ENSG00000104888'.
    var = adata.var
    sym2ens = {}
    if 'gene_symbol' in var.columns:
        for gs, ens in zip(var['gene_symbol'], var.index):
            full = str(gs)
            sym2ens[full] = ens                     # keep full form as fallback
            short = full.split('_ENSG')[0]
            if short != full:
                sym2ens[short] = ens                # also map bare symbol
    else:
        # var.index may be gene symbols directly
        sym2ens = {g: g for g in var.index}

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

    # Scale per-10k scVI values → per-1M (CPM) before log1p
    expr = np.log1p(expr * 100.0)

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
                 f'(log1p CPM; reference: {ref_label})',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fname = f'marker_validation{suffix}.png'
    plt.savefig(os.path.join(out, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  {fname}")


def make_marker_scatter_validation(tf, adata, out, max_ref=5000,
                                   ref_sources=None, age_lo=-np.inf, age_hi=np.inf,
                                   suffix='', period_label=''):
    """PC1-score scatter plot for class-remapped cells vs reference.

    For each major class remapping, computes PC1 of the old-class marker genes
    and PC1 of the new-class marker genes (fit on reference cells, then
    transformed for all three groups).  Scatter: x = old-class PC1,
    y = new-class PC1; points coloured by group (ref_old / remapped / ref_new).

    The axis titles include the top 5 genes by |loading|.  The full marker
    gene list used is printed below each subplot.
    """
    from sklearn.decomposition import PCA
    import scipy.sparse as sp

    if ref_sources is None:
        ref_sources = ['WANG']

    remap = tf[tf['is_class_remapped'] &
               (tf['age_years'] >= age_lo) &
               (tf['age_years'] < age_hi)].copy()
    if remap.empty:
        print(f"  No class-remapped cells in [{age_lo}, {age_hi}) — skipping scatter{suffix}.")
        return

    # Gene symbol → Ensembl ID mapping (same logic as make_marker_validation)
    var = adata.var
    sym2ens = {}
    if 'gene_symbol' in var.columns:
        for gs, ens in zip(var['gene_symbol'], var.index):
            full = str(gs)
            sym2ens[full] = ens
            short = full.split('_ENSG')[0]
            if short != full:
                sym2ens[short] = ens
    else:
        sym2ens = {g: g for g in var.index}

    top_remaps = (remap.groupby(['old_cell_class', 'new_cell_class'])
                  .size().sort_values(ascending=False).head(3))

    # Collect all marker genes needed across all remappings
    all_markers_needed = set()
    for (old_cls, new_cls), _ in top_remaps.items():
        for cls in [old_cls, new_cls]:
            all_markers_needed.update(MARKERS_SCATTER.get(cls, []))

    marker_ens = {sym: sym2ens[sym] for sym in all_markers_needed if sym in sym2ens}
    if not marker_ens:
        print(f"  No MARKERS_SCATTER genes found in adata — skipping scatter{suffix}.")
        return

    # Collect all cell positions needed
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
    ens_to_sym = {v: k for k, v in marker_ens.items()}
    var_indices = [ens_to_varidx[e] for e in valid_ens]
    ref_label = '+'.join(ref_sources)

    print(f"    [scatter {period_label or 'all'}] {len(all_positions)} cells × "
          f"{len(valid_ens)} markers …")

    # Load expression in chunks
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
    expr = np.log1p(expr * 100.0)

    n_remaps = len(top_remaps)
    fig, axes = plt.subplots(1, n_remaps, figsize=(5.5 * n_remaps, 5.5), squeeze=False)
    plt.subplots_adjust(bottom=0.30, wspace=0.35)

    for col_i, ((old_cls, new_cls), n_cells) in enumerate(top_remaps.items()):
        ax = axes[0, col_i]

        old_markers = [g for g in MARKERS_SCATTER.get(old_cls, [])
                       if marker_ens.get(g) in valid_ens]
        new_markers = [g for g in MARKERS_SCATTER.get(new_cls, [])
                       if marker_ens.get(g) in valid_ens]

        if not old_markers or not new_markers:
            ax.set_title(f'{old_cls} → {new_cls}\n(insufficient markers)', fontsize=10)
            ax.axis('off')
            continue

        # Column indices in expr for each marker set
        old_cols = [valid_ens.index(marker_ens[g]) for g in old_markers]
        new_cols = [valid_ens.index(marker_ens[g]) for g in new_markers]

        remap_sub = remap[(remap['old_cell_class'] == old_cls) &
                          (remap['new_cell_class'] == new_cls)]
        remap_local = np.array([pos_to_local[p] for p in remap_sub['h5ad_pos'].values
                                if p in pos_to_local])
        ref_old_local = np.array([pos_to_local[p] for p in ref_positions.get(old_cls, [])
                                  if p in pos_to_local])
        ref_new_local = np.array([pos_to_local[p] for p in ref_positions.get(new_cls, [])
                                  if p in pos_to_local])

        if len(ref_old_local) == 0 or len(ref_new_local) == 0:
            ax.set_title(f'{old_cls} → {new_cls}\n(no reference cells)', fontsize=10)
            ax.axis('off')
            continue

        # Fit PC1 on old-class reference, transform all groups
        pca_old = PCA(n_components=1)
        pca_old.fit(expr[ref_old_local][:, old_cols])
        all_local = np.concatenate([ref_old_local, remap_local, ref_new_local])
        x_all = pca_old.transform(expr[all_local][:, old_cols])[:, 0]
        n_ref_old = len(ref_old_local)
        n_remap   = len(remap_local)
        x_ref_old = x_all[:n_ref_old]
        x_remap   = x_all[n_ref_old:n_ref_old + n_remap]
        x_ref_new = x_all[n_ref_old + n_remap:]

        # Fit PC1 on new-class reference, transform all groups
        pca_new = PCA(n_components=1)
        pca_new.fit(expr[ref_new_local][:, new_cols])
        y_all = pca_new.transform(expr[all_local][:, new_cols])[:, 0]
        y_ref_old = y_all[:n_ref_old]
        y_remap   = y_all[n_ref_old:n_ref_old + n_remap]
        y_ref_new = y_all[n_ref_old + n_remap:]

        # Top 5 genes by |PC1 loading|
        def _top5(pca, genes):
            loadings = np.abs(pca.components_[0])
            idx = np.argsort(loadings)[::-1][:5]
            return [genes[i] for i in idx]

        top5_old = _top5(pca_old, old_markers)
        top5_new = _top5(pca_new, new_markers)

        # Scatter plot
        pt_size = max(2, min(10, 30000 // max(len(ref_old_local), 1)))
        ax.scatter(x_ref_old, y_ref_old, c='#377EB8', alpha=0.3, s=pt_size,
                   linewidths=0, rasterized=True,
                   label=f'Ref {old_cls} (n={len(ref_old_local):,})')
        ax.scatter(x_ref_new, y_ref_new, c='#4DAF4A', alpha=0.3, s=pt_size,
                   linewidths=0, rasterized=True,
                   label=f'Ref {new_cls} (n={len(ref_new_local):,})')
        ax.scatter(x_remap, y_remap, c='#D62728', alpha=0.7, s=pt_size * 1.5,
                   linewidths=0, rasterized=True, zorder=3,
                   label=f'Remapped (n={n_cells:,})')

        top5_old_str = ', '.join(top5_old)
        top5_new_str = ', '.join(top5_new)
        ax.set_xlabel(f'{old_cls} PC1  ({top5_old_str}, ++)', fontsize=8)
        ax.set_ylabel(f'{new_cls} PC1  ({top5_new_str}, ++)', fontsize=8)
        ax.set_title(f'{old_cls} → {new_cls}  (n={n_cells:,})', fontsize=10)
        ax.legend(fontsize=7, markerscale=2)
        ax.tick_params(labelsize=7)

        # Full marker list below the axes
        old_used_str = ', '.join(old_markers)
        new_used_str = ', '.join(new_markers)
        marker_txt = (f'{old_cls} markers: {old_used_str}\n'
                      f'{new_cls} markers: {new_used_str}')
        ax.text(0.5, -0.28, marker_txt, transform=ax.transAxes,
                fontsize=6, ha='center', va='top', style='italic',
                wrap=True)

    title_period = f' — {period_label}' if period_label else ''
    fig.suptitle(
        f'Marker PC1 scatter: remapped cells vs reference{title_period}\n'
        f'x = PC1 of original-class markers; y = PC1 of remapped-class markers\n'
        f'(log1p CPM, fit on ref cells; reference: {ref_label})',
        fontsize=10, y=1.01)
    fname = f'marker_scatter_validation{suffix}.png'
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

    # Guard against NaN/Inf ages in sparse remapping subsets.
    remap['age_years'] = pd.to_numeric(remap['age_years'], errors='coerce')
    remap = remap[np.isfinite(remap['age_years'])].copy()
    if remap.empty:
        print("  No finite ages in class-remapped cells — skipping age histogram.")
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
        age = pd.to_numeric(sub['age_years'], errors='coerce').clip(upper=30)
        conf = pd.to_numeric(sub['transfer_confidence'], errors='coerce')
        valid = np.isfinite(age.values) & np.isfinite(conf.values)
        age = age[valid]
        conf = conf[valid]

        if len(age) == 0:
            ax.text(0.5, 0.5, 'No finite age/conf data',
                ha='center', va='center', fontsize=8)
            ax.set_title(f'{oc} → {nc}  (n={n_cells:,})', fontsize=9)
            ax.set_xlim(-5, 30)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel('Age (years)', fontsize=8)
            ax.set_ylabel('Transfer confidence', fontsize=8)
            ax.tick_params(labelsize=7)
            continue

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
        print("\n── All-cells UMAP ──")
        make_umap_all(all_df, src_out, target_source=src)

        print("\n── Excitatory-cells UMAP ──")
        make_umap_excitatory(all_df, src_out, target_source=src)

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
