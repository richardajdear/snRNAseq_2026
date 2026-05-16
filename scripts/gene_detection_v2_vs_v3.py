"""
Per-gene detection comparison between 10x Chromium v2 and v3 chemistry.

Computes per-gene detection_rate, mean_counts and mean_cpm separately for V2 and V3
cells (sourced from Velmeshev, which has both chemistries; optionally PsychAD V3
can be appended for additional V3 cells). Outputs a CSV joined with AHBA C3+/C3-
GRN membership, a plain-text "well-detected in both" gene list, and diagnostic
PNGs.

Usage:
    # Test (10k):
    python scripts/gene_detection_v2_vs_v3.py \\
        --velmeshev /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_10k.h5ad \\
        --output-dir scripts/outputs/gene_detection_v2_vs_v3_TEST

    # Full (default paths):
    python scripts/gene_detection_v2_vs_v3.py
        # → uses VELMESHEV_PATH from read_data.py
        # → output dir: scripts/outputs/gene_detection_v2_vs_v3
"""

import argparse
import os
import sys
import gc
import numpy as np
import pandas as pd
import scipy.sparse as sp

import scanpy as sc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO_ROOT, 'code') not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, 'code'))

from pipeline.read_data import (
    VELMESHEV_PATH, VELMESHEV_META_DIR,
    PSYCHAD_AGING_PATH, PSYCHAD_HBCC_PATH,
    read_velmeshev_backed, read_psychad_backed,
)
from regulons import get_ahba_GRN


# ── Per-gene stats over chunked sparse counts ─────────────────────────────────

def per_gene_stats(adata_backed, cell_barcodes, label, chunk_size=50_000):
    """Compute per-gene detection_rate, mean_counts, mean_cpm for the given cells.

    Parameters
    ----------
    adata_backed : AnnData
        Backed-mode AnnData of raw counts.
    cell_barcodes : Index | array-like
        Cell barcodes to include.
    label : str
        Tag for logging ("V2" / "V3").
    chunk_size : int
        Cells per chunk during materialisation.

    Returns
    -------
    dict with arrays of length n_genes:
        detection_rate, mean_counts, mean_cpm, n_cells.
    """
    obs_names = adata_backed.obs_names
    mask = obs_names.isin(cell_barcodes)
    cell_idx = np.where(mask)[0]
    n_cells_used = int(len(cell_idx))
    n_genes = int(adata_backed.shape[1])

    print(f"  [{label}] {n_cells_used:,} cells × {n_genes:,} genes")

    if n_cells_used == 0:
        nan = np.full(n_genes, np.nan, dtype=np.float64)
        return dict(detection_rate=nan, mean_counts=nan, mean_cpm=nan,
                    n_cells=0)

    nonzero_count = np.zeros(n_genes, dtype=np.int64)
    sum_counts    = np.zeros(n_genes, dtype=np.float64)
    sum_cpm       = np.zeros(n_genes, dtype=np.float64)

    n_chunks = (n_cells_used + chunk_size - 1) // chunk_size
    for i_chunk in range(n_chunks):
        c_start = i_chunk * chunk_size
        c_end   = min(c_start + chunk_size, n_cells_used)
        # Sort indices for sequential HDF5 reads
        sub_idx = cell_idx[c_start:c_end]
        sub_idx = np.sort(sub_idx)
        chunk = adata_backed[sub_idx].to_memory()
        X = chunk.X
        # Coerce to sparse CSR (cells × genes)
        if not sp.issparse(X):
            X = sp.csr_matrix(np.asarray(X, dtype=np.float32))
        else:
            X = X.tocsr().astype(np.float32, copy=False)

        # Per-gene non-zero counts via CSC indptr diffs
        X_csc = X.tocsc()
        chunk_nonzero = np.diff(X_csc.indptr)  # length = n_genes
        chunk_sum     = np.asarray(X.sum(axis=0)).ravel()

        # Per-cell totals for CPM
        cell_totals = np.asarray(X.sum(axis=1)).ravel()
        cell_totals[cell_totals == 0] = 1.0
        # CPM = X / cell_totals * 1e6 (row-wise scale)
        inv_tot = (1e6 / cell_totals).astype(np.float32)
        # Multiply each row of CSR by inv_tot (broadcast via sp.diags)
        cpm = sp.diags(inv_tot, format='csr') @ X
        chunk_cpm_sum = np.asarray(cpm.sum(axis=0)).ravel()

        nonzero_count += chunk_nonzero.astype(np.int64)
        sum_counts    += chunk_sum
        sum_cpm       += chunk_cpm_sum

        if (i_chunk + 1) % 5 == 0 or (i_chunk + 1) == n_chunks:
            print(f"    chunk {i_chunk+1}/{n_chunks}  "
                  f"({c_end:,}/{n_cells_used:,} cells)")
        del chunk, X, X_csc, cpm, cell_totals, inv_tot
        gc.collect()

    return dict(
        detection_rate=nonzero_count / n_cells_used,
        mean_counts=sum_counts / n_cells_used,
        mean_cpm=sum_cpm / n_cells_used,
        n_cells=n_cells_used,
    )


# ── GRN membership ────────────────────────────────────────────────────────────

def load_grn_membership(ref_dir):
    """Return two sets of gene symbols: top-1000 C3+ and top-1000 C3-."""
    grn_file = os.path.join(ref_dir, 'ahba_dme_hcp_top8kgenes_weights.csv')
    grn = get_ahba_GRN(path_to_ahba_weights=grn_file, use_weights=False)
    c3pos = set(grn.loc[grn['Network'] == 'C3+', 'Gene'].astype(str).tolist())
    c3neg = set(grn.loc[grn['Network'] == 'C3-', 'Gene'].astype(str).tolist())
    print(f"AHBA GRN: top-1000 C3+ ({len(c3pos)} genes), top-1000 C3- ({len(c3neg)} genes)")
    return c3pos, c3neg


def _gene_symbol_series(adata_backed):
    """Return per-gene symbols best-effort: var['gene_symbol'] / 'feature_name' / var_names."""
    var = adata_backed.var
    for col in ('gene_symbol', 'feature_name', 'gene_symbols'):
        if col in var.columns:
            return var[col].astype(str)
    # Fallback: var_names
    return pd.Series(adata_backed.var_names.astype(str), index=adata_backed.var_names)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _color_by_grn(in_c3pos, in_c3neg):
    """One color per gene: C3+ red, C3- blue, other grey."""
    n = len(in_c3pos)
    colors = np.array(['#BBBBBB'] * n, dtype=object)
    colors[in_c3pos.values] = '#D62728'
    colors[in_c3neg.values] = '#1F77B4'
    return colors


def plot_scatter_detection(df, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
    colors = _color_by_grn(df['in_C3_pos_top1000'], df['in_C3_neg_top1000'])
    order = np.argsort(df['in_C3_pos_top1000'].astype(int) +
                       df['in_C3_neg_top1000'].astype(int))  # GRN on top
    ax.scatter(df['detection_v2'].values[order],
               df['detection_v3'].values[order],
               c=colors[order], s=4, alpha=0.6, linewidths=0)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.5, alpha=0.5)
    ax.set_xlabel('V2 detection rate')
    ax.set_ylabel('V3 detection rate')
    ax.set_title(f'Detection rate V2 vs V3 ({len(df):,} genes)')
    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', linestyle='', color='#D62728', label='C3+ top-1000'),
               Line2D([], [], marker='o', linestyle='', color='#1F77B4', label='C3- top-1000'),
               Line2D([], [], marker='o', linestyle='', color='#BBBBBB', label='other')]
    ax.legend(handles=handles, loc='upper left', fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_hist_detection(df, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    bins = np.linspace(0, 1, 40)
    ax.hist(df['detection_v2'], bins=bins, alpha=0.5, label='V2', color='steelblue')
    ax.hist(df['detection_v3'], bins=bins, alpha=0.5, label='V3', color='indianred')
    ax.set_xlabel('detection rate')
    ax.set_ylabel('# genes')
    ax.set_title('Per-gene detection rate distribution')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_hist_cpm(df, out_path):
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    bins = np.linspace(-2, 5, 50)
    ax.hist(np.log10(df['mean_cpm_v2'] + 1e-3), bins=bins, alpha=0.5,
            label='V2', color='steelblue')
    ax.hist(np.log10(df['mean_cpm_v3'] + 1e-3), bins=bins, alpha=0.5,
            label='V3', color='indianred')
    ax.set_xlabel(r'$\log_{10}$(mean CPM + 1e-3)')
    ax.set_ylabel('# genes')
    ax.set_title('Per-gene mean CPM distribution')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_lfc_vs_mean(df, out_path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    eps = 1e-3
    x = np.log10(0.5 * (df['detection_v2'] + df['detection_v3']) + eps)
    y = np.log2((df['detection_v3'] + eps) / (df['detection_v2'] + eps))
    colors = _color_by_grn(df['in_C3_pos_top1000'], df['in_C3_neg_top1000'])
    order = np.argsort(df['in_C3_pos_top1000'].astype(int) +
                       df['in_C3_neg_top1000'].astype(int))
    ax.scatter(x.values[order], y.values[order],
               c=colors[order], s=4, alpha=0.6, linewidths=0)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel(r'$\log_{10}$(mean detection across V2, V3)')
    ax.set_ylabel(r'$\log_2$(V3 / V2 detection)')
    ax.set_title('Per-gene detection LFC vs mean')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_grn_coverage(df, out_path):
    def _categorise(row):
        v2 = row['well_detected_v2']
        v3 = row['well_detected_v3']
        if v2 and v3:    return 'both'
        if v2 and not v3: return 'V2 only'
        if v3 and not v2: return 'V3 only'
        return 'neither'

    df = df.copy()
    df['coverage'] = df.apply(_categorise, axis=1)
    cats = ['both', 'V2 only', 'V3 only', 'neither']

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150, sharey=True)
    for ax, grn_col, title in zip(
            axes,
            ['in_C3_pos_top1000', 'in_C3_neg_top1000'],
            ['C3+ top-1000', 'C3- top-1000']):
        sub = df[df[grn_col]]
        counts = sub['coverage'].value_counts().reindex(cats, fill_value=0)
        ax.bar(cats, counts.values,
               color=['#2C7BB6', '#88AAEE', '#EE8866', '#BBBBBB'])
        ax.set_title(f'{title}: well-detected coverage (n={len(sub)})')
        ax.set_xlabel('chemistry coverage')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 1, str(int(v)), ha='center', fontsize=9)
    axes[0].set_ylabel('# genes')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--velmeshev', default=VELMESHEV_PATH,
                        help='Path to Velmeshev h5ad (defaults to full dataset).')
    parser.add_argument('--velmeshev-meta-dir', default=VELMESHEV_META_DIR,
                        help='Path to Velmeshev metadata TSV directory.')
    parser.add_argument('--include-psychad', action='store_true',
                        help='Also pool PsychAD V3 cells into the V3 statistics.')
    parser.add_argument('--psychad-aging', default=PSYCHAD_AGING_PATH)
    parser.add_argument('--psychad-hbcc',  default=PSYCHAD_HBCC_PATH)
    parser.add_argument('--ref-dir',
                        default=os.path.join(REPO_ROOT, 'reference'),
                        help='Reference dir containing AHBA weights CSV.')
    parser.add_argument('--output-dir',
                        default=os.path.join(REPO_ROOT, 'scripts/outputs/gene_detection_v2_vs_v3'),
                        help='Output directory.')
    parser.add_argument('--detect-min', type=float, default=0.05,
                        help='Minimum detection rate (fraction cells with count>0) per chemistry.')
    parser.add_argument('--lfc-max', type=float, default=1.0,
                        help='Maximum |log2(v3/v2)| of detection rate for robust_to_chemistry.')
    parser.add_argument('--chunk-size', type=int, default=50_000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # ── Load Velmeshev (has both V2 and V3) ──────────────────────────────────
    print(f"\n[1/4] Loading Velmeshev: {args.velmeshev}")
    vel_backed, vel_meta = read_velmeshev_backed(args.velmeshev, args.velmeshev_meta_dir)
    if vel_backed is None:
        raise RuntimeError(f"Failed to load Velmeshev from {args.velmeshev}")
    print(f"  Velmeshev cells in metadata: {len(vel_meta):,}")
    print(f"  Chemistry counts: "
          f"{dict(vel_meta['chemistry'].astype(str).value_counts())}")

    v2_barcodes = vel_meta.index[vel_meta['chemistry'].astype(str) == 'V2']
    v3_barcodes = vel_meta.index[vel_meta['chemistry'].astype(str) == 'V3']
    print(f"  V2 cells: {len(v2_barcodes):,}  |  V3 cells: {len(v3_barcodes):,}")

    # ── Per-gene stats ───────────────────────────────────────────────────────
    print("\n[2/4] Computing per-gene stats — V2")
    v2_stats = per_gene_stats(vel_backed, v2_barcodes, 'V2',
                              chunk_size=args.chunk_size)
    print("\n[2/4] Computing per-gene stats — V3 (Velmeshev)")
    v3_stats = per_gene_stats(vel_backed, v3_barcodes, 'V3',
                              chunk_size=args.chunk_size)

    gene_symbols = _gene_symbol_series(vel_backed)
    gene_ids = pd.Index(vel_backed.var_names.astype(str), name='gene_id')

    # Optionally pool PsychAD V3 cells
    if args.include_psychad:
        print("\n[2b/4] Adding PsychAD V3 cells to V3 statistics")
        aging_backed, hbcc_backed, hbcc_mask, ps_meta = read_psychad_backed(
            args.psychad_aging, args.psychad_hbcc)
        # Combine AGING + HBCC-unique → all V3
        # NOTE: PsychAD var_names typically differ from Velmeshev; intersect on symbols
        # We compute stats per dataset and combine on symbol overlap.
        ps_sym_aging = _gene_symbol_series(aging_backed)
        v3_psychad_aging = per_gene_stats(
            aging_backed,
            ps_meta.loc[ps_meta['source'] == 'PSYCHAD'].index.intersection(
                aging_backed.obs_names),
            'PsychAD-aging', chunk_size=args.chunk_size)

        # Build aging symbol -> stat lookup
        aging_lookup = pd.DataFrame({
            'symbol': ps_sym_aging.values,
            'nz': v3_psychad_aging['detection_rate'] * v3_psychad_aging['n_cells'],
            'sum_counts': v3_psychad_aging['mean_counts'] * v3_psychad_aging['n_cells'],
            'sum_cpm': v3_psychad_aging['mean_cpm'] * v3_psychad_aging['n_cells'],
        })

        # HBCC unique subset
        hbcc_obs = hbcc_backed.obs_names[hbcc_mask]
        v3_psychad_hbcc = per_gene_stats(
            hbcc_backed, hbcc_obs, 'PsychAD-hbcc-unique',
            chunk_size=args.chunk_size)
        ps_sym_hbcc = _gene_symbol_series(hbcc_backed)
        hbcc_lookup = pd.DataFrame({
            'symbol': ps_sym_hbcc.values,
            'nz': v3_psychad_hbcc['detection_rate'] * v3_psychad_hbcc['n_cells'],
            'sum_counts': v3_psychad_hbcc['mean_counts'] * v3_psychad_hbcc['n_cells'],
            'sum_cpm': v3_psychad_hbcc['mean_cpm'] * v3_psychad_hbcc['n_cells'],
        })

        n_aging = v3_psychad_aging['n_cells']
        n_hbcc  = v3_psychad_hbcc['n_cells']

        # Vel V3 baseline (per-symbol via gene_symbols mapping in Vel)
        vel_sym = gene_symbols.values
        v3_nz   = v3_stats['detection_rate'] * v3_stats['n_cells']
        v3_sum  = v3_stats['mean_counts']    * v3_stats['n_cells']
        v3_cpm  = v3_stats['mean_cpm']       * v3_stats['n_cells']

        vel_lookup = pd.DataFrame({
            'gene_id': gene_ids.values,
            'symbol':  vel_sym,
            'nz': v3_nz, 'sum_counts': v3_sum, 'sum_cpm': v3_cpm,
        })

        # Aggregate per symbol across the three sources, then map back to Velmeshev var_names
        all_lookup = pd.concat([
            vel_lookup.assign(n=v3_stats['n_cells']),
            aging_lookup.assign(n=n_aging),
            hbcc_lookup.assign(n=n_hbcc),
        ], ignore_index=True)
        agg = (all_lookup.groupby('symbol', sort=False)
               .agg(nz=('nz', 'sum'),
                    sum_counts=('sum_counts', 'sum'),
                    sum_cpm=('sum_cpm', 'sum'),
                    n=('n', 'sum'))
               .reset_index())
        agg['detection_rate'] = agg['nz'] / agg['n']
        agg['mean_counts']    = agg['sum_counts'] / agg['n']
        agg['mean_cpm']       = agg['sum_cpm'] / agg['n']
        # Project agg back to Velmeshev gene order
        vel_sym_series = pd.Series(vel_sym, index=gene_ids.values)
        pooled = agg.set_index('symbol').reindex(vel_sym_series.values).fillna(0.0)
        # Overwrite V3 stats with pooled
        v3_stats = dict(
            detection_rate=pooled['detection_rate'].values,
            mean_counts=pooled['mean_counts'].values,
            mean_cpm=pooled['mean_cpm'].values,
            n_cells=int(pooled['n'].iloc[0] if len(pooled) else 0),
        )
        print(f"  V3 (pooled with PsychAD): total cells now ~{int(agg['n'].iloc[0]) if len(agg) else 0:,} "
              f"(across Velmeshev + PsychAD)")
        try:
            aging_backed.file.close()
        except Exception:
            pass
        try:
            hbcc_backed.file.close()
        except Exception:
            pass
        del aging_backed, hbcc_backed, ps_meta, aging_lookup, hbcc_lookup, all_lookup, agg
        gc.collect()

    # ── Load AHBA GRN membership ─────────────────────────────────────────────
    print("\n[3/4] Loading AHBA GRN membership")
    c3pos_set, c3neg_set = load_grn_membership(args.ref_dir)

    in_c3pos = gene_symbols.isin(c3pos_set).values
    in_c3neg = gene_symbols.isin(c3neg_set).values
    print(f"  C3+ symbols found in adata.var: {in_c3pos.sum()} / {len(c3pos_set)}")
    print(f"  C3- symbols found in adata.var: {in_c3neg.sum()} / {len(c3neg_set)}")

    # ── Build output table ───────────────────────────────────────────────────
    print("\n[4/4] Building output table and plots")
    eps = 1e-6
    detection_v2 = v2_stats['detection_rate']
    detection_v3 = v3_stats['detection_rate']
    well_v2 = detection_v2 >= args.detect_min
    well_v3 = detection_v3 >= args.detect_min
    well_both = well_v2 & well_v3
    lfc = np.log2((detection_v3 + eps) / (detection_v2 + eps))
    robust = np.abs(lfc) <= args.lfc_max

    df = pd.DataFrame({
        'gene_id': gene_ids,
        'gene_symbol': gene_symbols.values,
        'detection_v2': detection_v2,
        'detection_v3': detection_v3,
        'mean_counts_v2': v2_stats['mean_counts'],
        'mean_counts_v3': v3_stats['mean_counts'],
        'mean_cpm_v2': v2_stats['mean_cpm'],
        'mean_cpm_v3': v3_stats['mean_cpm'],
        'well_detected_v2': well_v2,
        'well_detected_v3': well_v3,
        'well_detected_both': well_both,
        'robust_to_chemistry': robust,
        'log2_detection_lfc_v3_v2': lfc,
        'in_C3_pos_top1000': in_c3pos,
        'in_C3_neg_top1000': in_c3neg,
    })

    print(f"\nSummary:")
    print(f"  Total genes                 : {len(df):,}")
    print(f"  Well-detected in V2         : {int(well_v2.sum()):,}")
    print(f"  Well-detected in V3         : {int(well_v3.sum()):,}")
    print(f"  Well-detected in BOTH       : {int(well_both.sum()):,}")
    print(f"  Robust to chemistry         : {int(robust.sum()):,}")
    print(f"  C3+ in well-detected-both   : {int((in_c3pos & well_both).sum())} / {in_c3pos.sum()}")
    print(f"  C3- in well-detected-both   : {int((in_c3neg & well_both).sum())} / {in_c3neg.sum()}")

    # CSV
    csv_path = os.path.join(args.output_dir, 'gene_detection_stats.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Text files: well-detected genes (IDs and symbols separately).
    # Gene IDs (Ensembl) are dataset-specific; symbols travel across datasets,
    # which is what downstream cross-dataset filters need.
    txt_path = os.path.join(args.output_dir, 'well_detected_genes.txt')
    df.loc[df['well_detected_both'], 'gene_id'].to_csv(txt_path, index=False, header=False)
    print(f"Saved {txt_path}")

    sym_path = os.path.join(args.output_dir, 'well_detected_symbols.txt')
    symbols = (df.loc[df['well_detected_both'], 'gene_symbol']
                 .dropna().astype(str))
    symbols = symbols[symbols != 'nan']
    symbols = sorted(set(symbols))
    with open(sym_path, 'w') as fh:
        fh.write('\n'.join(symbols) + '\n')
    print(f"Saved {sym_path} ({len(symbols):,} unique symbols)")

    # Plots
    plot_scatter_detection(df, os.path.join(args.output_dir, 'scatter_detection_v2_vs_v3.png'))
    plot_hist_detection(df,    os.path.join(args.output_dir, 'hist_detection_per_chemistry.png'))
    plot_hist_cpm(df,          os.path.join(args.output_dir, 'hist_mean_cpm_per_chemistry.png'))
    plot_lfc_vs_mean(df,       os.path.join(args.output_dir, 'detection_lfc_vs_mean.png'))
    plot_grn_coverage(df,      os.path.join(args.output_dir, 'grn_coverage_summary.png'))

    try:
        vel_backed.file.close()
    except Exception:
        pass

    print("\nDone.")


if __name__ == '__main__':
    main()
