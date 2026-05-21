"""
01 — Batch-correction audit on the joint VelWangPsychAD pseudobulk.

Central question: does scVI/scANVI batch correction flip the developmental
GRN signal? In compare_Vel_PsychAD_genes (separate scANVI per dataset, no
cross-dataset batch correction) the V3-pooled child→adolescence AHBA C3+
Cohen's d is +0.77, p=0.011. In compare_Vel_PsychAD_integration_genes
(joint scANVI + batch correction) the same comparison flips to d=-0.81,
p=0.013.

The pseudobulk h5ad already stores `counts`, `scvi_normalized`, and
`scanvi_normalized` on the SAME donor × cell_class rows, so we can score
the GRN on each layer and pinpoint where the signal flips.

Outputs (scripts/integration_qc/outputs/batch_correction/):
  c3_scores_by_layer.parquet     — per-(donor, cell_class, layer) C3+/C3- score
  effect_sizes.csv               — child→adol Cohen's d / Wilcoxon p per (stratum, layer)
  fig01_c3pos_vs_age_excitatory.png
  fig02_c3pos_vs_age_all_classes.png
  fig03_per_donor_shift_vs_age.png
  fig04_pca_excitatory_by_layer.png
  fig05_marker_shift_vs_age.png
  interpretation.md              — auto-generated summary
"""
import os
import sys
import io
import contextlib
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'code'))

from environment import get_environment
from regulons import project_GRN, get_ahba_GRN


def _local_setup_grn(ref_dir, adata, gene_filter_symbols=None):
    """Local replacement for hvg_investigation.setup_grn that avoids the
    mygene dependency by mapping symbols → Ensembl via adata.var['gene_symbol'].

    The pseudobulk's gene_symbol column has the form SYMBOL or SYMBOL_ENSGxxx
    (uniquified). We strip the trailing _ENSG suffix to recover the symbol.
    """
    grn_file = os.path.join(ref_dir, 'ahba_dme_hcp_top8kgenes_weights.csv')
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file, use_weights=True)
    if gene_filter_symbols is not None:
        before = len(ahba_GRN)
        ahba_GRN = ahba_GRN.loc[ahba_GRN['Gene'].isin(set(gene_filter_symbols))].copy()
        print(f'GRN symbol filter: {before} -> {len(ahba_GRN)} rows')

    raw_sym = adata.var['gene_symbol'].astype(str).str.replace(r'_ENSG\d+$', '', regex=True)
    sym_to_ens = (pd.DataFrame({'sym': raw_sym.values, 'ens': adata.var_names})
                    .drop_duplicates(subset='sym', keep='first')
                    .set_index('sym')['ens'])
    before_rows = len(ahba_GRN)
    ahba_GRN = ahba_GRN.assign(_ens=lambda d: d['Gene'].map(sym_to_ens))
    ahba_GRN = ahba_GRN.dropna(subset=['_ens']).copy()
    ahba_GRN['Gene'] = ahba_GRN['_ens']
    ahba_GRN = ahba_GRN.drop(columns=['_ens'])
    total = ahba_GRN['Gene'].nunique()
    print(f'GRN→Ensembl mapping: {before_rows} rows → {len(ahba_GRN)} ({total} unique Ensembl)')
    return ahba_GRN, total

_env = get_environment()
RDS_DIR = _env['rds_dir']
REF_DIR = _env['ref_dir']

PSEUDOBULK = os.path.join(
    RDS_DIR,
    'Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel'
    '/pseudobulk_output/by_cell_class.h5ad')

GENE_FILTER_FILE = os.path.join(
    REPO_ROOT, 'scripts/outputs/gene_detection_v2_vs_v3/well_detected_symbols.txt')

OUT = os.path.join(REPO_ROOT, 'scripts/integration_qc/outputs/batch_correction')
os.makedirs(OUT, exist_ok=True)

NEURON_MARKERS = ['SLC17A7', 'SATB2', 'NEUROD6', 'TBR1', 'RBFOX3', 'CUX2', 'RORB',
                  'GAD1', 'GAD2', 'SLC32A1']
HOUSEKEEPING = ['ACTB', 'GAPDH', 'HPRT1', 'PPIA', 'B2M', 'RPL13A', 'YWHAZ',
                'SDHA', 'TBP', 'UBC']

LAYERS = ['counts_cpm', 'scvi_normalized', 'scanvi_normalized']

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _silent():
    return contextlib.redirect_stdout(io.StringIO())


def _to_dense(M):
    return M.toarray() if sp.issparse(M) else np.asarray(M)


def _set_layer(adata, layer):
    """Replace adata.X with the requested layer, CPM-normalising counts."""
    if layer == 'counts_cpm':
        adata.X = adata.layers['counts'].copy()
        sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
        adata.X = _to_dense(adata.X).astype(np.float32)
    else:
        adata.X = _to_dense(adata.layers[layer]).astype(np.float32)
    if 'highly_variable' in adata.var.columns:
        del adata.var['highly_variable']


def _project_one_layer(adata, ahba_GRN, layer):
    _set_layer(adata, layer)
    with _silent():
        project_GRN(adata, ahba_GRN, 'X_ahba',
                    use_residuals=False, use_highly_variable=False,
                    log_transform=False)
    cols = adata.uns['X_ahba_names']
    df = pd.DataFrame(adata.obsm['X_ahba'], index=adata.obs_names, columns=cols)
    keep = [c for c in ['C3+', 'C3-'] if c in df.columns]
    return df[keep].copy()


def cohens_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    s = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / s if s > 0 else np.nan


def child_adol_test(df, score_col, child=(1, 9), adol=(10, 20)):
    """Cohen's d and Wilcoxon p comparing childhood vs adolescence donors.

    Sign convention matches the notebook: positive d ⇒ childhood > adolescence
    (i.e., a downward trajectory across childhood→adolescence).
    """
    c = df.loc[(df['age_years'] >= child[0]) & (df['age_years'] < child[1]), score_col].values
    a = df.loc[(df['age_years'] >= adol[0]) & (df['age_years'] < adol[1]), score_col].values
    if len(c) < 3 or len(a) < 3:
        return dict(d=np.nan, p=np.nan, n_child=len(c), n_adol=len(a))
    d = cohens_d(c, a)
    try:
        _, p = mannwhitneyu(c, a, alternative='two-sided')
    except ValueError:
        p = np.nan
    return dict(d=d, p=p, n_child=len(c), n_adol=len(a))


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    warnings.filterwarnings('ignore')
    print(f'Loading pseudobulk: {PSEUDOBULK}')
    adata = sc.read_h5ad(PSEUDOBULK)
    print(f'  shape={adata.shape}, layers={list(adata.layers)}')

    # Restrict to Velmeshev + PsychAD (drop Wang as in the comparison notebook)
    keep = adata.obs['source'].isin(['VELMESHEV', 'PSYCHAD'])
    adata = adata[keep].copy()
    print(f'After source filter (VEL+PSYCHAD): {adata.shape}')

    # Build a single source-chemistry label that matches the notebook strata
    adata.obs['source_chem'] = (
        adata.obs['source'].astype(str) + '-' +
        adata.obs['chemistry'].astype(str))

    # Optional gene filter (well-detected in both V2 and V3) — matches
    # compare_Vel_PsychAD_integration_genes config when the file exists.
    gene_filter = None
    if os.path.exists(GENE_FILTER_FILE):
        with open(GENE_FILTER_FILE) as fh:
            gene_filter = sorted({line.strip() for line in fh if line.strip()})
        print(f'Gene filter applied: {len(gene_filter):,} symbols')
    else:
        print(f'Gene filter file not found ({GENE_FILTER_FILE}) — using full GRN')

    with _silent():
        ahba_GRN, n_grn = _local_setup_grn(REF_DIR, adata, gene_filter_symbols=gene_filter)
    print(f'GRN genes in adata: {n_grn}')

    # Score the GRN on every layer
    score_frames = []
    for layer in LAYERS:
        print(f'Projecting GRN on layer={layer} …')
        df_scores = _project_one_layer(adata, ahba_GRN, layer)
        df_scores['layer'] = layer
        df_scores['obs_names'] = df_scores.index
        score_frames.append(df_scores.reset_index(drop=True))
    scores = pd.concat(score_frames, ignore_index=True)

    meta_cols = ['individual', 'cell_class', 'age_years', 'sex',
                 'source', 'chemistry', 'source_chem', 'n_cells']
    meta = adata.obs[meta_cols].copy()
    meta['obs_names'] = adata.obs_names

    scores = scores.merge(meta, on='obs_names', how='left')
    scores_out = os.path.join(OUT, 'c3_scores_by_layer.csv')
    scores.to_csv(scores_out, index=False)
    print(f'Saved {scores_out}')

    # ── Effect-size table per (stratum, cell_class, layer)
    strata = {
        'Vel-V2':      lambda d: d['source_chem'] == 'VELMESHEV-V2',
        'Vel-V3':      lambda d: d['source_chem'] == 'VELMESHEV-V3',
        'Vel-all':     lambda d: d['source'] == 'VELMESHEV',
        'PsychAD-V3':  lambda d: d['source_chem'] == 'PSYCHAD-V3',
        'V3-pooled':   lambda d: d['chemistry'] == 'V3',
    }
    rows = []
    for ccls in scores['cell_class'].unique():
        sub_cc = scores[scores['cell_class'] == ccls]
        for layer in LAYERS:
            sub_layer = sub_cc[sub_cc['layer'] == layer]
            for sname, sf in strata.items():
                sub = sub_layer[sf(sub_layer)]
                stat = child_adol_test(sub, 'C3+')
                rows.append(dict(cell_class=ccls, layer=layer, stratum=sname, **stat))
    eff = pd.DataFrame(rows)
    eff_out = os.path.join(OUT, 'effect_sizes.csv')
    eff.to_csv(eff_out, index=False)
    print(f'Saved {eff_out}')

    # Sensitivity grid: scan (CHILD_START, CHILD_END, ADOL_START, ADOL_END)
    # mirroring the qmd's 4D grid, but here in coarse form. Report best |d|
    # per (cell_class, stratum, layer) so the headline number is comparable to
    # the notebook's grid-search optimum.
    grid = [(cs, ce, asx, ae)
            for cs in [1, 2, 3]
            for ce in [8, 9, 10]
            for asx in [10, 11, 12]
            for ae in [18, 20, 22]
            if ce <= asx <= ae]
    grid_rows = []
    for ccls in ['Excitatory']:
        sub_cc = scores[scores['cell_class'] == ccls]
        for layer in LAYERS:
            sub_layer = sub_cc[sub_cc['layer'] == layer]
            for sname, sf in strata.items():
                sub = sub_layer[sf(sub_layer)]
                best = dict(d=np.nan, p=np.nan, n_child=0, n_adol=0,
                            child=(np.nan, np.nan), adol=(np.nan, np.nan))
                for cs, ce, asx, ae in grid:
                    res = child_adol_test(sub, 'C3+', child=(cs, ce), adol=(asx, ae))
                    if res['n_child'] >= 3 and res['n_adol'] >= 3:
                        if np.isnan(best['d']) or abs(res['d']) > abs(best['d']):
                            best = dict(**res, child=(cs, ce), adol=(asx, ae))
                grid_rows.append(dict(cell_class=ccls, layer=layer, stratum=sname,
                                      best_d=best['d'], best_p=best['p'],
                                      n_child=best['n_child'], n_adol=best['n_adol'],
                                      child_window=best['child'], adol_window=best['adol']))
    grid_df = pd.DataFrame(grid_rows)
    grid_out = os.path.join(OUT, 'effect_sizes_grid_best.csv')
    grid_df.to_csv(grid_out, index=False)
    print(f'Saved {grid_out}')

    # ── Fig 1: C3+ vs age for Excitatory, three layers side by side
    exc = scores[scores['cell_class'] == 'Excitatory'].copy()
    palette = {'VELMESHEV-V2': '#d4a017',
               'VELMESHEV-V3': '#a87b00',
               'PSYCHAD-V3':   '#cc4040'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False)
    for ax, layer in zip(axes, LAYERS):
        sub = exc[exc['layer'] == layer]
        sns.scatterplot(data=sub, x='age_years', y='C3+', hue='source_chem',
                        palette=palette, s=28, alpha=0.85, ax=ax, legend=(ax is axes[-1]))
        for sc_label in sub['source_chem'].unique():
            ss = sub[sub['source_chem'] == sc_label].dropna(subset=['C3+'])
            if len(ss) >= 5:
                try:
                    from scipy.stats import linregress
                    res = linregress(ss['age_years'], ss['C3+'])
                    xs = np.linspace(ss['age_years'].min(), ss['age_years'].max(), 50)
                    ax.plot(xs, res.intercept + res.slope * xs, color=palette[sc_label], lw=1)
                except Exception:
                    pass
        ax.set_title(f'Excitatory · {layer}')
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('AHBA C3+ score')
        ax.set_xlim(-1, 40)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.suptitle('AHBA C3+ in Excitatory pseudobulks — by normalisation layer', y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig01_c3pos_vs_age_excitatory.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 2: All major cell classes × layers — small-multiples grid
    main_classes = ['Excitatory', 'Inhibitory', 'Astrocytes', 'Oligos', 'OPC', 'Microglia']
    sub_classes = [c for c in main_classes if c in scores['cell_class'].unique()]
    nrow, ncol = len(sub_classes), 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), sharex=False)
    for i, ccls in enumerate(sub_classes):
        for j, layer in enumerate(LAYERS):
            ax = axes[i, j] if nrow > 1 else axes[j]
            sub = scores[(scores['cell_class'] == ccls) & (scores['layer'] == layer)]
            sns.scatterplot(data=sub, x='age_years', y='C3+', hue='source_chem',
                            palette=palette, s=14, alpha=0.7, ax=ax,
                            legend=(i == 0 and j == ncol - 1))
            ax.set_xlim(-1, 40)
            if i == 0:
                ax.set_title(layer)
            if j == 0:
                ax.set_ylabel(f'{ccls}\nC3+')
            else:
                ax.set_ylabel('')
            ax.set_xlabel('Age' if i == len(sub_classes) - 1 else '')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig02_c3pos_vs_age_all_classes.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 3: Per-donor C3+ shift (scvi - counts_cpm) vs age, Excitatory
    pivot = (exc.pivot_table(index='obs_names', columns='layer', values='C3+')
               .merge(meta[['obs_names', 'source_chem', 'age_years']].drop_duplicates(),
                      on='obs_names', how='left'))
    pivot['shift_scvi'] = pivot['scvi_normalized'] - pivot['counts_cpm']
    pivot['shift_scanvi'] = pivot['scanvi_normalized'] - pivot['counts_cpm']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, col, title in zip(axes,
                              ['shift_scvi', 'shift_scanvi'],
                              ['scvi_normalized − counts(CPM)', 'scanvi_normalized − counts(CPM)']):
        sns.scatterplot(data=pivot, x='age_years', y=col, hue='source_chem',
                        palette=palette, s=22, alpha=0.85, ax=ax, legend=(ax is axes[-1]))
        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_title(title)
        ax.set_xlabel('Age (years)')
        ax.set_ylabel('Per-donor C3+ shift (Excitatory)')
        ax.set_xlim(-1, 40)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.suptitle('How batch correction moves each donor (Excitatory)', y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig03_per_donor_shift_vs_age.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 4: PCA of Excitatory pseudobulks per layer, colored by age & source
    exc_mask = adata.obs['cell_class'] == 'Excitatory'
    exc_adata = adata[exc_mask].copy()
    pca_panels = []
    for layer in LAYERS:
        _set_layer(exc_adata, layer)
        X = _to_dense(exc_adata.X)
        # standardise gene-wise to make layers comparable
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        pcs = PCA(n_components=2, random_state=0).fit_transform(X)
        pca_panels.append((layer, pcs))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, (layer, pcs) in enumerate(pca_panels):
        df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
        df['age_years'] = exc_adata.obs['age_years'].values
        df['source_chem'] = (exc_adata.obs['source'].astype(str) + '-' +
                             exc_adata.obs['chemistry'].astype(str)).values
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='age_years', palette='viridis',
                        s=26, ax=axes[0, j], legend=(j == 2))
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='source_chem', palette=palette,
                        s=26, ax=axes[1, j], legend=(j == 2))
        axes[0, j].set_title(f'{layer} · color = age')
        axes[1, j].set_title(f'{layer} · color = source-chemistry')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig04_pca_excitatory_by_layer.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ── Fig 5: Per-donor mean expression shift for markers + housekeeping vs age
    def _marker_means(symbol_list):
        sym2ens = dict(zip(adata.var['gene_symbol'].astype(str), adata.var_names))
        ensembl = [sym2ens[s] for s in symbol_list if s in sym2ens]
        if not ensembl:
            return None
        gene_idx = adata.var_names.get_indexer(ensembl)
        rows = {}
        for layer in LAYERS:
            _set_layer(adata, layer)
            X = _to_dense(adata.X)
            rows[layer] = X[:, gene_idx].mean(axis=1)
        df = pd.DataFrame(rows, index=adata.obs_names)
        df['age_years'] = adata.obs['age_years'].values
        df['source_chem'] = (adata.obs['source'].astype(str) + '-' +
                             adata.obs['chemistry'].astype(str)).values
        df['cell_class'] = adata.obs['cell_class'].values
        df['shift_scvi'] = df['scvi_normalized'] - df['counts_cpm']
        df['shift_scanvi'] = df['scanvi_normalized'] - df['counts_cpm']
        return df, ensembl

    panels = [('Neuron markers', NEURON_MARKERS),
              ('Housekeeping',   HOUSEKEEPING)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row, (label, syms) in enumerate(panels):
        got = _marker_means(syms)
        if got is None:
            axes[row, 0].text(0.5, 0.5, f'{label}: no symbol matches', ha='center')
            continue
        df, found = got
        exc_only = df[df['cell_class'] == 'Excitatory']
        for col, shift_col in enumerate(['shift_scvi', 'shift_scanvi']):
            ax = axes[row, col]
            sns.scatterplot(data=exc_only, x='age_years', y=shift_col,
                            hue='source_chem', palette=palette, s=22, alpha=0.85,
                            ax=ax, legend=(row == 0 and col == 1))
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.set_title(f'{label} · {shift_col} (Excitatory)')
            ax.set_xlabel('Age (years)')
            ax.set_ylabel(f'mean expr shift\n[{len(found)} genes]')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig05_marker_shift_vs_age.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ── Interpretation summary
    def _df_to_md(df):
        df = df.copy()
        if df.index.name is not None and df.index.name not in df.columns:
            df = df.reset_index()
        cols = list(df.columns)
        out = ['| ' + ' | '.join(cols) + ' |',
               '|' + '|'.join(['---'] * len(cols)) + '|']
        for _, r in df.iterrows():
            out.append('| ' + ' | '.join(
                f'{v:.4g}' if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else str(v)
                for v in r.values) + ' |')
        return '\n'.join(out)

    pivot_d = (eff[(eff['cell_class'] == 'Excitatory') & (eff['stratum'] == 'V3-pooled')]
                 .set_index('layer')[['d', 'p', 'n_child', 'n_adol']])
    interp = [
        '# 01 — Batch-correction audit · interpretation',
        '',
        f'Pseudobulk: `{os.path.basename(PSEUDOBULK)}`',
        'GRN: `ahba_dme_hcp_top8kgenes_weights.csv`',
        f'Gene filter: {"yes ("+str(len(gene_filter))+" symbols)" if gene_filter else "no"}',
        '',
        '## V3-pooled Excitatory child→adolescence (fixed window 1–9 / 10–20)',
        'Cohen\'s d sign convention: positive ⇒ childhood > adolescence (downward trajectory).',
        '',
        _df_to_md(pivot_d),
        '',
        '## Excitatory best-of-grid (mirroring the notebook\'s 4D sensitivity grid)',
        'Cohen\'s d with largest |d| over (CHILD_START × CHILD_END × ADOL_START × ADOL_END).',
        '',
        _df_to_md(grid_df),
        '',
        '## Interpretation cues',
        '',
        '- If `counts_cpm` shows d>0 (positive child→adol drop) while `scvi_normalized`/`scanvi_normalized` reverse it, batch correction is responsible for the flip observed between the non-integrated and joint integrations.',
        '- Inspect `fig01` (C3+ vs age, Excitatory) — does the slope change sign across layers?',
        '- Inspect `fig03` — are PsychAD adolescent donors pushed UP while Velmeshev adolescents are pushed DOWN by batch correction? If so, scVI is mistaking developmental signal for batch effect.',
        '- Inspect `fig05` — do neuron markers and housekeeping genes show similar shifts, or only neuron markers? Targeted distortion of the regulon set is more concerning than uniform shrinkage.',
        '- Inspect `fig04` — on raw `counts_cpm` we expect dataset clustering on PC1; on `scvi_normalized` we expect age to become the dominant axis. If age still does not drive PC1 after correction, the integration may be over-aggressive.',
        '',
        '## Full effect-size table',
        '',
        _df_to_md(eff),
    ]
    with open(os.path.join(OUT, 'interpretation.md'), 'w') as fh:
        fh.write('\n'.join(interp))
    print(f'Saved interpretation.md')

    print('\nDone.')


if __name__ == '__main__':
    main()
