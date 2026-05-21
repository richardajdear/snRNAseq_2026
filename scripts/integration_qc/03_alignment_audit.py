"""
03 — scANVI alignment audit on the joint Vel + Wang + PsychAD integration.

Needs the HPC-exported integrated_obs.csv.gz (run hpc_export_integrated_obs.py
on the HPC first, scp to outputs/alignment/integrated_obs.csv.gz).

If the file is absent the script still runs the per-cell-class diagnostics
on the local pseudobulk (limited audit), but skips the per-cell ones.

Outputs (scripts/integration_qc/outputs/alignment/):
  fine_confusion_velmeshev.csv
  fine_confusion_psychad.csv
  broad_confusion_velmeshev.csv
  broad_confusion_psychad.csv
  wang_age_coverage.csv
  confidence_threshold_sweep.csv
  cross_dataset_excitatory_correlation.csv
  marker_enrichment.csv
  fig01_broad_confusion_per_dataset.png
  fig02_confidence_by_age_class.png
  fig03_wang_age_coverage.png
  fig04_confidence_threshold_sweep.png
  fig05_marker_enrichment.png
  fig06_cross_dataset_excitatory_correlation.png
  recommendations.md
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'code'))

from environment import get_environment
_env = get_environment()
RDS_DIR = _env['rds_dir']

PB_JOINT = os.path.join(RDS_DIR,
    'Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel'
    '/pseudobulk_output/by_cell_class.h5ad')

OUT = os.path.join(REPO_ROOT, 'scripts/integration_qc/outputs/alignment')
os.makedirs(OUT, exist_ok=True)
OBS_FILE = os.path.join(OUT, 'integrated_obs.csv.gz')

BROAD_MAP = {
    # Wang fine → broad
    'EN-L2_3-IT': 'Excitatory', 'EN-L3-IT': 'Excitatory', 'EN-L4-IT': 'Excitatory',
    'EN-L5-ET': 'Excitatory', 'EN-L5-IT': 'Excitatory', 'EN-L5_6-NP': 'Excitatory',
    'EN-L6-CT': 'Excitatory', 'EN-L6-IT': 'Excitatory', 'EN-L6b': 'Excitatory',
    'EN-Newborn': 'Excitatory', 'EN-IT-Immature': 'Excitatory',
    'EN-ET-Immature': 'Excitatory', 'EN-CT-Immature': 'Excitatory',
    'IN-MGE-PV': 'Inhibitory', 'IN-MGE-SST': 'Inhibitory',
    'IN-CGE-VIP': 'Inhibitory', 'IN-CGE-LAMP5': 'Inhibitory',
    'IN-CGE-SNCG': 'Inhibitory', 'IN-Immature': 'Inhibitory',
    'Astrocyte': 'Astrocytes', 'Astrocyte-Fibrous': 'Astrocytes',
    'Astrocyte-Protoplasmic': 'Astrocytes',
    'Oligodendrocyte': 'Oligos', 'Oligodendrocyte-Immature': 'Oligos',
    'OPC': 'OPC', 'Microglia': 'Microglia', 'Endothelial': 'Endothelial',
    'RG-oRG': 'Other', 'RG-tRG': 'Other', 'RG-vRG': 'Other', 'IPC': 'Other',
    'Pericyte': 'Endothelial', 'SMC': 'Endothelial', 'VLMC': 'Endothelial',
}


def _broad(label):
    s = str(label)
    if pd.isna(label):
        return None
    if s in BROAD_MAP:
        return BROAD_MAP[s]
    sl = s.lower()
    if sl.startswith('en') or sl.startswith('ex') or 'excit' in sl:
        return 'Excitatory'
    if sl.startswith('in') or 'inhib' in sl:
        return 'Inhibitory'
    if 'astro' in sl:
        return 'Astrocytes'
    if 'oligo' in sl:
        return 'Oligos'
    if 'opc' in sl:
        return 'OPC'
    if 'micro' in sl or 'mglia' in sl or 'macrop' in sl or 'pvm' in sl:
        return 'Microglia'
    if 'endo' in sl or 'vasc' in sl or 'peri' in sl or 'vlmc' in sl or 'smc' in sl:
        return 'Endothelial'
    return 'Other'


def _add_age_bin(df, col='age_years'):
    bins   = [-1, 1, 2, 5, 10, 15, 20, 25, 30, 50, 70, 100]
    labels = ['<1', '1-2', '2-5', '5-10', '10-15', '15-20',
              '20-25', '25-30', '30-50', '50-70', '70+']
    df['age_bin'] = pd.cut(df[col], bins=bins, labels=labels)
    return df


# ────────────────────────────────────────────────────────────────────────────
def per_cell_audit(obs):
    """Audits that need per-cell obs (require HPC export)."""
    print('Running per-cell audits …')
    obs = obs.copy()
    obs = _add_age_bin(obs)
    obs['aligned_broad'] = obs['cell_type_aligned'].map(_broad)
    obs['orig_broad'] = obs['cell_class_original'].astype(str)

    # 1. Confusion matrices per dataset
    confs = {}
    for src in ['VELMESHEV', 'PSYCHAD']:
        sub = obs[obs['source'] == src]
        if sub.empty:
            continue
        # Fine confusion (rows = cell_type_raw, cols = cell_type_aligned)
        if 'cell_type_raw' in sub.columns:
            fine = pd.crosstab(sub['cell_type_raw'], sub['cell_type_aligned'])
            fine.to_csv(os.path.join(OUT, f'fine_confusion_{src.lower()}.csv'))
        # Broad confusion
        broad = pd.crosstab(sub['orig_broad'], sub['aligned_broad'])
        broad.to_csv(os.path.join(OUT, f'broad_confusion_{src.lower()}.csv'))
        confs[src] = broad
        print(f'  {src}: broad confusion saved (shape {broad.shape})')

    # Figure 1: Broad confusion heatmaps (row-normalized)
    if confs:
        fig, axes = plt.subplots(1, len(confs), figsize=(5.5 * len(confs), 4.4))
        if len(confs) == 1:
            axes = [axes]
        for ax, (src, c) in zip(axes, confs.items()):
            row_pct = c.div(c.sum(axis=1), axis=0)
            sns.heatmap(row_pct, ax=ax, cmap='magma', annot=True, fmt='.2f',
                        cbar=False, vmin=0, vmax=1, annot_kws={'size': 7})
            ax.set_title(f'{src} broad original→aligned (row %)')
            ax.set_xlabel('aligned_broad')
            ax.set_ylabel('orig_broad')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'fig01_broad_confusion_per_dataset.png'),
                    dpi=160, bbox_inches='tight')
        plt.close(fig)

    # 2. Confidence distribution by (dataset, age_bin, orig_broad)
    if 'cell_type_aligned_confidence' in obs.columns:
        conf_long = obs.dropna(subset=['cell_type_aligned_confidence']).copy()
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        for ax, src in zip(axes, ['VELMESHEV', 'PSYCHAD']):
            sub = conf_long[conf_long['source'] == src]
            sub = sub[sub['orig_broad'].isin(['Excitatory', 'Inhibitory',
                                              'Astrocytes', 'Oligos', 'OPC', 'Microglia'])]
            if sub.empty:
                ax.set_title(f'{src} (no data)')
                continue
            sns.boxplot(data=sub, x='age_bin', y='cell_type_aligned_confidence',
                        hue='orig_broad', ax=ax, fliersize=1)
            ax.set_title(f'{src} · scANVI confidence by age bin × original broad class')
            ax.set_ylabel('alignment confidence')
            ax.set_xlabel('age bin')
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'fig02_confidence_by_age_class.png'),
                    dpi=140, bbox_inches='tight')
        plt.close(fig)

    # 3. Wang age coverage
    wang = obs[obs['source'] == 'WANG']
    if not wang.empty:
        cov = wang.groupby('age_bin', observed=True).size().rename('n_cells').to_frame()
        cov.to_csv(os.path.join(OUT, 'wang_age_coverage.csv'))
        fig, ax = plt.subplots(figsize=(8, 3.4))
        cov['n_cells'].plot(kind='bar', ax=ax, color='#666')
        ax.set_ylabel('n cells')
        ax.set_title('Wang reference age coverage (cells per bin)')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'fig03_wang_age_coverage.png'),
                    dpi=160, bbox_inches='tight')
        plt.close(fig)

    # 4. Confidence-threshold sweep on Excitatory% vs age
    sweep_rows = []
    for thr in [0.0, 0.5, 0.7, 0.9]:
        sub = obs.copy()
        if 'cell_type_aligned_confidence' in sub.columns:
            sub = sub[sub['cell_type_aligned_confidence'].fillna(0) >= thr]
        for src in ['VELMESHEV', 'PSYCHAD']:
            d_sub = sub[sub['source'] == src]
            # Excitatory % per individual
            ind_tot = d_sub.groupby('individual').size().rename('total')
            ind_exc = (d_sub[d_sub['aligned_broad'] == 'Excitatory']
                            .groupby('individual').size().rename('exc'))
            comp = pd.concat([ind_tot, ind_exc], axis=1).fillna(0)
            comp['pct_exc'] = comp['exc'] / comp['total'].replace(0, np.nan) * 100
            ages = (obs[obs['source'] == src][['individual', 'age_years']]
                      .drop_duplicates('individual').set_index('individual'))
            comp = comp.join(ages)
            comp = comp.dropna(subset=['age_years', 'pct_exc'])
            if len(comp) >= 5:
                from scipy.stats import spearmanr
                rho, p = spearmanr(comp['age_years'], comp['pct_exc'])
                sweep_rows.append(dict(threshold=thr, source=src, n_donors=len(comp),
                                       rho=rho, p=p,
                                       mean_pct=float(comp['pct_exc'].mean())))
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(os.path.join(OUT, 'confidence_threshold_sweep.csv'), index=False)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for src, sub in sweep.groupby('source'):
        ax.plot(sub['threshold'], sub['rho'], 'o-', label=src)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Confidence threshold')
    ax.set_ylabel('Spearman ρ (Excitatory% vs age, within source)')
    ax.set_title('Confidence-threshold sweep: does filtering low-confidence cells '
                 'converge the trends across sources?')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig04_confidence_threshold_sweep.png'),
                dpi=160, bbox_inches='tight')
    plt.close(fig)

    return confs


def pseudobulk_audits():
    """Cross-dataset pseudobulk audits — usable without HPC obs."""
    print('Running pseudobulk audits …')
    import scanpy as sc
    import scipy.sparse as sp
    adata = sc.read_h5ad(PB_JOINT)
    print(f'  pseudobulk: {adata.shape}')

    # Marker enrichment heatmap
    panels = {
        'Excitatory': ['SLC17A7', 'SATB2', 'NEUROD6', 'TBR1', 'RBFOX3', 'CUX2', 'RORB'],
        'Inhibitory': ['GAD1', 'GAD2', 'SLC32A1', 'PVALB', 'SST', 'VIP'],
        'Astrocytes': ['AQP4', 'GFAP', 'SLC1A2', 'ALDH1L1'],
        'Oligos':     ['PLP1', 'MOG', 'MBP', 'CNP'],
        'OPC':        ['PDGFRA', 'CSPG4', 'VCAN'],
        'Microglia':  ['CSF1R', 'P2RY12', 'C1QB', 'AIF1'],
    }

    sym2ens = (adata.var['gene_symbol']
                 .astype(str)
                 .str.replace(r'_ENSG\d+$', '', regex=True)
                 .reset_index().drop_duplicates('gene_symbol', keep='first')
                 .set_index('gene_symbol')['index'])

    # CPM-normalize counts
    Xc = adata.layers['counts']
    Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
    row_sums = Xc.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    Xc_cpm = Xc / row_sums * 1e6

    # Build (source × cell_class) × marker-panel mean expression table
    rows = []
    for src in ['VELMESHEV', 'PSYCHAD']:
        for ccls in ['Excitatory', 'Inhibitory', 'Astrocytes', 'Oligos', 'OPC', 'Microglia']:
            mask = ((adata.obs['source'] == src) & (adata.obs['cell_class'] == ccls)).values
            if mask.sum() == 0:
                continue
            sub = Xc_cpm[mask]
            for panel, syms in panels.items():
                idx = [adata.var_names.get_loc(sym2ens[s]) for s in syms
                       if s in sym2ens.index and sym2ens[s] in adata.var_names]
                if not idx:
                    continue
                vals = sub[:, idx]
                rows.append(dict(source=src, cell_class=ccls, panel=panel,
                                 mean_expr=float(np.log1p(vals).mean()),
                                 n_genes_found=len(idx), n_pseudobulks=int(mask.sum())))
    marker_df = pd.DataFrame(rows)
    marker_df.to_csv(os.path.join(OUT, 'marker_enrichment.csv'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, src in zip(axes, ['VELMESHEV', 'PSYCHAD']):
        sub = marker_df[marker_df['source'] == src]
        if sub.empty:
            ax.set_title(f'{src} no data')
            continue
        pv = sub.pivot(index='panel', columns='cell_class', values='mean_expr')
        pv = pv.reindex(['Excitatory', 'Inhibitory', 'Astrocytes', 'Oligos', 'OPC', 'Microglia'])
        sns.heatmap(pv, cmap='magma', annot=True, fmt='.2f', ax=ax, cbar=False)
        ax.set_title(f'{src}: log1p(CPM) mean of marker panels by aligned cell class')
        ax.set_xlabel('aligned cell_class')
        ax.set_ylabel('marker panel')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig05_marker_enrichment.png'),
                dpi=160, bbox_inches='tight')
    plt.close(fig)

    # Cross-dataset Excitatory transcriptome correlation per age bin
    obs = adata.obs.copy()
    obs = _add_age_bin(obs)
    exc_mask = obs['cell_class'] == 'Excitatory'
    rows = []
    bins = ['<1', '1-2', '2-5', '5-10', '10-15', '15-20']
    for b in bins:
        vel_idx = np.where(exc_mask & (obs['source'] == 'VELMESHEV') & (obs['age_bin'] == b))[0]
        psy_idx = np.where(exc_mask & (obs['source'] == 'PSYCHAD') & (obs['age_bin'] == b))[0]
        if len(vel_idx) >= 2 and len(psy_idx) >= 2:
            vel_mean = Xc_cpm[vel_idx].mean(axis=0)
            psy_mean = Xc_cpm[psy_idx].mean(axis=0)
            v = np.log1p(vel_mean); p_ = np.log1p(psy_mean)
            r = float(np.corrcoef(v, p_)[0, 1])
            rows.append(dict(age_bin=b, n_vel=len(vel_idx), n_psy=len(psy_idx),
                             pearson_log_cpm=r))
    cross = pd.DataFrame(rows)
    cross.to_csv(os.path.join(OUT, 'cross_dataset_excitatory_correlation.csv'),
                 index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    if not cross.empty:
        ax.bar(cross['age_bin'], cross['pearson_log_cpm'], color='#4F8DBF')
        for i, r in cross.iterrows():
            ax.text(i, r['pearson_log_cpm'] + 0.005,
                    f"n_v={r['n_vel']}\nn_p={r['n_psy']}",
                    ha='center', fontsize=7)
        ax.set_ylim(0.5, 1.02)
        ax.set_ylabel('Pearson r')
        ax.set_xlabel('age bin')
        ax.set_title('Cross-dataset Excitatory pseudobulk transcriptome correlation\n'
                     '(log1p CPM mean expression, Velmeshev vs PsychAD)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig06_cross_dataset_excitatory_correlation.png'),
                dpi=160, bbox_inches='tight')
    plt.close(fig)

    return cross, marker_df


def main():
    warnings.filterwarnings('ignore')
    have_obs = os.path.exists(OBS_FILE)
    print(f'Per-cell obs file present? {have_obs}  ({OBS_FILE})')

    if have_obs:
        obs = pd.read_csv(OBS_FILE)
        confs = per_cell_audit(obs)
    else:
        print('Per-cell audits will be skipped. Run hpc_export_integrated_obs.py on'
              ' the HPC and scp the resulting integrated_obs.csv.gz to:')
        print(f'   {OBS_FILE}')
        confs = {}

    cross, markers = pseudobulk_audits()

    # ── Recommendations
    recs = ['# 03 — Alignment audit · recommendations', '']
    if have_obs:
        for src, c in confs.items():
            if 'Excitatory' in c.index:
                row = c.loc['Excitatory']
                tot = row.sum()
                kept = row.get('Excitatory', 0)
                recs.append(f'- **{src}**: {kept/tot:.1%} of cells originally labelled '
                            f'Excitatory are kept as Excitatory by joint scANVI '
                            f'(remainder distributed across: {", ".join(c.columns[c.loc["Excitatory"]>0][:5])}).')
        recs.append('')
    recs += [
        '## Cross-dataset Excitatory pseudobulk correlation per age bin',
        '',
        cross.to_string(index=False) if not cross.empty else '(no bins with N≥2 per dataset)',
        '',
        '## Recommended actions',
        '',
        '1. If broad confusion shows >5% off-diagonal mass for Vel/PsychAD original Excitatory → aligned non-Excitatory: implement semi-supervised redesign using `shared_fine_labels.csv` (Script 04) so the model is supervised by all three datasets, not Wang alone.',
        '2. If `fig02_confidence_by_age_class.png` shows low confidence in PsychAD young donors: apply a confidence threshold (e.g., 0.5) at the pseudobulk stage. Implementation: add `adata = adata[adata.obs["cell_type_aligned_confidence"] >= 0.5]` before pseudobulk aggregation in `code/pipeline/pseudobulk.py`.',
        '3. If `fig03_wang_age_coverage.png` shows gaps (e.g., 15+ y): augment Wang reference with a curated subset of adult PsychAD cells (use their original `cell_class_original` as ground truth).',
        '4. If `fig04_confidence_threshold_sweep.png` shows trends converge as threshold rises: the alignment quality is part of the cross-dataset divergence. Combine #1 + #2 above.',
        '5. If `fig05_marker_enrichment.png` shows neuron markers under-expressed in PsychAD Excitatory vs Velmeshev Excitatory: the label "Excitatory" is not capturing the same biology in both datasets — fully implement #1.',
        '6. If `fig06_cross_dataset_excitatory_correlation.png` shows low correlation in young bins: Excitatory mean expression differs across datasets at those ages, which is consistent with the per-cell relabel observation in script 02 (joint scANVI removes young PsychAD Excitatory cells).',
        '',
    ]
    with open(os.path.join(OUT, 'recommendations.md'), 'w') as fh:
        fh.write('\n'.join(recs))
    print('Saved recommendations.md')
    print('Done.')


if __name__ == '__main__':
    main()
