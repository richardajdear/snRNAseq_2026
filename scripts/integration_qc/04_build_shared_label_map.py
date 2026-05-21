"""
04 — Build a shared fine-label vocabulary across Velmeshev × Wang × PsychAD.

Currently `code/pipeline/downsample.py:288-306` sets cell_type_for_scanvi
to the Wang fine label for Wang cells and "Unknown" for Velmeshev and PsychAD
cells. That makes scANVI a one-reference supervised model — Velmeshev and
PsychAD labels are then inferred by embedding kNN, which (as Script 02 shows)
mis-routes PsychAD young donors away from Excitatory entirely.

This script reads each dataset's native fine label column and produces a
single mapping CSV (`shared_fine_labels.csv`) so a future pipeline change can
set cell_type_for_scanvi from the same shared vocabulary across all three
datasets where a clean mapping exists. Unmappable cells stay 'Unknown'.

Outputs (scripts/integration_qc/outputs/label_map/):
  shared_fine_labels.csv            — mapping table
  fine_label_coverage.csv           — fraction of cells covered per dataset
  README_label_map.md               — how to use the CSV in the pipeline
"""
import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'code'))

from environment import get_environment
_env = get_environment()
RDS_DIR = _env['rds_dir']

VEL  = os.path.join(RDS_DIR, 'Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad')
PSY  = os.path.join(RDS_DIR, 'Cam_PsychAD/RNAseq/Aging_Cohort_100k_PFC_lessOld.h5ad')
WANG_HPC = 'Cam_snRNAseq/wang/wang_100k_PFC_lessOld.h5ad'  # may not exist locally

OUT = os.path.join(REPO_ROOT, 'scripts/integration_qc/outputs/label_map')
os.makedirs(OUT, exist_ok=True)


# Hand-curated mapping. shared_label gives a shared vocabulary; broad_class is
# the broad cell class. Cells whose dataset fine label is not in any row of
# the appropriate column (vel_cell_type / wang_type_updated / psychad_subclass)
# will be left as 'Unknown' for scANVI training (broad-class is still tracked
# separately in the pipeline for QC purposes).
MAPPING = [
    # shared_label,           broad_class,    vel_cell_type,    wang_type_updated,                  psychad_subclass
    ('EN_L2_3_IT',           'Excitatory',  'L2-3',           'EN-L2_3-IT',                       'EN_L2_3_IT'),
    ('EN_L4_IT',             'Excitatory',  'L4',             'EN-L4-IT',                         ''),
    # Vel "L5" and "L6" are coarse layer-only labels that don't distinguish
    # ET/IT/CT subtypes Wang and PsychAD resolve. Map them to the most common
    # subtype (IT) and leave the alternative shared labels (ET, CT) unsupervised
    # for Velmeshev. See reference/shared_fine_labels.csv for the consumed copy.
    ('EN_L5_ET',             'Excitatory',  '',               'EN-L5-ET',                         'EN_L5_ET'),
    ('EN_L5_IT',             'Excitatory',  'L5',             'EN-L5-IT',                         'EN_L3_5_IT_1|EN_L3_5_IT_2|EN_L3_5_IT_3'),
    ('EN_L5_6_NP',           'Excitatory',  '',               'EN-L5_6-NP',                       'EN_L5_6_NP'),
    ('EN_L6_CT',             'Excitatory',  '',               'EN-L6-CT',                         'EN_L6_CT'),
    ('EN_L6_IT',             'Excitatory',  'L5-6-IT|L6',     'EN-L6-IT',                         'EN_L6_IT_1|EN_L6_IT_2'),
    ('EN_L6B',               'Excitatory',  '',               'EN-L6b',                           'EN_L6B'),
    ('EN_Newborn',           'Excitatory',  'Progenitors',    'EN-Newborn|EN-IT-Immature|EN-ET-Immature|EN-CT-Immature', ''),
    ('EN_SP',                'Excitatory',  'SP',             '',                                 ''),

    ('IN_PV',                'Inhibitory',  'PV|PV_MP',       'IN-MGE-PV',                        'IN_PVALB|IN_PVALB_CHC'),
    ('IN_SST',               'Inhibitory',  'SST|SST_RELN',   'IN-MGE-SST',                       'IN_SST'),
    ('IN_VIP',               'Inhibitory',  'VIP',            'IN-CGE-VIP',                       'IN_VIP'),
    ('IN_LAMP5',             'Inhibitory',  '',               'IN-CGE-LAMP5',                     'IN_LAMP5_LHX6|IN_LAMP5_RELN'),
    ('IN_SNCG_ADARB2',       'Inhibitory',  '',               'IN-CGE-SNCG',                      'IN_ADARB2'),
    ('IN_RELN',              'Inhibitory',  'RELN',           '',                                 ''),
    ('IN_CALB2',             'Inhibitory',  'CALB2|CCK|NOS|SV2C|INT|Interneurons', '',            ''),
    ('IN_Immature',          'Inhibitory',  '',               'IN-Immature',                      ''),

    ('Astro',                'Astrocytes',  'Fibrous_astrocytes|Protoplasmic_astrocytes', 'Astrocyte|Astrocyte-Fibrous|Astrocyte-Protoplasmic', 'Astro'),
    ('Oligo',                'Oligos',      'Oligos',         'Oligodendrocyte|Oligodendrocyte-Immature', 'Oligo'),
    ('OPC',                  'OPC',         'OPC',            'OPC',                              'OPC'),
    ('Microglia',            'Microglia',   'Microglia',      'Microglia',                        'Micro|Adaptive|PVM'),
    ('Endo',                 'Endothelial', '',               'Endothelial|Pericyte|SMC|VLMC',    'Endo|PC|SMC|VLMC'),
    ('Glial_progenitor',     'Other',       'Glial_progenitors', 'RG-oRG|RG-tRG|RG-vRG|IPC',     ''),
]


def _split(cell):
    return [s for s in str(cell).split('|') if s]


def _coverage(adata_path, fine_col, mapping_col_index):
    """Return (fraction_covered, n_cells, n_covered, unmapped_examples)."""
    import anndata as ad
    if not os.path.exists(adata_path):
        return None
    a = ad.read_h5ad(adata_path, backed='r')
    fine = a.obs[fine_col].astype(str)
    mapped_set = set()
    for row in MAPPING:
        for fl in _split(row[mapping_col_index]):
            mapped_set.add(fl)
    is_mapped = fine.isin(mapped_set)
    unmapped_vals = fine[~is_mapped].value_counts().head(10).to_dict()
    return dict(n_cells=len(fine),
                n_covered=int(is_mapped.sum()),
                fraction=float(is_mapped.mean()),
                fine_col=fine_col,
                unmapped_top10=unmapped_vals)


def main():
    rows = []
    for shared, broad, vel, wang, psy in MAPPING:
        rows.append(dict(shared_label=shared, broad_class=broad,
                         vel_cell_type=vel, wang_type_updated=wang,
                         psychad_subclass=psy))
    df = pd.DataFrame(rows)
    csv_out = os.path.join(OUT, 'shared_fine_labels.csv')
    df.to_csv(csv_out, index=False)
    print(f'Saved {csv_out}')

    # Coverage assessment
    print('Velmeshev coverage:')
    vel_cov = _coverage(VEL, fine_col='cell_type', mapping_col_index=2)
    print(f'  {vel_cov}')

    print('PsychAD coverage:')
    psy_cov = _coverage(PSY, fine_col='subclass', mapping_col_index=4)
    print(f'  {psy_cov}')

    # Wang typically not local — skipped
    wang_cov = None
    print('Wang: skipped (raw h5ad not local). Run this script on HPC for Wang coverage.')

    cov_df = []
    for name, c in [('Velmeshev', vel_cov), ('PsychAD', psy_cov), ('Wang', wang_cov)]:
        if c is None:
            cov_df.append(dict(dataset=name, n_cells=np.nan, n_covered=np.nan,
                               fraction=np.nan, fine_col='', unmapped_top10=''))
        else:
            cov_df.append(dict(dataset=name, **{k: v for k, v in c.items()
                                                if k != 'unmapped_top10'},
                                unmapped_top10=str(c['unmapped_top10'])))
    pd.DataFrame(cov_df).to_csv(os.path.join(OUT, 'fine_label_coverage.csv'), index=False)
    print(f"Saved {os.path.join(OUT, 'fine_label_coverage.csv')}")

    # Usage README
    readme = [
        '# shared_fine_labels.csv — usage',
        '',
        'Columns:',
        '- `shared_label`: the target vocabulary used to supervise scANVI across all three datasets.',
        '- `broad_class`: broad cell class (Excitatory, Inhibitory, Astrocytes, Oligos, OPC, Microglia, Endothelial, Other).',
        '- `vel_cell_type`: pipe-separated values matched against Velmeshev raw `cell_type`.',
        '- `wang_type_updated`: pipe-separated values matched against Wang raw `Type-updated` (verify column name on HPC).',
        '- `psychad_subclass`: pipe-separated values matched against PsychAD raw `subclass`.',
        '',
        'Pipe (`|`) acts as OR. Empty string ⇒ no mapping for that dataset.',
        '',
        '## How to consume in `code/pipeline/downsample.py`',
        '',
        'Replace the block at lines ~288–306 that currently sets:',
        '',
        '```python',
        'adata.obs["cell_type_for_scanvi"] = np.where(',
        '    adata.obs["dataset"] == "WANG",',
        '    adata.obs["cell_type_raw"].astype(str),',
        '    "Unknown",',
        ')',
        '```',
        '',
        'with logic that consults this CSV per dataset, e.g.:',
        '',
        '```python',
        'mapping = pd.read_csv(SHARED_LABELS_CSV)',
        '',
        'def _map_for(dataset, raw_label, col):',
        '    for _, row in mapping.iterrows():',
        '        if raw_label in str(row[col]).split("|"):',
        '            return row["shared_label"]',
        '    return "Unknown"',
        '',
        'mask_v = adata.obs["dataset"] == "VELMESHEV"',
        'mask_w = adata.obs["dataset"] == "WANG"',
        'mask_p = adata.obs["dataset"] == "PSYCHAD"',
        'adata.obs.loc[mask_v, "cell_type_for_scanvi"] = (',
        '    adata.obs.loc[mask_v, "cell_type_raw"]',
        '    .map(lambda x: _map_for("VELMESHEV", x, "vel_cell_type")))',
        '... similar for WANG and PSYCHAD ...',
        '```',
        '',
        '## Coverage check',
        '',
        'See `fine_label_coverage.csv`. Aim for ≥ 80% of cells in each dataset to map cleanly.',
        'The unmapped_top10 column lists the most common unmapped labels — extend `MAPPING` in this script to absorb them.',
        '',
        '## Why this addresses the divergence',
        '',
        'Script 02 showed that the joint scANVI relabels PsychAD young-donor cells away from Excitatory entirely (10 donors with 0% Excitatory in joint vs 2–9% in PsychAD-only scANVI). The root cause is that the model is supervised only by Wang fine labels, and Wang\'s perinatal Excitatory phenotypes do not transfer cleanly to PsychAD young donors. Supervising scANVI with all three datasets\' biological labels (via this CSV) anchors the latent space to each dataset\'s structure and should preserve PsychAD\'s Excitatory neurons.',
    ]
    with open(os.path.join(OUT, 'README_label_map.md'), 'w') as fh:
        fh.write('\n'.join(readme))
    print('Saved README_label_map.md')
    print('Done.')


if __name__ == '__main__':
    main()
