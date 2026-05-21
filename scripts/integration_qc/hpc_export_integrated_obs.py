"""
HPC-side helper: export a slim per-cell obs table from the joint integrated
h5ad so the alignment audit can run locally.

Usage on HPC:
    cd /home/rajd2/rds/hpc-work/snRNAseq_2026
    PYTHONPATH=code conda run -n scvi python scripts/integration_qc/hpc_export_integrated_obs.py

Outputs a CSV next to the integrated h5ad: integrated_obs.csv.gz
Roughly 50–80 MB compressed; scp to local
scripts/integration_qc/outputs/alignment/integrated_obs.csv.gz
"""
import os
import sys

import anndata as ad
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, 'code'))

from environment import get_environment
_env = get_environment()
RDS_DIR = _env['rds_dir']

INTEGRATED = os.path.join(
    RDS_DIR,
    'Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel'
    '/scvi_output/integrated.h5ad')

OUT_CSV = os.path.join(os.path.dirname(INTEGRATED), 'integrated_obs.csv.gz')

WANT_COLS = [
    'source', 'dataset', 'age_years', 'individual',
    'cell_class', 'cell_class_original',
    'cell_type_raw', 'cell_type_for_scanvi',
    'cell_type_aligned', 'cell_type_aligned_confidence',
    'chemistry', 'sex', 'region',
]


def main():
    print(f'Reading obs from: {INTEGRATED}')
    a = ad.read_h5ad(INTEGRATED, backed='r')
    obs = a.obs.copy()
    cols = [c for c in WANT_COLS if c in obs.columns]
    missing = [c for c in WANT_COLS if c not in obs.columns]
    if missing:
        print(f'WARNING: missing columns in obs: {missing}')
    obs_out = obs[cols].copy()
    obs_out['barcode'] = obs.index
    print(f'Writing {OUT_CSV}  ({len(obs_out):,} rows × {len(obs_out.columns)} cols)')
    obs_out.to_csv(OUT_CSV, index=False, compression='gzip')
    print('Done.')


if __name__ == '__main__':
    main()
