#!/bin/bash
#SBATCH --job-name=age_bins_diag
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/age_bins_%j.log
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/age_bins_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

set -euo pipefail

WORK_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake_full.sif"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
H5AD="${DATA_DIR}/Cam_snRNAseq/integrated/VelWangPsychAD_100k_source-chemistry/scvi_output/integrated.h5ad"

mkdir -p "${WORK_DIR}/scripts/outputs"

singularity exec \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n shortcake_default \
    python3 -u - <<'PYEOF'
import anndata as ad
import numpy as np
import pandas as pd

h5ad = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_100k_source-chemistry/scvi_output/integrated.h5ad"
print(f"Loading (backed): {h5ad}")
adata = ad.read_h5ad(h5ad, backed='r')
print(f"Shape: {adata.n_obs} cells x {adata.n_vars} genes")
print(f"obs columns: {list(adata.obs.columns)}")
print(f"obsm keys:   {list(adata.obsm.keys())}")

age = adata.obs['age_years'].astype(float)
print(f"\nage_years: min={age.min():.4f}  max={age.max():.2f}  nulls={age.isna().sum()}")

print("\n--- Unique age values (sorted) with cell counts ---")
vc = age.value_counts().sort_index()
for v, c in vc.items():
    print(f"  {v:8.4f}  →  {c:6d} cells")

# Prenatal ages (< 0 or fractional < 0.5)
prenatal = age[age < 0.5]
print(f"\nPrenatal/neonatal (<0.5 yrs): {len(prenatal)} cells")
print(f"  unique values: {sorted(prenatal.unique())}")

# Check cell type key
ct_key = 'cell_type_aligned' if 'cell_type_aligned' in adata.obs.columns else None
if ct_key:
    print(f"\n--- Cell type distribution (top 30) ---")
    ct_vc = adata.obs[ct_key].value_counts().head(30)
    for ct, n in ct_vc.items():
        print(f"  {ct:<45s}  {n:6d}")

adata.file.close()
print("\nDone.")
PYEOF
