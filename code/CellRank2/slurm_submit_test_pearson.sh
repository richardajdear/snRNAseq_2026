#!/bin/bash
#SBATCH --job-name=cr2_test_pearson
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/cr2_test_pearson_%j.log
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/cr2_test_pearson_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

set -euo pipefail

WORK_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake_full.sif"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
INPUT_H5AD="${DATA_DIR}/Cam_snRNAseq/integrated/VelWangPsychAD_100k_pearson/scvi_output/integrated.h5ad"
TEST_H5AD="${WORK_DIR}/tmp/pearson_5k_test.h5ad"
CONFIG="code/CellRank2/pearson_test_config.yaml"

echo "========================================"
echo "CellRank 2 Pipeline — TEST (5 000 cells)"
echo "Config:  ${CONFIG}"
echo "Workdir: ${WORK_DIR}"
echo "SIF:     ${SIF}"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "========================================"

mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/tmp"

# ── Downsample to 5 000 cells ─────────────────────────────────────────────────
echo "Downsampling to 5 000 cells → ${TEST_H5AD}"
singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n shortcake_default \
    python3 -u -c "
import anndata as ad, numpy as np
rng = np.random.default_rng(42)
# backed='r' memory-maps the file — only the selected rows are loaded into RAM
adata = ad.read_h5ad('${INPUT_H5AD}', backed='r')
print(f'Full dataset: {adata.n_obs} cells x {adata.n_vars} genes')
n = min(5000, adata.n_obs)
idx = np.sort(rng.choice(adata.n_obs, size=n, replace=False))
adata[idx].to_memory().write_h5ad('${TEST_H5AD}')
adata.file.close()
print(f'Saved {n} cells to ${TEST_H5AD}')
"

# ── Run CellRank 2 pipeline on the subset ─────────────────────────────────────
singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    --env "LD_LIBRARY_PATH=/opt/micromamba/envs/shortcake_default/lib" \
    "${SIF}" \
    micromamba run -n shortcake_default \
    env PYTHONPATH="code" python3 -u -m CellRank2.run_pipeline --config "${CONFIG}"

echo "Done."
