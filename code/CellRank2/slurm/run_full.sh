#!/bin/bash
#SBATCH --job-name=cr2_full
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/cr2_full_%j.log
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/cr2_full_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

# For GPU-accelerated OT (requires GPU partition access):
#   Change partition to e.g. ampere, change account to your GPU account,
#   and add:  #SBATCH --gres=gpu:1
# The pipeline auto-detects GPU via ot_device: auto in the config.

set -euo pipefail

WORK_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake_full.sif"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIG="code/CellRank2/config.yaml"

echo "========================================"
echo "CellRank 2 Pipeline — full run"
echo "Config:  ${CONFIG}"
echo "Workdir: ${WORK_DIR}"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "========================================"

mkdir -p "${WORK_DIR}/logs"

# JAX prefers not to preallocate the full GPU memory pool
export XLA_PYTHON_CLIENT_PREALLOCATE=false

singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    --env "LD_LIBRARY_PATH=/opt/micromamba/envs/shortcake_default/lib" \
    "${SIF}" \
    micromamba run -n shortcake_default \
    env PYTHONPATH="code" python3 -u -m CellRank2.run_pipeline --config "${CONFIG}"

echo "Done."
