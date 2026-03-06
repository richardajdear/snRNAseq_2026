#!/bin/bash
#SBATCH --job-name=scvi_100k
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/scvi_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/scvi_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --account=vertes-sl2-gpu

set -euo pipefail

# --- Configuration ---
WORK_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake.sif"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIG="${1:-code/scVI/hpc_config.yaml}"

# --- Ensure directories exist ---
mkdir -p "${WORK_DIR}/logs"

# --- Job info ---
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Config: ${CONFIG}"
echo "Start: $(date)"
echo "========================================"

# --- Run via Singularity with GPU ---
singularity exec --nv \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n shortcake_default \
    python3 -m code.scVI.run_pipeline --config "${CONFIG}"

echo "End: $(date)"
