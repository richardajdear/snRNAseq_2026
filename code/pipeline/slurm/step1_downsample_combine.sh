#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/step1_prep_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/step1_prep_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIG="${CONFIG:-code/pipeline/hpc_config.yaml}"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "STEP 1: Downsample + Combine"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Config:    ${CONFIG}"
echo "Start:     $(date)"
echo "========================================"

singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m pipeline.run_pipeline \
        --config "${CONFIG}" \
        --steps downsample combine

echo "Step 1 complete: $(date)"
