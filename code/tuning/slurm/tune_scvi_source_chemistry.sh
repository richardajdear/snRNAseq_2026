#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/tune_scvi_source_chemistry_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/tune_scvi_source_chemistry_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --account=vertes-sl2-gpu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIG="${CONFIG:-code/tuning/source-chemistry_tuning_config.yaml}"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "scVI hyperparameter tuning (source-chemistry)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "Config:    ${CONFIG}"
echo "Start:     $(date)"
echo "========================================"
_JOB_START=$(date +%s)

singularity exec --nv \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m tuning.tune_scvi_batch \
        --config "${CONFIG}"

_ELAPSED=$(( $(date +%s) - _JOB_START ))
echo "========================================"
echo "Tuning complete"
echo "Elapsed: $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s"
echo "End:     $(date)"
echo "========================================"
