#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/step1_prep_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/step1_prep_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
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
_JOB_START=$(date +%s)

singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m pipeline.run_pipeline \
        --config "${CONFIG}" \
        --steps downsample combine

_ELAPSED=$(( $(date +%s) - _JOB_START ))
_TIME_LIMIT=$(squeue -j "${SLURM_JOB_ID}" -h -o "%l" 2>/dev/null || echo "N/A")
_MAX_RSS="N/A"
if _tmp=$(sstat --jobs="${SLURM_JOB_ID}.batch" --format=MaxRSS --noheader 2>/dev/null); then
    _MAX_RSS=$(echo "$_tmp" | awk 'NR==1{print $1}')
fi
_ALLOC_MEM_GB=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
echo "========================================"
echo "Resource usage:"
echo "  Time:   $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s  /  ${_TIME_LIMIT} allocated"
echo "  Memory: ${_MAX_RSS} peak RSS  /  ${_ALLOC_MEM_GB}G allocated"
echo "========================================"
echo "Step 1 complete: $(date)"
