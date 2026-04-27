#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/tuning_diagnostics_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/tuning_diagnostics_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=icelake
#SBATCH --mem=8G
#SBATCH --account=vertes-sl2-cpu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

INPUT_DIR="${INPUT_DIR:-${DATA_DIR}/Cam_snRNAseq/integrated/VelWangPsychAD_100k_source-chemistry/scvi_tuning_round4}"
OUTPUT_DIR="${OUTPUT_DIR:-${INPUT_DIR}/plots}"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "scVI tuning diagnostics (plots)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Input:     ${INPUT_DIR}"
echo "Output:    ${OUTPUT_DIR}"
echo "Start:     $(date)"
echo "========================================"
_JOB_START=$(date +%s)

singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m tuning.tuning_diagnostics \
        --input_dir "${INPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}"

_ELAPSED=$(( $(date +%s) - _JOB_START ))
echo "========================================"
echo "Diagnostics complete"
echo "Elapsed: $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s"
echo "End:     $(date)"
echo "========================================"
