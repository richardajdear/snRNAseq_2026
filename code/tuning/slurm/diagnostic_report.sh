#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/diagnostic_report_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/diagnostic_report_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=icelake
#SBATCH --mem=64G
#SBATCH --account=vertes-sl2-cpu

# Diagnostic report for scVI/scANVI input data characterisation.
#
# Loads integrated.h5ad in backed mode and produces:
#   data_availability.csv, chemistry_age_crosstab.csv+png,
#   anchor_donors.csv, donor_age_structure.csv+png,
#   cell_type_labels.csv, excitatory_lineage_proxy.csv,
#   marker_expression.csv+png, confound_summary.txt
#
# Usage:
#   sbatch code/tuning/slurm/diagnostic_report.sh
#   # override input/output:
#   sbatch --export=ALL,INPUT=/path/to.h5ad,OUTPUT=/path/to/dir \
#          code/tuning/slurm/diagnostic_report.sh

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

INPUT="${INPUT:-${DATA_DIR}/Cam_snRNAseq/integrated/VelWangPsychAD_100k_source-chemistry/scvi_output/integrated.h5ad}"
OUTPUT="${OUTPUT:-${DATA_DIR}/Cam_snRNAseq/diagnostics}"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "scVI diagnostic report"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Input:   ${INPUT}"
echo "Output:  ${OUTPUT}"
echo "Start:   $(date)"
echo "========================================"
_JOB_START=$(date +%s)

singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m tuning.diagnostic_report \
        --input  "${INPUT}" \
        --output "${OUTPUT}"

_ELAPSED=$(( $(date +%s) - _JOB_START ))
_MAX_RSS="N/A"
if _tmp=$(sstat --jobs="${SLURM_JOB_ID}.batch" --format=MaxRSS --noheader 2>/dev/null); then
    _MAX_RSS=$(echo "$_tmp" | awk 'NR==1{print $1}')
fi
_ALLOC_MEM_GB=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
echo "========================================"
echo "Resource usage:"
echo "  Time:   $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s  /  02:00:00 allocated"
echo "  Memory: ${_MAX_RSS} peak RSS  /  ${_ALLOC_MEM_GB}G allocated"
echo "========================================"
echo "Diagnostic report complete: $(date)"
echo "Output written to: ${OUTPUT}"
