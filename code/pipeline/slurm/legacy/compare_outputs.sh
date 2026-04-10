#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/compare_outputs_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/compare_outputs_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=icelake-himem
#SBATCH --mem=256G
#SBATCH --account=vertes-sl2-cpu

# Compare two scVI integration runs (seurat vs pearson_residuals HVG selection).
# CPU-only: LISI/ASW on a 30k subsample of the scANVI latent space.
# Both integrated.h5ad files must exist before running.
#
# Output is placed in:
#   integrated/compare_outputs/<RUN1>--<RUN2>/
# where run names are derived automatically from the input paths.
#
# Usage:
#   sbatch code/pipeline/slurm/compare_outputs.sh
#
#   # Custom inputs:
#   sbatch --export=ALL,SEURAT=<path>,PEARSON=<path> \
#          code/pipeline/slurm/compare_outputs.sh

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

SEURAT="${SEURAT:-${DATA_DIR}/Cam_snRNAseq/integrated/VelWangPsychAD_100k/scvi_output/integrated.h5ad}"
PEARSON="${PEARSON:-${DATA_DIR}/Cam_snRNAseq/integrated/VelWangPsychAD_100k_pearson/scvi_output/integrated.h5ad}"

# Derive run names and output dir automatically from input paths
# e.g. .../integrated/VelWangPsychAD_100k/scvi_output/integrated.h5ad → VelWangPsychAD_100k
SEURAT_RUN=$(basename "$(dirname "$(dirname "${SEURAT}")")")
PEARSON_RUN=$(basename "$(dirname "$(dirname "${PEARSON}")")")
INTEGRATED_DIR=$(dirname "$(dirname "$(dirname "${SEURAT}")")")
OUTPUT_DIR="${INTEGRATED_DIR}/compare_outputs/${SEURAT_RUN}--${PEARSON_RUN}"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "STEP: compare_outputs (seurat vs pearson)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Seurat:    ${SEURAT}"
echo "Pearson:   ${PEARSON}"
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
    env PYTHONPATH="code" python3 -m pipeline.compare_outputs \
        --seurat      "${SEURAT}" \
        --pearson     "${PEARSON}" \
        --output_dir  "${OUTPUT_DIR}"

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
echo "compare_outputs complete: $(date)"
