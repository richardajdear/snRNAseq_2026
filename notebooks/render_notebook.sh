#!/bin/bash
#SBATCH --job-name=render_notebook
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/render_notebook_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/render_notebook_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --partition=icelake

# Usage: sbatch render_notebook.sh <path/to/notebook.qmd>
#   or for login-node testing: bash render_notebook.sh <path/to/notebook.qmd>
#
# The notebook path can be absolute or relative to the repo root
# (/home/rajd2/rds/hpc-work/snRNAseq_2026).

set -euo pipefail

NOTEBOOK="${1:-}"
if [[ -z "$NOTEBOOK" ]]; then
    echo "Usage: $0 <path/to/notebook.qmd>" >&2
    exit 1
fi

# Resolve to absolute path
if [[ "$NOTEBOOK" != /* ]]; then
    NOTEBOOK="/home/rajd2/rds/hpc-work/snRNAseq_2026/${NOTEBOOK}"
fi

NOTEBOOK_DIR="$(dirname "$NOTEBOOK")"
NOTEBOOK_FILE="$(basename "$NOTEBOOK")"

REPO_ROOT="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake.sif"
QUARTO_DIR="/usr/local/Cluster-Apps/ceuadmin/quarto/1.7.13"
CONDA_ENV="shortcake_default"
PYTHON_BIN="/opt/micromamba/envs/${CONDA_ENV}/bin/python3"

echo "========================================"
echo "Rendering: $NOTEBOOK"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Output:    $NOTEBOOK_DIR"
echo "Start:     $(date)"
echo "========================================"
_JOB_START=$(date +%s)

singularity exec \
    --pwd "$NOTEBOOK_DIR" \
    --bind "${QUARTO_DIR}:/quarto" \
    --env "R_LIBS_USER=/home/rajd2/R/library" \
    "$SIF" \
    micromamba run -n "$CONDA_ENV" \
    bash -c "QUARTO_PYTHON=${PYTHON_BIN} /quarto/bin/quarto render '${NOTEBOOK_FILE}'"

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
echo "Done: ${NOTEBOOK_FILE%.qmd} rendered successfully."
