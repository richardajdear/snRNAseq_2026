#!/bin/bash
#SBATCH --job-name=render_notebook
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/render_notebook_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/render_notebook_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --partition=icelake

# Low-level render worker: renders a single .qmd with optional params file and output dir.
# Prefer render_single.sh for day-to-day use; this script is called by it.
#
# Usage:
#   sbatch render_notebook.sh <notebook.qmd> [params.yaml] [output_dir]
#   bash   render_notebook.sh <notebook.qmd> [params.yaml] [output_dir]
#
# All paths can be absolute or relative to the repo root.

set -euo pipefail

NOTEBOOK="${1:-}"
PARAMS_FILE="${2:-}"
OUTPUT_DIR="${3:-}"
FORCE="${4:-}"
OUTPUT_FILE="${5:-}"  # Optional: override output filename (e.g. my_config.html)

if [[ -z "$NOTEBOOK" ]]; then
    echo "Usage: $0 <path/to/notebook.qmd> [params.yaml] [output_dir] [--force] [output_file.md]" >&2
    exit 1
fi

REPO_ROOT="/home/rajd2/rds/hpc-work/snRNAseq_2026"

# Resolve to absolute paths
[[ "$NOTEBOOK"    != /* ]] && NOTEBOOK="${REPO_ROOT}/${NOTEBOOK}"
[[ -n "$PARAMS_FILE" && "$PARAMS_FILE" != /* ]] && PARAMS_FILE="${REPO_ROOT}/${PARAMS_FILE}"
[[ -n "$OUTPUT_DIR"  && "$OUTPUT_DIR"  != /* ]] && OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"

NOTEBOOK_DIR="$(dirname "$NOTEBOOK")"
NOTEBOOK_FILE="$(basename "$NOTEBOOK")"

# Build optional quarto flags
# Note: --execute-params requires papermill; we use NOTEBOOK_PARAMS env var instead.
PARAMS_ENV=""
[[ -n "$PARAMS_FILE" ]] && PARAMS_ENV="NOTEBOOK_PARAMS=${PARAMS_FILE}"

OUTPUT_ARG=""
if [[ -n "$OUTPUT_DIR" && -n "$OUTPUT_FILE" ]]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_ARG="--output '${OUTPUT_FILE}'"
elif [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

CACHE_FLAG=""
[[ "$FORCE" == "--force" ]] && CACHE_FLAG="--cache-refresh"

SIF="/home/rajd2/rds/hpc-work/shortcake.sif"
QUARTO_DIR="/usr/local/Cluster-Apps/ceuadmin/quarto/1.7.13"
CONDA_ENV="shortcake_default"
PYTHON_BIN="/opt/micromamba/envs/${CONDA_ENV}/bin/python3"

# When OUTPUT_DIR is set, copy the notebook there before rendering so that
# Quarto's intermediate files (*.quarto_ipynb, *_files/) land in the
# job-specific output directory rather than the shared templates directory.
# Without this, concurrent renders of the same template collide on those paths.
if [[ -n "$OUTPUT_DIR" ]]; then
    RENDER_NOTEBOOK="${OUTPUT_DIR}/${NOTEBOOK_FILE}"
    cp "${NOTEBOOK}" "${RENDER_NOTEBOOK}"
    RENDER_PWD="${OUTPUT_DIR}"
    _CLEANUP_NOTEBOOK=1
else
    RENDER_NOTEBOOK="${NOTEBOOK_FILE}"
    RENDER_PWD="${NOTEBOOK_DIR}"
    _CLEANUP_NOTEBOOK=0
fi

echo "========================================"
echo "Rendering: $NOTEBOOK"
[[ -n "$PARAMS_FILE"  ]] && echo "Params:    $PARAMS_FILE"
[[ -n "$OUTPUT_DIR"   ]] && echo "Output:    $OUTPUT_DIR"
[[ -n "$OUTPUT_FILE"  ]] && echo "Filename:  $OUTPUT_FILE"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "========================================"
_JOB_START=$(date +%s)

singularity exec \
    --pwd "$RENDER_PWD" \
    --bind "${QUARTO_DIR}:/quarto" \
    --env "R_LIBS_USER=/home/rajd2/R/library" \
    ${PARAMS_ENV:+--env "$PARAMS_ENV"} \
    "$SIF" \
    micromamba run -n "$CONDA_ENV" \
    bash -c "QUARTO_PYTHON=${PYTHON_BIN} /quarto/bin/quarto render '${RENDER_NOTEBOOK}' ${OUTPUT_ARG} ${CACHE_FLAG}"

# Remove the notebook copy; keep all generated output files
[[ "${_CLEANUP_NOTEBOOK}" -eq 1 ]] && rm -f "${RENDER_NOTEBOOK}"

_ELAPSED=$(( $(date +%s) - _JOB_START ))
_TIME_LIMIT=$(squeue -j "${SLURM_JOB_ID}" -h -o "%l" 2>/dev/null || echo "N/A")
_MAX_RSS_GB="N/A"
if _tmp=$(sstat --jobs="${SLURM_JOB_ID}.batch" --format=MaxRSS --noheader 2>/dev/null); then
    _MAX_RSS_GB=$(echo "$_tmp" | awk 'NR==1 && NF {
        val = $1
        unit = substr(val, length(val))
        num = substr(val, 1, length(val)-1) + 0
        if      (unit == "K") gb = num / 1048576
        else if (unit == "M") gb = num / 1024
        else if (unit == "G") gb = num
        else                  gb = num / 1073741824
        printf "%.1f G", gb
    }')
fi
_ALLOC_MEM_GB=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
echo "========================================"
echo "Resource usage:"
echo "  Time:   $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s  /  ${_TIME_LIMIT} allocated"
echo "  Memory: ${_MAX_RSS_GB} peak RSS  /  ${_ALLOC_MEM_GB}G allocated"
echo "========================================"
echo "Done: ${NOTEBOOK_FILE%.qmd} rendered successfully."
