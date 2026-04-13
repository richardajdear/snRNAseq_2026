#!/bin/bash
#SBATCH --job-name=run_script
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/run_script_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/run_script_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=10G
#SBATCH --partition=icelake

# Usage: sbatch scripts/run_script.sh <path/to/script.py>
#   e.g. sbatch scripts/run_script.sh scripts/debug_scanvi_pipeline.py
#   e.g. sbatch --mem=300G scripts/run_script.sh scripts/debug_scanvi_pipeline.py
#
# The script path can be absolute or relative to the repo root.

set -euo pipefail

SCRIPT="${1:-}"
if [[ -z "$SCRIPT" ]]; then
    echo "Usage: $0 <path/to/script.py>" >&2
    exit 1
fi

if [[ "$SCRIPT" != /* ]]; then
    SCRIPT="/home/rajd2/rds/hpc-work/snRNAseq_2026/${SCRIPT}"
fi

REPO_ROOT="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake.sif"
CONDA_ENV="shortcake_default"
PYTHON_BIN="/opt/micromamba/envs/${CONDA_ENV}/bin/python3"

echo "========================================"
echo "Script:  $SCRIPT"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "========================================"
_JOB_START=$(date +%s)

singularity exec \
    --pwd "$REPO_ROOT" \
    --env "R_LIBS_USER=/home/rajd2/R/library" \
    "$SIF" \
    micromamba run -n "$CONDA_ENV" \
    "$PYTHON_BIN" -u "$SCRIPT"

_ELAPSED=$(( $(date +%s) - _JOB_START ))
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
echo "  Time:   $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s"
echo "  Memory: ${_MAX_RSS_GB} peak RSS  /  ${_ALLOC_MEM_GB}G allocated"
echo "========================================"
