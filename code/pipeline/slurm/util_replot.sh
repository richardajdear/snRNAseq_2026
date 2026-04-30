#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_util_replot.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_util_replot.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=icelake
#SBATCH --mem=64G
#SBATCH --account=vertes-sl2-cpu

# Utility: regenerate scVI output plots from an existing integrated.h5ad.
#
# NOT part of the normal pipeline chain. Use this when you want to regenerate
# scvi_output/plots/ (umaps_latent.png, umaps_inferred.png, pca_latent.png,
# pca_inferred.png) without re-running scVI/scANVI training or inference.
#
# Requires that integrated.h5ad already exists with all obsm keys (X_pca_raw,
# X_scVI, X_scANVI, X_pca_scvi_inferred, X_pca_scanvi_inferred, X_umap_*).
# If X_pca_scvi_latent / X_pca_scanvi_latent are missing they will be computed
# on the fly by plot_pca_grids() before plotting.
#
# Reads:  <output_dir>/scvi_output/integrated.h5ad  (via scvi_config.yaml)
# Writes: <output_dir>/scvi_output/plots/*.png
#
# Usage:
#   SCVI_CONFIG=/path/to/scvi_output/config.yaml
#   sbatch --export=ALL,SCVI_CONFIG="${SCVI_CONFIG}" \
#          code/pipeline/slurm/util_replot.sh

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
SCVI_CONFIG="${SCVI_CONFIG:-}"

if [[ -z "${SCVI_CONFIG}" ]]; then
    echo "ERROR: SCVI_CONFIG must be set (path to scvi_output/config.yaml)" >&2
    exit 1
fi

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "UTILITY: Regenerate scVI plots"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "SCVI config: ${SCVI_CONFIG}"
echo "Start:     $(date)"
echo "========================================"
_JOB_START=$(date +%s)

singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m scVI.run_pipeline \
        --config "${SCVI_CONFIG}" \
        --steps plot

_ELAPSED=$(( $(date +%s) - _JOB_START ))
_TIME_LIMIT=$(squeue -j "${SLURM_JOB_ID}" -h -o "%l" 2>/dev/null || echo "N/A")
_ALLOC_MEM_GB=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
_rss_gb="N/A"
if _tmp=$(sstat --jobs="${SLURM_JOB_ID}.batch" --format=MaxRSS --noheader 2>/dev/null); then
    _rss_kb=$(echo "$_tmp" | awk 'NR==1{gsub(/[^0-9]/,"",$1); print $1+0}')
    if [[ -n "$_rss_kb" && "$_rss_kb" -gt 0 ]]; then
        _rss_gb=$(awk "BEGIN{printf \"%.2f\", ${_rss_kb}/1048576}")
    fi
fi
echo "========================================"
echo "Resource usage:"
echo "  Time:   $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s  /  ${_TIME_LIMIT} allocated"
echo "  Memory: ${_rss_gb}G peak RSS  /  ${_ALLOC_MEM_GB}G allocated"
echo "========================================"
echo "Utility complete: $(date)"
