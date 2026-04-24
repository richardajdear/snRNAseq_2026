#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/retroactive_scvi_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/retroactive_scvi_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=vertes-sl2-gpu
#
# Retroactively add scvi_normalized layer and regenerate UMAP plots.
# Runs on GPU (ampere) because:
#   1. GPU nodes reliably have RDS mounted (some CPU nodes don't).
#   2. scVI get_normalized_expression is GPU-accelerated (~1h vs many hours on CPU).
#
# Required env vars:
#   OUTPUT_DIR  — path to the scvi_output directory (contains config.yaml,
#                 integrated.h5ad, scvi_model/)
#
# Usage:
#   sbatch --export=ALL,\
# OUTPUT_DIR=/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_100k_prepost_tuning3/scvi_output \
#     code/pipeline/slurm/step_retroactive_scvi_layer.sh

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

if [[ -z "${OUTPUT_DIR:-}" ]]; then
    echo "ERROR: OUTPUT_DIR must be set (path to scvi_output directory)" >&2
    exit 1
fi

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "RETROACTIVE: add scvi_normalized + regenerate UMAPs/plots"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES:-none}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "Start:      $(date)"
echo "========================================"
_JOB_START=$(date +%s)

# Verify SIF is accessible before invoking singularity — give a clear error
# if RDS is not mounted on this node rather than silently doing nothing.
if [[ ! -f "${SIF}" ]]; then
    echo "ERROR: SIF not found: ${SIF}" >&2
    echo "  RDS may not be mounted on $(hostname). Requeue on a different node." >&2
    exit 1
fi

echo "SIF found: ${SIF}"
echo "Launching singularity exec..."

singularity exec --nv \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m pipeline.retroactive_add_scvi_layer \
        --output_dir "${OUTPUT_DIR}" \
        --overwrite_h5ad
SING_EXIT=$?
echo "Singularity exit code: ${SING_EXIT}"

# Verify the layer was actually added — singularity can exit 0 without doing
# anything if the container process fails silently (Python import error, etc.)
python3 - <<'PYCHECK'
import sys, h5py, os
path = os.environ["OUTPUT_DIR"] + "/integrated.h5ad"
with h5py.File(path, "r") as f:
    layers = list(f.get("layers", {}).keys())
if "scvi_normalized" not in layers:
    print(f"ERROR: scvi_normalized not found in {path}", file=sys.stderr)
    print(f"  Layers present: {layers}", file=sys.stderr)
    sys.exit(1)
print(f"Verified: scvi_normalized present in {path}")
PYCHECK

_ELAPSED=$(( $(date +%s) - _JOB_START ))
_TIME_LIMIT=$(squeue -j "${SLURM_JOB_ID}" -h -o "%l" 2>/dev/null || echo "N/A")
_ALLOC_MEM_GB=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
_rss_gb="N/A"
if _tmp=$(sstat --jobs="${SLURM_JOB_ID}.batch" --format=MaxRSS --noheader 2>/dev/null); then
    _rss_kb=$(echo "$_tmp" | awk 'NR==1{gsub(/[^0-9]/,"",$1); print $1+0}')
    [[ -n "$_rss_kb" && "$_rss_kb" -gt 0 ]] && \
        _rss_gb=$(awk "BEGIN{printf \"%.2f\", ${_rss_kb}/1048576}")
fi
echo "========================================"
echo "Resource usage:"
echo "  Time:   $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s  /  ${_TIME_LIMIT} allocated"
echo "  Memory: ${_rss_gb}G peak RSS  /  ${_ALLOC_MEM_GB}G allocated"
echo "========================================"
echo "Retroactive fix complete: $(date)"
