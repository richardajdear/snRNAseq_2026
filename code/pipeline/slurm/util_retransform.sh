#!/bin/bash
# util_retransform.sh — Re-run scVI inference + pseudobulk with a different transform_batch.
#
# Uses an existing trained scVI + scANVI model to produce new normalised expression
# (scvi_normalized, scanvi_normalized) under a different reference batch, then
# aggregates the result into pseudobulk h5ads.
#
# Required env vars:
#   SCVI_CONFIG  — path to the transform-batch scVI config yaml
#                  (e.g. scvi_config_transformPsychAD.yaml)
#   PB_OUTPUT    — output directory for pseudobulk h5ads
#                  (e.g. <base>/pseudobulk_output_transformPsychAD)
#   PB_CONFIG    — path to the pipeline_config.yaml (for pseudobulk group definitions)
#
# Usage:
#   BASE=/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_excitatory_1y+_tuning4
#
#   sbatch \
#     --export=ALL,\
#   SCVI_CONFIG="${BASE}/scvi_config_transformPsychAD.yaml",\
#   PB_OUTPUT="${BASE}/pseudobulk_output_transformPsychAD",\
#   PB_CONFIG="${BASE}/pipeline_config.yaml" \
#     code/pipeline/slurm/util_retransform.sh
#
#   sbatch \
#     --export=ALL,\
#   SCVI_CONFIG="${BASE}/scvi_config_transformVelmeshev.yaml",\
#   PB_OUTPUT="${BASE}/pseudobulk_output_transformVelmeshev",\
#   PB_CONFIG="${BASE}/pipeline_config.yaml" \
#     code/pipeline/slurm/util_retransform.sh

#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_util_retransform.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_util_retransform.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --account=vertes-sl2-gpu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

for var in SCVI_CONFIG PB_OUTPUT PB_CONFIG; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: ${var} must be set" >&2
        exit 1
    fi
done

mkdir -p "${WORK_DIR}/logs"

# Extract paths from the scVI config for pre-flight checks.
# scvi_model_dir and scanvi_model_dir are absolute paths in these configs.
SCVI_MODEL_DIR=$(awk '/^scvi_model_dir:/{print $2; exit}' "${SCVI_CONFIG}")
SCANVI_MODEL_DIR=$(awk '/^scanvi_model_dir:/{print $2; exit}' "${SCVI_CONFIG}")
SCVI_OUTPUT_DIR=$(awk '/^output_dir:/{print $2; exit}' "${SCVI_CONFIG}")
TRANSFORM_BATCH=$(awk '/^transform_batch:/{print $2; exit}' "${SCVI_CONFIG}")

echo "========================================"
echo "UTILITY: Re-run inference with transform_batch=${TRANSFORM_BATCH}"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "SCVI config: ${SCVI_CONFIG}"
echo "Output dir:  ${SCVI_OUTPUT_DIR}"
echo "PB output:   ${PB_OUTPUT}"
echo "Start:       $(date)"
echo "========================================"
_JOB_START=$(date +%s)

# Pre-flight: verify original models exist
if [[ ! -d "${SCVI_MODEL_DIR}" ]]; then
    echo "ERROR: scvi_model not found: ${SCVI_MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -d "${SCANVI_MODEL_DIR}" ]]; then
    echo "ERROR: scanvi_model not found: ${SCANVI_MODEL_DIR}" >&2
    exit 1
fi
echo "scvi_model:   ${SCVI_MODEL_DIR}  [found]"
echo "scanvi_model: ${SCANVI_MODEL_DIR} [found]"

# --- STEP 1: scVI inference + umap + plot + save ---
echo ""
echo "--- scVI inference (transform_batch=${TRANSFORM_BATCH}) ---"
singularity exec --nv \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m scVI.run_pipeline \
        --config "${SCVI_CONFIG}"

INTEGRATED_H5AD="${SCVI_OUTPUT_DIR}/integrated.h5ad"
if [[ ! -f "${INTEGRATED_H5AD}" ]]; then
    echo "ERROR: scVI.run_pipeline exited 0 but integrated.h5ad not found: ${INTEGRATED_H5AD}" >&2
    exit 1
fi
echo "scVI inference complete. Output: ${INTEGRATED_H5AD}"

# --- STEP 2: Pseudobulk aggregation ---
echo ""
echo "--- Pseudobulk aggregation → ${PB_OUTPUT} ---"
singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m pipeline.pseudobulk \
        --input  "${INTEGRATED_H5AD}" \
        --output "${PB_OUTPUT}" \
        --config "${PB_CONFIG}" \
        --overwrite

echo "Pseudobulk complete."

_ELAPSED=$(( $(date +%s) - _JOB_START ))
_TIME_LIMIT=$(squeue -j "${SLURM_JOB_ID}" -h -o "%l" 2>/dev/null || echo "N/A")
_ALLOC_MEM_GB=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
_rss_gb="N/A"; _rss_pct="N/A"
if _tmp=$(sstat --jobs="${SLURM_JOB_ID}.batch" --format=MaxRSS --noheader 2>/dev/null); then
    _rss_kb=$(echo "$_tmp" | awk 'NR==1{gsub(/[^0-9]/,"",$1); print $1+0}')
    if [[ -n "$_rss_kb" && "$_rss_kb" -gt 0 ]]; then
        _rss_gb=$(awk  "BEGIN{printf \"%.2f\", ${_rss_kb}/1048576}")
        [[ ${_ALLOC_MEM_GB} -gt 0 ]] && \
            _rss_pct=$(awk "BEGIN{printf \"%.0f\", (${_rss_kb}/1048576)/${_ALLOC_MEM_GB}*100}")
    fi
fi
echo "========================================"
echo "Resource usage:"
echo "  Time:   $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s  /  ${_TIME_LIMIT} allocated"
echo "  Memory: ${_rss_gb}G peak RSS  /  ${_ALLOC_MEM_GB}G allocated (${_rss_pct}%)"
echo "========================================"
echo "Utility complete: $(date)"
echo ""
echo "Outputs:"
echo "  Plots:      ${SCVI_OUTPUT_DIR}/plots/"
echo "  Pseudobulk: ${PB_OUTPUT}/"
