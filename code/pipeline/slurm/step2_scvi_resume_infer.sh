#!/bin/bash
# step2_scvi_resume_infer.sh — Resume scVI pipeline from inference only.
#
# Use when scVI + scANVI training completed successfully but the job timed out
# before integrated.h5ad was fully written. Both trained models must exist in
# scvi_output/scvi_model/ and scvi_output/scanvi_model/.
#
# This calls scVI.run_pipeline directly (bypassing pipeline.run_pipeline) with
# --steps prep infer umap plot save, which loads the saved models from disk and
# skips all training. It reads the scvi_config.yaml that was written by the
# original step2 job into the output directory.
#
# Usage (from project root):
#   sbatch --export=ALL,SCVI_CONFIG=<path/to/scvi_config.yaml> \
#          code/pipeline/slurm/step2_scvi_resume_infer.sh
#
# Or pass SCVI_CONFIG as an environment variable before submitting:
#   SCVI_CONFIG=/path/to/.../scvi_output/config.yaml
#   sbatch --export=ALL,SCVI_CONFIG="${SCVI_CONFIG}" \
#          code/pipeline/slurm/step2_scvi_resume_infer.sh

#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/step2_scvi_resume_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/step2_scvi_resume_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --account=vertes-sl2-gpu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

# Default to the tuning2 run's auto-generated scvi_config.yaml
SCVI_CONFIG="${SCVI_CONFIG:-/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_100k_postnatal_source-chemistry_tuning2/scvi_output/config.yaml}"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "STEP 2 RESUME: scVI inference + umap + save (no training)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "Config:    ${SCVI_CONFIG}"
echo "Start:     $(date)"
echo "========================================"
_JOB_START=$(date +%s)

# Verify models exist before starting
SCVI_OUTPUT_DIR=$(awk '/^output_dir:/{print $2; exit}' "${SCVI_CONFIG}")
if [[ ! -d "${SCVI_OUTPUT_DIR}/scvi_model" ]]; then
    echo "ERROR: scvi_model not found at ${SCVI_OUTPUT_DIR}/scvi_model" >&2
    exit 1
fi
if [[ ! -d "${SCVI_OUTPUT_DIR}/scanvi_model" ]]; then
    echo "ERROR: scanvi_model not found at ${SCVI_OUTPUT_DIR}/scanvi_model" >&2
    exit 1
fi
echo "scvi_model:   ${SCVI_OUTPUT_DIR}/scvi_model  [found]"
echo "scanvi_model: ${SCVI_OUTPUT_DIR}/scanvi_model [found]"

# Remove corrupt/incomplete integrated.h5ad if present so the save step
# writes a fresh file rather than appending to a broken one.
INTEGRATED_H5AD="${SCVI_OUTPUT_DIR}/integrated.h5ad"
if [[ -f "${INTEGRATED_H5AD}" ]]; then
    echo "Removing corrupt/incomplete integrated.h5ad: ${INTEGRATED_H5AD}"
    rm -f "${INTEGRATED_H5AD}"
fi

echo "Launching singularity exec (SIF: ${SIF})..."
singularity exec --nv \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m scVI.run_pipeline \
        --config "${SCVI_CONFIG}" \
        --steps prep infer umap plot save
SING_EXIT=$?
echo "Singularity exec finished (exit code: ${SING_EXIT})"

# Verify output was produced
if [[ ! -f "${INTEGRATED_H5AD}" ]]; then
    echo "ERROR: singularity exited ${SING_EXIT} but integrated.h5ad was not created." >&2
    echo "  Expected: ${INTEGRATED_H5AD}" >&2
    exit 1
fi
echo "Output verified: ${INTEGRATED_H5AD}"

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
echo "Resume step complete: $(date)"
echo ""
echo "Next: run diagnostics + pseudobulk with:"
echo "  sbatch --export=ALL,WORK_DIR=${WORK_DIR},CONFIG=<pipeline_config.yaml> \\"
echo "         code/pipeline/slurm/step4_pseudobulk.sh"
