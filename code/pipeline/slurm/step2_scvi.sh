#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/step2_scvi_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/step2_scvi_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --account=vertes-sl2-gpu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIG="${CONFIG:-code/pipeline/configs/source_hpc_config.yaml}"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "STEP 2: scVI + scANVI training and inference (diagnostics run separately as step 3)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "Config:    ${CONFIG}"
echo "Start:     $(date)"
echo "========================================"
_JOB_START=$(date +%s)

# pipeline.run_pipeline step=scvi generates a temporary scvi_config.yaml and
# calls scVI.run_pipeline internally (including scANVI when enabled). Diagnostics
# are NOT run here — they run as a separate CPU step 3 (step3_diagnostics.sh).
echo "Launching singularity exec (SIF: ${SIF})..."
singularity exec --nv \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m pipeline.run_pipeline \
        --config "${CONFIG}" \
        --steps scvi
SING_EXIT=$?
echo "Singularity exec finished (exit code: ${SING_EXIT})"

# Verify that scVI actually produced its output.  Singularity can exit 0 even
# when the container process silently did nothing (e.g. RDS not accessible on
# this node), so we check for the expected artifact independently.
OUTPUT_DIR=$(awk '/^output_dir:/{print $2; exit}' "${WORK_DIR}/${CONFIG}")
INTEGRATED_H5AD="${OUTPUT_DIR}/scvi_output/integrated.h5ad"
if [[ ! -f "${INTEGRATED_H5AD}" ]]; then
    echo "ERROR: singularity exited ${SING_EXIT} but integrated.h5ad was not created." >&2
    echo "  Expected: ${INTEGRATED_H5AD}" >&2
    echo "  Possible causes: RDS not mounted inside container on this node, micromamba" >&2
    echo "  environment not found, or Python import failure producing no output." >&2
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
echo "Step 2 complete: $(date)"
