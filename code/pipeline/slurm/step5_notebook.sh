#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step5_notebook.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step5_notebook.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
CONFIG="${CONFIG:-code/pipeline/configs/source_hpc_config.yaml}"
# Override these at submission time if needed:
NOTEBOOK_TEMPLATE="${NOTEBOOK_TEMPLATE:-notebooks/templates/grn_dev.qmd}"
PSEUDOBULK_GROUP="${PSEUDOBULK_GROUP:-by_cell_class}"

RENDER_SCRIPT="${WORK_DIR}/notebooks/render_notebook.sh"

# Derive experiment name from config filename stem
CONFIG_STEM=$(basename "${CONFIG}" .yaml)

# Parse output_dir from config (same pattern as step2_scvi.sh)
OUTPUT_DIR=$(awk '/^output_dir:/{print $2; exit}' "${WORK_DIR}/${CONFIG}")

TEMPLATE_PATH="${WORK_DIR}/${NOTEBOOK_TEMPLATE}"
RESULTS_DIR="${WORK_DIR}/notebooks/results/${CONFIG_STEM}"
PARAMS_FILE="${RESULTS_DIR}/${CONFIG_STEM}_params.yaml"
OUTPUT_FILE="${CONFIG_STEM}.md"
PSEUDOBULK_FILE="${OUTPUT_DIR}/pseudobulk_output/${PSEUDOBULK_GROUP}.h5ad"

mkdir -p "${RESULTS_DIR}"

# Verify pseudobulk output exists before starting
if [[ ! -f "${PSEUDOBULK_FILE}" ]]; then
    echo "ERROR: pseudobulk file not found: ${PSEUDOBULK_FILE}" >&2
    echo "  Ensure pseudobulk step completed and PSEUDOBULK_GROUP='${PSEUDOBULK_GROUP}' is correct." >&2
    exit 1
fi

# Generate params YAML for the notebook
cat > "${PARAMS_FILE}" << EOF
EXPERIMENT_NAME: ${CONFIG_STEM}
PSEUDOBULK_FILE: ${PSEUDOBULK_FILE}
EOF
echo "Params written: ${PARAMS_FILE}"

echo "========================================"
echo "STEP 5: Notebook render"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Config:    ${CONFIG}"
echo "Template:  ${TEMPLATE_PATH}"
echo "Params:    ${PARAMS_FILE}"
echo "Output:    ${RESULTS_DIR}/${OUTPUT_FILE}"
echo "Start:     $(date)"
echo "========================================"
_JOB_START=$(date +%s)

bash "${RENDER_SCRIPT}" "${TEMPLATE_PATH}" "${PARAMS_FILE}" "${RESULTS_DIR}" '' "${OUTPUT_FILE}"

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
echo "Step 5 complete: $(date)"
