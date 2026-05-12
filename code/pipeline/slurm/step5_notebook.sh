#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step5_notebook.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step5_notebook.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
CONFIG="${CONFIG:-code/pipeline/configs/source_hpc_config.yaml}"
# Override these at submission time if needed:
NOTEBOOK_TEMPLATE="${NOTEBOOK_TEMPLATE:-notebooks/templates/grn_dev_v2.qmd}"
PSEUDOBULK_GROUP="${PSEUDOBULK_GROUP:-by_cell_class}"
# PSEUDOBULK_FILE and EXPERIMENT_NAME can be set directly to bypass CONFIG
# parsing (used by util_retransform.sh when submitting a dependency notebook).

RENDER_SCRIPT="${WORK_DIR}/notebooks/render_notebook.sh"

# Derive experiment name and pseudobulk file.
# Direct-mode: if PSEUDOBULK_FILE is provided, use it as-is and skip CONFIG
# parsing. EXPERIMENT_NAME defaults to the grandparent dir of PSEUDOBULK_FILE
# (i.e. the retransform subdir, e.g. retransform_velmeshev_v3).
if [[ -n "${PSEUDOBULK_FILE:-}" ]]; then
    _direct_mode=true
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$(dirname "$(dirname "${PSEUDOBULK_FILE}")")")}"
else
    _direct_mode=false
    CONFIG_STEM=$(basename "${CONFIG}" .yaml)
    OUTPUT_DIR=$(awk '/^output_dir:/{print $2; exit}' "${WORK_DIR}/${CONFIG}")
    # Allow notebook section of pipeline config to override pseudobulk group and experiment name
    _nb_overrides=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('${WORK_DIR}/${CONFIG}'))
nb = cfg.get('notebook', {})
print(nb.get('pseudobulk_group', '') or '')
print(nb.get('experiment_name', '') or '')
" 2>/dev/null || printf '\n')
    _pb_group=$(echo "${_nb_overrides}" | sed -n '1p')
    _exp_name=$(echo "${_nb_overrides}" | sed -n '2p')
    [[ -n "${_pb_group}" ]] && PSEUDOBULK_GROUP="${_pb_group}"
    PSEUDOBULK_FILE="${OUTPUT_DIR}/pseudobulk_output/${PSEUDOBULK_GROUP}.h5ad"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-${_exp_name:-${CONFIG_STEM}}}"
fi

TEMPLATE_PATH="${WORK_DIR}/${NOTEBOOK_TEMPLATE}"
TEMPLATE_STEM=$(basename "${NOTEBOOK_TEMPLATE}" .qmd)
RESULTS_DIR="${WORK_DIR}/notebooks/results/${EXPERIMENT_NAME}"
PARAMS_FILE="${RESULTS_DIR}/${TEMPLATE_STEM}_params.yaml"
OUTPUT_FILE="${TEMPLATE_STEM}.md"

mkdir -p "${RESULTS_DIR}"

# Verify pseudobulk output exists before starting
if [[ ! -f "${PSEUDOBULK_FILE}" ]]; then
    echo "ERROR: pseudobulk file not found: ${PSEUDOBULK_FILE}" >&2
    echo "  Ensure pseudobulk step completed and PSEUDOBULK_GROUP='${PSEUDOBULK_GROUP}' is correct." >&2
    exit 1
fi

# Generate params YAML for the notebook
cat > "${PARAMS_FILE}" << EOF
EXPERIMENT_NAME: ${EXPERIMENT_NAME}
PSEUDOBULK_FILE: ${PSEUDOBULK_FILE}
EOF

# Append optional cell-class filter params from notebook config section.
# When cell_class_col is set to '' in the config, the notebook skips filtering
# (needed for single-class datasets that are already pre-filtered).
# Skipped in direct mode (no CONFIG to read from).
if [[ "${_direct_mode}" == "false" ]]; then
    _cell_class_col=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('${WORK_DIR}/${CONFIG}'))
nb = cfg.get('notebook', {})
print(nb.get('cell_class_col', '__NOTSET__'))
" 2>/dev/null || echo "__NOTSET__")
    if [[ "${_cell_class_col}" != "__NOTSET__" ]]; then
        _cell_class_val=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('${WORK_DIR}/${CONFIG}'))
print(cfg.get('notebook', {}).get('cell_class_value', ''))
" 2>/dev/null || echo "")
        echo "CELL_CLASS_COL: '${_cell_class_col}'" >> "${PARAMS_FILE}"
        echo "CELL_CLASS_VALUE: '${_cell_class_val}'" >> "${PARAMS_FILE}"
    fi
fi
echo "Params written: ${PARAMS_FILE}"

echo "========================================"
echo "STEP 5: Notebook render"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Experiment:   ${EXPERIMENT_NAME}"
[[ "${_direct_mode}" == "false" ]] && echo "Config:       ${CONFIG}"
echo "Pseudobulk:   ${PSEUDOBULK_FILE}"
echo "Template:     ${TEMPLATE_PATH}"
echo "Params:       ${PARAMS_FILE}"
echo "Output:       ${RESULTS_DIR}/${OUTPUT_FILE}"
echo "Start:        $(date)"
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
