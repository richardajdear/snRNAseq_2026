#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step4_notebook.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step4_notebook.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
CONFIG="${CONFIG:-code/pipeline/configs/source_hpc_config.yaml}"
# Track whether NOTEBOOK_TEMPLATE was explicitly set by the caller (env var takes
# priority over the config value; config takes priority over the default below).
_NOTEBOOK_TEMPLATE_ENVSET="${NOTEBOOK_TEMPLATE+set}"
NOTEBOOK_TEMPLATE="${NOTEBOOK_TEMPLATE:-notebooks/templates/grn_dev_v2.qmd}"
PSEUDOBULK_GROUP="${PSEUDOBULK_GROUP:-by_cell_class}"
# PSEUDOBULK_FILE and EXPERIMENT_NAME can be set directly to bypass CONFIG
# parsing (used by util_retransform.sh when submitting a dependency notebook).

RENDER_SCRIPT="${WORK_DIR}/notebooks/render_notebook.sh"

# Derive experiment name and pseudobulk file(s).
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
    # Allow notebook section of pipeline config to override template, pseudobulk group,
    # and experiment name. template: takes priority over the default but not an explicit
    # env-var override (tracked via _NOTEBOOK_TEMPLATE_ENVSET above).
    _nb_overrides=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('${WORK_DIR}/${CONFIG}'))
nb = cfg.get('notebook', {})
print(nb.get('pseudobulk_group', '') or '')
print(nb.get('experiment_name', '') or '')
print(nb.get('template', '') or '')
" 2>/dev/null || printf '\n\n')
    _pb_group=$(echo "${_nb_overrides}" | sed -n '1p')
    _exp_name=$(echo "${_nb_overrides}" | sed -n '2p')
    _nb_template=$(echo "${_nb_overrides}" | sed -n '3p')
    [[ -n "${_pb_group}" ]] && PSEUDOBULK_GROUP="${_pb_group}"
    [[ -n "${_nb_template}" && "${_NOTEBOOK_TEMPLATE_ENVSET}" != "set" ]] && \
        NOTEBOOK_TEMPLATE="${_nb_template}"
    PSEUDOBULK_FILE="${OUTPUT_DIR}/pseudobulk_output/${PSEUDOBULK_GROUP}.h5ad"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-${_exp_name:-${CONFIG_STEM}}}"
fi

TEMPLATE_PATH="${WORK_DIR}/${NOTEBOOK_TEMPLATE}"
TEMPLATE_STEM=$(basename "${NOTEBOOK_TEMPLATE}" .qmd)
RESULTS_DIR="${WORK_DIR}/notebooks/results/${EXPERIMENT_NAME}"
PARAMS_FILE="${RESULTS_DIR}/${TEMPLATE_STEM}_params.yaml"
OUTPUT_FILE="${TEMPLATE_STEM}.md"

mkdir -p "${RESULTS_DIR}"

# Generate params YAML for the notebook.
# When notebook.pseudobulk_inputs is a list in the config, write PSEUDOBULK_INPUTS;
# otherwise fall back to the single PSEUDOBULK_FILE path (backward compat).
if [[ "${_direct_mode}" == "false" ]]; then
    python3 - "${WORK_DIR}/${CONFIG}" "${OUTPUT_DIR}" "${EXPERIMENT_NAME}" "${PARAMS_FILE}" << 'PYEOF'
import sys, yaml
config_path, output_dir, experiment_name, params_file = sys.argv[1:]

cfg  = yaml.safe_load(open(config_path))
nb   = cfg.get('notebook', {})

params = {'EXPERIMENT_NAME': experiment_name}

pb_inputs = nb.get('pseudobulk_inputs')
if pb_inputs:
    MAX_INPUTS = 4
    if len(pb_inputs) > MAX_INPUTS:
        print(f"WARNING: pseudobulk_inputs has {len(pb_inputs)} entries; "
              f"only the first {MAX_INPUTS} will be used.", file=sys.stderr)
        pb_inputs = pb_inputs[:MAX_INPUTS]
    resolved = []
    for entry in pb_inputs:
        import os
        file_val = entry.get('file', '')
        if file_val and os.path.isabs(file_val):
            fpath = file_val
        else:
            group = entry.get('group') or entry.get('name', 'by_cell_class')
            fpath = f"{output_dir}/pseudobulk_output/{group}.h5ad"
        resolved.append({
            'name':             entry.get('name', os.path.splitext(os.path.basename(fpath))[0]),
            'file':             fpath,
            'cell_class_col':   entry.get('cell_class_col',  'cell_class') or '',
            'cell_class_value': entry.get('cell_class_value', 'Excitatory') or '',
        })
    params['PSEUDOBULK_INPUTS'] = resolved
else:
    # Single-input: PSEUDOBULK_FILE already set by shell; write it directly.
    # Read it from shell via the pre-computed value embedded below.
    params['PSEUDOBULK_FILE'] = '__PSEUDOBULK_FILE_PLACEHOLDER__'
    cell_class_col = nb.get('cell_class_col', '__NOTSET__')
    if cell_class_col != '__NOTSET__':
        params['CELL_CLASS_COL']   = nb.get('cell_class_col',  '') or ''
        params['CELL_CLASS_VALUE'] = nb.get('cell_class_value', '') or ''

with open(params_file, 'w') as fh:
    yaml.dump(params, fh, default_flow_style=False)
print('multi' if 'PSEUDOBULK_INPUTS' in params else 'single')
PYEOF
    # Replace placeholder with the actual shell-resolved PSEUDOBULK_FILE path.
    sed -i "s|__PSEUDOBULK_FILE_PLACEHOLDER__|${PSEUDOBULK_FILE}|" "${PARAMS_FILE}"
else
    # Direct mode: always single-file.
    cat > "${PARAMS_FILE}" << EOF
EXPERIMENT_NAME: ${EXPERIMENT_NAME}
PSEUDOBULK_FILE: ${PSEUDOBULK_FILE}
EOF
fi

# Verify at least the primary pseudobulk file exists (single-input path).
if grep -q "^PSEUDOBULK_FILE:" "${PARAMS_FILE}" 2>/dev/null; then
    if [[ ! -f "${PSEUDOBULK_FILE}" ]]; then
        echo "ERROR: pseudobulk file not found: ${PSEUDOBULK_FILE}" >&2
        echo "  Ensure pseudobulk step completed and PSEUDOBULK_GROUP='${PSEUDOBULK_GROUP}' is correct." >&2
        exit 1
    fi
fi
echo "Params written: ${PARAMS_FILE}"

echo "========================================"
echo "STEP 4: Notebook render"
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
echo "Step 4 complete: $(date)"
