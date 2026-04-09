#!/bin/bash
#SBATCH --job-name=fix_cell_class
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/fix_cell_class_%A_%a.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/fix_cell_class_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl3-cpu

# Retroactive fix: add cell_class_original and regenerate diagnostic plots for
# multiple pipeline runs.
#
# This is a Slurm array job: each array element processes one config file.
# The CONFIGS_FILE should list one pipeline config YAML path per line.
#
# Usage — fix all configs listed in a file:
#
#   # Create a file with one config per line:
#   cat > /tmp/configs_to_fix.txt << 'EOF'
#   code/pipeline/hpc_config.yaml
#   code/pipeline/pearson_hpc_config.yaml
#   code/pipeline/dataset_hpc_config.yaml
#   EOF
#
#   # Submit (auto-sizes the array from the number of lines):
#   cd /home/rajd2/rds/hpc-work/snRNAseq_2026
#   N=$(grep -c . /tmp/configs_to_fix.txt)
#   sbatch --array=1-${N} \
#          --export=ALL,CONFIGS_FILE=/tmp/configs_to_fix.txt \
#          code/pipeline/slurm/step_retroactive_fix.sh
#
# Usage — pass configs directly (space-separated, quoted):
#
#   cd /home/rajd2/rds/hpc-work/snRNAseq_2026
#   sbatch --array=1-3 \
#          --export=ALL,CONFIGS="code/pipeline/hpc_config.yaml code/pipeline/pearson_hpc_config.yaml code/pipeline/dataset_hpc_config.yaml" \
#          code/pipeline/slurm/step_retroactive_fix.sh
#
# Options forwarded to fix_cell_class_original.py via environment variables:
#   FORCE=1            — pass --force (re-apply even if already fixed)
#   NO_DIAGNOSTICS=1   — pass --no_diagnostics (fix h5ad only, skip plot regen)
#   NO_PSEUDOBULK=1    — skip the pseudobulk re-aggregation step

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIGS_FILE="${CONFIGS_FILE:-}"
CONFIGS="${CONFIGS:-}"

mkdir -p "${WORK_DIR}/logs"

# ── Resolve the config for this array element ──────────────────────────────
ARRAY_IDX="${SLURM_ARRAY_TASK_ID:-1}"

if [[ -n "${CONFIGS_FILE}" ]]; then
    # Support both absolute paths and paths relative to WORK_DIR
    [[ "${CONFIGS_FILE}" = /* ]] && _CF="${CONFIGS_FILE}" || _CF="${WORK_DIR}/${CONFIGS_FILE}"
    CONFIG=$(sed -n "${ARRAY_IDX}p" "${_CF}")
elif [[ -n "${CONFIGS}" ]]; then
    # Convert space-separated string to indexed access
    read -ra _CONFIGS_ARR <<< "${CONFIGS}"
    CONFIG="${_CONFIGS_ARR[$((ARRAY_IDX - 1))]}"
else
    echo "ERROR: set CONFIGS_FILE or CONFIGS before submitting." >&2
    exit 1
fi

if [[ -z "${CONFIG}" ]]; then
    echo "ERROR: No config found for array index ${ARRAY_IDX}." >&2
    exit 1
fi

# ── Extra flags ──────────────────────────────────────────────────────────────
EXTRA_FLAGS=""
[[ "${FORCE:-0}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --force"
[[ "${NO_DIAGNOSTICS:-0}" == "1" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --no_diagnostics"

echo "========================================"
echo "Retroactive cell_class fix"
echo "Job ID:   ${SLURM_JOB_ID:-local}  Array: ${ARRAY_IDX}"
echo "Node:     $(hostname)"
echo "Config:   ${CONFIG}"
echo "Flags:    ${EXTRA_FLAGS:-none}"
echo "Start:    $(date)"
echo "========================================"
_JOB_START=$(date +%s)

singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m pipeline.fix_cell_class_original \
        --config "${CONFIG}" \
        ${EXTRA_FLAGS}

echo "Fix step complete: $(date)"

# ── Pseudobulk re-run ────────────────────────────────────────────────────────
if [[ "${NO_PSEUDOBULK:-0}" != "1" ]]; then
    echo ""
    echo "========================================"
    echo "Re-running pseudobulk with corrected cell_class"
    echo "========================================"

    # Parse output_dir and pseudobulk.output_dir from the config
    _DIRS=$(singularity exec \
        --pwd "${WORK_DIR}" \
        --bind "${DATA_DIR}:${DATA_DIR}" \
        --bind "${WORK_DIR}:${WORK_DIR}" \
        "${SIF}" \
        micromamba run -n scvi-scgen-scmomat-unitvelo \
        python3 -c "
import yaml, os
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
od = cfg['output_dir']
pb = cfg.get('pseudobulk', {}).get('output_dir') or os.path.join(od, 'pseudobulk_output')
print(od + '|' + pb)
")

    _PIPELINE_OUTPUT_DIR="${_DIRS%%|*}"
    _PB_OUTPUT_DIR="${_DIRS##*|}"
    _INTEGRATED="${_PIPELINE_OUTPUT_DIR}/scvi_output/integrated.h5ad"

    echo "  Pipeline output dir: ${_PIPELINE_OUTPUT_DIR}"
    echo "  Pseudobulk output:   ${_PB_OUTPUT_DIR}"
    echo "  integrated.h5ad:     ${_INTEGRATED}"

    singularity exec \
        --pwd "${WORK_DIR}" \
        --bind "${DATA_DIR}:${DATA_DIR}" \
        --bind "${WORK_DIR}:${WORK_DIR}" \
        "${SIF}" \
        micromamba run -n scvi-scgen-scmomat-unitvelo \
        env PYTHONPATH="code" python3 -m pipeline.pseudobulk \
            --input  "${_INTEGRATED}" \
            --output "${_PB_OUTPUT_DIR}" \
            --config "${CONFIG}" \
            --overwrite

    echo "Pseudobulk complete: $(date)"
fi

_ELAPSED=$(( $(date +%s) - _JOB_START ))
echo "========================================"
echo "Resource usage:"
echo "  Time: $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s"
echo "========================================"
echo "All steps complete: $(date)"
