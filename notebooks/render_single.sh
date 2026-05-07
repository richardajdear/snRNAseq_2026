#!/bin/bash
# See README.md at the repo root for full documentation and usage guidelines.
# Render a single notebook experiment by config name.
#
# Usage:
#   bash render_single.sh <config_name> [template_override] [--force] [--local]
#
# <config_name> is the YAML filename without extension, e.g.:
#   sensitivity_chemistry_scANVI
#   pseudobulk_gap_excitatory_postnatal
#
# The template is inferred from a comment on the first line of the config YAML:
#   # Template: <template_name>
# where <template_name> matches a file in notebooks/templates/ (without .qmd).
# Override with a second argument (pass "" to use the inferred template).
#
# --force  deletes the CACHE_DIR from the config before rendering, forcing a
#          full re-run of the projection pipeline (overwrites the cache).
#
# --local  run directly in the current shell instead of submitting to SLURM.
#          On the HPC login node this is useful for interactive tests
#          (Singularity is still used, so the environment matches production).
#          On a local workstation (no sbatch/singularity) this flag is optional —
#          local execution is detected automatically.
#
# Execution modes:
#   HPC default   bash render_single.sh <config>           → auto-submits via sbatch
#   HPC direct    bash render_single.sh <config> --local   → runs in current shell
#   MacBook       bash render_single.sh <config>           → auto-detects, runs directly
#
# Output goes to:  notebooks/results/<config_name>/
# Logs go to:      logs/render_<config_name>_%j.{out,err}  (sbatch mode only)
#
# Examples:
#   bash render_single.sh sensitivity_chemistry_scANVI
#   bash render_single.sh sensitivity_chemistry_scANVI "" --force
#   bash render_single.sh sensitivity_chemistry_scANVI --local

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIGS_DIR="${REPO_ROOT}/notebooks/configs"
TEMPLATES_DIR="${REPO_ROOT}/notebooks/templates"
RESULTS_DIR="${REPO_ROOT}/notebooks/results"
LOGS_DIR="${REPO_ROOT}/logs"

CONFIG_NAME=""
TEMPLATE_OVERRIDE=""
FORCE=""
LOCAL=""
_pos=0
for _arg in "$@"; do
    case "$_arg" in
        --force) FORCE="--force" ;;
        --local) LOCAL="--local" ;;
        *)
            (( ++_pos ))
            case $_pos in
                1) CONFIG_NAME="$_arg" ;;
                2) TEMPLATE_OVERRIDE="$_arg" ;;
            esac
            ;;
    esac
done

if [[ -z "$CONFIG_NAME" ]]; then
    echo "Usage: $0 <config_name> [template_override] [--force] [--local]" >&2
    echo ""
    echo "Available configs:"
    ls "${CONFIGS_DIR}"/*.yaml 2>/dev/null \
        | xargs -n1 basename | sed 's/\.yaml$//'
    exit 1
fi

CONFIG_FILE="${CONFIGS_DIR}/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# If not already inside a SLURM job and --local was not requested, submit via
# sbatch (HPC default).  On a local workstation with no sbatch this block is
# skipped and execution falls through directly.
if [[ -z "${SLURM_JOB_ID:-}" && -z "$LOCAL" ]]; then
    if command -v sbatch >/dev/null 2>&1; then
        mkdir -p "$LOGS_DIR"
        exec sbatch \
            --job-name="render_${CONFIG_NAME}" \
            --output="${LOGS_DIR}/%j_render_${CONFIG_NAME}.out" \
            --error="${LOGS_DIR}/%j_render_${CONFIG_NAME}.err" \
            --time=00:10:00 \
            --mem=32G \
            --partition=icelake \
            "$0" "$@"
    fi
    # No sbatch available (local workstation): fall through and run directly.
fi

# Infer template from first-line comment "# Template: <name>"
if [[ -n "$TEMPLATE_OVERRIDE" ]]; then
    TEMPLATE_NAME="$TEMPLATE_OVERRIDE"
else
    TEMPLATE_NAME=$(head -1 "$CONFIG_FILE" | sed -n 's/^# Template: *//p')
    if [[ -z "$TEMPLATE_NAME" ]]; then
        echo "Error: no template found in ${CONFIG_FILE}." >&2
        echo "  Add a first-line comment: # Template: <template_name>" >&2
        echo "  Or pass a template as second argument." >&2
        exit 1
    fi
fi

TEMPLATE="${TEMPLATES_DIR}/${TEMPLATE_NAME}.qmd"
if [[ ! -f "$TEMPLATE" ]]; then
    echo "Error: template not found: ${TEMPLATE}" >&2
    echo "  Available templates:"
    ls "${TEMPLATES_DIR}"/*.qmd 2>/dev/null | xargs -n1 basename || echo "  (none)"
    exit 1
fi

OUTPUT_DIR="${RESULTS_DIR}/${CONFIG_NAME}"
mkdir -p "$OUTPUT_DIR"

# --force: delete the application-level CACHE_DIR so the projection pipeline re-runs
if [[ "$FORCE" == "--force" ]]; then
    CACHE_DIR_VAL=$(grep '^CACHE_DIR:' "$CONFIG_FILE" | awk '{print $2}')
    if [[ -n "$CACHE_DIR_VAL" && -d "$CACHE_DIR_VAL" ]]; then
        echo "Removing cache: $CACHE_DIR_VAL"
        rm -rf "$CACHE_DIR_VAL"
    fi
fi

PARAMS_SNAPSHOT="${OUTPUT_DIR}/${TEMPLATE_NAME}_params.yaml"
cp "$CONFIG_FILE" "$PARAMS_SNAPSHOT"

echo "========================================"
echo "Config:    ${CONFIG_NAME}"
echo "Template:  ${TEMPLATE_NAME}"
echo "Params:    ${PARAMS_SNAPSHOT}"
echo "Output:    ${OUTPUT_DIR}"
[[ "$FORCE" == "--force" ]] && echo "Mode:      FORCE (cache cleared)"
echo "========================================"

bash "${REPO_ROOT}/notebooks/render_notebook.sh" \
    "$TEMPLATE" \
    "$PARAMS_SNAPSHOT" \
    "$OUTPUT_DIR" \
    "$FORCE"
