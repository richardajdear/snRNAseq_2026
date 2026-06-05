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
# Configs live under notebooks/templates/<template>/configs/<config_name>.yaml.
# The template is the parent directory of the config's configs/ folder, so no
# first-line comment is needed. Override with a second argument (pass "" to use
# the inferred template).
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

REPO_ROOT="${RENDER_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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
    echo "Available configs (grouped by template):"
    for _td in "${TEMPLATES_DIR}"/*/configs; do
        [[ -d "$_td" ]] || continue
        _tpl=$(basename "$(dirname "$_td")")
        for _cf in "$_td"/*.yaml; do
            [[ -f "$_cf" ]] || continue
            printf "  %s  (%s)\n" "$(basename "$_cf" .yaml)" "$_tpl"
        done
    done
    exit 1
fi

# Locate the config under templates/*/configs/<name>.yaml.  Error on no match
# or on collisions (same config name under multiple templates).
_MATCHES=()
for _c in "${TEMPLATES_DIR}"/*/configs/"${CONFIG_NAME}.yaml"; do
    [[ -f "$_c" ]] && _MATCHES+=("$_c")
done
if [[ ${#_MATCHES[@]} -eq 0 ]]; then
    echo "Error: config not found: ${CONFIG_NAME}.yaml" >&2
    echo "  Searched: ${TEMPLATES_DIR}/*/configs/${CONFIG_NAME}.yaml" >&2
    exit 1
elif [[ ${#_MATCHES[@]} -gt 1 ]]; then
    echo "Error: config name '${CONFIG_NAME}' is ambiguous; found in multiple templates:" >&2
    printf '  %s\n' "${_MATCHES[@]}" >&2
    echo "  Rename one of them to make it unique." >&2
    exit 1
fi
CONFIG_FILE="${_MATCHES[0]}"

# If not already inside a SLURM job and --local was not requested, submit via
# sbatch (HPC default).  On a local workstation with no sbatch this block is
# skipped and execution falls through directly.
if [[ -z "${SLURM_JOB_ID:-}" && -z "$LOCAL" ]]; then
    if command -v sbatch >/dev/null 2>&1; then
        mkdir -p "$LOGS_DIR"
        export RENDER_REPO_ROOT="${REPO_ROOT}"
        exec sbatch \
            --job-name="render_${CONFIG_NAME}" \
            --output="${LOGS_DIR}/%j_render_${CONFIG_NAME}.out" \
            --error="${LOGS_DIR}/%j_render_${CONFIG_NAME}.err" \
            --time=00:30:00 \
            --mem=32G \
            --partition=icelake \
            --account=vertes-sl2-cpu \
            "$(realpath "${BASH_SOURCE[0]}")" "$@"
    fi
    # No sbatch available (local workstation): fall through and run directly.
fi

# Template is the parent of the config's configs/ folder
# (notebooks/templates/<template>/configs/<config>.yaml).
_INFERRED_TEMPLATE=$(basename "$(dirname "$(dirname "$CONFIG_FILE")")")
if [[ -n "$TEMPLATE_OVERRIDE" ]]; then
    if [[ "$TEMPLATE_OVERRIDE" != "$_INFERRED_TEMPLATE" ]]; then
        echo "Warning: template override '${TEMPLATE_OVERRIDE}' disagrees with config location" >&2
        echo "  (config is under template '${_INFERRED_TEMPLATE}'). Using override." >&2
    fi
    TEMPLATE_NAME="$TEMPLATE_OVERRIDE"
else
    TEMPLATE_NAME="$_INFERRED_TEMPLATE"
fi

TEMPLATE="${TEMPLATES_DIR}/${TEMPLATE_NAME}/${TEMPLATE_NAME}.qmd"
if [[ ! -f "$TEMPLATE" ]]; then
    echo "Error: template not found: ${TEMPLATE}" >&2
    echo "  Available templates:"
    ls -d "${TEMPLATES_DIR}"/*/ 2>/dev/null | xargs -n1 basename || echo "  (none)"
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
