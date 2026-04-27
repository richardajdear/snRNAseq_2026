#!/bin/bash
# See README.md at the repo root for full documentation and usage guidelines.
# Render a single notebook experiment by config name.
#
# Usage:
#   bash render_single.sh <config_name> [template_override] [--force]
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
# Always invoke with `bash`, not `sbatch` directly — the script submits itself
# to SLURM with the correct per-config log paths automatically.
#
# Output goes to:  notebooks/results/<config_name>/
# Logs go to:      logs/render_<config_name>_%j.{out,err}
#
# Examples:
#   bash render_single.sh sensitivity_chemistry_scANVI
#   bash render_single.sh sensitivity_chemistry_scANVI "" --force

#SBATCH --job-name=render_notebook
#SBATCH --time=00:10:00
#SBATCH --mem=10G
#SBATCH --partition=icelake

set -euo pipefail

CONFIG_NAME="${1:-}"
TEMPLATE_OVERRIDE="${2:-}"
FORCE="${3:-}"

if [[ -z "$CONFIG_NAME" ]]; then
    echo "Usage: $0 <config_name> [template_override]" >&2
    echo ""
    echo "Available configs:"
    ls /home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/configs/*.yaml 2>/dev/null \
        | xargs -n1 basename | sed 's/\.yaml$//'
    exit 1
fi

REPO_ROOT="/home/rajd2/rds/hpc-work/snRNAseq_2026"
CONFIGS_DIR="${REPO_ROOT}/notebooks/configs"
TEMPLATES_DIR="${REPO_ROOT}/notebooks/templates"
RESULTS_DIR="${REPO_ROOT}/notebooks/results"
LOGS_DIR="${REPO_ROOT}/logs"

CONFIG_FILE="${CONFIGS_DIR}/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# If not already inside a SLURM job, re-submit via sbatch so that --output and
# --error can be set dynamically to include the config name.  The exec replaces
# this shell process — nothing below runs until the re-submitted job starts.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p "$LOGS_DIR"
    exec sbatch \
        --job-name="render_${CONFIG_NAME}" \
        --output="${LOGS_DIR}/render_${CONFIG_NAME}_%j.out" \
        --error="${LOGS_DIR}/render_${CONFIG_NAME}_%j.err" \
        --time=02:00:00 \
        --mem=200G \
        --partition=icelake \
        "$0" "$@"
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

echo "========================================"
echo "Config:    ${CONFIG_NAME}"
echo "Template:  ${TEMPLATE_NAME}"
echo "Params:    ${CONFIG_FILE}"
echo "Output:    ${OUTPUT_DIR}"
[[ "$FORCE" == "--force" ]] && echo "Mode:      FORCE (cache cleared)"
echo "========================================"

bash "${REPO_ROOT}/notebooks/render_notebook.sh" \
    "$TEMPLATE" \
    "$CONFIG_FILE" \
    "$OUTPUT_DIR" \
    "$FORCE"
