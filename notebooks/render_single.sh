#!/bin/bash
# See README.md at the repo root for full documentation and usage guidelines.
# Render a single notebook experiment by config name.
#
# Usage:
#   sbatch render_single.sh <config_name> [template_override]
#   bash   render_single.sh <config_name> [template_override]   # login-node testing
#
# <config_name> is the YAML filename without extension, e.g.:
#   sensitivity_chemistry_scANVI
#   hvg_investigation_combined_scANVI
#
# The template is inferred from a comment on the first line of the config YAML:
#   # Template: <template_name>
# where <template_name> matches a file in notebooks/templates/ (without .qmd).
# Override with a second argument, e.g.: sensitivity_gap_analysis
#
# Output goes to:  notebooks/results/<config_name>/
# Logs go to:      logs/render_<config_name>_%j.{out,err}
#
# Examples:
#   sbatch render_single.sh sensitivity_chemistry_scANVI
#   bash   render_single.sh hvg_investigation_combined_scANVI

#SBATCH --job-name=render_notebook
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --partition=icelake

set -euo pipefail

CONFIG_NAME="${1:-}"
TEMPLATE_OVERRIDE="${2:-}"

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

# Set SLURM log paths now that we know the config name
# (only used when submitted via sbatch; ignored when run with bash)
export SBATCH_OUTPUT="${LOGS_DIR}/render_${CONFIG_NAME}_%j.out"
export SBATCH_ERROR="${LOGS_DIR}/render_${CONFIG_NAME}_%j.err"

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

echo "========================================"
echo "Config:    ${CONFIG_NAME}"
echo "Template:  ${TEMPLATE_NAME}"
echo "Params:    ${CONFIG_FILE}"
echo "Output:    ${OUTPUT_DIR}"
echo "========================================"

bash "${REPO_ROOT}/notebooks/render_notebook.sh" \
    "$TEMPLATE" \
    "$CONFIG_FILE" \
    "$OUTPUT_DIR"
