#!/bin/bash
# Render all notebook experiments (or a filtered subset) as separate SLURM jobs.
#
# Usage:
#   bash render_all.sh [pattern] [--force] [--local]
#
# [pattern] is an optional glob to restrict which configs are rendered, e.g.:
#   bash render_all.sh                                # render everything
#   bash render_all.sh 'sensitivity_*'               # all sensitivity notebooks
#   bash render_all.sh '*scANVI*'                     # all scANVI variants
#
# --force  deletes the CACHE_DIR for each config before rendering, forcing a
#          full re-run of the projection pipeline (overwrites the cache).
#
# --local  run each render directly in the current shell instead of submitting
#          to SLURM (see render_single.sh --local for details).
#   bash render_all.sh '*scANVI*' --force --local    # rerun scANVI interactively
#
# Without --local on HPC, each config is submitted as a separate SLURM job.
# With --local (or on a local workstation), configs are rendered sequentially.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATES_DIR="${REPO_ROOT}/notebooks/templates"

PATTERN="*"
FORCE=""
LOCAL=""
for _arg in "$@"; do
    case "$_arg" in
        --force) FORCE="--force" ;;
        --local) LOCAL="--local" ;;
        *)       PATTERN="$_arg" ;;
    esac
done

mapfile -t CONFIGS < <(ls "${TEMPLATES_DIR}"/*/configs/${PATTERN}.yaml 2>/dev/null || true)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No configs found matching: ${TEMPLATES_DIR}/*/configs/${PATTERN}.yaml" >&2
    exit 1
fi

echo "Submitting ${#CONFIGS[@]} render job(s)..."
for config_file in "${CONFIGS[@]}"; do
    name=$(basename "$config_file" .yaml)
    output=$(bash "${REPO_ROOT}/notebooks/render_single.sh" "$name" "" ${FORCE:+"$FORCE"} ${LOCAL:+"$LOCAL"})
    echo "  ${name}  → ${output}"
done
