#!/bin/bash
# Render all notebook experiments (or a filtered subset) as separate SLURM jobs.
#
# Usage:
#   bash render_all.sh [pattern]
#
# [pattern] is an optional glob to restrict which configs are rendered, e.g.:
#   bash render_all.sh                          # render everything
#   bash render_all.sh 'sensitivity_*'          # all sensitivity notebooks
#   bash render_all.sh '*scANVI*'               # all scANVI variants
#
# Each config in notebooks/configs/ that matches the pattern is submitted as a
# separate SLURM job via render_single.sh.

set -euo pipefail

PATTERN="${1:-*}"
REPO_ROOT="/home/rajd2/rds/hpc-work/snRNAseq_2026"
CONFIGS_DIR="${REPO_ROOT}/notebooks/configs"

mapfile -t CONFIGS < <(ls "${CONFIGS_DIR}/${PATTERN}.yaml" 2>/dev/null || true)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No configs found matching: ${CONFIGS_DIR}/${PATTERN}.yaml" >&2
    exit 1
fi

echo "Submitting ${#CONFIGS[@]} render job(s)..."
for config_file in "${CONFIGS[@]}"; do
    name=$(basename "$config_file" .yaml)
    job_id=$(sbatch --parsable \
        --output="${REPO_ROOT}/logs/render_${name}_%j.out" \
        --error="${REPO_ROOT}/logs/render_${name}_%j.err" \
        "${REPO_ROOT}/notebooks/render_single.sh" "$name")
    echo "  Submitted ${name}  → job ${job_id}"
done
