#!/usr/bin/env bash
# Syncs integrated pipeline outputs between HPC and local machine.
# Run from local Mac. Pulls HPC→local by default.
#
# Usage: ./sync_data.sh [--push] [--plots] [--dry-run]
#   --push     Push local→HPC (default: pull HPC→local)
#   --plots    Include scanvi_diagnostics/ plot directories
#   --dry-run  Show what would be transferred without doing it

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
HPC_USER="rajd2"
HPC_HOST="login.hpc.cam.ac.uk"
HPC_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated"
LOCAL_DIR="/Users/richard/Git/snRNAseq_2026/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated"
# ──────────────────────────────────────────────────────────────────────────────

PUSH=false
PLOTS=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --push)    PUSH=true ;;
        --plots)   PLOTS=true ;;
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown option: $arg"; echo "Usage: $0 [--push] [--plots] [--dry-run]"; exit 1 ;;
    esac
done

# Build rsync filter rules (order matters — first match wins)
FILTERS=(
    # Always exclude these directories
    "--exclude=pipeline_test/"
    "--exclude=compare_outputs/"
    "--exclude=scvi_output/"
)

if ! $PLOTS; then
    FILTERS+=("--exclude=scanvi_diagnostics/")
fi

FILTERS+=(
    # Include config and log files
    "--include=*.yaml"
    "--include=*.log"

    # Include only the specific pseudobulk file (by name, anywhere)
    "--include=by_cell_class.h5ad"

    # Exclude all other h5ad files
    "--exclude=*.h5ad"

    # Recurse into all remaining directories
    "--include=*/"

    # Exclude everything else
    "--exclude=*"
)

RSYNC_OPTS=(-avz --progress "${FILTERS[@]}")

if $DRY_RUN; then
    RSYNC_OPTS+=(--dry-run)
fi

HPC_PATH="${HPC_USER}@${HPC_HOST}:${HPC_DIR}/"

if $PUSH; then
    echo "Pushing: local → HPC"
    $DRY_RUN && echo "(dry run)"
    rsync "${RSYNC_OPTS[@]}" "${LOCAL_DIR}/" "${HPC_PATH}"
else
    echo "Pulling: HPC → local"
    $DRY_RUN && echo "(dry run)"
    mkdir -p "${LOCAL_DIR}"
    rsync "${RSYNC_OPTS[@]}" "${HPC_PATH}" "${LOCAL_DIR}/"
fi
