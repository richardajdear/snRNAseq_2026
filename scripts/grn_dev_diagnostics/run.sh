#!/bin/bash
# Thin wrapper to run a python script inside the shortcake_default env
# without going through SLURM. Usage:
#   bash scripts/grn_dev_diagnostics/run.sh scripts/grn_dev_diagnostics/foo.py [args...]
set -euo pipefail
REPO_ROOT="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake.sif"
ENV="shortcake_default"
PY="/opt/micromamba/envs/${ENV}/bin/python3"

SCRIPT="$1"; shift
if [[ "$SCRIPT" != /* ]]; then
    SCRIPT="${REPO_ROOT}/${SCRIPT}"
fi

singularity exec \
    --pwd "$REPO_ROOT" \
    --env "PYTHONUNBUFFERED=1" \
    "$SIF" \
    micromamba run -n "$ENV" "$PY" -u "$SCRIPT" "$@"
