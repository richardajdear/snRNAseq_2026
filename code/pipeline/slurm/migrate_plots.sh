#!/bin/bash
# One-off migration: move scvi_output/plots/ → plots/all/ for all integrated runs.
# Safe to re-run: skips any run that doesn't have scvi_output/plots/.
set -euo pipefail

INTEGRATED="${1:-/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated}"

echo "Migrating plots in: ${INTEGRATED}"
migrated=0
for run_dir in "${INTEGRATED}"/*/; do
    src="${run_dir}scvi_output/plots"
    dst="${run_dir}plots/all"
    if [[ -d "${src}" ]]; then
        mkdir -p "${dst}"
        mv "${src}"/* "${dst}/"
        rmdir "${src}"
        echo "  Migrated: $(basename ${run_dir})"
        (( migrated++ )) || true
    fi
done
echo "Done. Migrated ${migrated} run(s)."
