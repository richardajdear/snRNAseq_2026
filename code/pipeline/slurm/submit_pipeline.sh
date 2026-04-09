#!/bin/bash
# submit_pipeline.sh — Submit the full snRNAseq pipeline as chained SLURM jobs.
#
# Steps:
#   1 (CPU)  Downsample + combine  →  per_dataset/*.h5ad, combined.h5ad
#   2 (GPU)  scVI + scANVI         →  scvi_output/integrated.h5ad
#
# Optional rerun:
#   3 (GPU)  scanvi-only rerun     →  refreshes cell_type_aligned on integrated.h5ad
#
# Usage:
#   cd /home/rajd2/rds/hpc-work/snRNAseq_2026
#   bash code/pipeline/slurm/submit_pipeline.sh [config]
#
# The config defaults to code/pipeline/hpc_config.yaml.
# Individual steps can be resubmitted independently using the step scripts.
# To run scanvi-only after step 2, submit step3_label_transfer.sh manually.

set -euo pipefail

WORK_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
CONFIG="${1:-code/pipeline/hpc_config.yaml}"

echo "========================================"
echo "Submitting snRNAseq pipeline"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "========================================"

mkdir -p "${WORK_DIR}/logs"

# Step 1: Downsample + Combine (CPU)
JID1=$(sbatch --parsable \
    --job-name=snrna_prep \
    --export=ALL,WORK_DIR="${WORK_DIR}",CONFIG="${CONFIG}" \
    "${WORK_DIR}/code/pipeline/slurm/step1_downsample_combine.sh")
echo "Step 1 (downsample+combine) submitted: job ${JID1}"

# Step 2: scVI (+ scANVI if enabled in config) — depends on step 1
JID2=$(sbatch --parsable \
    --dependency=afterok:${JID1} \
    --job-name=snrna_scvi \
    --export=ALL,WORK_DIR="${WORK_DIR}",CONFIG="${CONFIG}" \
    "${WORK_DIR}/code/pipeline/slurm/step2_scvi.sh")
echo "Step 2 (scVI+scANVI)       submitted: job ${JID2}  [depends on ${JID1}]"

# Step 3: Pseudobulk (CPU) — depends on step 2
JID3=$(sbatch --parsable \
    --dependency=afterok:${JID2} \
    --job-name=snrna_pseudobulk \
    --export=ALL,WORK_DIR="${WORK_DIR}",CONFIG="${CONFIG}" \
    "${WORK_DIR}/code/pipeline/slurm/step4_pseudobulk.sh")
echo "Step 3 (pseudobulk)        submitted: job ${JID3}  [depends on ${JID2}]"

echo ""
echo "Pipeline chain: ${JID1} → ${JID2} → ${JID3}"
echo "Optional scanvi-only rerun (after step 2):"
echo "  sbatch --dependency=afterok:${JID2} --job-name=snrna_scanvi \
    --export=ALL,WORK_DIR=${WORK_DIR},CONFIG=${CONFIG} \
    ${WORK_DIR}/code/pipeline/slurm/step3_label_transfer.sh"
echo "Monitor with: squeue -u \$(whoami)"
