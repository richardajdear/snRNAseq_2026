#!/bin/bash
# submit_pipeline.sh — Submit the full snRNAseq pipeline as chained SLURM jobs.
#
# Steps (in order):
#   1 (CPU)  Downsample + combine  →  per_dataset/*.h5ad, combined.h5ad
#   2 (GPU)  scVI + scANVI         →  scvi_output/integrated.h5ad + plots
#   3 (CPU)  Diagnostics           →  scanvi_diagnostics/ (label-transfer QC)
#   4 (CPU)  Pseudobulk            →  pseudobulk_output/*.h5ad
#   5 (CPU)  Notebook              →  notebooks/results/<config>/ (if config has notebook:)
#
# Each step is idempotent: re-submitting picks up where a partial run left off.
# The scvi step (step 2) is complete only when BOTH integrated.h5ad AND
# scanvi_model/ exist. If the scVI model exists but scANVI is missing, simply
# re-submit step2_scvi.sh (or this script) and scANVI will resume without
# retraining scVI.
#
# Usage:
#   cd /home/rajd2/rds/hpc-work/snRNAseq_2026
#   bash code/pipeline/slurm/submit_pipeline.sh [config]
#
# The config defaults to code/pipeline/configs/source_hpc_config.yaml.
# Individual steps can be resubmitted independently using the step scripts.
#
# Utility scripts (submit manually only when needed):
#   util_scanvi_rerun.sh     — force-retrain scANVI after updating label mappings
#                              (use --overwrite; not needed for simple resumption)
#   util_retransform.sh      — re-run inference with a different transform_batch
#                              then aggregate pseudobulk (unique use case)
#   util_replot.sh           — regenerate scVI plots without any retraining
#   step2_scvi_resume_infer.sh — resume after BOTH models exist but h5ad is missing

set -euo pipefail

WORK_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
CONFIG="${1:-code/pipeline/configs/source_hpc_config.yaml}"

echo "========================================"
echo "Submitting snRNAseq pipeline"
echo "Config: ${CONFIG}"
echo "Work dir: ${WORK_DIR}"
echo "========================================"

mkdir -p "${WORK_DIR}/logs"

# Step 1: Downsample + Combine (CPU)
JID1=$(sbatch --parsable \
    --job-name=step1_scvi \
    --export=ALL,WORK_DIR="${WORK_DIR}",CONFIG="${CONFIG}" \
    "${WORK_DIR}/code/pipeline/slurm/step1_downsample_combine.sh")
echo "Step 1 (downsample+combine) submitted: job ${JID1}"

# Step 2: scVI + scANVI (GPU) — depends on step 1
JID2=$(sbatch --parsable \
    --dependency=afterok:${JID1} \
    --job-name=step2_scvi \
    --export=ALL,WORK_DIR="${WORK_DIR}",CONFIG="${CONFIG}" \
    "${WORK_DIR}/code/pipeline/slurm/step2_scvi.sh")
echo "Step 2 (scVI+scANVI)        submitted: job ${JID2}  [depends on ${JID1}]"

# Step 3: Diagnostics (CPU) — depends on step 2
JID3=$(sbatch --parsable \
    --dependency=afterok:${JID2} \
    --job-name=step3_scvi \
    --export=ALL,WORK_DIR="${WORK_DIR}",CONFIG="${CONFIG}" \
    "${WORK_DIR}/code/pipeline/slurm/step3_diagnostics.sh")
echo "Step 3 (diagnostics)        submitted: job ${JID3}  [depends on ${JID2}]"

# Step 4: Pseudobulk (CPU) — depends on step 3
JID4=$(sbatch --parsable \
    --dependency=afterok:${JID3} \
    --job-name=step4_scvi \
    --export=ALL,WORK_DIR="${WORK_DIR}",CONFIG="${CONFIG}" \
    "${WORK_DIR}/code/pipeline/slurm/step4_pseudobulk.sh")
echo "Step 4 (pseudobulk)         submitted: job ${JID4}  [depends on ${JID3}]"

# Step 5: Notebook render (CPU) — depends on step 4, only if config has notebook: section
CHAIN="${JID1} → ${JID2} → ${JID3} → ${JID4}"
if grep -q '^notebook:' "${WORK_DIR}/${CONFIG}" 2>/dev/null; then
    JID5=$(sbatch --parsable \
        --dependency=afterok:${JID4} \
        --job-name=step5_scvi \
        --export=ALL,WORK_DIR="${WORK_DIR}",CONFIG="${CONFIG}" \
        "${WORK_DIR}/code/pipeline/slurm/step5_notebook.sh")
    echo "Step 5 (notebook)           submitted: job ${JID5}  [depends on ${JID4}]"
    CHAIN="${CHAIN} → ${JID5}"
fi

echo ""
echo "Pipeline chain: ${CHAIN}"
echo "Monitor with: squeue -u \$(whoami)"
