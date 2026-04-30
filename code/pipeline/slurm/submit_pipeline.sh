#!/bin/bash
# submit_pipeline.sh — Submit the snRNAseq pipeline as chained SLURM jobs.
#
# Steps (in order):
#   1 (CPU)  Downsample + combine  →  per_dataset/*.h5ad, combined.h5ad
#   2 (GPU)  scVI + scANVI         →  scvi_output/integrated.h5ad + plots
#   3 (CPU)  Diagnostics           →  scanvi_diagnostics/ (label-transfer QC)
#   4 (CPU)  Pseudobulk            →  pseudobulk_output/*.h5ad
#   5 (CPU)  Notebook              →  notebooks/results/<config>/
#
# Which steps are submitted is controlled by the 'steps:' key in the config
# YAML. Only the listed steps are submitted; the dependency chain is built
# dynamically across the submitted subset.  If 'steps:' is absent, the
# default is: downsample, combine, scvi, diagnostics, pseudobulk.
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

# Parse 'steps:' from the config YAML; fall back to the full default pipeline.
STEPS=$(python3 -c "
import yaml
with open('${WORK_DIR}/${CONFIG}') as f:
    cfg = yaml.safe_load(f)
steps = cfg.get('steps', ['downsample', 'combine', 'scvi', 'diagnostics', 'pseudobulk'])
print(' '.join(steps))
")
echo "Steps: ${STEPS}"
echo ""

has_step() { [[ " ${STEPS} " == *" $1 "* ]]; }

PREV_JID=""
CHAIN=""

_submit() {
    local label="$1"
    local script="$2"
    local dep_flag=""
    if [[ -n "${PREV_JID}" ]]; then
        dep_flag="--dependency=afterok:${PREV_JID}"
    fi
    local jid
    jid=$(sbatch --parsable \
        ${dep_flag} \
        --job-name="${script%.sh}" \
        --export=ALL,WORK_DIR="${WORK_DIR}",CONFIG="${CONFIG}" \
        "${WORK_DIR}/code/pipeline/slurm/${script}")
    local dep_info=""
    [[ -n "${PREV_JID}" ]] && dep_info="  [depends on ${PREV_JID}]"
    echo "${label} submitted: job ${jid}${dep_info}"
    PREV_JID="${jid}"
    CHAIN="${CHAIN:+${CHAIN} → }${jid}"
}

# Step 1: Downsample + Combine (CPU)
if has_step downsample || has_step combine; then
    _submit "Step 1 (downsample+combine)" "step1_downsample_combine.sh"
fi

# Step 2: scVI + scANVI (GPU)
if has_step scvi; then
    _submit "Step 2 (scVI+scANVI)       " "step2_scvi.sh"
fi

# Step 3: Diagnostics (CPU)
if has_step diagnostics; then
    _submit "Step 3 (diagnostics)       " "step3_diagnostics.sh"
fi

# Step 4: Pseudobulk (CPU)
if has_step pseudobulk; then
    _submit "Step 4 (pseudobulk)        " "step4_pseudobulk.sh"
fi

# Step 5: Notebook render (CPU)
if has_step notebook; then
    _submit "Step 5 (notebook)          " "step5_notebook.sh"
fi

if [[ -z "${CHAIN}" ]]; then
    echo "WARNING: no recognised steps found in config (got: ${STEPS})" >&2
    echo "Known steps: downsample, combine, scvi, diagnostics, pseudobulk, notebook" >&2
    exit 1
fi

echo ""
echo "Pipeline chain: ${CHAIN}"
echo "Monitor with: squeue -u \$(whoami)"
