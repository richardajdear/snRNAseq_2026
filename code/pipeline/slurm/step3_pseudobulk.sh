#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step3_pseudobulk.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step3_pseudobulk.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

# ── Environment ────────────────────────────────────────────────────────────────
# Required env vars (must be set before sbatch):
#   CONFIG          path to pipeline YAML config (relative to WORK_DIR)
#
# Optional env vars:
#   MANUAL_ANNOTATOR  set to 1 to run marker-based cell annotation before
#                     pseudobulk. Annotation runs on the raw source h5ad files
#                     listed in CONFIG's 'sources' section; results are saved to
#                     pseudobulk_output/manual_annotations.parquet and then
#                     passed to pseudobulk as --annotation-file.
#                     Requires annotation_by_markers.py (code/) and the
#                     by_cell_class_manual group defined in CONFIG.
#   WORK_DIR          project root (default: /home/rajd2/rds/hpc-work/snRNAseq_2026)
#   SIF               Singularity image (default: shortcake_scvi.sif)
#
# Example submissions:
#   # Normal pseudobulk (existing cell_class column):
#   CONFIG=code/pipeline/configs/PsychAD_noage_tuning5.yaml \
#     sbatch code/pipeline/slurm/step3_pseudobulk.sh
#
#   # With marker-based manual annotation:
#   CONFIG=code/pipeline/configs/PsychAD_noage_tuning5.yaml \
#   MANUAL_ANNOTATOR=1 \
#     sbatch code/pipeline/slurm/step3_pseudobulk.sh

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIG="${CONFIG:-code/pipeline/configs/source_hpc_config.yaml}"
MANUAL_ANNOTATOR="${MANUAL_ANNOTATOR:-0}"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "STEP 3: Pseudobulk aggregation"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Node:           $(hostname)"
echo "Config:         ${CONFIG}"
echo "ManualAnnotator:${MANUAL_ANNOTATOR}"
echo "Start:          $(date)"
echo "========================================"
_JOB_START=$(date +%s)

# ── Singularity wrapper ────────────────────────────────────────────────────────
_SING="singularity exec
    --pwd ${WORK_DIR}
    --bind ${DATA_DIR}:${DATA_DIR}
    --bind ${WORK_DIR}:${WORK_DIR}
    ${SIF}
    micromamba run -n scvi-scgen-scmomat-unitvelo"

_PY="env PYTHONPATH=code python3"

# ── Derive output directory from config ───────────────────────────────────────
echo "Resolving output_dir from config ..."
OUTPUT_DIR=$(${_SING} ${_PY} -c \
    "import yaml; print(yaml.safe_load(open('${CONFIG}'))['output_dir'])")
echo "  output_dir: ${OUTPUT_DIR}"

INTEGRATED="${OUTPUT_DIR}/scvi_output/integrated.h5ad"
PB_OUTPUT="${OUTPUT_DIR}/pseudobulk_output"

# Use the pipeline_config.yaml copy in output_dir (written by run_pipeline.py).
# Fall back to the original CONFIG if the copy doesn't exist yet.
if [[ -f "${OUTPUT_DIR}/pipeline_config.yaml" ]]; then
    PIPELINE_CFG="${OUTPUT_DIR}/pipeline_config.yaml"
else
    PIPELINE_CFG="${CONFIG}"
fi
echo "  pipeline_cfg: ${PIPELINE_CFG}"

# ── Validate integrated.h5ad exists ───────────────────────────────────────────
if [[ ! -f "${INTEGRATED}" ]]; then
    echo "ERROR: integrated.h5ad not found: ${INTEGRATED}" >&2
    echo "       Run --steps scvi first." >&2
    exit 1
fi

# ── Main logic ────────────────────────────────────────────────────────────────
if [[ "${MANUAL_ANNOTATOR}" == "1" ]]; then

    ANNOT_FILE="${PB_OUTPUT}/manual_annotations.parquet"
    mkdir -p "${PB_OUTPUT}"

    # Step 3a — marker-based annotation on source files
    # Skip if parquet already exists (allows re-running pseudobulk without
    # re-annotating ~2M cells).
    if [[ -f "${ANNOT_FILE}" ]]; then
        echo ""
        echo "Step 3a: Annotation parquet already exists, skipping:"
        echo "  ${ANNOT_FILE}"
    else
        echo ""
        echo "Step 3a: Running marker-based cell annotation ..."
        echo "  Source paths read from: ${CONFIG}"
        echo "  Output:                 ${ANNOT_FILE}"

        ${_SING} ${_PY} code/annotation_by_markers.py \
            --config  "${CONFIG}" \
            --no-age-filter \
            --save    "${ANNOT_FILE}"

        echo "  Annotation complete: $(date)"
    fi

    # Step 3b — pseudobulk with manual annotation
    echo ""
    echo "Step 3b: Running pseudobulk with manual annotations ..."

    ${_SING} ${_PY} -m pipeline.pseudobulk \
        --input           "${INTEGRATED}" \
        --output          "${PB_OUTPUT}" \
        --config          "${PIPELINE_CFG}" \
        --annotation-file "${ANNOT_FILE}"

else

    # Normal pseudobulk via run_pipeline.py orchestrator
    echo ""
    echo "Running pseudobulk (standard) ..."

    ${_SING} ${_PY} -m pipeline.run_pipeline \
        --config "${CONFIG}" \
        --steps pseudobulk

fi

# ── Resource usage ─────────────────────────────────────────────────────────────
_ELAPSED=$(( $(date +%s) - _JOB_START ))
_TIME_LIMIT=$(squeue -j "${SLURM_JOB_ID}" -h -o "%l" 2>/dev/null || echo "N/A")
_ALLOC_MEM_GB=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
_rss_gb="N/A"; _rss_pct="N/A"
if _tmp=$(sstat --jobs="${SLURM_JOB_ID}.batch" --format=MaxRSS --noheader 2>/dev/null); then
    _rss_kb=$(echo "$_tmp" | awk 'NR==1{gsub(/[^0-9]/,"",$1); print $1+0}')
    if [[ -n "$_rss_kb" && "$_rss_kb" -gt 0 ]]; then
        _rss_gb=$(awk  "BEGIN{printf \"%.2f\", ${_rss_kb}/1048576}")
        [[ ${_ALLOC_MEM_GB} -gt 0 ]] && \
            _rss_pct=$(awk "BEGIN{printf \"%.0f\", (${_rss_kb}/1048576)/${_ALLOC_MEM_GB}*100}")
    fi
fi
echo ""
echo "========================================"
echo "Resource usage:"
echo "  Time:   $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s  /  ${_TIME_LIMIT} allocated"
echo "  Memory: ${_rss_gb}G peak RSS  /  ${_ALLOC_MEM_GB}G allocated (${_rss_pct}%)"
echo "========================================"
echo "Step 3 complete: $(date)"
