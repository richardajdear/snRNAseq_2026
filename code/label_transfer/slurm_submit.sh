#!/bin/bash
#SBATCH --job-name=label_transfer
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/label_transfer_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/label_transfer_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=icelake
#SBATCH --account=VERTES-SL2-CPU

# No GPU required — kNN is CPU-only.
# For the 100k dataset (~220k cells) this completes in <10 min.
# For larger datasets, increase --mem and --time as needed.

set -euo pipefail

WORK_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake_scvi.sif"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

INPUT="${1:-${DATA_DIR}/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld/scvi_output/integrated.h5ad}"
OUTPUT_DIR="${2:-${DATA_DIR}/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld/label_transfer}"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Input:   ${INPUT}"
echo "Output:  ${OUTPUT_DIR}"
echo "Start:   $(date)"
echo "========================================"

RUN="singularity exec \
    --pwd ${WORK_DIR} \
    --bind ${DATA_DIR}:${DATA_DIR} \
    --bind ${WORK_DIR}:${WORK_DIR} \
    ${SIF} \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH=code"

echo "── Label transfer ──"
${RUN} python3 -m label_transfer.run_transfer \
    --input "${INPUT}" \
    --output_dir "${OUTPUT_DIR}" \
    --reference_sources WANG

echo ""
echo "── Diagnostics ──"
${RUN} python3 -m label_transfer.diagnostics \
    --all_labels "${OUTPUT_DIR}/all_cell_labels.csv" \
    --transfer   "${OUTPUT_DIR}/transferred_labels.csv" \
    --input      "${INPUT}" \
    --output_dir "${OUTPUT_DIR}/diagnostics"

echo ""
echo "Finished: $(date)"
