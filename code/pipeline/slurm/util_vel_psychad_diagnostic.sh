#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_vel_psychad_diag.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_vel_psychad_diag.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=icelake
#SBATCH --mem=16G
#SBATCH --account=vertes-sl2-cpu

# Cross-assignment diagnostic for VelPsychAD_psychad_labels_age5.
# Reads only obs metadata (h5py), so 16G is ample.
# CONFIG must point to the psychad_labels config so we can derive the output_dir.

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_scvi.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIG="${CONFIG:-code/pipeline/configs/VelPsychAD_psychad_labels_age5.yaml}"

mkdir -p "${WORK_DIR}/logs"

OUTPUT_DIR=$(awk '/^output_dir:/{print $2; exit}' "${WORK_DIR}/${CONFIG}")
H5AD="${OUTPUT_DIR}/scvi_output/integrated.h5ad"
OUT_DIR="${OUTPUT_DIR}/cross_assignment_diagnostic"

echo "========================================"
echo "Vel-PsychAD cross-assignment diagnostic"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "H5AD:    ${H5AD}"
echo "OutDir:  ${OUT_DIR}"
echo "Start:   $(date)"
echo "========================================"

singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m pipeline.diagnostics.vel_psychad_cross_assignment \
        --h5ad "${H5AD}" \
        --out_dir "${OUT_DIR}"

echo "========================================"
echo "Diagnostic complete: $(date)"
