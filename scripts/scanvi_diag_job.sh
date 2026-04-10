#!/bin/bash
#SBATCH --job-name=scanvi_diag
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/scanvi_diag_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/scanvi_diag_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cclake
#SBATCH --mem=64G
#SBATCH --account=vertes-sl3-cpu

set -euo pipefail

WORK_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake_scvi.sif"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
INPUT_H5AD="${DATA_DIR}/Cam_snRNAseq/integrated/VelWangPsychAD_100k_pearson/scvi_output/integrated.h5ad"
OUT_DIR="${DATA_DIR}/Cam_snRNAseq/integrated/VelWangPsychAD_100k_pearson/scanvi_diagnostics"

echo "Job ID: ${SLURM_JOB_ID}, node: $(hostname), start: $(date)"

singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n scvi-scgen-scmomat-unitvelo \
    env PYTHONPATH="code" python3 -m pipeline.scanvi_diagnostics \
        --input "${INPUT_H5AD}" \
        --output_dir "${OUT_DIR}" \
        --confidence_threshold 0.5

echo "Done: $(date)"
