#!/bin/bash
#SBATCH --job-name=cr2_pearson
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/cr2_pearson_%j.log
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/cr2_pearson_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

set -euo pipefail

WORK_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake_full.sif"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIG="code/CellRank2/pearson_config.yaml"

echo "========================================"
echo "CellRank 2 Pipeline — pearson (full)"
echo "Config:  ${CONFIG}"
echo "Workdir: ${WORK_DIR}"
echo "SIF:     ${SIF}"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "========================================"

mkdir -p "${WORK_DIR}/logs"

singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    --env "LD_LIBRARY_PATH=/opt/micromamba/envs/shortcake_default/lib" \
    "${SIF}" \
    micromamba run -n shortcake_default \
    env PYTHONPATH="code" python3 -u -m CellRank2.run_pipeline --config "${CONFIG}"

echo "Done."
