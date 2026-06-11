#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step5_test_gpu_umaps.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step5_test_gpu_umaps.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=vertes-sl2-gpu

# Quick smoke test for the GPU UMAP step on a 5 k-cell subsample.
#
# Hard-coded to Vel_prepost_noage_tuning5.  Output goes to a _test_gpu_umaps/
# subdirectory alongside the production plots/ dir so production outputs are
# never overwritten.
#
# Usage:
#   sbatch code/pipeline/slurm/step5_test_gpu_umaps.sh

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_full.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIG="code/pipeline/configs/Vel_prepost_noage_tuning5.yaml"

# Resolve output_dir from config
OUTPUT_DIR=$(awk '/^output_dir:/{print $2; exit}' "${WORK_DIR}/${CONFIG}")
INTEGRATED_H5AD="${OUTPUT_DIR}/scvi_output/integrated.h5ad"
TEST_OUT="${OUTPUT_DIR}/_test_gpu_umaps"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "TEST: GPU UMAP smoke test (5k cells)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "Input:     ${INTEGRATED_H5AD}"
echo "Output:    ${TEST_OUT}"
echo "Start:     $(date)"
echo "========================================"

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

singularity exec --nv \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n shortcake_rapidsc \
    env PYTHONPATH="code" python3 -m pipeline.gpu_umaps \
        --config "${CONFIG}" \
        --input  "${INTEGRATED_H5AD}" \
        --output_dir "${TEST_OUT}" \
        --n_cells 5000

echo "GPU usage after job:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

echo "========================================"
echo "Output files:"
find "${TEST_OUT}" -name "*.png" | sort
echo "========================================"
echo "Test complete: $(date)"
