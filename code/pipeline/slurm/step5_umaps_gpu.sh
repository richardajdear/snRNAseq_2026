#!/bin/bash
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step5_umaps_gpu.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/%j_step5_umaps_gpu.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --mem=212G
#SBATCH --account=vertes-sl2-gpu

# GPU-accelerated UMAP computation for integrated.h5ad subsets.
#
# Reads the diagnostic_umaps section from CONFIG and produces per-subset
# UMAP grids in <output_dir>/plots/<subset_name>/.  One subset is loaded
# at a time; GPU memory is freed between subsets.
#
# Requires:
#   - step2 (scVI+scANVI) complete; integrated.h5ad must exist
#   - shortcake_full.sif with shortcake_rapidsc environment
#
# Reads:  <output_dir>/scvi_output/integrated.h5ad
# Writes: <output_dir>/plots/
#
# For CPU-only scANVI diagnostics (scanvi_diagnostics/) run step6_diagnostics.sh.
#
# Usage:
#   sbatch --export=ALL,CONFIG=code/pipeline/configs/Vel_prepost_noage_tuning5.yaml \
#          code/pipeline/slurm/step5_umaps_gpu.sh

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/rajd2/rds/hpc-work/snRNAseq_2026}"
SIF="${SIF:-/home/rajd2/rds/hpc-work/shortcake_full.sif}"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
CONFIG="${CONFIG:-code/pipeline/configs/source_hpc_config.yaml}"

mkdir -p "${WORK_DIR}/logs"

echo "========================================"
echo "STEP 5: GPU UMAP computation"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "Config:    ${CONFIG}"
echo "Start:     $(date)"
echo "========================================"
_JOB_START=$(date +%s)

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

singularity exec --nv \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n shortcake_rapidsc \
    env PYTHONPATH="code" python3 -m pipeline.run_pipeline \
        --config "${CONFIG}" \
        --steps gpu_umaps

echo "GPU usage after job:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null || true

_ELAPSED=$(( $(date +%s) - _JOB_START ))
_TIME_LIMIT=$(squeue -j "${SLURM_JOB_ID}" -h -o "%l" 2>/dev/null || echo "N/A")
_MAX_RSS="N/A"
if _tmp=$(sstat --jobs="${SLURM_JOB_ID}.batch" --format=MaxRSS --noheader 2>/dev/null); then
    _MAX_RSS=$(echo "$_tmp" | awk 'NR==1{print $1}')
fi
_ALLOC_MEM_GB=$(( ${SLURM_MEM_PER_NODE:-0} / 1024 ))
echo "========================================"
echo "Resource usage:"
echo "  Time:   $(( _ELAPSED/3600 ))h $(( (_ELAPSED%3600)/60 ))m $(( _ELAPSED%60 ))s  /  ${_TIME_LIMIT} allocated"
echo "  Memory: ${_MAX_RSS} peak RSS  /  ${_ALLOC_MEM_GB}G allocated"
echo "========================================"
echo "Step 5 GPU UMAPs complete: $(date)"
