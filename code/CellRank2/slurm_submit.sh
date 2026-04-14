#!/bin/bash
#SBATCH --job-name=cellrank2
#SBATCH --output=logs/cellrank2_%j.log
#SBATCH --error=logs/cellrank2_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -p cclake-himem,icelake-himem

set -euo pipefail

CONFIG="${1:-code/CellRank2/hpc_config.yaml}"
CONDA_BASE="${CONDA_BASE:-${HOME}/mambaforge}"
MAMBA_ENV="${MAMBA_ENV:-cellrank2}"
WORKDIR="${CELLRANK_WORKDIR:-$(pwd)}"
NCPUS="${SLURM_CPUS_PER_TASK:-8}"

echo "========================================"
echo "CellRank 2 Pipeline"
echo "Config:  ${CONFIG}"
echo "Workdir: ${WORKDIR}"
echo "Env:     ${MAMBA_ENV}"
echo "CPUs:    ${NCPUS}"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "========================================"

cd "${WORKDIR}"

module purge 2>/dev/null || true

# Activate mamba/conda environment (provides petsc4py/slepc4py for krylov solver)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${MAMBA_ENV}"

mkdir -p logs

# Avoid OpenMP conflict between conda-forge llvm-openmp (petsc/slepc) and PyTorch bundled libomp
export KMP_DUPLICATE_LIB_OK=TRUE

# Run with MPI so SLEPc can use all available CPUs for the Schur decomposition
PYTHONPATH=code mpirun -n "${NCPUS}" python -m CellRank2.run_pipeline --config "${CONFIG}"

echo "Done."
