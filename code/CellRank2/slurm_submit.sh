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
VENV="${CELLRANK_VENV:-${HOME}/.venvs/scvi}"
WORKDIR="${CELLRANK_WORKDIR:-$(pwd)}"

echo "========================================"
echo "CellRank 2 Pipeline"
echo "Config:  ${CONFIG}"
echo "Workdir: ${WORKDIR}"
echo "Venv:    ${VENV}"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "========================================"

cd "${WORKDIR}"

module purge 2>/dev/null || true
module load python/3.11 cuda/12.1 2>/dev/null || true

source "${VENV}/bin/activate"

mkdir -p logs

PYTHONPATH=code python -m CellRank2.run_pipeline --config "${CONFIG}"

echo "Done."
