#!/bin/bash
#SBATCH --job-name=diagnose_scvi_batch
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/diagnose_scvi_batch_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/diagnose_scvi_batch_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=icelake

set -euo pipefail

echo "Starting scVI batch correction diagnosis..."
singularity exec /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default \
    python -u /home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/diagnose_scvi_batch.py

echo "Done."
