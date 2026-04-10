#!/bin/bash
#SBATCH --job-name=verify_batch
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/verify_batch_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/verify_batch_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=icelake

set -euo pipefail

echo "Starting batch correction verification..."
singularity exec /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default \
    python -u /home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/verify_batch_correction.py

echo "Done."
