#!/bin/bash
#SBATCH --job-name=diagnose_pc1_age
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/diagnose_pc1_age_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/diagnose_pc1_age_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=icelake

set -euo pipefail

echo "Diagnosing PC1 age vs batch confound..."
singularity exec /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default \
    python -u /home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/diagnose_pc1_age_batch.py

echo "Done."
