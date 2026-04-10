#!/bin/bash
#SBATCH --job-name=diagnose_grn
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/diagnose_grn_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/diagnose_grn_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=1
#SBATCH --partition=icelake
#SBATCH --account=VERTES-SL2-CPU

set -euo pipefail

echo "Diagnosing ahbaC3 GRN source differences..."
singularity exec /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default \
    python -u /home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/diagnose_grn_batch_effect.py

echo "Done."
