#!/bin/bash
#SBATCH --job-name=umap_comparison
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/umap_comparison_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/umap_comparison_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=icelake

set -euo pipefail

echo "Computing UMAP comparison (raw / X_scVI / scvi_normalized→PCA)..."
singularity exec /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default \
    python -u /home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/umap_comparison.py

echo "Done."
