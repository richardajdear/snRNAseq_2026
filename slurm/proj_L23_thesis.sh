#!/bin/bash
#SBATCH --job-name=L23_thesis
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/L23_thesis_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/L23_thesis_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --partition=icelake

set -euo pipefail

cd /home/rajd2/rds/hpc-work/snRNAseq_2026

singularity exec \
    --pwd /home/rajd2/rds/hpc-work/snRNAseq_2026 \
    /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default \
    python3 code/project_velmeshev_L23_thesis.py
