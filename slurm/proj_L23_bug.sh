#!/bin/bash
#SBATCH --job-name=proj_L23_bug
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/proj_L23_bug_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/proj_L23_bug_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --partition=icelake

set -euo pipefail

cd /home/rajd2/rds/hpc-work/snRNAseq_2026

singularity exec \
    --pwd /home/rajd2/rds/hpc-work/snRNAseq_2026 \
    /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default \
    python3 code/project_velmeshev_L23.py --filter-fc --batch-key chemistry --thesis-bug
