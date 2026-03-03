#!/bin/bash
#SBATCH --job-name=vel_300k_noHVG
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/vel_300k_noHVG_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/vel_300k_noHVG_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --partition=icelake

set -euo pipefail

cd /home/rajd2/rds/hpc-work/snRNAseq_2026

echo "Running full projection without HVG on VelWangPsychad_300k_PFC_lessOld..."

singularity exec \
    --pwd /home/rajd2/rds/hpc-work/snRNAseq_2026 \
    /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default \
    python3 code/process_and_project.py \
    --input "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/VelWangPsychad_300k_PFC_lessOld.h5ad" \
    --output "notebooks/ahbaC3_projection/vel_300k_noHVG.csv" \
    --grn "reference/ahba_dme_hcp_top8kgenes_weights.csv" \
    --region "all" \
    --all-genes \
    --no-log

echo "Done!"
