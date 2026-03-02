#!/bin/bash
#SBATCH --job-name=vel_noHVG
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/vel_noHVG_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/vel_noHVG_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=100G
#SBATCH --partition=icelake

set -euo pipefail

cd /home/rajd2/rds/hpc-work/snRNAseq_2026

# 1. Project Velmeshev Without HVG
singularity exec \
    --pwd /home/rajd2/rds/hpc-work/snRNAseq_2026 \
    /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default \
    python3 code/process_and_project.py \
    --input /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_PFC_lessOld.h5ad \
    --output notebooks/ahbaC3_projection/vel_noHVG.csv \
    --grn reference/ahba_dme_hcp_top8kgenes_weights.csv \
    --region "prefrontal cortex" \
    --no-log \
    --all-genes

# 2. Render HTML Report
singularity exec \
    --pwd /home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection \
    /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default \
    Rscript -e "rmarkdown::render('analysis.Rmd', output_file='analysis_vel_noHVG.html')"
