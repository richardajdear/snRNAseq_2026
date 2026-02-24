#!/bin/bash
# Script to knit the AHBA C3 Analysis Rmd using Singularity
# This script can be called from any directory.

# Configuration
SINGULARITY_IMAGE="/home/rajd2/rds/hpc-work/shortcake.sif"
PROJECT_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
RMD_PATH="${PROJECT_DIR}/notebooks/ahbaC3_projection/analysis.Rmd"

# Rendering Command
# We use LD_PRELOAD to avoid libjpeg version conflicts in the environment
singularity exec --env LD_PRELOAD=/opt/micromamba/envs/shortcake_default/lib/libjpeg.so.8 \
    --pwd "${PROJECT_DIR}" \
    "${SINGULARITY_IMAGE}" \
    micromamba run -n shortcake_default \
    Rscript -e "rmarkdown::render('${RMD_PATH}')"
