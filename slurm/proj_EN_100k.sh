#!/bin/bash
#SBATCH --job-name=proj_EN_100k
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/proj_EN_100k_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/proj_EN_100k_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=icelake

set -euo pipefail

echo "Starting projection..."
singularity exec /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python /home/rajd2/rds/hpc-work/snRNAseq_2026/code/project_combined_postnatal_EN_100k_thesis.py

echo "Starting Rmd knit..."
cd /home/rajd2/rds
singularity exec /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default Rscript -e "rmarkdown::render('hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/analysis_combined_postnatal_EN_100k.Rmd', output_file='analysis_combined_postnatal_EN_100k.html')"
echo "Done!"
