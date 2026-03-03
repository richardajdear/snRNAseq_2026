#!/bin/bash
#SBATCH --job-name=fix_var
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/fix_var_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/fix_var_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --partition=icelake

cd /home/rajd2/rds/hpc-work/snRNAseq_2026
export SINGULARITY_IMAGE=/home/rajd2/rds/hpc-work/shortcake.sif

echo "Starting fix_var..."
singularity exec --cleanenv $SINGULARITY_IMAGE micromamba run -n shortcake_default python -u code/fix_100k_var.py
echo "Done fix_var."
