#!/bin/bash
#SBATCH --job-name=extract_meta
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/meta_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/meta_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=20G
#SBATCH --partition=icelake

singularity exec /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python hpc-work/snRNAseq_2026/code/extract_all_meta.py

