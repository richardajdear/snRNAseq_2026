#!/bin/bash
#SBATCH --job-name=Verify_Output
#SBATCH --output=logs/verify_output_%j.out
#SBATCH --error=logs/verify_output_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --partition=cclake

OUTPUT_FILE="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_postnatal_full.h5ad"
SCRIPT="/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/verify_output.py"

mkdir -p logs
echo "Verifying output: $OUTPUT_FILE"
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPT $OUTPUT_FILE
