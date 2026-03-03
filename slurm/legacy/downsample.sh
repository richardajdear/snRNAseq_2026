#!/bin/bash
#SBATCH --job-name=downsample
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/downsample_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/downsample_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=80G
#SBATCH --partition=cclake

INPUT_FILE="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"
OUTPUT_FILE="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort_100k.h5ad"
SCRIPT="/home/rajd2/rds/hpc-work/snRNAseq_2026/code/downsample.py"

mkdir -p /home/rajd2/rds/hpc-work/snRNAseq_2026/logs

echo "Downsampling HBCC to 100k..."
date

singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPT \
    --input $INPUT_FILE \
    --output $OUTPUT_FILE \
    --n_cells 100000

if [ $? -eq 0 ]; then
    echo "Success!"
else
    echo "Failed!"
    exit 1
fi
