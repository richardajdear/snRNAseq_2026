#!/bin/bash
#SBATCH --job-name=ds_vel_100k
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/downsample_vel_100k_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/downsample_vel_100k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --partition=icelake

# Paths
CODE_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/code"
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

INPUT="$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_100k.h5ad"
OUTPUT="$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad"
TYPE="Velmeshev"

mkdir -p /home/rajd2/rds/hpc-work/snRNAseq_2026/logs

echo "========================================================"
echo "Starting Downsample (PFC only, <40 all, >=40 10%) for Velmeshev 100k"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "========================================================"

singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default \
    python -u $CODE_DIR/downsample.py \
    --input $INPUT \
    --output $OUTPUT \
    --dataset_type $TYPE \
    --pfc_only \
    --age_downsample

if [ $? -ne 0 ]; then
    echo "Job Failed"
    exit 1
fi

echo "Job Completed Successfully"
