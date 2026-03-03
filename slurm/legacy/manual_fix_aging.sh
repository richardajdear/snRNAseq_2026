#!/bin/bash
#SBATCH --job-name=manual_aging
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/manual_aging_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/manual_aging_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=10G
#SBATCH --partition=cclake

echo "Running manual downsample test for Aging..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default \
    python code/downsample.py \
    --input /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort_10k.h5ad \
    --output /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort_10k_PFC_lessOld.h5ad \
    --dataset_type Aging \
    --pfc_only \
    --age_downsample

if [ $? -eq 0 ]; then
    echo "Success!"
else
    echo "Failed!"
    exit 1
fi
