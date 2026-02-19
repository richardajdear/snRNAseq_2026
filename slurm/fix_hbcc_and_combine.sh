#!/bin/bash
#SBATCH --job-name=fix_hbcc
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/fix_hbcc_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/fix_hbcc_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --partition=icelake

CODE_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/code"
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
SIF="/home/rajd2/rds/hpc-work/shortcake.sif"

echo "========================================================"
echo "Step 1: Re-downsample HBCC (fix corrupted file)"
echo "========================================================"

singularity exec --cleanenv $SIF micromamba run -n shortcake_default \
    python -u $CODE_DIR/downsample.py \
    --input $BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_10k.h5ad \
    --output $BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_10k_PFC_lessOld.h5ad \
    --dataset_type HBCC \
    --pfc_only \
    --age_downsample

if [ $? -ne 0 ]; then echo "HBCC Downsample Failed"; exit 1; fi

echo "========================================================"
echo "Step 2: Combine all datasets"
echo "========================================================"

singularity exec --cleanenv $SIF micromamba run -n shortcake_default \
    python -u $CODE_DIR/combine_data.py \
    --output $BASE_DIR/Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld.h5ad \
    --velmeshev_path $BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_10k_PFC_lessOld.h5ad \
    --wang_path $BASE_DIR/Cam_snRNAseq/wang/wang_10k_PFC_lessOld.h5ad \
    --aging_path $BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_10k_PFC_lessOld.h5ad \
    --hbcc_path $BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_10k_PFC_lessOld.h5ad

if [ $? -ne 0 ]; then echo "Combine Failed"; exit 1; fi

echo "========================================================"
echo "Step 3: Extract metadata and check ages"
echo "========================================================"

singularity exec --cleanenv $SIF micromamba run -n shortcake_default \
    python -u /home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/age_distribution_v2/extract_meta.py \
    --input $BASE_DIR/Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld.h5ad \
    --output /home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/age_distribution_v2/metadata.csv

echo "All steps complete!"
