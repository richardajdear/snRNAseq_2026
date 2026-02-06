#!/bin/bash
#SBATCH --job-name=Test_40k_Combine
#SBATCH --output=logs/test_combine_%j.out
#SBATCH --error=logs/test_combine_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --mem=20G
#SBATCH --partition=icelake

# Paths
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined"
OUTPUT_COMBINED="$BASE_DIR/combined_40k_test.h5ad"
SCRIPTS_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts"

# Input Data Paths (10k Subsets)
VELMESHEV="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev10k.h5ad"
WANG="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/wang/wang_10k.h5ad"
AGING="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort_10k.h5ad"
HBCC="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort_10k.h5ad"

mkdir -p logs

echo "========================================================"
echo "Starting Test Combine (4x10k Subsets)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================================"

echo "Combining datasets..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/read_and_combine.py \
    --postnatal \
    --output $OUTPUT_COMBINED \
    --velmeshev_path $VELMESHEV \
    --wang_path $WANG \
    --aging_path $AGING \
    --hbcc_path $HBCC

if [ $? -eq 0 ]; then
    echo "Combination successful!"
    ls -lh $OUTPUT_COMBINED
else
    echo "Combination failed!"
    exit 1
fi
