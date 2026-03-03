#!/bin/bash
#SBATCH --job-name=combine_10k
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/combine_10k_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/combine_10k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --partition=icelake

# Paths
# We use absolute paths inside the script, which is fine on the cluster
CODE_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/code"
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

# Input Data Paths (New 10k Donor-Downsampled versions)
VELMESHEV="$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_10k_PFC_lessOld.h5ad"
WANG="$BASE_DIR/Cam_snRNAseq/wang/wang_10k_PFC_lessOld.h5ad"
AGING="$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_10k_PFC_lessOld.h5ad"
HBCC="$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_10k_PFC_lessOld.h5ad"

# Output
OUTPUT_COMBINED="$BASE_DIR/Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld.h5ad"

mkdir -p /home/rajd2/rds/hpc-work/snRNAseq_2026/logs
mkdir -p $(dirname $OUTPUT_COMBINED)

echo "========================================================"
echo "Starting Combine for 10k Donor-Downsampled Data"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================================"

echo "Combining datasets..."
# Note: We do NOT pass --postnatal, to keep prenatal ages in Velmeshev/Wang.
# The inputs are already downsampled by age/region, so no further filtering needed.

singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default \
    python -u $CODE_DIR/combine_data.py \
    --output $OUTPUT_COMBINED \
    --velmeshev_path $VELMESHEV \
    --wang_path $WANG \
    --aging_path $AGING \
    --hbcc_path $HBCC

if [ $? -ne 0 ]; then
    echo "Combine Failed"
    exit 1
fi

echo "Combine Complete!"
