#!/bin/bash
#SBATCH --job-name=combine_ice
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/combine_ice_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/combine_ice_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=384G
#SBATCH --cpus-per-task=60
#SBATCH --partition=icelake-himem

# Paths
CODE_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/code"
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

# Input Data Paths (Full, Downsampled)
VELMESHEV="$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_PFC_lessOld.h5ad"
WANG="$BASE_DIR/Cam_snRNAseq/wang/wang_PFC_lessOld.h5ad"
AGING="$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_PFC_lessOld.h5ad"
HBCC="$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_PFC_lessOld.h5ad"

# Output 
OUTPUT_COMBINED="$BASE_DIR/Cam_snRNAseq/combined/VelWangPsychad_PFC_lessOld.h5ad"

mkdir -p /home/rajd2/rds/hpc-work/snRNAseq_2026/logs
mkdir -p $(dirname $OUTPUT_COMBINED)

echo "========================================================"
echo "Starting Combine Job (Icelake Himem)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================================"

for f in $VELMESHEV $WANG $AGING $HBCC; do
    if [ ! -f "$f" ]; then echo "Error: Input file $f does not exist."; exit 1; fi
    echo "Input: $f ($(du -h $f | cut -f1))"
done

# Combine using --direct_load to skip re-processing metadata
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default \
    python -u $CODE_DIR/combine_data.py \
    --direct_load \
    --output $OUTPUT_COMBINED \
    --velmeshev_path $VELMESHEV \
    --wang_path $WANG \
    --aging_path $AGING \
    --hbcc_path $HBCC

if [ $? -ne 0 ]; then echo "Combine Failed"; exit 1; fi

echo "========================================================"
echo "Combine Complete!"
echo "Output: $OUTPUT_COMBINED"
echo "========================================================"
