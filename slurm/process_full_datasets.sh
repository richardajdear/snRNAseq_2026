#!/bin/bash
#SBATCH --job-name=process_full
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/process_full_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/process_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=180G
#SBATCH --partition=icelake

set -euo pipefail

# Paths
CODE_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/code"
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
SIF="/home/rajd2/rds/hpc-work/shortcake.sif"

# Input Data Paths (Full)
VELMESHEV="$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev.h5ad"
WANG="$BASE_DIR/Cam_snRNAseq/wang/wang.h5ad"
AGING="$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad"
HBCC="$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"

# Output Itermediate Paths
VEL_OUT="$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_PFC_lessOld.h5ad"
WANG_OUT="$BASE_DIR/Cam_snRNAseq/wang/wang_PFC_lessOld.h5ad"
AGING_OUT="$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_PFC_lessOld.h5ad"
HBCC_OUT="$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_PFC_lessOld.h5ad"

COMBINED_OUT="$BASE_DIR/Cam_snRNAseq/combined/VelWangPsychad_PFC_lessOld.h5ad"

mkdir -p /home/rajd2/rds/hpc-work/snRNAseq_2026/logs
mkdir -p $(dirname $COMBINED_OUT)

echo "========================================================"
echo "Starting Full Dataset Processing"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Filtered Inputs: PFC only, Keep all donors <40, Keep 20% of donors >=40"
echo "========================================================"

run_downsample() {
    INPUT=$1
    OUTPUT=$2
    TYPE=$3
    
    echo "--------------------------------------------------------"
    echo "Processing $TYPE..."
    echo "In: $INPUT"
    echo "Out: $OUTPUT"
    
    if [ ! -f "$INPUT" ]; then
        echo "Error: Input file $INPUT does not exist."
        exit 1
    fi
    
    # Remove existing output to force recreation
    rm -f "$OUTPUT"
    
    singularity exec --cleanenv $SIF micromamba run -n shortcake_default \
        python -u $CODE_DIR/downsample.py \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --dataset_type "$TYPE" \
        --pfc_only \
        --age_downsample
        
    if [ ! -f "$OUTPUT" ]; then
        echo "Error: Output file $OUTPUT was NOT created!"
        exit 1
    fi
    echo "Done $TYPE. Size: $(du -h "$OUTPUT" | cut -f1)"
}

# 1. Velmeshev
run_downsample "$VELMESHEV" "$VEL_OUT" "Velmeshev"

# 2. Wang
run_downsample "$WANG" "$WANG_OUT" "Wang"

# 3. Aging
run_downsample "$AGING" "$AGING_OUT" "Aging"

# 4. HBCC
run_downsample "$HBCC" "$HBCC_OUT" "HBCC"

echo "--------------------------------------------------------"
echo "All downsampling complete. Starting Combination..."
echo "Out: $COMBINED_OUT"

# Remove existing combined file to force recreation
rm -f "$COMBINED_OUT"

# 5. Combine using --direct_load for concat_on_disk
singularity exec --cleanenv $SIF micromamba run -n shortcake_default \
    python -u $CODE_DIR/combine_data.py \
    --direct_load \
    --output "$COMBINED_OUT" \
    --velmeshev_path "$VEL_OUT" \
    --wang_path "$WANG_OUT" \
    --aging_path "$AGING_OUT" \
    --hbcc_path "$HBCC_OUT"
    
if [ ! -f "$COMBINED_OUT" ]; then
    echo "Error: Combined output was NOT created!"
    exit 1
fi

echo "========================================================"
echo "Full Processing Complete!"
echo "Combined file: $COMBINED_OUT ($(du -h "$COMBINED_OUT" | cut -f1))"
echo "========================================================"
