#!/bin/bash
#SBATCH --job-name=process_full
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/process_full_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/process_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=180G
#SBATCH --partition=icelake

# Paths
CODE_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/code"
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

# Input Data Paths (Full)
VELMESHEV="$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev.h5ad"
WANG="$BASE_DIR/Cam_snRNAseq/wang/wang.h5ad"
AGING="$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad"
HBCC="$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"

# Output Directory (Same as inputs or specific?)
# User asked for {source_name}_PFC_lessOld.h5ad
# We will save them in the same directory as source for organization, or 'combined/intermediate'?
# User said "Call these datasets {source_name}_PFC_lessOld.h5ad."
# Usually better to keep them near source.
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
    
    singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default \
        python -u $CODE_DIR/downsample.py \
        --input $INPUT \
        --output $OUTPUT \
        --dataset_type $TYPE \
        --pfc_only \
        --age_downsample
        
    if [ $? -ne 0 ]; then
        echo "Error processing $TYPE"
        exit 1
    fi
    echo "Done $TYPE."
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

# 5. Combine
# Note: DO NOT pass --postnatal here, because we want to keep the 10% old cells we just filtered for.
# combine_data.py with --postnatal forces < 40.
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default \
    python -u $CODE_DIR/combine_data.py \
    --output $COMBINED_OUT \
    --velmeshev_path "$VEL_OUT" \
    --wang_path "$WANG_OUT" \
    --aging_path "$AGING_OUT" \
    --hbcc_path "$HBCC_OUT"
    
if [ $? -ne 0 ]; then
    echo "Error in Combination"
    exit 1
fi

echo "========================================================"
echo "Full Processing Complete!"
echo "Combined file: $COMBINED_OUT"
