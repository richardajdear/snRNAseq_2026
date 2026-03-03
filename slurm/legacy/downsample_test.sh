#!/bin/bash
#SBATCH --job-name=downsample_test
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/downsample_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/downsample_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=10G
#SBATCH --partition=icelake

# Paths
CODE_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/code"
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"

# Input Data Paths (10k Test versions)
# Note: These paths are derived from slurm/combine.sh
# Check if files exist
VELMESHEV_10k="$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_10k.h5ad"
WANG_10k="$BASE_DIR/Cam_snRNAseq/wang/wang_10k.h5ad"
AGING_10k="$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_10k.h5ad"
HBCC_10k="$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_10k.h5ad"

for f in $VELMESHEV_10k $WANG_10k $AGING_10k $HBCC_10k; do
    if [ ! -f "$f" ]; then echo "Error: Input file $f does not exist."; exit 1; fi
done

mkdir -p /home/rajd2/rds/hpc-work/snRNAseq_2026/logs

echo "========================================================"
echo "Starting Downsample Test (PFC only, <40 all, >=40 10%)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================================"

# Run singularity command wrapper
run_downsample() {
    INPUT=$1
    OUTPUT=$2
    TYPE=$3
    
    echo "Processing $TYPE..."
    echo "  In: $INPUT"
    echo "  Out: $OUTPUT"
    
    singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default \
        python -u $CODE_DIR/downsample.py \
        --input $INPUT \
        --output $OUTPUT \
        --dataset_type $TYPE \
        --pfc_only \
        --age_downsample
        
    if [ $? -ne 0 ]; then echo "Failed to process $TYPE"; exit 1; fi
}

# 1. Velmeshev
# Output name: [source_name]_10k_PFC_lessOld
VELMESHEV_OUT="${VELMESHEV_10k%.*}_PFC_lessOld.h5ad"
run_downsample $VELMESHEV_10k $VELMESHEV_OUT "Velmeshev"

# 2. Wang
WANG_OUT="${WANG_10k%.*}_PFC_lessOld.h5ad"
run_downsample $WANG_10k $WANG_OUT "Wang"

# 3. Aging
aging_out="${AGING_10k%.*}_PFC_lessOld.h5ad"
run_downsample $AGING_10k $aging_out "Aging"

# 4. HBCC
hbcc_out="${HBCC_10k%.*}_PFC_lessOld.h5ad"
run_downsample $HBCC_10k $hbcc_out "HBCC"

echo "All downsampling jobs completed."
