#!/bin/bash
#SBATCH --job-name=proc_100k
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/proc_100k_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/proc_100k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --partition=icelake

set -euo pipefail

CODE_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/code"
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
SIF="/home/rajd2/rds/hpc-work/shortcake.sif"

mkdir -p /home/rajd2/rds/hpc-work/snRNAseq_2026/logs

echo "========================================================"
echo "100k Intermediate Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================================"

# Helper: create 100k random subsample from full dataset
create_100k() {
    local FULL=$1
    local OUT=$2
    local NAME=$3
    
    if [ -f "$OUT" ]; then
        echo "  $NAME 100k already exists: $OUT. Skipping."
        return 0
    fi
    
    echo "  Creating $NAME 100k from $FULL..."
    singularity exec --cleanenv $SIF micromamba run -n shortcake_default \
        python -u $CODE_DIR/downsample.py \
        --input "$FULL" \
        --output "$OUT" \
        --dataset_type "$NAME" \
        --n_cells 100000
    
    # Verify file was created
    if [ ! -f "$OUT" ]; then
        echo "  ERROR: Output file $OUT was NOT created!"
        exit 1
    fi
    echo "  Done: $(du -h $OUT | cut -f1)"
}

# Helper: apply PFC + lessOld downsampling
apply_pfc_lessold() {
    local INPUT=$1
    local OUTPUT=$2
    local TYPE=$3
    
    if [ ! -f "$INPUT" ]; then
        echo "  ERROR: Input file $INPUT does not exist!"
        exit 1
    fi
    
    echo "  Applying PFC + lessOld to $TYPE..."
    singularity exec --cleanenv $SIF micromamba run -n shortcake_default \
        python -u $CODE_DIR/downsample.py \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --dataset_type "$TYPE" \
        --pfc_only \
        --age_downsample
    
    if [ ! -f "$OUTPUT" ]; then
        echo "  ERROR: Output file $OUTPUT was NOT created!"
        exit 1
    fi
    echo "  Done: $(du -h $OUTPUT | cut -f1)"
}

echo ""
echo "======== Step 1: Create missing 100k datasets ========"

# Wang 100k
create_100k "$BASE_DIR/Cam_snRNAseq/wang/wang.h5ad" \
            "$BASE_DIR/Cam_snRNAseq/wang/wang_100k.h5ad" \
            "Wang"

# Aging 100k
create_100k "$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad" \
            "$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_100k.h5ad" \
            "Aging"

echo ""
echo "======== Step 2: Apply PFC + lessOld to 100k datasets ========"

# Velmeshev already done
if [ -f "$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad" ]; then
    echo "  Velmeshev 100k PFC_lessOld already exists. Skipping."
else
    apply_pfc_lessold "$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_100k.h5ad" \
                      "$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad" \
                      "Velmeshev"
fi

# Wang
apply_pfc_lessold "$BASE_DIR/Cam_snRNAseq/wang/wang_100k.h5ad" \
                  "$BASE_DIR/Cam_snRNAseq/wang/wang_100k_PFC_lessOld.h5ad" \
                  "Wang"

# Aging
apply_pfc_lessold "$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_100k.h5ad" \
                  "$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_100k_PFC_lessOld.h5ad" \
                  "Aging"

# HBCC
apply_pfc_lessold "$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_100k.h5ad" \
                  "$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_100k_PFC_lessOld.h5ad" \
                  "HBCC"

echo ""
echo "======== Step 3: Combine all 100k PFC_lessOld datasets ========"

COMBINED_OUT="$BASE_DIR/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld.h5ad"
mkdir -p $(dirname $COMBINED_OUT)

singularity exec --cleanenv $SIF micromamba run -n shortcake_default \
    python -u $CODE_DIR/combine_data.py \
    --direct_load \
    --output "$COMBINED_OUT" \
    --velmeshev_path "$BASE_DIR/Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad" \
    --wang_path "$BASE_DIR/Cam_snRNAseq/wang/wang_100k_PFC_lessOld.h5ad" \
    --aging_path "$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_100k_PFC_lessOld.h5ad" \
    --hbcc_path "$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_100k_PFC_lessOld.h5ad"

if [ ! -f "$COMBINED_OUT" ]; then
    echo "ERROR: Combined output was NOT created!"
    exit 1
fi

echo ""
echo "========================================================"
echo "100k Pipeline Complete!"
echo "Combined: $COMBINED_OUT ($(du -h $COMBINED_OUT | cut -f1))"
echo "========================================================"
