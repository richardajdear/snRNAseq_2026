#!/bin/bash
#SBATCH --job-name=proc_100k
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/proc_100k_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/proc_100k_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --partition=icelake

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
    FULL=$1
    OUT=$2
    NAME=$3
    
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
    
    if [ $? -ne 0 ]; then echo "  FAILED: $NAME 100k creation"; return 1; fi
    echo "  Done: $(du -h $OUT | cut -f1)"
}

# Helper: apply PFC + lessOld downsampling
apply_pfc_lessold() {
    INPUT=$1
    OUTPUT=$2
    TYPE=$3
    
    echo "  Applying PFC + lessOld to $TYPE..."
    singularity exec --cleanenv $SIF micromamba run -n shortcake_default \
        python -u $CODE_DIR/downsample.py \
        --input "$INPUT" \
        --output "$OUTPUT" \
        --dataset_type "$TYPE" \
        --pfc_only \
        --age_downsample
    
    if [ $? -ne 0 ]; then echo "  FAILED: $TYPE PFC+lessOld"; return 1; fi
    echo "  Done: $(du -h $OUTPUT | cut -f1)"
}

echo ""
echo "======== Step 1: Create missing 100k datasets ========"

# Wang 100k
create_100k "$BASE_DIR/Cam_snRNAseq/wang/wang.h5ad" \
            "$BASE_DIR/Cam_snRNAseq/wang/wang_100k.h5ad" \
            "Wang"
if [ $? -ne 0 ]; then exit 1; fi

# Aging 100k
create_100k "$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad" \
            "$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_100k.h5ad" \
            "Aging"
if [ $? -ne 0 ]; then exit 1; fi

echo ""
echo "======== Step 2: Apply PFC + lessOld to 100k datasets ========"

# Velmeshev already done
echo "  Velmeshev 100k PFC_lessOld already exists. Skipping."

# Wang
apply_pfc_lessold "$BASE_DIR/Cam_snRNAseq/wang/wang_100k.h5ad" \
                  "$BASE_DIR/Cam_snRNAseq/wang/wang_100k_PFC_lessOld.h5ad" \
                  "Wang"
if [ $? -ne 0 ]; then exit 1; fi

# Aging
apply_pfc_lessold "$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_100k.h5ad" \
                  "$BASE_DIR/Cam_PsychAD/RNAseq/Aging_Cohort_100k_PFC_lessOld.h5ad" \
                  "Aging"
if [ $? -ne 0 ]; then exit 1; fi

# HBCC
apply_pfc_lessold "$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_100k.h5ad" \
                  "$BASE_DIR/Cam_PsychAD/RNAseq/HBCC_Cohort_100k_PFC_lessOld.h5ad" \
                  "HBCC"
if [ $? -ne 0 ]; then exit 1; fi

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

if [ $? -ne 0 ]; then echo "Combine Failed"; exit 1; fi

echo ""
echo "========================================================"
echo "100k Pipeline Complete!"
echo "Combined: $COMBINED_OUT"
echo "========================================================"
