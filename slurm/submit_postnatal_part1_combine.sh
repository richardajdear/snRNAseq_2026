#!/bin/bash
#SBATCH --job-name=Post_Part1_Combine
#SBATCH --output=logs/postnatal_part1_%j.out
#SBATCH --error=logs/postnatal_part1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=180G
#SBATCH --partition=cclake

# Paths
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined"
OUTPUT_COMBINED="$BASE_DIR/combined_postnatal_full.h5ad"
OUTPUT_PROCESSED="$BASE_DIR/combined_postnatal_full_processed.h5ad"
RESULTS_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/results_full"
SCRIPTS_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts"

# Input Data Paths (Full)
VELMESHEV="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev.h5ad"
WANG="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/wang/wang.h5ad"
AGING="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad"
HBCC="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"

mkdir -p $RESULTS_DIR
mkdir -p logs

echo "========================================================"
echo "Starting Postnatal-Only Part 1: Combine & Process"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================================"

# Control Steps
export SKIP_STEP1=true
export SKIP_STEP2=false

# 1. Combine (Postnatal Filter)
if [ "$SKIP_STEP1" == "true" ]; then
    echo "[Step 1] Skipping combination as requested. Using existing $OUTPUT_COMBINED"
else
    echo "[Step 1] Combining FULL datasets (Postnatal Only, Age < 40)..."
    singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/read_and_combine.py \
        --postnatal \
        --max_age 40 \
        --output $OUTPUT_COMBINED \
        --velmeshev $VELMESHEV \
        --wang $WANG \
        --aging $AGING \
        --hbcc $HBCC

    if [ $? -ne 0 ]; then echo "Step 1 Failed"; exit 1; fi
fi

# 2. Process
if [ "$SKIP_STEP2" == "true" ]; then
    echo "[Step 2] Skipping processing as requested. Using existing $OUTPUT_PROCESSED"
else
    echo "[Step 2] Processing data (Normalize, Log, HVG, PCA)..."
    # Reducing HVGs to 2000 and using simple 'seurat' flavor to fix OOM.
    # 'seurat_v3' uses raw counts and spikes memory. 'seurat' uses log-norm data.
    # Added --hvg_subset 200000 to drastically reduce memory usage during HVG selection.
    singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/process_data.py \
        --input $OUTPUT_COMBINED \
        --output $OUTPUT_PROCESSED \
        --n_top_genes 2000 \
        --flavor seurat \
        --hvg_subset 200000
    
    if [ $? -ne 0 ]; then echo "Step 2 Failed"; exit 1; fi
fi

# 3. Plot (Pre-Harmony)
echo "[Step 3] Plotting UMAP (Pre-Harmony)..."
# Using standard settings for quick check
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/plot_umap.py \
    --input $OUTPUT_PROCESSED \
    --output $RESULTS_DIR/UMAP_Postnatal_Full_NoHarmony_Grid.png \
    --colors source dataset chemistry lineage age_log2 age_category \
    --recompute

if [ $? -ne 0 ]; then echo "Step 3 Failed"; exit 1; fi

echo "Part 1 Complete!"
