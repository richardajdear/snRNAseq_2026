#!/bin/bash
#SBATCH --job-name=Postnatal_Full_Workflow
#SBATCH --output=logs/postnatal_full_%j.out
#SBATCH --error=logs/postnatal_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --partition=cclake

# Paths - Updated for Full Run
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined"
OUTPUT_COMBINED="$BASE_DIR/combined_postnatal_full.h5ad"
OUTPUT_PROCESSED="$BASE_DIR/combined_postnatal_full_processed.h5ad"
OUTPUT_HARMONY="$BASE_DIR/combined_postnatal_full_harmony.h5ad"
RESULTS_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/results_full"
SCRIPTS_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts"

# Input Data Paths (Full)
VELMESHEV="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev.h5ad"
WANG="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/wang/wang.h5ad"
# Note: Using Aging_Cohort.h5ad as Roussos full based on directory search 
ROUSSOS="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad"

mkdir -p $RESULTS_DIR
mkdir -p logs

echo "========================================================"
echo "Starting Postnatal-Only FULL Workflow"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================================"

# 1. Combine (Postnatal Filter)
echo "[Step 1] Combining FULL datasets (Postnatal Only)..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/read_and_combine.py \
    --postnatal \
    --output $OUTPUT_COMBINED \
    --velmeshev $VELMESHEV \
    --wang $WANG \
    --roussos $ROUSSOS

if [ $? -ne 0 ]; then echo "Step 1 Failed"; exit 1; fi

# 2. Process
echo "[Step 2] Processing data (Normalize, Log, HVG, PCA)..."
# Increase HVG to 10k or standard? Using default 10k from previous runs but could be more for full data.
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/process_data.py \
    --input $OUTPUT_COMBINED \
    --output $OUTPUT_PROCESSED \
    --n_top_genes 10000

if [ $? -ne 0 ]; then echo "Step 2 Failed"; exit 1; fi

# 3. Plot (Pre-Harmony)
echo "[Step 3] Plotting UMAP (Pre-Harmony)..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/plot_umap.py \
    --input $OUTPUT_PROCESSED \
    --output $RESULTS_DIR/UMAP_Postnatal_Full_NoHarmony_Grid.png \
    --colors source dataset chemistry lineage age_log2 age_category \
    --recompute # Force recompute neighbors/umap on full set just in case

if [ $? -ne 0 ]; then echo "Step 3 Failed"; exit 1; fi

# 4. Harmony Integration
echo "[Step 4] Running Harmony Integration (Full)..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/run_harmony.py \
    --input $OUTPUT_PROCESSED \
    --output $OUTPUT_HARMONY \
    --batch_key source

if [ $? -ne 0 ]; then echo "Step 4 Failed"; exit 1; fi

# 5. Plot (Post-Harmony)
echo "[Step 5] Plotting UMAP (Post-Harmony)..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/plot_umap.py \
    --input $OUTPUT_HARMONY \
    --output $RESULTS_DIR/UMAP_Postnatal_Full_Harmony_Grid.png \
    --colors source dataset chemistry lineage age_log2 age_category

if [ $? -ne 0 ]; then echo "Step 5 Failed"; exit 1; fi

echo "========================================================"
echo "Full Workflow Complete!"
echo "DATE: $(date)"
echo "========================================================"
