#!/bin/bash
#SBATCH --job-name=Post_Harmony_3Dat
#SBATCH --output=logs/postnatal_part2_3datasets_%j.out
#SBATCH --error=logs/postnatal_part2_3datasets_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --partition=cclake

# Paths
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined"
INPUT_PROCESSED="$BASE_DIR/combined_postnatal_3datasets_processed.h5ad"
OUTPUT_HARMONY="$BASE_DIR/combined_postnatal_3datasets_harmony.h5ad"
RESULTS_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/results_3datasets"
SCRIPTS_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts"

mkdir -p $RESULTS_DIR
mkdir -p logs

echo "========================================================"
echo "Starting Postnatal Part 2: Harmony (3 Datasets)"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================================"

# 1. Plot Pre-Harmony UMAP
echo "[Step 1] Plotting Pre-Harmony UMAP..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/plot_umap.py \
    --input $INPUT_PROCESSED \
    --output $RESULTS_DIR/UMAP_Postnatal_3Datasets_NoHarmony_Grid.png \
    --colors source dataset chemistry lineage age_log2 age_category \
    --recompute \
    --use_rep X_pca

if [ $? -ne 0 ]; then echo "Pre-Harmony Plot Failed"; exit 1; fi

# 2. Run Harmony
echo "[Step 2] Running Harmony on 'dataset'..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/run_harmony.py \
    --input $INPUT_PROCESSED \
    --output $OUTPUT_HARMONY \
    --batch_key dataset

if [ $? -ne 0 ]; then echo "Harmony Failed"; exit 1; fi

# 3. Plot Post-Harmony UMAP
echo "[Step 3] Plotting Post-Harmony UMAP..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/plot_umap.py \
    --input $OUTPUT_HARMONY \
    --output $RESULTS_DIR/UMAP_Postnatal_3Datasets_Harmony_Grid.png \
    --colors source dataset chemistry lineage age_log2 age_category \
    --recompute \
    --use_rep X_pca_harmony

if [ $? -ne 0 ]; then echo "Post-Harmony Plot Failed"; exit 1; fi

echo "Part 2 (3 Datasets) Complete!"
