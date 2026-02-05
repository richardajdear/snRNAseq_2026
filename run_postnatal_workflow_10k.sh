#!/bin/bash
# run_postnatal_workflow_10k.sh
# Automates the creation, processing, and integration of the Postnatal-Only dataset (10k test).

# Paths
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined"
OUTPUT_COMBINED="$BASE_DIR/combined_postnatal_10k.h5ad"
OUTPUT_PROCESSED="$BASE_DIR/combined_postnatal_10k_processed.h5ad"
OUTPUT_HARMONY="$BASE_DIR/combined_postnatal_10k_harmony.h5ad"
RESULTS_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/results"
SCRIPTS_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts"

mkdir -p $RESULTS_DIR

echo "========================================================"
echo "Starting Postnatal-Only 10k Workflow"
echo "========================================================"

# 1. Combine (Postnatal Filter)
echo "[Step 1] Combining datasets (Postnatal Only)..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/read_and_combine.py \
    --postnatal \
    --output $OUTPUT_COMBINED

if [ $? -ne 0 ]; then echo "Step 1 Failed"; exit 1; fi

# 2. Process
echo "[Step 2] Processing data (Normalize, Log, HVG, PCA)..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/process_data.py \
    --input $OUTPUT_COMBINED \
    --output $OUTPUT_PROCESSED

if [ $? -ne 0 ]; then echo "Step 2 Failed"; exit 1; fi

# 3. Plot (Pre-Harmony)
echo "[Step 3] Plotting UMAP (Pre-Harmony)..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/plot_umap.py \
    --input $OUTPUT_PROCESSED \
    --output $RESULTS_DIR/UMAP_Postnatal_10k_NoHarmony_Grid.png \
    --colors source dataset chemistry lineage age_log2 age_category

if [ $? -ne 0 ]; then echo "Step 3 Failed"; exit 1; fi

# 4. Harmony Integration
echo "[Step 4] Running Harmony Integration..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/run_harmony.py \
    --input $OUTPUT_PROCESSED \
    --output $OUTPUT_HARMONY \
    --batch_key source

if [ $? -ne 0 ]; then echo "Step 4 Failed"; exit 1; fi

# 5. Plot (Post-Harmony)
echo "[Step 5] Plotting UMAP (Post-Harmony)..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/plot_umap.py \
    --input $OUTPUT_HARMONY \
    --output $RESULTS_DIR/UMAP_Postnatal_10k_Harmony_Grid.png \
    --colors source dataset chemistry lineage age_log2 age_category

if [ $? -ne 0 ]; then echo "Step 5 Failed"; exit 1; fi

echo "========================================================"
echo "Workflow Complete!"
echo "Outputs:"
echo "  - Combined: $OUTPUT_COMBINED"
echo "  - Processed: $OUTPUT_PROCESSED"
echo "  - Harmony: $OUTPUT_HARMONY"
echo "  - Plot (No Harmony): $RESULTS_DIR/UMAP_Postnatal_10k_NoHarmony_Grid.png"
echo "  - Plot (Harmony): $RESULTS_DIR/UMAP_Postnatal_10k_Harmony_Grid.png"
echo "========================================================"
