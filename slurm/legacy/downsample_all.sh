#!/bin/bash
#SBATCH --job-name=downsample_all
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/downsample_all_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/downsample_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=80G
#SBATCH --partition=icelake

# Paths
BASE_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined"
INPUT_FILE="$BASE_DIR/combined_postnatal_full_harmony.h5ad"
SCRIPTS_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/code"

mkdir -p /home/rajd2/rds/hpc-work/snRNAseq_2026/logs

echo "========================================================"
echo "Creating Downsampled Subsets"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "========================================================"

# 1. 10k Random
echo "[Step 1] Creating 10k Random Subset..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/downsample.py \
    --input $INPUT_FILE \
    --output "$BASE_DIR/combined_postnatal_full_harmony_10k.h5ad" \
    --n_cells 10000

if [ $? -ne 0 ]; then echo "Step 1 Failed"; exit 1; fi

# 2. 100k Random
echo "[Step 2] Creating 100k Random Subset..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/downsample.py \
    --input $INPUT_FILE \
    --output "$BASE_DIR/combined_postnatal_full_harmony_100k.h5ad" \
    --n_cells 100000

if [ $? -ne 0 ]; then echo "Step 2 Failed"; exit 1; fi

# 3. 10k Excitatory
echo "[Step 3] Creating 10k Excitatory Subset..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/downsample.py \
    --input $INPUT_FILE \
    --output "$BASE_DIR/combined_postnatal_full_harmony_EN_10k.h5ad" \
    --n_cells 10000 \
    --filter_column lineage \
    --filter_value Excitatory

if [ $? -ne 0 ]; then echo "Step 3 Failed"; exit 1; fi

# 4. 100k Excitatory
echo "[Step 4] Creating 100k Excitatory Subset..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPTS_DIR/downsample.py \
    --input $INPUT_FILE \
    --output "$BASE_DIR/combined_postnatal_full_harmony_EN_100k.h5ad" \
    --n_cells 100000 \
    --filter_column lineage \
    --filter_value Excitatory

if [ $? -ne 0 ]; then echo "Step 4 Failed"; exit 1; fi

echo "All subsets created successfully."
