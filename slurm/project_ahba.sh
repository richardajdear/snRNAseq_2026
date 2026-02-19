#!/bin/bash
#SBATCH --job-name=project_ahba
#SBATCH --output=logs/project_ahba_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=cclake

# AHBA C3 Projection Job
# Usage: sbatch slurm/project_ahba.sh

export SINGULARITY_IMAGE=/home/rajd2/rds/hpc-work/shortcake.sif

# --- Configuration ---
INPUT="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad"
OUTPUT="notebooks/ahbaC3_projection/vel_100k_projection.csv"
GRN="reference/ahba_dme_hcp_top8kgenes_weights.csv"
REGION="prefrontal cortex"

echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Region: $REGION"

singularity exec --cleanenv $SINGULARITY_IMAGE micromamba run -n shortcake_default \
    python -u code/process_and_project.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --grn "$GRN" \
    --region "$REGION" \
    --no-log

echo "Job Complete. $(date)"
