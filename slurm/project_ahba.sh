#!/bin/bash
#SBATCH --job-name=project_ahba
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/project_ahba_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=4:00:00
#SBATCH --partition=icelake

# AHBA C3 Projection Job
# Usage: sbatch hpc-work/snRNAseq_2026/slurm/project_ahba.sh

cd /home/rajd2/rds/hpc-work/snRNAseq_2026

export SINGULARITY_IMAGE=/home/rajd2/rds/hpc-work/shortcake.sif

# --- Configuration ---
INPUT="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/VelWangPsychad_300k_PFC_lessOld.h5ad"
OUTPUT="notebooks/ahbaC3_projection/projection_results_300k.csv"
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
