#!/bin/bash
#SBATCH --job-name=ahbaC3
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/projection_full_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/projection_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --partition=icelake

# Environments
INPUT_FILE="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_postnatal_full_processed.h5ad"
REF_MAPPING="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort_10k.h5ad"
OUTPUT_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026/results_full_pfc"
OUTPUT_CSV="${OUTPUT_DIR}/projection.csv"
OUTPUT_PLOT="${OUTPUT_DIR}/projection_plot.png"

SCRIPT_PY="/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/project_genes.py"
SCRIPT_R="/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/plot_projection.R"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "Starting Projection Analysis Pipeline"
date

# Step 1: Python Projection
echo "Step 1: Running Python Projection..."
singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPT_PY \
    --input $INPUT_FILE \
    --output $OUTPUT_CSV \
    --filter_column region \
    --filter_value "prefrontal cortex" \
    --ref_mapping_file $REF_MAPPING


if [ $? -ne 0 ]; then
    echo "Python script failed."
    exit 1
fi

# Step 2: R Plotting
echo "Step 2: Running R Plotting..."

singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default Rscript $SCRIPT_R \
    --input $OUTPUT_CSV \
    --output $OUTPUT_PLOT \
    --title "Developmental expression of AHBA C3 in Postnatal PFC" \
    --color_var "source"

if [ $? -ne 0 ]; then
    echo "R script failed."
    exit 1
fi

echo "Pipeline Completed Successfully!"
date
