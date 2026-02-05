#!/bin/bash
#SBATCH --job-name=Test_HBCC_Inplace
#SBATCH --output=logs/test_hbcc_inplace_full_%j.out
#SBATCH --error=logs/test_hbcc_inplace_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --mem=180G
#SBATCH --partition=cclake

HBCC_FULL="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"
SCRIPT="/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/test_read_hbcc_inplace.py"

mkdir -p logs

echo "Testing reading FULL HBCC Cohort (In-Place Mode)..."
date

singularity exec --cleanenv /home/rajd2/rds/hpc-work/shortcake.sif micromamba run -n shortcake_default python -u $SCRIPT $HBCC_FULL

if [ $? -eq 0 ]; then
    echo "Success!"
else
    echo "Failed!"
    exit 1
fi
