#!/bin/bash
#SBATCH --output=outputs/job_%j.out
#SBATCH --error=outputs/job_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=cclake

# Default Singularity Image
export SINGULARITY_IMAGE=/home/rajd2/rds/hpc-work/shortcake.sif

# Usage Information
if [ -z "$SCRIPT" ]; then
    echo "Error: SCRIPT environment variable not set."
    echo "Usage: sbatch --export=SCRIPT=path/to/script.py,ARGS='--arg1 val1' slurm/submit_job.sh"
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Running Script: $SCRIPT"
echo "Arguments: $ARGS"
echo "Date: $(date)"

# Determine interpreter based on file extension
if [[ "$SCRIPT" == *.py ]]; then
    CMD="python -u"
elif [[ "$SCRIPT" == *.R || "$SCRIPT" == *.r ]]; then
    CMD="Rscript"
else
    echo "Unknown script extension: $SCRIPT"
    exit 1
fi

echo "Detailed Command: singularity exec --cleanenv $SINGULARITY_IMAGE micromamba run -n shortcake_default $CMD $SCRIPT $ARGS"

# Run script inside singularity
singularity exec --cleanenv $SINGULARITY_IMAGE micromamba run -n shortcake_default $CMD $SCRIPT $ARGS

echo "Job Complete."
