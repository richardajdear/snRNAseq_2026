#!/bin/bash
#SBATCH --job-name=cellrank2
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/cellrank2_%j.out
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/cellrank2_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=cclake
#SBATCH --account=vertes-sl3-cpu

# Usage:
#   cd /home/rajd2/rds/hpc-work/snRNAseq_2026
#   sbatch slurm/cellrank2.sh [config_path]
#
# Default config: code/CellRank2/default_config.yaml

CONFIG=${1:-code/CellRank2/default_config.yaml}

source /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

SINGULARITY_IMAGE=/home/rajd2/rds/hpc-work/shortcake.sif

singularity exec --bind /home/rajd2/rds "$SINGULARITY_IMAGE" \
    bash -c "
        source /opt/conda/etc/profile.d/mamba.sh
        mamba activate shortcake_default
        cd /home/rajd2/rds/hpc-work/snRNAseq_2026
        PYTHONPATH=code python -m CellRank2.run_pipeline --config $CONFIG
    "
