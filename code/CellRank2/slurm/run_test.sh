#!/bin/bash
#SBATCH --job-name=cr2_test
#SBATCH --output=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/cr2_test_%j.log
#SBATCH --error=/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/cr2_test_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=icelake
#SBATCH --account=vertes-sl2-cpu

# Downsamples to ~5 000 excitatory neurons, then runs the full CellRank 2
# pipeline on that subset as a quick end-to-end validation.

set -euo pipefail

WORK_DIR="/home/rajd2/rds/hpc-work/snRNAseq_2026"
SIF="/home/rajd2/rds/hpc-work/shortcake_full.sif"
DATA_DIR="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
INPUT_H5AD="${DATA_DIR}/Cam_snRNAseq/integrated/VelWangPsychAD_100k_source-chemistry/scvi_output/integrated.h5ad"
TEST_H5AD="${WORK_DIR}/code/CellRank2/test_results/source_chemistry_5k_excit_test.h5ad"
TEST_OUTPUT_DIR="${WORK_DIR}/code/CellRank2/test_results/cellrank_test_output"

echo "========================================"
echo "CellRank 2 Pipeline — TEST (5 000 excitatory neurons)"
echo "Workdir: ${WORK_DIR}"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURMD_NODENAME:-$(hostname)}"
echo "========================================"

mkdir -p "${WORK_DIR}/logs" "${WORK_DIR}/code/CellRank2/test_results"

# ── Downsample to 5 000 excitatory neurons ────────────────────────────────────
echo "Downsampling to 5 000 excitatory neurons → ${TEST_H5AD}"
singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    "${SIF}" \
    micromamba run -n shortcake_default \
    python3 -u -c "
import anndata as ad, numpy as np

rng = np.random.default_rng(42)
adata = ad.read_h5ad('${INPUT_H5AD}', backed='r')
print(f'Full dataset: {adata.n_obs} cells x {adata.n_vars} genes')

# Subset to excitatory neurons only (mirrors cell_type_filter_pattern in config)
ct = adata.obs['cell_type_aligned'].astype(str)
excit_mask = ct.str.contains('excit|EN-', case=False, na=False)
excit_idx = np.where(excit_mask)[0]
print(f'Excitatory neurons: {len(excit_idx)}')

n = min(5000, len(excit_idx))
chosen = np.sort(rng.choice(excit_idx, size=n, replace=False))
adata[chosen].to_memory().write_h5ad('${TEST_H5AD}')
adata.file.close()
print(f'Saved {n} cells to ${TEST_H5AD}')
"

# ── Run CellRank 2 pipeline on the subset ─────────────────────────────────────
# Pass cell_type_filter_pattern="" to skip re-filtering (already subsetted above),
# and override output_dir to a temporary location.
singularity exec \
    --pwd "${WORK_DIR}" \
    --bind "${DATA_DIR}:${DATA_DIR}" \
    --bind "${WORK_DIR}:${WORK_DIR}" \
    --env "LD_LIBRARY_PATH=/opt/micromamba/envs/shortcake_default/lib" \
    "${SIF}" \
    micromamba run -n shortcake_default \
    env PYTHONPATH="code" python3 -u -m CellRank2.run_pipeline \
        --config "code/CellRank2/config.yaml" \
        --input_h5ad "${TEST_H5AD}" \
        --output_dir "${TEST_OUTPUT_DIR}" \
        --cell_type_filter_pattern "" \
        --n_macrostates 6 \
        --ot_max_iterations 500

echo "Done. Outputs in ${TEST_OUTPUT_DIR}"
