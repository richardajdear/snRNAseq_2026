# snRNAseq_2026

Single-nucleus RNA-seq analysis of postmortem human brain samples. Goal: characterize developmental trajectories of gene regulatory networks (AHBA C3) across the lifespan, with a focus on adolescence.

**Datasets:** Velmeshev, Wang, HBCC (PsychAD), Aging (PsychAD) — combined ~1M cells, integrated via scVI/scANVI for batch correction and cell type label transfer.

---

## For AI Agents

When working in this codebase:

- **Packages** are in the singularity image `hpc-work/shortcake.sif`, environment `shortcake_default` (micromamba). Always use this environment for Python and R.
- **Submit jobs with `sbatch`** from the repo root (`snRNAseq_2026/`) so logs write to `logs/`. Default partition: `cclake`; use `icelake` if busy.
- **Test on the login node first.** If a command takes more than ~2 minutes, cancel and submit via `sbatch`. Use backed-mode (`backed='r'`) for h5ad files and small subsets (1k–10k cells) for validation.
- **Large data files** live in RDS (`rds-cam-psych-transc-Pb9UGUlrwWc/`) and are not committed to git. Only code, config, and small reference files go in this repo.
- **Keep `scripts/` tidy.** Scripts there are for agent-driven debugging and validation. Periodically move old ones to `scripts/legacy/`. Keep notable outputs in `scripts/outputs/`.
- **Logs older than ~1 week** can be deleted. `fix_cell_class_*`, old projection runs, etc. are safe to remove once confirmed complete.
- This README is the authoritative guide to repo structure. Consult it before creating new files or directories.

---

## Repository Structure

```
snRNAseq_2026/
├── code/                        # All analysis code
│   ├── pipeline/                # Main data pipeline (downsample → combine → scVI → scANVI → pseudobulk)
│   │   ├── run_pipeline.py      # Orchestrator — START HERE for pipeline runs
│   │   ├── downsample.py        # Step 1: per-dataset filtering & downsampling
│   │   ├── combine_data.py      # Step 2: concatenate datasets
│   │   ├── read_data.py         # Dataset readers (Velmeshev, Wang, HBCC, Aging)
│   │   ├── scanvi_diagnostics.py # Post-scANVI label transfer diagnostics
│   │   ├── pseudobulk.py        # Pseudobulk aggregation
│   │   ├── label_transfer/      # Label transfer utilities (transfer.py is active)
│   │   ├── slurm/               # SLURM job scripts: step1–5 (normal chain) + util_* (ad-hoc)
│   │   ├── *.yaml               # Pipeline configs (hpc_config.yaml is the production config)
│   │   └── legacy/              # Retired pipeline utilities
│   │
│   ├── scVI/                    # scVI/scANVI batch correction module
│   │   ├── run_pipeline.py      # Entry point (called as subprocess by pipeline/)
│   │   ├── config.py            # PipelineConfig dataclass
│   │   ├── train.py, inference.py, visualize.py, data.py, utils.py
│   │   └── *.yaml               # scVI configs
│   │
│   ├── plotting/                # R plotting scripts (sourced by notebooks)
│   │   ├── hvg_plots.r          # HVG investigation and sensitivity plots
│   │   ├── sensitivity_gap_plots.r  # Gap model sensitivity plots
│   │   ├── thesis_plots.r       # Publication-quality UMAP/AHBA plots
│   │   └── age_plots.r, plot_projection.R
│   │
│   ├── environment.py           # HPC/local path resolution — imported by pipeline & notebooks
│   ├── process_data.py          # Normalization, HVG selection, PCA
│   ├── regulons.py              # AHBA C3 GRN loading and projection
│   ├── metadata_utils.py        # Cell type / age metadata utilities
│   ├── hvg_investigation.py     # HVG projection pipeline (used by notebooks)
│   │
│   ├── legacy_analysis/         # One-off projection scripts from earlier analyses
│   ├── legacy_standalone/       # Utility scripts with no active callers
│   └── legacy_scVI/             # Deprecated monolithic scVI implementation
│
├── notebooks/                   # Quarto analysis notebooks
│   ├── templates/               # Canonical notebook templates (start here for new analyses)
│   │   ├── sensitivity_analysis.qmd      # Standard sensitivity model + optional diagnostics
│   │   └── sensitivity_gap_analysis.qmd  # Gap model sensitivity (chemistry/dataset batch)
│   ├── configs/                 # One YAML per experiment — the only thing you change per run
│   ├── results/                 # Rendered outputs (one subdirectory per experiment)
│   ├── render_single.sh         # Render one notebook: sbatch render_single.sh <config_name>
│   ├── render_all.sh            # Render all (or a glob subset): bash render_all.sh '*scANVI*'
│   ├── render_notebook.sh       # Low-level SLURM worker (called by render_single.sh)
│   └── LEGACY/                  # Old per-experiment .qmd and .ipynb notebooks
│
├── scripts/                     # Agent debugging & validation scripts (non-pipeline)
│   ├── run_script.sh            # Generic SLURM wrapper for scripts
│   ├── diagnose_*.py            # Batch correction / scVI diagnostics
│   ├── outputs/                 # Diagnostic results
│   └── legacy/                  # Older scripts no longer in active use
│
├── slurm/                       # Retired top-level SLURM scripts
│   └── legacy/                  # Feb–Mar 2026 projection and diagnostic jobs
│
├── logs/                        # SLURM job logs (auto-cleaned periodically)
├── reference/                   # Small reference files (GRN weights, gene lists)
└── results/                     # Analysis outputs for review
```

---

## Running the Pipeline

The main pipeline processes 4 datasets through 5 steps. All steps are driven by a YAML config.

**Example config:** `code/pipeline/configs/excitatory_1y+_tuning4.yaml`

```bash
# Submit the full 5-step chain
cd snRNAseq_2026
bash code/pipeline/slurm/submit_pipeline.sh code/pipeline/configs/excitatory_1y+_tuning4.yaml

# Or run individual steps (each takes CONFIG as an env var)
sbatch --export=ALL,CONFIG=code/pipeline/configs/excitatory_1y+_tuning4.yaml \
       code/pipeline/slurm/step1_downsample_combine.sh
sbatch --export=ALL,CONFIG=... code/pipeline/slurm/step2_scvi.sh
sbatch --export=ALL,CONFIG=... code/pipeline/slurm/step3_diagnostics.sh
sbatch --export=ALL,CONFIG=... code/pipeline/slurm/step4_pseudobulk.sh
sbatch --export=ALL,CONFIG=... code/pipeline/slurm/step5_notebook.sh
```

**Steps:**
1. `downsample.py` — Filter and subsample each source to `n_cells`; writes `per_dataset/*.h5ad`
2. `combine_data.py` — Concatenate to `combined.h5ad` (inner join on genes)
3. `scVI/run_pipeline.py` — Train scVI, run scANVI label transfer, compute UMAPs + PCA plots, save `integrated.h5ad`
4. `scanvi_diagnostics.py` — Validate label transfer quality; writes `scanvi_diagnostics/`
5. `pseudobulk.py` — Aggregate `integrated.h5ad` to donor-level pseudobulk
6. Notebook render *(if `notebook:` section present in config)*

**Key outputs** (under the `output_dir` in the config):
- `scvi_output/integrated.h5ad` — scVI/scANVI corrected, with `cell_type_aligned` labels
- `scvi_output/plots/` — UMAP and PCA comparison grids (`umaps_*.png`, `pca_*.png`)
- `scanvi_diagnostics/` — Label transfer QC plots and tables
- `pseudobulk_output/` — Donor-level aggregates

**Utility scripts** (not in normal chain — submit manually when needed):

```bash
# Re-run scANVI label transfer without retraining scVI
sbatch --export=ALL,CONFIG=... code/pipeline/slurm/util_scanvi_rerun.sh

# Regenerate scvi_output/plots/ from an existing integrated.h5ad (CPU, fast)
SCVI_CONFIG=.../scvi_output/config.yaml
sbatch --export=ALL,SCVI_CONFIG="${SCVI_CONFIG}" code/pipeline/slurm/util_replot.sh

# Resume scVI pipeline from inference onwards (after a GPU timeout during training)
sbatch --export=ALL,SCVI_CONFIG="${SCVI_CONFIG}" code/pipeline/slurm/step2_scvi_resume_infer.sh
```

### Overnight scVI hyperparameter tuning (source-chemistry)

Use the dedicated tuning module under `code/tuning/`:

```bash
# Submit 12h GPU tuning job
sbatch code/tuning/slurm/tune_scvi_source_chemistry.sh

# Optional: override config path
sbatch --export=ALL,CONFIG=code/tuning/source-chemistry_tuning_config.yaml \
  code/tuning/slurm/tune_scvi_source_chemistry.sh
```

The tuner writes:
- `trial_results.csv` — one row per trial with objective + metrics
- `best_hyperparameters.yaml` — best scVI settings to copy into pipeline config
- `tuning.log` — detailed training/metric log

The objective is age-aware: batch mixing is evaluated separately by age bins, with prenatal bins upweighted.

---

## Running Analysis Notebooks

Notebooks use a **template + config** architecture. To run an analysis:

```bash
# Render a single experiment (most common)
sbatch notebooks/render_single.sh sensitivity_chemistry_scANVI

# Render all scANVI sensitivity variants
bash notebooks/render_all.sh '*scANVI*'

# Render everything
bash notebooks/render_all.sh
```

**Adding a new experiment** — create one YAML in `notebooks/configs/`:

```yaml
# Template: sensitivity_gap_analysis        ← required: tells render_single.sh which template
EXPERIMENT_NAME: my_new_experiment
DATA_FILE: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/...path/to/integrated.h5ad
SCVI_LAYER: scanvi_normalized              # "" = raw, "scvi_normalized", "scanvi_normalized"
SOURCE_LABEL: combined
N_VALUES: [1000, 2000, 4000, 6000, 8000, 10000]
FILTER_CELL_TYPES: null                    # or list of cell_type_aligned values
CACHE_DIR: ""                              # leave "" to auto-create under results/
```

Output goes to `notebooks/results/<EXPERIMENT_NAME>/`.

**Available templates:**
- `sensitivity_gap_analysis` — Gap model sensitivity (most analyses). Use for chemistry/dataset batch variants.
- `sensitivity_analysis` — Standard sensitivity model. Set `INCLUDE_DIAGNOSTICS: true` for HVG investigation notebooks.

**How parameters work:** `render_single.sh` sets `$NOTEBOOK_PARAMS` to the config YAML path. The template reads it via `os.environ.get('NOTEBOOK_PARAMS')`. No papermill required.

---

## Code Architecture

### Path Resolution

`code/environment.py` detects HPC vs local and returns canonical paths:
```python
from environment import get_environment
env = get_environment()
rds_dir  = env['rds_dir']   # /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc  (HPC)
code_dir = env['code_dir']  # /home/rajd2/rds/hpc-work/snRNAseq_2026/code
ref_dir  = env['ref_dir']   # /home/rajd2/rds/hpc-work/snRNAseq_2026/reference
```

Notebooks and pipeline scripts add `code/` to `sys.path` for bare imports:
```python
sys.path.insert(0, os.path.join(_repo_root, 'code'))
from environment import get_environment
from hvg_investigation import load_single_scvi, setup_grn, ...
```

R scripts are sourced via `code_dir`:
```r
source(file.path(code_dir, 'plotting', 'hvg_plots.r'))
source(file.path(code_dir, 'plotting', 'sensitivity_gap_plots.r'))
```

### scVI Module

`code/scVI/` is called as a subprocess by `pipeline/run_pipeline.py`. Configuration is passed via a YAML file parsed by `scVI/config.py` (`PipelineConfig` dataclass). Do not import `scVI/` modules directly from outside the module.

### Legacy Directories

`legacy_*/` directories contain older code kept for reference/reproducibility. Do not import from them in new code.

---

## Environment

- **Singularity image:** `/home/rajd2/rds/hpc-work/shortcake.sif`
- **Conda env:** `shortcake_default` (Python 3.10, scanpy, scvi-tools, rpy2, R + ggplot2/patchwork)
- **Quarto:** `/usr/local/Cluster-Apps/ceuadmin/quarto/1.7.13`
- **SLURM partitions:** `cclake` (default), `icelake`
- **R libraries:** `/home/rajd2/R/library`

Running interactively inside the container:
```bash
singularity exec --bind /usr/local/Cluster-Apps/ceuadmin/quarto/1.7.13:/quarto \
    --env R_LIBS_USER=/home/rajd2/R/library \
    /home/rajd2/rds/hpc-work/shortcake.sif \
    micromamba run -n shortcake_default python
```
