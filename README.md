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

### Pipeline Architecture (quick reference)

The main pipeline is a 5-step SLURM chain. Steps are submitted by `code/pipeline/slurm/submit_pipeline.sh` using a YAML config. Each step runs as an independent SLURM job; the chain is built dynamically via `afterok` dependencies.

**Step order and key data flows:**

| Step | Script | Input → Output | Resources |
|------|--------|---------------|-----------|
| 1 | `step1_downsample_combine.sh` | raw h5ads → `per_dataset/*.h5ad`, `combined.h5ad` | CPU, 1h, 64GB |
| 2 | `step2_scvi.sh` | `combined.h5ad` → `scvi_output/integrated.h5ad` + plots | **GPU**, 12h, 200GB |
| 3 | `step3_pseudobulk.sh` | `integrated.h5ad` → `pseudobulk_output/*.h5ad` | CPU, 1h, 128GB |
| 4 | `step4_notebook.sh` | `pseudobulk_output/` → `notebooks/results/<exp>/` | CPU, 10m, 10GB |
| 5 | `step5_diagnostics.sh` | `integrated.h5ad` → `scanvi_diagnostics/` | CPU, 3h, 228GB |

**Why diagnostics runs last (step 5):** it is time-intensive (recomputes UMAPs) but nothing downstream depends on it. Pseudobulk and notebook can complete independently.

**Config YAML controls which steps run** via the `steps:` key (e.g. `steps: [scvi, pseudobulk, notebook, diagnostics]`). If absent, defaults to all steps. Step *names* (`downsample`, `combine`, `scvi`, `pseudobulk`, `notebook`, `diagnostics`) are stable — the numbering in script filenames reflects execution order.

**Output directory** is set per-config via `output_dir:` (an absolute path on RDS). All step outputs land there: `scvi_output/`, `pseudobulk_output/`, `scanvi_diagnostics/`, `per_dataset/`, `combined.h5ad`.

**Utility scripts** (submit manually, not part of the normal chain):
- `util_scanvi_rerun.sh` — force-retrain scANVI after updating label mappings
- `util_retransform.sh` — re-run inference with a different `transform_batch`, then pseudobulk
- `util_replot.sh` — regenerate `scvi_output/plots/` without retraining
- `step2_scvi_resume_infer.sh` — resume inference when both models exist but `integrated.h5ad` is missing

---

## Repository Structure

```
snRNAseq_2026/
├── code/                        # All analysis code
│   ├── pipeline/                # Main data pipeline (downsample → combine → scVI → pseudobulk → diagnostics)
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
sbatch --export=ALL,CONFIG=... code/pipeline/slurm/step3_pseudobulk.sh
sbatch --export=ALL,CONFIG=... code/pipeline/slurm/step4_notebook.sh
sbatch --export=ALL,CONFIG=... code/pipeline/slurm/step5_diagnostics.sh
```

**Steps:**
1. `downsample.py` — Filter and subsample each source to `n_cells`; writes `per_dataset/*.h5ad`
2. `combine_data.py` — Concatenate to `combined.h5ad` (inner join on genes)
3. `scVI/run_pipeline.py` — Train scVI, run scANVI label transfer, compute UMAPs + PCA plots, save `integrated.h5ad`
4. `pseudobulk.py` — Aggregate `integrated.h5ad` to donor-level pseudobulk
5. Notebook render *(if `notebook:` section present in config)*
6. `scanvi_diagnostics.py` — Validate label transfer quality; writes `scanvi_diagnostics/` *(runs last — time-intensive UMAPs; nothing downstream depends on it)*

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

**Adding a new experiment** — drop one YAML next to the template that should render it, at `notebooks/templates/<template>/configs/<config_name>.yaml`. The template is inferred from the directory, so no `# Template:` comment is needed.

```yaml
EXPERIMENT_NAME: my_new_experiment
DATA_FILE: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/...path/to/integrated.h5ad
SCVI_LAYER: scanvi_normalized              # "" = raw, "scvi_normalized", "scanvi_normalized"
SOURCE_LABEL: combined
N_VALUES: [1000, 2000, 4000, 6000, 8000, 10000]
FILTER_CELL_TYPES: null                    # or list of cell_type_aligned values
CACHE_DIR: ""                              # leave "" to auto-create under results/
```

Output goes to `notebooks/results/<EXPERIMENT_NAME>/`. `render_single.sh <config_name>` finds the config by walking `notebooks/templates/*/configs/`; if the same `<config_name>` is used under two templates it errors out, so keep names unique.

**Available templates** (each in its own dir under `notebooks/templates/`):
- `grn_dev_v2` — current pseudobulk GRN-projection notebook (post-tuning5 datasets).
- `grn_dev` — older pseudobulk GRN-projection (tuning2 datasets).
- `grn_dev_compare_datasets` — side-by-side comparison of two pseudobulk datasets.
- `grn_dev_multi` — multi-input GRN-projection (used by HPC pipeline configs).
- `grn_dev_pc1_vs_ahba` — PC1 vs AHBA C3 comparison.
- `grn_projection` — GRN projection on cell-level (non-pseudobulk) data.
- `psychad_diagnostic_report` — PsychAD relabelling diagnostics.

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
