# CLAUDE.md — snRNAseq_2026

Guidance for AI coding agents (Claude Code) working in this repository.

## Environment

This is a CSD3 HPC project. The actual compute environment lives in a
Singularity SIF, NOT in the login-node user environment.

- Singularity image: `/home/rajd2/rds/hpc-work/shortcake.sif`
- Conda env inside it: `shortcake_default`
- Python interpreter: `/opt/micromamba/envs/shortcake_default/bin/python3`
- Repo root: `/home/rajd2/rds/hpc-work/snRNAseq_2026`

The login-node base Python does NOT have anndata/scanpy/scvi-tools/mygene.
Always invoke through Singularity.

Invocation patterns:

```bash
# Inline (login node) — ONLY for fast (<60 s) jobs that touch small inputs.
singularity exec \
  --pwd /home/rajd2/rds/hpc-work/snRNAseq_2026 \
  --env PYTHONUNBUFFERED=1 \
  /home/rajd2/rds/hpc-work/shortcake.sif \
  micromamba run -n shortcake_default \
  /opt/micromamba/envs/shortcake_default/bin/python3 -u path/to/script.py

# SLURM (compute node) — for anything bigger, see "When to use sbatch"
cd /home/rajd2/rds/hpc-work/snRNAseq_2026
sbatch --time=00:30:00 --mem=120G \
  scripts/run_script.sh scripts/path/to/script.py
```

## When inline runs get killed on the login node

The login node enforces aggressive limits (CPU minutes, memory, total
wall time, and "background drift" — long-lived non-interactive
processes). The following will be killed *silently* (the python process
exits with no stderr, only "shell cwd was reset" appears):

1. **Any read of a multi-GB h5ad file.** The `integrated.h5ad` files
   under `Cam_snRNAseq/integrated/<config>/scvi_output/` are 60-90 GB
   sparse matrices. Even `anndata.read_h5ad(..., backed='r')` paged the
   index into memory and got killed during the PsychAD load on the
   login node.
2. **Any background `python` task left running for >2 minutes.**
   `Bash(run_in_background=True)` with a heavy script will be reaped
   even if the agent goes on to other work; the parent CLI is detached
   from the login session.
3. **Polling loops with `sleep`.** Long `sleep` + retry chains get
   flagged as idle background drift. (The harness's auto-mode also
   blocks `sleep N && ...` patterns; use `Bash(run_in_background=True)`
   or `Monitor`.)
4. **scVI/scANVI training**, **pseudobulk aggregation**, any
   scanpy/scvi pipeline step that touches `layers['counts']` for >100k
   cells.

Symptoms to recognise that you have been killed by the node, not by
your code: bash returns `exit code 0`, the output file only has the
first 1–3 `print(..., flush=True)` lines, and you never see a
Python traceback.

## When to use `sbatch`

Use `sbatch scripts/run_script.sh <script.py>` whenever the script:
- opens any `.h5ad` from `Cam_snRNAseq/integrated/*/scvi_output/`
- reads `layers['counts']`, `layers['scvi_normalized']`, or
  `layers['scanvi_normalized']` for >50k cells
- needs >30 GB RAM
- is expected to take >2 minutes
- does any scVI/scANVI training, marker annotation, or pseudobulk
  aggregation
- runs the pipeline (`code/pipeline/...`)

For inline (login-node) work it is safe to:
- read parquets, CSVs, small pseudobulk h5ads (the
  `pseudobulk_output/*by_donor.h5ad` and `*by_cell_class*.h5ad` files
  are <500 MB)
- read the AHBA reference tables under `reference/`
- compute statistics on per-gene tables already cached in
  `scripts/grn_dev_diagnostics/outputs/`
- query mygene for small (<200) symbol lookups

`scripts/run_script.sh` defaults to 30 min / 200 G on the
`icelake / vertes-sl2-cpu` partition; override with
`sbatch --time=HH:MM:SS --mem=NNG scripts/run_script.sh ...`.

Job output lands at
`/home/rajd2/rds/hpc-work/snRNAseq_2026/logs/<jobid>_run_script.{out,err}`.
Job submission is typically queued for a few minutes; use
`squeue -u $USER` to check status (PD = pending, R = running, CG =
completing).

## Codebase quick reference

Pipeline entrypoints:
- `code/pipeline/configs/<NAME>.yaml` — per-run integration config
  (source h5ads, scVI hyperparams, pseudobulk groups, notebook
  inputs). Each config produces a directory under
  `/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/<NAME>/`.
- `code/pipeline/slurm/submit_pipeline.sh <config.yaml>` runs the
  full pipeline (downsample → scvi → scanvi → pseudobulk → notebook).
- `code/annotation_by_markers.py` — marker-based cell type annotation
  (RBFOX3/DCX/RBFOX1/GAD1/GAD2/SLC32A1; raw count thresholds).
  Reads source h5ads directly via h5py.
- `code/regulons.py::get_ahba_GRN` — AHBA C3+/C3-/etc. weighted GRN
  loading. Weights live in `reference/ahba_dme_hcp_top8kgenes_weights.csv`.
- `code/hvg_investigation.py::setup_grn` — wraps the GRN + symbol→ensembl
  mapping for projection.

Pseudobulk h5ads (per integration config):
- `all_cells_by_donor.h5ad` — donor × all_cells
- `by_cell_class.h5ad` — (donor × cell_class) using inherited labels
- `by_cell_class_manual.h5ad` — (donor × marker_annotation) using the
  marker-based classifier
- `ExN_manual_by_donor.h5ad` — donor × (ExN_mature ∪ ExN_immature ∪
  ExN_weak); the input to grn_dev_multi.md
- All pseudobulks store raw integer sums in `layers['counts']` plus
  scVI/scANVI means in additional layers; CPM-normalise from
  `layers['counts']` for projection.

Active investigations:
- `scripts/grn_dev_diagnostics/` — diagnostic battery for the
  PsychAD-vs-Velmeshev AHBA C3+ developmental disagreement.
  **Read `outputs/FINAL_REPORT.md` in full before touching this analysis.**
  Report reframed 2026-06-08 around the biological question: "does C3+
  (externally derived = positive tail of AHBA 3rd spatial component, Dear et
  al. 2024 NatNeuro; NOT data-driven, so aligning it with a cell-maturity
  axis is cross-modal validation, NOT circular) represent childhood+adolescent
  synaptic maturation beyond birth differentiation?" Three evidence strands:
  (1) STRONG — C3+ rises with single-cell maturity (ρ+0.4..0.8); (2) MODERATE
  — childhood→adol within-state decline, V3-pair fuzzy d +0.46; (3) SUGGESTIVE
  — childhood-elevated genes enriched for C3+ membership (w_age_axis.py: Vel-V3
  hypergeom p=3e-12 clean, Vel-V2 p=3e-24 but confounded, PsychAD p=0.47 ABSENT;
  enrichment is membership-not-weight, ρ(C3+weight,age_d)≈0).
  - W findings (job 30241257): child vs adol ExN ARE separable in scVI latent
    (grouped-CV AUC 0.93 Vel-V2 / 0.67 Vel-V3 / 0.62 PsychAD; max|ρ(latent,age)|
    0.5–0.83) — a real data-driven age axis, strong in Velmeshev, WEAK in
    PsychAD. The 9-gene module's own child→adol d is FLAT in deep cohorts
    (PsychAD −0.10, V3 −0.05) = it's an EARLY-differentiation index saturated
    before our window; right tool to REMOVE the maturity confound, WRONG tool
    to MEASURE late maturation. Vel-V2 has a genome-wide child-shift
    (background age_d −0.34) → its big drop is partly technical; DEMOTED from
    "the big lead". Decisive next step = a DATA-DRIVEN late-maturation axis
    (supervised age direction from the §4.1 classifier, then pseudotime).
  - Crucial resolved results (2026-06-07):
  - The AHBA C3+ network's child→adolescent **drop is a neuronal-MATURITY
    effect**: it lives in immature ExN and fades as they mature. The naive
    all-ExN aggregate disagreed across cohorts (PsychAD-V3 −0.18 vs Vel-V3
    +0.58) because PsychAD's FANS NeuN+ prep under-samples shallow/immature
    nuclei, masking the childhood peak.
  - Fix = stratify by a 9-marker mature-module (mean log1p-CP10k of
    NEUROD2/SATB2/BCL11B/MEF2C/NEFM/NEFH/SYT1/SNAP25/MAP2) and take the
    **least-mature quintile (q0)**. No depth window needed.
  - Combined result: **V3-pair fuzzy d = +0.46** (PsychAD-V3 +0.49,
    Vel-V3 +0.49; Vel-V2 +2.56 = shallow-library amplification).
    Donor-robust (LOO +0.36…+0.64), pan-layer.
  - **Depth was a proxy for maturity** (Spearman ρ(UMI,maturity)=+0.65);
    the earlier depth-window result (+0.32) is superseded. Use maturity,
    not a depth window.
  - Nuance (report §3.5): the drop is a WITHIN-state age decline, not
    compositional — cross-sectionally C3+ RISES with maturity (ρ≈+0.4..+0.8),
    so maturation opposes the drop; stratifying by maturity reveals it.
    Childhood elevation is cohort-robust only in the least-mature decile
    (mature deciles are depth-confounded), which is why q0 is used.
  - Pitfall: a *two-gene* normalised maturity ratio (RBFOX3−DCX) is
    depth-confounded and gave a false-null; use a multi-marker MEAN module
    or a low-detection count. Take the EXTREME q0, not an above/below-median
    split (median dilutes to +0.19).
  - Next step scoped in §5/Appendix D: a DATA-DRIVEN late-maturation axis —
    first the supervised age direction from the §4.1 classifier (project C3+
    onto it), then a scVI-latent **pseudotime** maturity axis (watch for scVI
    over-correcting maturity into "batch"; re-embed ExN-only; marker-anchored
    DPT root). The 9-gene module is early-diff and the WRONG axis for this.
  - Join key: `outputs/r_per_cell_cache_v4.parquet` carries `cell_key`,
    markers, `mature_module`, `layer`, `per_cell_c3` for all 95,605 cells.
  - Companion derivations: `R_REPORT.md` (maturity), `J/K/L/F2_REPORT.md`.

## Memory & cache locations

- Per-conversation memory: `/home/rajd2/.claude/projects/-home-rajd2-rds/memory/`
- C3+ Ensembl mapping cache: `scripts/grn_dev_diagnostics/outputs/ahba_c3plus_ensembl.parquet`
- Per-gene Cohen's d tables (PsychAD vs Velmeshev) cached at
  `scripts/grn_dev_diagnostics/outputs/c_per_gene_*.parquet`
- All diagnostic plots and CSVs land in
  `scripts/grn_dev_diagnostics/outputs/`

## Conventions

- Cohen's d sign: positive = childhood > adolescence. Pass
  `cohens_d(child, adol)` (NOT alphabetical).
- Age windows: childhood vs adolescence. Latest C3+ analyses use a FUZZY
  boundary — compute d at cut-offs {8,9,10,11,12} y and report the mean
  ("fuzzy d"); window is [1, b) child, [b, 25) adol. (Older scripts used a
  fixed [1,9)/[9,25) split; the story is identical.)
- "PsychAD" alone refers to `PsychAD_noage_tuning5`; "Velmeshev" alone
  refers to `Vel_prepost_noage_tuning5`. Other configs are listed
  explicitly.
- `transform_batch=null` is correct (by design) for single-chemistry
  configs (PsychAD is all V3); Velmeshev's config uses
  `transform_batch: VELMESHEV-V3` because it mixes V2 and V3 cells.
