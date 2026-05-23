# scANVI v3 Task List

- [x] Phase A — reference/shared_fine_labels.csv refinement
  - [x] A1: Remove CCK|NOS|SV2C|INT|Interneurons from IN_CALB2 vel_cell_type
  - [x] A2: Delete Glial_progenitor row
  - [x] A3: Collapse EN layer subtypes into EN_L2_3, EN_L4, EN_L5, EN_L6
- [x] Phase B — downsample.py: --unlabel_below_age
- [x] Phase C — age_years covariate (YAML config only)
- [x] Phase D — downsample.py: --max_age + run_pipeline.py plumbing
- [x] Phase E — n_samples_per_label: config.py + train.py + run_pipeline.py plumbing
- [x] Phase E2a — n_latent: 30 → 50 (YAML config only)
- [x] Phase E2b — dispersion: gene-batch (config.py + train.py)
- [x] Phase E2c — transform_batch as list (config.py type change + run_pipeline.py: drop hardcoded setdefault)
- [x] Phase E3 — Skip scvi_normalized (YAML + _is_scvi_step_complete fix in run_pipeline.py)
- [x] Phase F — composition_check.py (new module + wired into step_pseudobulk)
- [x] Phase G — semisup3_age_tuning5.yaml (new config)
- [x] Tests — updated test_production_csv + 2 new tests (unlabel_below_age, collapsed EN labels)
- [x] Tests pass — 19/19 PASSED, 0 failed
- [x] All Python files parse with no syntax errors
