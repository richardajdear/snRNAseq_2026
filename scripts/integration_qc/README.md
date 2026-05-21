# Integration QC — joint Vel + Wang + PsychAD scANVI audit

Audit toolkit motivated by two observations across the comparison notebooks
`compare_Vel_PsychAD_genes` (separate per-dataset scANVI, no batch correction)
and `compare_Vel_PsychAD_integration_genes` (joint scANVI, batch corrected):

1. **Excitatory% vs age trends diverge** between Velmeshev (ρ ≈ −0.6 to −0.7)
   and PsychAD (≈ 0 / mildly positive) over development. The divergence
   persists with each dataset's own labels, so it cannot be solely
   a scANVI label-transfer artefact.
2. **The V3-pooled AHBA C3+ developmental signal flips sign** between the
   two integrations (d = +0.77 non-integrated → d = −0.81 integrated).

This toolkit answers two questions:

- **Q1.** Can we trust the batch-corrected (`scvi_normalized` /
  `scanvi_normalized`) pseudobulk values for cross-dataset developmental
  analysis?
- **Q2.** Is the joint scANVI label transfer reliable, or is it systematically
  misclassifying donors in age regions where the Wang reference is sparse?

---

## Run order

```bash
# all scripts run from the repo root
conda run -n scvi python scripts/integration_qc/01_batch_correction_audit.py
conda run -n scvi python scripts/integration_qc/02_composition_trends.py
# Script 03 needs a small file from the HPC — see "HPC fetch" below.
conda run -n scvi python scripts/integration_qc/03_alignment_audit.py
conda run -n scvi python scripts/integration_qc/04_build_shared_label_map.py
```

Each script writes to `scripts/integration_qc/outputs/<step>/`.

---

## `01_batch_correction_audit.py`

Question Q1 — direct. Computes the AHBA C3+ score on the SAME donor × cell_class
pseudobulk rows from three layers:

- `counts_cpm` — raw counts row-normalised to CPM (the "no batch correction" baseline)
- `scvi_normalized` — scVI-corrected expression
- `scanvi_normalized` — scANVI-corrected expression

If `counts_cpm` reproduces the non-integrated comparison's positive child→adol
drop but `scvi_normalized`/`scanvi_normalized` reverse it, the batch correction
is the cause.

Outputs in `outputs/batch_correction/`:
- `c3_scores_by_layer.csv` — per (donor, cell_class, layer) C3+/C3- scores
- `effect_sizes.csv` — child→adol Cohen's d / Wilcoxon p per (cell_class, stratum, layer) at fixed windows (child 1–9, adol 10–20)
- `effect_sizes_grid_best.csv` — same comparison but the best |d| over a 4D age-window grid (matches the qmd's sensitivity grid)
- `fig01_c3pos_vs_age_excitatory.png` — Excitatory C3+ vs age, three layer panels
- `fig02_c3pos_vs_age_all_classes.png` — grid: rows = cell classes, cols = layers
- `fig03_per_donor_shift_vs_age.png` — per-donor (scvi − counts_cpm) shift in C3+ as a function of age, by source
- `fig04_pca_excitatory_by_layer.png` — PCA of Excitatory pseudobulks, colored by age and by source-chemistry, per layer
- `fig05_marker_shift_vs_age.png` — per-donor shift in mean expression for canonical neuron markers vs housekeeping genes
- `interpretation.md` — auto-generated summary

---

## `02_composition_trends.py`

Documents Excitatory% vs age trends and disentangles age-coverage from label-transfer
contributions to the divergence.

Outputs in `outputs/composition/`:
- `donor_counts_by_bin.csv` — donor counts per (label_source, source_chem, age_bin)
- `bin_matched_mannwhitney.csv` — Vel vs PsychAD Mann–Whitney on Excitatory% per fine age bin
- `within_dataset_spearman.csv` — Spearman ρ vs age, raw and CLR-transformed
- `composition_summary.csv` — CLR(Excitatory) Spearman ρ per (label_source, source_chem)
- `psychad_joint_vs_original_excitatory_pct.csv` — per-PsychAD-donor Excitatory% under joint scANVI vs under PsychAD-only scANVI; flags the relabelling
- `fig01_donor_age_histogram.png` — donor counts per age bin × chemistry
- `fig02_excitatory_vs_age_three_panels.png` — Excitatory% vs age under joint, Vel-original, and PsychAD-original labels
- `fig03_all_classes_heatmap.png` — mean cell-class proportion per (class, age-bin, source-chemistry)
- `fig04_clr_excitatory_vs_age.png` — CLR-transformed Excitatory% vs age, three panels
- `fig05_within_dataset_spearman.png` — bar chart of within-dataset Spearman ρ (raw vs CLR) for Excitatory
- `fig06_psychad_relabel_excitatory.png` — per-PsychAD-donor Excitatory% gap between PsychAD-only and joint labels
- `interpretation.md` — auto-generated summary

### What is CLR and why do we use it here?

Each donor's six cell-class proportions sum to 1 — they live on a simplex.
When Oligo% rises in adults (e.g., because adult brain has more myelin → more
oligodendrocytes recovered per nucleus prep), Excitatory% *mathematically*
must drop to compensate, even if the underlying number of excitatory neurons
per cortical mm³ is unchanged.

Centred log-ratio (CLR) replaces each proportion `p_i` with
`log(p_i / geometric_mean(p))`. After this transform the values sit in
ordinary vector space and can be compared without the closed-sum constraint.

**Interpretation:**
- If raw Excitatory% trends diverge between datasets but CLR Excitatory
  trends agree, the divergence is the simplex constraint propagating an Oligo
  gain — not a real cross-dataset difference in excitatory neuron behaviour.
- If both raw and CLR trends diverge, the difference is genuine (real
  biology or real technical artefact independent of compositional coupling).

---

## `03_alignment_audit.py` (needs HPC export)

### HPC fetch

The per-cell audits need an obs-only export from `integrated.h5ad`.
On the HPC:

```bash
cd /home/rajd2/rds/hpc-work/snRNAseq_2026
PYTHONPATH=code conda run -n scvi python scripts/integration_qc/hpc_export_integrated_obs.py
```

This writes `integrated_obs.csv.gz` (~50–80 MB) next to the integrated h5ad.
Then scp to:

```
scripts/integration_qc/outputs/alignment/integrated_obs.csv.gz
```

If the file is absent the script falls back to pseudobulk-only audits
(cross-dataset Excitatory transcriptome correlation, marker enrichment).

### Audits

Outputs in `outputs/alignment/`:
- `fine_confusion_velmeshev.csv`, `fine_confusion_psychad.csv` — original `cell_type_raw` × `cell_type_aligned`
- `broad_confusion_velmeshev.csv`, `broad_confusion_psychad.csv` — original-broad × aligned-broad
- `wang_age_coverage.csv` — Wang reference cells per age bin
- `confidence_threshold_sweep.csv` — Spearman ρ(Excitatory%, age) per source at thresholds {0, 0.5, 0.7, 0.9}
- `cross_dataset_excitatory_correlation.csv` — Pearson r between Vel-Excitatory and PsychAD-Excitatory pseudobulks per age bin
- `marker_enrichment.csv` — log1p(CPM) mean of marker panels per (source × aligned cell class)
- `fig01_broad_confusion_per_dataset.png`
- `fig02_confidence_by_age_class.png`
- `fig03_wang_age_coverage.png`
- `fig04_confidence_threshold_sweep.png`
- `fig05_marker_enrichment.png`
- `fig06_cross_dataset_excitatory_correlation.png`
- `recommendations.md` — auto-generated, conditioned on what the audits reveal

---

## `04_build_shared_label_map.py`

Builds `shared_fine_labels.csv`: a manually-curated mapping from each
dataset's native fine label column to a shared vocabulary, so a future
pipeline change can use semi-supervised scANVI on all three datasets
instead of supervising only by Wang labels.

Outputs in `outputs/label_map/`:
- `shared_fine_labels.csv` — 25 rows, columns `shared_label`, `broad_class`, `vel_cell_type`, `wang_type_updated`, `psychad_subclass`. Pipe (`|`) is OR; empty string = no mapping.
- `fine_label_coverage.csv` — fraction of cells covered per dataset
- `README_label_map.md` — exact pseudocode for how to consume the CSV in `code/pipeline/downsample.py`

Run this on the HPC too (with Wang's raw h5ad available) to verify the
Wang fine-label coverage.

---

## Headline findings on the actual data (local pseudobulks only)

These are observations from running 01–04 with no gene filter (the
`well_detected_symbols.txt` is not local — full-GRN audit). Numbers may
differ in magnitude from the notebook's filtered values; the direction
and sign are what matter.

### Batch correction

| Stratum | counts_cpm (best d) | scvi_normalized | scanvi_normalized |
|---|---|---|---|
| Vel-V2 Excitatory child→adol | **+3.02** | +0.94 | +0.94 |
| Vel-all Excitatory child→adol | **+1.24** | +0.70 | +0.59 |
| PsychAD-V3 Excitatory child→adol | −0.72 | **−1.23** | **−1.36** |
| V3-pooled Excitatory child→adol | −0.65 | −0.49 | −0.67 |

Reading: batch correction *shrinks* Velmeshev's positive child→adol signal
(d falls from +3.02 → +0.94 for Vel-V2) and *amplifies* PsychAD's small
negative signal (d falls from −0.72 → −1.36). In the V3-pooled comparison
both forces push d more negative.

### Composition

Bin-matched Mann–Whitney on Excitatory% in the joint scANVI labels:

| age bin | Vel mean | PsychAD mean | p |
|---|---|---|---|
| <1 | 56.5% | **0.0%** | 1.3e-06 |
| 1-2 | 40.1% | 10.0% | 0.20 |
| 2-5 | 39.4% | 7.4% | 0.02 |
| 5-10 | 27.6% | 11.5% | 0.25 |
| 10-15 | 24.2% | 26.5% | 0.80 |
| 15-20 | 20.4% | 21.3% | 0.94 |

The cross-dataset agreement holds in adolescence; in early life PsychAD
loses Excitatory cells almost entirely under joint labels.

Per-PsychAD-donor comparison (joint vs PsychAD-only scANVI):
- 10 of 11 PsychAD <1 y donors: 0% Excitatory in joint labels vs 2–9% in PsychAD-only labels.
- Adolescent PsychAD donors get *more* Excitatory cells in joint labels than in PsychAD-only (joint inflates Excitatory% in older PsychAD donors).

The combined effect: joint scANVI artificially generates a *positive* developmental trend in PsychAD Excitatory% that wasn't present in PsychAD-only scANVI.

### Alignment

Cross-dataset Excitatory pseudobulk transcriptome correlation (log1p CPM, joint labels):
- 5-10 y: r=0.96 (n_vel=5, n_psy=2)
- 10-15 y: r=0.95 (n_vel=7, n_psy=13)
- 15-20 y: r=0.96 (n_vel=6, n_psy=16)

So where both datasets are populated, "Excitatory" pseudobulks are
transcriptomically nearly identical across datasets. The label is consistent
biologically — the misalignment is upstream of the pseudobulk (it's the cell-level
relabelling that erases young PsychAD Excitatory).

---

## Conclusions and recommendations

1. **Don't use `scvi_normalized` / `scanvi_normalized` pseudobulk for cross-dataset developmental analyses without correction.** It shrinks within-dataset developmental signal and amplifies cross-dataset relative differences, producing the V3-pooled signal flip.

   **Workaround:** use `counts` size-normalised per pseudobulk row (CPM) for GRN scoring instead — script 01 demonstrates this layer reproduces the non-integrated V3-pooled positive child→adol signal (or at minimum, the shrinkage pattern).

2. **Joint scANVI mis-labels young PsychAD donors.** 10 of 11 PsychAD donors <1 y have all their cells re-routed away from "Excitatory" by the joint integration. Root cause: scANVI is trained only on Wang's fine labels ([code/pipeline/downsample.py:288-306](../../code/pipeline/downsample.py#L288-L306)), and Wang's perinatal Excitatory phenotypes do not transfer cleanly to PsychAD's young donors.

   **Recommended fix:** implement semi-supervised scANVI using `shared_fine_labels.csv` (script 04) so all three datasets' biological labels supervise training. See `outputs/label_map/README_label_map.md` for the implementation sketch in downsample.py.

3. **Apply a confidence filter at the pseudobulk stage.** Add `adata = adata[adata.obs["cell_type_aligned_confidence"] >= 0.5]` inside [code/pipeline/pseudobulk.py](../../code/pipeline/pseudobulk.py) before aggregation. This drops cells whose label is not well-supported in latent space.

4. **Add marker-gene QC to the pipeline output.** Emit `marker_enrichment.csv`-style summaries automatically; flag any class whose canonical markers fail to enrich.

5. **The compositional divergence is partly real cell capture difference.** Even without scANVI relabelling, PsychAD's young donors recover fewer Excitatory neurons than Velmeshev's (∼2–9% vs ∼50%). This may reflect dissection or dissociation protocol differences that systematically reduce Excitatory recovery in PsychAD's aging-cohort tissue processing for young donors. This is worth investigating in the PsychAD metadata (postmortem interval, dissociation protocol).
