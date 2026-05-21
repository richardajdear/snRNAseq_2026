# 03 — Alignment audit · recommendations

## Cross-dataset Excitatory pseudobulk correlation per age bin

age_bin  n_vel  n_psy  pearson_log_cpm
   5-10      5      2         0.959037
  10-15      7     13         0.945102
  15-20      6     16         0.955775

## Recommended actions

1. If broad confusion shows >5% off-diagonal mass for Vel/PsychAD original Excitatory → aligned non-Excitatory: implement semi-supervised redesign using `shared_fine_labels.csv` (Script 04) so the model is supervised by all three datasets, not Wang alone.
2. If `fig02_confidence_by_age_class.png` shows low confidence in PsychAD young donors: apply a confidence threshold (e.g., 0.5) at the pseudobulk stage. Implementation: add `adata = adata[adata.obs["cell_type_aligned_confidence"] >= 0.5]` before pseudobulk aggregation in `code/pipeline/pseudobulk.py`.
3. If `fig03_wang_age_coverage.png` shows gaps (e.g., 15+ y): augment Wang reference with a curated subset of adult PsychAD cells (use their original `cell_class_original` as ground truth).
4. If `fig04_confidence_threshold_sweep.png` shows trends converge as threshold rises: the alignment quality is part of the cross-dataset divergence. Combine #1 + #2 above.
5. If `fig05_marker_enrichment.png` shows neuron markers under-expressed in PsychAD Excitatory vs Velmeshev Excitatory: the label "Excitatory" is not capturing the same biology in both datasets — fully implement #1.
6. If `fig06_cross_dataset_excitatory_correlation.png` shows low correlation in young bins: Excitatory mean expression differs across datasets at those ages, which is consistent with the per-cell relabel observation in script 02 (joint scANVI removes young PsychAD Excitatory cells).
