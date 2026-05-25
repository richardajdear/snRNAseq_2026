# PsychAD relabel diagnostic — summary

## Composition check results (from composition_check.py)

| source-chemistry | age_bin | age_run EN% | no_age_run EN% | Δ pp |
|------------------|---------|------------:|---------------:|-----:|
| PSYCHAD-V3 | <1y    |  5.46 |  5.16 | -0.30 |
| PSYCHAD-V3 | 1-5y   | 11.90 | 11.90 |  0.00 |
| PSYCHAD-V3 | 5-18y  | 18.13 | 18.16 | +0.03 |
| PSYCHAD-V3 | 18-30y | 22.68 | 22.70 | +0.02 |
| PSYCHAD-V3 | 30-50y | 31.43 | 31.43 |  0.00 |

**Key finding:** Removing the `age_years` covariate had essentially no effect.
The EN% gradient monotonically *increases* with age — opposite of biology —
and is reproduced almost identically by two independently-trained models.
→ H1 (age covariate erased developmental signal) is **rejected**.

## Hypotheses

| # | Hypothesis | Status | Key output |
|---|-----------|--------|-----------|
| H1 | age covariate erased latent dev signal | REJECTED | S1 test, C4 latent movement |
| H2 | PsychAD-adult labels dominate supervision | see D3/D5 | predicted_vs_raw, confusion |
| H5 | Vel IN removal → IN class is PsychAD-only | see D5 | confusion EN→IN cross |
| H6 | PsychAD 5-18y dev cells labelled adult-IN | see D5 | confusion diagonal |
| H7 | PsychAD subclass is IN-heavy by default | see D3 | raw column marginals |

## Output files

**Per-run (`age_run/` and `no_age_run/`):**
- `psychad_predictions_by_age_bin.csv` — D1: top predicted labels per PsychAD age bin
- `psychad_confidence_summary.csv` — D2: confidence quartiles
- `psychad_under1y_predicted_vs_raw_crosstab.csv` — D3: scANVI pred vs raw subclass
- `marker_means.csv` — D4: EN/IN marker expression per predicted class
- `scanvi_confusion_on_labelled.csv` — D5: classifier accuracy on training labels
- `latent_neighbor_composition.csv` — D6: are PsychAD <1y neighbours PsychAD adults?
- `psychad_under1y_confidence_hist.png` — D7: confidence distributions
- `umap_psychad_under1y.png` — D9: UMAP with PsychAD <1y highlighted

**Cross-run (`comparison/`):**
- `psychad_under1y_prediction_changes.csv` — C1: which cells changed prediction?
- `psychad_under1y_confidence_delta.csv` — C2: confidence deltas
- `latent_movement_distribution.csv` — C4: L2 shift in X_scANVI between runs

## Next step

Once `errors.log` is clean and D3/D4/D5/D6 are reviewed, choose v4 approach:
- **S2**: drop ALL PsychAD labels (Wang+Vel as sole reference) — addresses H2/H5/H6
- **S2b**: keep only PsychAD non-neuronal labels (Micro/Endo/Astro/Oligo/OPC)
- **S3**: scArches/scPoli reference-query architecture