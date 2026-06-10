# PsychAD cell-type annotation: provenance, comparison, and recommendation

_Companion to `REPORT.md`. Addresses: what exactly is the "marker-based" annotation, where did it
come from, and which labels should the within-EN analysis use?_ · pushed to github `main`.

## Why this matters

PsychAD was built for dementia/aging studies, and its **native** cell-type labels come from an
**aging reference**. If young-donor neurons are mislabeled, any "within-EN" developmental analysis
could be measuring a labeling artefact rather than biology. There are **three** labelings in play,
with very different provenance — and an earlier draft of `REPORT.md` conflated two of them. This
document sets the record straight with code-level provenance and UMAP evidence.

## The three labelings (provenance)

| label | where it comes from | depends on aging reference? | depends on scVI/scANVI embedding? |
|---|---|---|---|
| **native** `cell_type_raw` (= PsychAD `subclass`) | PsychAD's own published annotation (aging/dementia reference) | **Yes** (it *is* the reference) | No |
| **`cell_type_aligned`** | scANVI label transfer, **trained on `cell_type_raw`** (config `PsychAD_noage_tuning5.yaml`: `scanvi_label_transfer.label_column: cell_type_raw`), run on the scVI latent | **Yes** — inherits the native reference as training labels | Yes (scVI latent + scANVI) |
| **`marker_annotation`** | `code/annotation_by_markers.py` — a hard-threshold marker classifier on **raw counts**, read directly from the raw per-dataset h5ads via h5py | **No** | **No** |

### Exact logic of `marker_annotation` (`code/annotation_by_markers.py`)

Per cell, on **raw integer counts** (PsychAD: HBCC/Aging h5ads; the `manual_annotations.parquet`
sidecar is its cached output), applied in this order:

1. **InN** if `max(GAD1, GAD2, SLC32A1) ≥ 10` — the deliberately high "GAD>10" cutoff (reduces
   ambient-RNA false positives).
2. **ExN** if `RBFOX3 ≥ 1` or `DCX ≥ 1` (neuron-exclusive; overrides glial signal as ambient):
   `DCX`-only → **ExN_immature**; `RBFOX3` → **ExN_mature**.
3. **Glia** (only if no neuronal marker): Astro (`AQP4`/`GFAP`) > Oligo (`MBP`/`PLP1`) >
   Micro (`CX3CR1`/`P2RY12`) > OPC (`PDGFRA`), each at `≥ 1` count.
4. **ExN_weak** if `RBFOX1 ≥ 1` and no glia.
5. else **Unknown**.

So `marker_annotation` is genuinely **independent** of both the aging reference and the scANVI
embedding — but it is a crude per-cell classifier with one important caveat: it is **dropout-/
depth-sensitive**. A real neuron whose single-cell `RBFOX3`/`DCX` counts happen to be 0 (common at
low depth, e.g. V2 or shallow cells) is *not* called ExN — it falls to glia or **Unknown**. So
`marker_annotation` EN fractions are a **lower bound** that degrades with sequencing depth.

## What the analysis actually used

- **Steps 1–2 (within-EN trajectory & per-gene program)** used **`cell_type_aligned`** (EN subtype
  labels: EN_L2_3_IT, …). These are **reference-derived** (scANVI trained on the native subclass) —
  *not* reference-independent. The earlier `REPORT.md` Appendix wording ("independent of the aging
  reference") was correct for `marker_annotation` but was wrongly implied for the subtype labels;
  this is now corrected.
- **The composition sanity check (A1)** used **`marker_annotation`** (reference-independent) — this
  is the appropriate, conservative check that young EN fractions are not absurd.
- **The purity check (A2)** showed the `cell_type_aligned` EN pseudobulks express EN markers high /
  IN+glia markers low at all ages — i.e. whatever its subtype-level accuracy, `cell_type_aligned`
  EN is genuinely *neuronal*.

**Why the conclusion is robust to this.** The deflationary result is also reproduced by the
independent `grn_dev_diagnostics` line using **native** `cell_class == Excitatory` + raw counts (no
scANVI), and that line additionally showed an alternative kNN-vote "principled ExN" definition is
InN-contaminated (so it was *not* used). The within-EN C3 result does not depend on which EN
definition is used.

## UMAP comparison _(job 30337068)_

UMAP computed on the **scVI/scANVI latent space** (`obsm['X_scANVI']` if present, else `X_scVI`) —
the same embedding used for integration and label transfer — on an 80k-cell subsample
(young-oversampled so <10y cells are visible). Coloured by native / aligned / marker labelings, by
age, and by **ground-truth marker genes** (SLC17A7, GAD1, AQP4, RBFOX3) computed as log1p(CPM) from
raw counts.

> **Pending** — figures `s07_umap_annotations.png`, `s07_en_fraction_by_labeling.png`, and the
> young-donor native×marker confusion table will be embedded here when job 30337068 completes.
> Key question they resolve: does the native/aligned labeling actually under-call EN in young
> donors (the "~5% EN" claim), and where do those cells sit relative to the SLC17A7+ neuron island?

## Recommendation _(to be finalised with UMAP evidence)_

- For **composition / EN-membership**, prefer the reference-independent **`marker_annotation`**
  (with the depth caveat in mind — read it as a depth-attenuated lower bound on EN fraction).
- For **EN subtype / layer** resolution, `cell_type_aligned` is the only option, but treat
  young-donor *subtype* assignments cautiously (validated as EN-pure, not as correctly-layered).
- The safest within-EN analyses condition on EN *membership* via the reference-independent marker
  call and use subtype only as a covariate — which is the direction the main report's robustness
  work is heading.
