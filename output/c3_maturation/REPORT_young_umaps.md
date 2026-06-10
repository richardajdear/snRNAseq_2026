# Young-donor (<5y) UMAPs: does the ExN definition hold up?

_Companion to `REPORT.md` / `REPORT_annotation.md`. Dedicated visual check of cell-type labeling in
the youngest donors, where the aging-reference labels are most suspect._ · github `main`.

## Question

In PsychAD, native labels (and `cell_type_aligned`, scANVI-trained on them) call only ~10% of <2y
cells excitatory, vs ~50% by a marker classifier (`REPORT_annotation.md`). Is the native labeling
*wrong* in young donors, and does the discrepancy make visual sense? We look at the **youngest
donors (<5y)** directly, per dataset, and ask whether the unsupervised clusters agree with the
native labels or with the marker labels — using the differentiation markers as ground truth.

## Method (be explicit)

- **Cells:** ALL cells from donors aged **<5y**, per dataset, taken from the integrated objects.
  PsychAD = `PsychAD_noage_tuning5`; Velmeshev-V3 = `Vel_prepost_noage_tuning5` filtered to
  `chemistry == V3` (Herring + Ramos sub-sources; excludes V2-U01).
- **UMAP representation:** recomputed (scanpy `neighbors`+`umap`, default params, n_neighbors=15) on
  the subset's **`X_scVI` latent** — the *unsupervised*, batch-corrected scVI embedding from each
  dataset's own integration run. We deliberately use scVI, **not** scANVI, because scANVI is trained
  on the native labels we are scrutinising; scVI lets the data cluster independently of those labels.
  _(The integrated objects also carry `X_scANVI`, `X_pca_*`, and precomputed `X_umap_*`.)_
- **Labels shown:** native broad (`cell_class`), native fine (`cell_type_raw`/`subclass` for PsychAD,
  `Cell_Type` for Velmeshev), and our **marker** label computed here from raw counts with the
  `code/annotation_by_markers.py` logic (InN if max(GAD1,GAD2,SLC32A1)≥10; ExN_mature if RBFOX3≥1;
  ExN_immature if DCX≥1 & RBFOX3<1; glia by AQP4/GFAP/MBP/PLP1/CX3CR1/P2RY12/PDGFRA≥1).
- **Marker genes (log1p CPM):** differentiation axis — SOX2 (progenitor), MKI67 (cycling), DCX /
  STMN2 (immature/migrating neuron), NEUROD6 (neuronal diff.), RBFOX3 (mature neuron); ExN identity —
  SLC17A7, SATB2; InN identity — GAD1, GAD2, SLC32A1, DLX2; glia — AQP4, PDGFRA, PLP1, CSF1R.

## PsychAD (<5y)

> _Figure `s09_young_umap_psychad.png` and native×marker crosstab — pending job 30341254._

Key things to read off:
- Do the SLC17A7+/SATB2+ islands (true ExN) get labeled EN by native, or scattered into glia/IN?
- Where do DCX+/STMN2+ (immature neurons) sit, and what does native call them? (the hypothesised
  late-maturing population)
- Is there a SOX2+/MKI67+ progenitor pole, and does it explain any "Unknown"/glia calls?

## Velmeshev-V3 (<5y)

> _Figure `s09_young_umap_velmeshev_v3.png` and crosstab — pending job 30341254._

Velmeshev's native labels come from a **developmental** atlas (not an aging reference), so this is
the positive control: if native↔marker agree well here but disagree in PsychAD, that pinpoints the
problem as PsychAD's aging-reference labeling specifically.

## Verdict

> _To be written from the figures: does the EN-% discrepancy make visual sense, which labeling
> tracks the marker-defined neuron territory, and what should the within-EN ExN definition be for
> each dataset._
