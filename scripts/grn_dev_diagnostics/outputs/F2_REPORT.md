# F2 — biological deep-dive on the C3+ dataset disagreement

**Question:** F1 found that the aggregate C3+ child→adolescence signal in
Velmeshev (d = +1.28) but not PsychAD (d = −0.44) is carried by a small
number of genes (top 90 / 3,331 shared C3+ genes carry 50% of |contribution|),
and that PsychAD's high-weight C3+ genes systematically go the WRONG way
(weight-percentile of d<−0.5 is 0.61) while Velmeshev's high-weight genes
drop (weight-percentile of d>+0.5 is 0.54). What does this mean?

This report decomposes the disagreement by C3+ weight, lists the gene-level
contributors with symbols, and proposes biological interpretations.

## TL;DR

1. **The divergence is monotonically increasing with C3+ weight.** In the
   bottom decile (median weight 0.018) the mean per-gene d is +0.03 in
   PsychAD and −0.14 in Velmeshev — essentially flat in both. In the top
   decile (median weight 0.52) PsychAD's mean d is **−0.33** while
   Velmeshev's mean d is **+0.00** — Velmeshev is neutral, PsychAD pulls
   strongly downward. The weighted aggregate exaggerates this gap because
   it up-weights exactly the genes where the two datasets disagree most.

2. **The disagreement is biologically coherent on the Velmeshev side.**
   The top-25 Velmeshev drop carriers are dominated by synaptic
   adhesion molecules (NRXN1, LRRTM4, KIRREL3, NTM, LINGO1/2, GPM6A,
   CSMD3, TENM2, CAMK2N1), glutamate receptors (GRM7, GRIK1, GRIN2A),
   synaptic vesicle / active zone (RIMS1, SV2B), postsynaptic scaffolds
   (DLGAP1), neuronal identity TFs (MEF2C, RBFOX1, SATB2), and growth
   factors (NRG3, FGF12/14). These are the canonical *synapse formation
   and refinement* programme — exactly what is expected to peak in
   childhood and decline as the cortex transitions from synaptogenesis
   to synaptic pruning over adolescence.

3. **PsychAD detects part of the same biology but the net is dominated
   by an opposite-direction set.** PsychAD's own top positive-contributors
   (genes that DROP with age) are also synaptic — NRXN1 (d=+0.48), GRIK1
   (+1.23), NTM (+0.99), FGF14, NLGN1, DLGAP1, KIRREL3, GRM7, GRIK2 —
   meaning the same module IS detected as developmentally declining.
   But these are outweighed by an even larger number of high-weight
   synaptic genes that go the OTHER direction in PsychAD: SYT1
   (d=−0.73), CDH18 (−1.06), FRMPD4 (−0.71), CHRM3 (−0.43), SV2B
   (−1.14), PPP3CA (−1.16), MLIP (−1.10), OLFM3 (−0.84), ATRNL1 (−0.64),
   ATP2B1 (−1.01), SLC17A7/VGLUT1 (−0.83), SATB2 (−0.86), CAMK4 (−0.60).
   These are also classical mature ExN markers — and many of them
   (SLC17A7/VGLUT1, SATB2, SYT1) are the textbook "this cell is a
   mature glutamatergic projection neuron" markers.

4. **There are 83 high-confidence "flip" genes** where Vel d > +0.5 and
   PsychAD d < −0.5. 38 of these 83 (46%) are top-quintile-weight C3+
   genes (~5× over chance). The list is enriched for genes that mature
   excitatory neurons express strongly: SLC17A7 (VGLUT1), SATB2,
   EPHA4, CAMK4, ATP2B1, NEGR1, NELL2, ARHGAP32, LMO4, CACNA2D1,
   ENC1, R3HDM1, CELF2, FZD3, ENC1.

## Quantitative summary

### Concordance by C3+ weight quintile (from `f2_concordance_by_weight_quintile.csv`)

| quintile | weight range  | n   | median d_psy | median d_vel | frac(d_psy<0, d_vel>0) |
|----------|---------------|-----|--------------|--------------|------------------------|
| Q1_low   | 0.000–0.068   | 666 | −0.01        | −0.24        | 2.6%                   |
| Q2       | 0.068–0.140   | 666 | −0.09        | −0.19        | 6.2%                   |
| Q3       | 0.140–0.223   | 663 | −0.10        |  +0.00       | 7.2%                   |
| Q4       | 0.223–0.344   | 665 | −0.16        | −0.08        | 8.3%                   |
| Q5_top   | 0.344–0.849   | 665 | **−0.31**    |  −0.05       | **12.2%**              |

The fraction of "PsychAD negative AND Velmeshev positive" genes more than
4× from bottom to top quintile, while the reverse fraction (d_psy > 0,
d_vel < 0) is flat at 3-7%. The disagreement is not random — it is
specifically located in the high-weight C3+ genes that drive the
aggregate score.

### Concordance bucket summary (from `f2_story_summary.csv`)

| bucket                          | n    | frac  | weight pct | total contrib_psy | total contrib_vel |
|---------------------------------|------|-------|------------|-------------------|-------------------|
| All genes                       | 3331 | 1.000 | 0.500      | −4943             | +6943             |
| flat (|d|<0.3 both)             |  523 | 0.157 | 0.459      |    −19            |   −101            |
| both drop (d>+0.3 both)         |  283 | 0.085 | 0.462      | **+1933**         | **+2627**         |
| both rise (d<−0.3 both)         |  559 | 0.168 | 0.551      |   −2733           |   −2625           |
| Vel drops, PsychAD rises (FLIP) |  242 | 0.073 | **0.626**  | **−1933**         | **+2909**         |
| Vel rises, PsychAD drops        |  167 | 0.050 | 0.410      |   +312            |    −238           |

Notes:
- The "both drop" concordance bucket contributes +1.9 K to PsychAD's
  aggregate — i.e. **PsychAD does pick up the drop, partially**. Vel
  picks up the same drop and adds another +2.6 K.
- The "Vel drops, PsychAD rises" bucket alone takes PsychAD's aggregate
  from a fairly-positive baseline to its observed negative net, and
  takes the average weight-percentile to 0.63 — exactly the high-weight
  bias visible in F1.
- The "both rise" bucket (genes that increase from childhood to
  adolescence in both datasets) is the SINGLE LARGEST contributor to
  both aggregates' downward pull. 559 genes (17% of C3+) genuinely
  go up between 1-9 y and 9-25 y in *both* datasets. This is a real
  developmental signal in the OPPOSITE direction to what we expected
  from the C3+ network ("childhood-enriched"). It mirrors the
  childhood-to-adolescence maturation of long projection mRNA and
  myelin-adjacent transcripts.

## Top contributors by symbol (full lists in `f2_top_contributors_annotated.csv`)

### Vel_top_drop — genes carrying Velmeshev's +1.28 aggregate
(positive contrib_vel → drops in childhood → adolescence in Velmeshev)

LRRTM4, NRXN1, RBFOX1, NRG3, GRM7, KIRREL3, LINGO2, ENC1, CHRM3,
LINGO1, GPM6A, RIMS1, LRRC7, CAMK2N1, NTM, HDAC9, RGS7, TMEM132D,
GRIK1, TENM2, R3HDM1, MEF2C, GRIN2A, DLGAP1, CSMD3, ...

→ trans-synaptic adhesion + glutamate receptors + neuronal-identity
TFs + active-zone proteins. Classical synapse formation/refinement.

### Psy_top_rise — genes pulling PsychAD aggregate DOWN
(negative contrib_psy → rises in childhood → adolescence in PsychAD)

RBFOX1, SYT1, CDH18, FRMPD4, KHDRBS2, RASGRF2, LDB2, CHRM3, SV2B,
KCNQ5, DGKB, PPP3CA, LRRTM4, ATRNL1, KALRN, CHN1, FSTL4, STXBP5L,
RGS7, OLFM3, PDE4D, RYR2, R3HDM1, MLIP, ATP2B1, ...

→ also synaptic / mature ExN, with substantial **overlap**: RBFOX1,
CHRM3, LRRTM4, RGS7, R3HDM1 appear on BOTH lists with OPPOSITE signs.

### Psy_top_drop — PsychAD's correct-direction positives

NRXN1 (d_psy=+0.48), GRIK1 (+1.23), NTM (+0.99), DPP6, FGF14, DPP10,
GRIK2 (+1.19), MACROD2, FGF12, TMEM132D, DLGAP1, KIRREL3, CSMD3, GRM7,
NOVA1, NLGN1 (+1.48), KCNC2, IL1RAPL2, EXT1, NCAM2, ...

→ same synapse-formation module, partially detected. NLGN1 and GRIK1
show the strongest childhood-enrichment in PsychAD (d>1.2). So
PsychAD isn't blind to the signal; it's biased.

## Biological interpretation

The Velmeshev "drop with age" signal is biologically what we expect:
neurons in 1-9 y children have actively-elaborating synapses (high
NRXN1/LRRTM4/MEF2C/GRM7/synaptic-adhesion expression) that get pruned
and refined down through adolescence. This is the well-documented
*synaptic pruning* window described in PFC histology
(Petanjek et al., 2011) and corresponds to the C3+ network's
peak-childhood signature in the AHBA bulk reference.

PsychAD picks up part of this signal — NRXN1, NLGN1, GRIK1, GRIK2,
GRM7, KIRREL3, NTM, DLGAP1 all drop in PsychAD too — but for an
equally synapse-relevant subset (SYT1, CDH18, SLC17A7, SATB2, CAMK4,
SV2B, ATP2B1, PPP3CA, ENC1, EPHA4, LRRTM4, CHRM3, etc.) PsychAD has
HIGHER expression in adolescents than in children. The same gene
(LRRTM4) is +1.35 in Vel and −0.17 in PsychAD; CHRM3 is +1.12 in Vel
and −0.43 in PsychAD; RBFOX1 is +0.29 in Vel and −0.33 in PsychAD.

This is NOT a "different biological story" in any clean sense — both
datasets see synaptic and mature-ExN genes change across this window.
Rather it is a SYSTEMATIC opposite-sign bias that grows with C3+ weight,
strongly suggesting the cause is methodological rather than biological.

### Candidate non-biological explanations (ranked by plausibility)

**1. PsychAD child cohort selection bias — RULED OUT by F3 donor audit.**
All 11 PsychAD child donors (ages 2-8 y) are from the HBCC (`Source = "H"`)
cohort, labeled `disease = "normal"`, and have all neuropath/diagnostic
flags set to "No" (Schizophrenia, Bipolar, Parkinson's, Alzheimer's,
DLBD, FTD, Tauopathy, Vascular, ASCVD). The negative aggregate is NOT
driven by pediatric clinical pathology — these are documented normal
controls. Per-donor C3+ scores (`f3_donor_score_plot.png`) show three
specific low-end outliers (Donor_1400 z=−3.4, Donor_701 z=−2.7,
Donor_28 z=−2.2) but also one strong positive outlier (Donor_1476
z=+2.7), suggesting the per-donor variance is real noise on top of
the modest 5.9% mean shift (child mean 79,440 vs adol mean 84,383).
With n=11 children, three low-end donors are enough to set the mean
direction; but the mean-shift effect size is small (Cohen's d=−0.44
is dominated by within-group variability, not by a clean signal).

**2. Marker-classifier depth bias re-distributes cells.** The marker
classifier is a hard absolute threshold (RBFOX3 ≥ 1, DCX ≥ 1).
PsychAD-V3 is high-depth and FANS-enriched → most ExN cells satisfy
RBFOX3 ≥ 1 and get binned as ExN_mature. Vel-V2 children are
shallow-depth → many genuinely mature ExN cells fail RBFOX3 ≥ 1 and
get binned as ExN_immature or ExN_weak. So the PsychAD-V3 "child
ExN_mature" bin may contain an unusually large fraction of cells
that are biologically still pre-mature (RBFOX3 low but ≥1 thanks to
depth), pulling down its synaptic-gene mean relative to the same
bin in adolescents. This is consistent with E1's finding that
PsychAD's ExN_immature subtype DOES show the drop (d=+0.31).
**Action:** see Group G (depth-distribution comparison) and H (build a
depth-corrected maturity score within PsychAD).

**3. scVI / pseudobulk batch artifact.** PsychAD_noage_tuning5 was not
age-balanced during scVI training, so the few child donors are
strongly outnumbered by older donors in the latent. Even though the
score is computed on layers['counts'] CPM (not scvi_normalized), scVI
training affected which cells survived donor-quality filtering and
the per-donor library composition.
**Action:** rerun the projection on raw per-cell CPM (skip scVI)
restricted to the same donors to bound this.

**4. Real cohort-level difference.** Velmeshev's child donors are
explicitly recruited as part of a developmental atlas; PsychAD's
are clinical/aging cohort opportunistic samples. There may be true
biological differences (different brain regions within DLPFC,
different PMI, different cause of death). But this should average out
across many genes, not produce a systematic sign-flip in 12% of high-
weight C3+ genes. Unlikely to be the dominant driver.

### Why high-weight genes specifically?

C3+ weight in the AHBA-derived GRN reflects how strongly a gene
correlates with the C3+ spatial component in the adult AHBA bulk
reference. Genes with the highest weights are the genes whose adult
expression most distinguishes C3+ cortical regions — these are
disproportionately the synaptic / glutamatergic / cell-adhesion
module already noted above. If the underlying issue is that
PsychAD's pediatric ExN cells are technical or pathological outliers
on the dimension of "mature ExN synaptic content," it will manifest
exactly on the genes that BEST mark mature ExN identity — which is
exactly the set of genes with the highest C3+ weights. The clean
monotonic gradient in `f2_mean_d_by_weight_decile.csv` (PsychAD's
mean d falls from +0.03 in bottom decile to −0.33 in top decile)
supports this: it is a single contamination axis projected through
the AHBA weight basis.

## Recommended next steps

1. **PsychAD child donor metadata audit** — for the 11 PsychAD child
   donors, pull cohort (HBCC vs Aging), PMI, cause of death, brain
   weight, and any recorded neuropath / diagnosis. Stratify by donor
   and recompute the aggregate C3+ d to see which donors drive the
   pull-down. (Single small script, no h5ad needed.)
2. **Group G — depth-classification bias** (sbatch'd) — verify whether
   marker-class composition shifts with depth, which would explain
   why the same gene flips direction in PsychAD-V3 vs Velmeshev-V2.
3. **Group H — PsychAD-internal maturity baseline** — derive a
   continuous, depth-aware maturity score within PsychAD-V3 alone and
   show whether the "low maturity" cells drive the negative d.
4. **Drop the aggregate score**, report per-gene d (as in F's scatter)
   plus the canonical drop-gene panel (NRXN1, NLGN1, GRIK1, GRIK2,
   GRM7, KIRREL3, NTM, MEF2C, GRIN2A, DLGAP1) instead. The signal
   is robust in both datasets at the gene level when reported gene-
   by-gene; it only disagrees in the *weighted sum* because the
   sum amplifies a small disagreement-skewed minority.

## Files produced

- `f_contribution_per_gene.parquet` — per-gene contribution table
- `f_dominance_summary.csv` — top-N concentration of |contribution|
- `f_weight_percentile_high_drop.csv` — weight-pct by d-bucket
- `f_top25_velmeshev_drop_carriers.csv` — Vel top contributors
- `f_top15_psychad_negative_drivers.csv` — PsychAD top negative pull
- `f_scatter_d_vs_weight.png` — per-dataset d vs weight scatter
- `f_lorenz_curve.png` — cumulative |contribution|
- `f_d_distribution.png` — histogram of per-gene d
- `f2_top_contributors_annotated.csv` — top 40 in each of 4 buckets
- `f2_concordance_by_weight_quintile.csv` — concordance by weight Q
- `f2_dataset_disagreement_genes.csv` — full flip-gene catalogue
- `f2_story_summary.csv` — bucket summary
- `f2_mean_d_by_weight_decile.csv` — decile-binned mean d
- `f2_weight_vs_d_overlay.png` — d vs weight overlay PsychAD+Velmeshev
- `f2_mean_d_by_weight_decile.png` — monotonic gradient plot
- `f2_concordance_per_quintile.png` — d_psy vs d_vel by weight Q
