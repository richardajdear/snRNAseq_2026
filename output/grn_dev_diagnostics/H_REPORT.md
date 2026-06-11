# H — within-PsychAD maturity baseline + recommendations

User asked: before applying complex algorithms (lineage tracing,
cross-dataset label transfer), can we show the immature/mature
distinction is biologically meaningful **inside PsychAD-V3 alone**?
And what's a better way to derive it that controls for the depth bias
documented in G?

## Baseline: does the existing classifier carry real maturity biology in PsychAD-V3?

**Yes — partially. Validated by three independent checks.**

### H3. Continuous depth-normalised score validates discrete bins

A continuous per-cell score
`maturity = log1p(RBFOX3 / total_UMI × 1e4) − log1p(DCX / total_UMI × 1e4)`
removes per-cell library size and gives a signed scalar (positive =
mature-leaning, negative = immature-leaning).

| marker_annotation | n_cells | mean_score | median_score | sd_score |
|-------------------|--------:|-----------:|-------------:|---------:|
| ExN_mature        | 27,132  |  **+1.18** |       +1.19  |    0.69  |
| ExN_immature      |  3,452  |  **−1.03** |       −0.98  |    0.38  |
| ExN_weak          |  1,711  |    0.00    |        0.00  |    0.00  |

→ ExN_mature cells score 2.2 SD above ExN_immature cells on the
depth-normalised mature-vs-immature axis. The classifier does what
its name says, within PsychAD-V3. (The 0.00 for ExN_weak is mechanical:
those cells have RBFOX3=0 AND DCX=0 by definition, so both CP10K
terms are 0.)

### H1. Subtype mix shifts with donor age, in the expected direction

| age (y) | ExN_immature frac | ExN_mature frac | ExN_weak frac |
|--------:|------------------:|----------------:|--------------:|
|       2 |             16.6% |          71.5%  |        11.8%  |
|       3 |         **38.2%** |          56.7%  |         5.0%  |
|       4 |             10.8% |          84.2%  |         5.1%  |
|       5 |             21.3% |          74.3%  |         4.4%  |
|       6 |             18.5% |          76.1%  |         5.4%  |
|       8 |              8.6% |          78.1%  |        13.4%  |
|      13 |             10.2% |          86.6%  |         3.2%  |
|      16 |             10.5% |          83.7%  |         5.8%  |
|      18 |              8.3% |          88.2%  |         3.5%  |
|      22 |              9.9% |          85.8%  |         4.3%  |

→ Childhood (2-8 y) ExN_immature fraction averages 19% (range 9-38%)
vs adolescence (13-24 y) average 10% (range 7-13%). The classifier
does pick up developmental maturation.

→ Age 3 (Donor_1400) is the extreme outlier with 38% ExN_immature.
This single donor, also the F3 score outlier (z = −3.4), drives a
disproportionate share of the negative aggregate.

### H4. Per-age median maturity correlates with age

- Per-cell Spearman(age, maturity_score) = **+0.048** (significant
  at p≈10⁻¹⁷ but tiny effect size; per-cell variance is dominated by
  within-donor noise, not age).
- Per-age-median Spearman = **+0.579** (p = 0.007). Across age bins,
  median per-cell maturity increases from ~0.7 in toddlers to ~1.2 by
  late adolescence (`h_donor_maturity_vs_age.png`).

→ A REAL but modest age trend exists within PsychAD-V3 alone. The
signal is partially detectable through the marker classifier.

## Caveat: the classifier ALSO carries the depth artifact within PsychAD-V3 itself

Subtype mix at matched UMI depth (PsychAD-V3 only):

| UMI bin       | ExN_immature | ExN_mature | ExN_weak |
|---------------|-------------:|-----------:|---------:|
| 1-3 k         |        14.6% |     47.0%  |   38.3%  |
| 5-8 k         |        21.5% |     69.3%  |    9.1%  |
| 12-20 k       |         9.8% |     89.7%  |    0.5%  |
| 20-100 k      |         2.2% |     97.8%  |    0.0%  |

→ The same depth-bias pattern from G operates *inside* PsychAD too:
shallow cells get routed to weak/immature, deep cells to mature.

So the within-PsychAD baseline says: the classifier carries (a) a
modest but real maturity signal, (b) a strong depth artifact, and
(c) a single outlier donor (Donor_1400) that drags the child group
substantially.

## Recommendations for a better maturity annotation

### 1. CONTINUOUS depth-normalised score (no extra data needed)

The simple `log1p(RBFOX3_CP10K) − log1p(DCX_CP10K)` score used here is
a first pass. It validates the discrete bins, removes per-cell depth
bias, and gives a usable scalar that can be sliced by quartile or
regressed against age. **Recommend: use this as a sanity overlay on
the marker_annotation output going forward, and report results both
ways (discrete and continuous).**

Cost: zero — uses the same three marker genes already extracted.

### 2. Multi-gene neuron-maturity MODULE score (small extension)

Use a panel rather than a single gene to reduce noise:

  **Mature ExN module** (positive set):
   SYT1, SNAP25, SLC17A7 (VGLUT1), RBFOX3, NEFL, NEFM, NEFH, MAPT,
   SATB2, SLC17A6, BSN, GRIN2A, GRIN2B, CAMK2A
  **Immature ExN module** (negative set):
   DCX, NEUROD2, NEUROD6, SOX11, STMN2, TBR1, EOMES, PAX6, NEUROG2,
   ELAVL3, NHLH1

Score = mean log1p CP10K of mature − mean log1p CP10K of immature.

This is the standard "module score" approach (à la `sc.tl.score_genes`).
With a panel, single-marker capture failures don't flip a cell's
classification. **Should retire the binary threshold classifier
entirely.** Pseudobulks should aggregate by maturity quartile of this
continuous score, not by marker_annotation bin.

Cost: one sbatch job per dataset to pull ~25 gene columns from the
integrated h5ads — same machinery as G.

### 3. Within-donor rank-normalised markers (eliminates cross-dataset depth bias)

For each marker, compute the cell's PERCENTILE within its own donor
(or within its own UMI-bin × donor cell). This makes the score
invariant to absolute capture efficiency — a cell in PsychAD's
"top 10% RBFOX3 expression for my donor" maps to a cell in Velmeshev's
"top 10% RBFOX3 for my donor," regardless of the 4× detection-rate
gap documented in G.

Cost: same as #2 plus a per-donor groupby.

### 4. PsychAD-internal pseudotime (sidestep marker labels entirely)

Run diffusion pseudotime on PsychAD-V3 ExN-like cells using the existing
scVI latent. The endpoint with highest age correlation defines
"maturity"; project all cells onto that axis. This bypasses marker
thresholds entirely and uses the full transcriptome via scVI.
Strength: should align with Velmeshev when applied jointly.
Weakness: requires re-loading the integrated h5ad and running
`sc.tl.diffmap` / `sc.tl.dpt` (sbatch).

### 5. Why label transfer from Wang failed (and what to do instead)

User noted that scANVI label transfer from Wang→PsychAD assigned most
PsychAD cells to inhibitory classes (likely due to ambient GAD1/GAD2
contamination — known issue with PsychAD's FANS prep). Two ways
forward that avoid this trap:

- **Reference label transfer using an EN-only reference panel.**
  Restrict the Wang reference to clearly excitatory subtypes only,
  then transfer. Cells that don't classify confidently as a specific
  EN subtype get label "ambiguous EN" rather than being forced into
  an InN class. This puts the burden on the classifier to admit
  uncertainty.
- **Use the marker classifier ONLY to filter out InN/glia, then use a
  continuous maturity score** (#2 or #3) to rank what remains.
  This is essentially what we are already doing implicitly — make it
  explicit and continuous instead of binary.

### Decision tree for the immediate path forward

For the AHBA C3+ developmental question specifically, before
attempting #4 or #5:

1. **Re-do the GRN projection using a continuous maturity-quartile
   stratification instead of `marker_annotation`.** Pseudobulk per
   donor × maturity-quartile. Check whether the high-maturity quartile
   in PsychAD now agrees with Velmeshev's ExN_mature signal. This
   uses only the work already done in H and G.
2. **Repeat the F2 analysis stratified by depth quartile.** If at
   matched depth quartiles the disagreement goes away, the answer is
   "weighted-aggregate depth artifact" and the deliverable shifts to
   reporting per-gene d for the canonical drop panel.
3. **If neither (1) nor (2) closes the gap, only then invest in #2
   (module score) or #4 (pseudotime).** Those are bigger commitments
   and only justified if the simpler controls fail.

## Files

- `h_continuous_maturity_per_cell.parquet` — per-cell PsychAD-V3 ExN
  continuous maturity score + raw markers
- `h_continuous_vs_discrete.csv` — mean continuous score per
  marker_annotation
- `h_psychad_subtype_vs_age.csv` — per-age subtype mix
- `h_psychad_subtype_at_matched_depth.csv` — same as G4 but
  PsychAD-V3 only
- `h_subtype_score_distributions.png` — continuous score histograms
  per discrete subtype
- `h_subtype_mix_vs_age.png` — subtype fractions vs donor age
- `h_donor_maturity_vs_age.png` — per-age median maturity vs age
