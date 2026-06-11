# G — depth dependence of the marker-based ExN classifier

**Question:** the marker classifier in `code/annotation_by_markers.py`
uses absolute raw-count thresholds (RBFOX3 ≥ 1, DCX ≥ 1, RBFOX1 ≥ 1).
These are inherently library-size dependent: a shallow V2 cell with
1500 UMI may fail RBFOX3 ≥ 1 even when the underlying neuron is
biologically mature. Does this bias the ExN subtype mix differently
across PsychAD-V3 / Vel-V2 / Vel-V3?

## Group depth & marker detection (overall, age 1-25 y)

| group           | n_cells | median UMI | mean UMI | P(RBFOX3≥1) | P(DCX≥1) | P(RBFOX1≥1) | P(GAD1≥1) |
|-----------------|---------|-----------|----------|-------------|----------|-------------|-----------|
| PsychAD-V3      | 76,814  | 7,748     | 13,919   | 0.415       | 0.274    | 0.919       | 0.304     |
| Velmeshev-V3    | 70,164  | 3,993     | 7,105    | 0.372       | 0.211    | 0.773       | 0.211     |
| Velmeshev-V2    | 57,216  | 1,648     | 4,391    | 0.360       | 0.185    | 0.728       | 0.187     |

PsychAD-V3 is **5× deeper** in median UMI than Vel-V2 and 2× deeper than
Vel-V3.

## ExN subtype mix per group

| group         | ExN_mature | ExN_immature | ExN_weak |
|---------------|-----------:|-------------:|---------:|
| PsychAD-V3    |     84.0%  |      10.7%   |    5.3%  |
| Velmeshev-V3  |     81.6%  |       9.0%   |    9.4%  |
| Velmeshev-V2  |     59.4%  |       8.3%   |   32.4%  |

Vel-V2's ExN_weak fraction is **6× larger** than PsychAD-V3's. With
shallow libraries, RBFOX3 and DCX both fail the ≥1 threshold in 40-50%
of cells, sending them to the "weak" RBFOX1-only bin.

## At MATCHED depth, classification is mostly depth-driven (G4)

`g4_classification_vs_depth.png` — fraction of ExN cells in each
subtype, stratified by UMI bin:

UMI bin [1000, 2000):
| group       | mature | immature | weak |
|-------------|-------:|---------:|-----:|
| PsychAD-V3  |  37%   |   12%    |  51% |
| Vel-V2      |  35%   |   12%    |  53% |
| Vel-V3      |  51%   |   10%    |  39% |

UMI bin [5000, 8000):
| group       | mature | immature | weak |
|-------------|-------:|---------:|-----:|
| PsychAD-V3  |  69%   |   22%    |   9% |
| Vel-V2      |  82%   |    9%    |   8% |
| Vel-V3      |  81%   |   16%    |   3% |

UMI bin [12000, 20000):
| group       | mature | immature | weak |
|-------------|-------:|---------:|-----:|
| PsychAD-V3  |  90%   |   10%    | 0.5% |
| Vel-V2      |  98%   |  1.4%    | 0.2% |
| Vel-V3      |  97%   |  2.6%    | 0.3% |

→ At shallow depth, the subtype mix is very similar across the three
groups; the dominant difference is depth itself. **This confirms the
classifier output is depth-determined.**

## But — at MATCHED depth, RBFOX3 detection differs by DATASET (G3)

| UMI bin | PsychAD-V3 P(RBFOX3≥1) | Vel-V2 P(RBFOX3≥1) | Vel-V3 P(RBFOX3≥1) |
|---------|------------------------|--------------------|--------------------|
| 1-2 k   | 6.3%                   | 13.4%              | 11.0%              |
| 2-3 k   | 7.9%                   | 25.9%              | 14.4%              |
| 3-5 k   | 11.4%                  | 53.3%              | 20.7%              |
| 5-8 k   | 18.9%                  | 76.5%              | 39.2%              |
| 8-12 k  | 35.0%                  | 88.6%              | 75.0%              |
| 12-20 k | 67.6%                  | 97.2%              | 94.0%              |
| 20-40 k | 94.2%                  | 99.6%              | 99.3%              |

This is a **massive dataset-level offset**. At 5-8 k UMI, Velmeshev-V2
detects RBFOX3 in 76% of cells; PsychAD-V3 detects it in 19% — a 4×
gap at matched depth. At 8-12 k, the gap is 89% vs 35%. PsychAD-V3's
detection only catches up at >40 k UMI.

The same gap exists for DCX (35% Vel-V2 vs 10% PsychAD-V3 at 5-8 k UMI)
but is much smaller for RBFOX1 (Vel ≈ PsychAD at most depth bins).

→ This is **not** explainable by depth alone. RBFOX3 (and DCX) are
captured *less efficiently* in PsychAD's data at the same per-cell UMI.
RBFOX1 capture is comparable. The classifier is therefore biased
toward calling fewer PsychAD-V3 cells "ExN_mature" at any given depth
than it would call Velmeshev cells — except that PsychAD's overall
shift to deeper UMI compensates.

## Why?

Plausible mechanisms (need verification):

1. **3' end positioning.** RBFOX3 mRNA in 10x v3 chemistry can be
   under-captured if its 3' UTR has a poly-A internal signal or short
   3' tail. If PsychAD's library prep (e.g. FANS-sorted nuclei +
   different reagent batches) differentially recovers shorter 3' tags,
   the RBFOX3 vs RBFOX1 detection ratio will shift between datasets.
2. **FANS sort effect.** PsychAD's FANS-purified NeuN (RBFOX3)
   positive nuclei would actually be expected to INCREASE RBFOX3
   capture in the sorted fraction — so this would predict the
   opposite. Unless what's being sorted has lower nuclear RBFOX3
   protein but still surface marker reactivity.
3. **Ambient profile differences.** Velmeshev's unsorted nuclei may
   have higher ambient RBFOX3 from lysed neurons, "rescuing"
   RBFOX3 ≥ 1 calls in cells that aren't really expressing it
   (esp. astrocytes, oligos in the same depth bin). This would inflate
   Velmeshev's detection rate at matched depth without it being more
   "real."
4. **Different mapping references / gene model boundaries** between
   the two source preprocessing pipelines. PsychAD's Aging/HBCC source
   h5ads were processed through PsychAD's own pipeline; Velmeshev's
   were processed by the original Velmeshev lab. Different
   transcript definitions could move the RBFOX3 count.

## Combined story (with F2)

- F2 said: PsychAD's high-weight C3+ genes go in the opposite
  direction (negative d) than Velmeshev's, and the gap grows
  monotonically with C3+ weight.
- E1 said: within PsychAD's ExN_immature subtype, d is +0.31 (drop
  direction); within ExN_mature, d is −0.29 (anti-drop).
- G now says: PsychAD's ExN_mature bin is conservative (only cells
  with RBFOX3 ≥ 1 raw counts, which at PsychAD's depth captures only
  35-68% of cells in the 5-20 k UMI bins), so PsychAD's "mature" bin
  is enriched for the deepest, highest-quality cells. PsychAD's
  "immature" and "weak" bins by contrast collect cells that failed
  RBFOX3 but may be biologically mature neurons whose RBFOX3 was
  undersampled.
- F2 + G together suggest: in **PsychAD's `ExN_mature` bin, the few
  child donors contribute disproportionately deep and "selectively
  RBFOX3-positive" cells**. The synaptic-gene signal in those cells
  doesn't track a clean developmental gradient because the bin is
  itself a depth-and-quality filtered subset, not a biological maturity
  class.

**Net diagnosis:** the marker classifier is not a robust maturity bin.
Its output depends on (a) per-cell UMI depth and (b) dataset-specific
capture efficiency for RBFOX3. The biology is real in both datasets
at the per-gene level (NRXN1, NLGN1, GRIK1/2, GRM7, etc. all drop in
both PsychAD and Velmeshev). The weighted aggregate disagreement is
the result of (i) the aggregate up-weighting genes that are
particularly sensitive to mature/non-mature cell composition, and
(ii) the marker classifier producing dataset-specific subpopulations
inside the "ExN_mature" bin.

## Files

- `g_per_cell_markers.parquet` — 204,194 cells × (6 markers, total UMI, age, chemistry, marker_annotation)
- `g_group_summary.csv` — per-group overall detection rates + depth
- `g_exn_mix_per_group.csv` — overall ExN subtype mix per group
- `g3_detection_vs_depth.csv` + `g3_detection_vs_depth.png` — P(marker ≥ 1) per UMI bin per group
- `g4_classification_vs_depth.csv` + `g4_classification_vs_depth.png` — ExN subtype mix per UMI bin per group
- `g1_total_umi.png` — UMI distribution per group
- `g2_marker_counts.png` — raw-count histograms of RBFOX3/DCX/RBFOX1 per group
