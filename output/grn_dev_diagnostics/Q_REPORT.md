# Q — multi-marker maturation analysis and binary-vs-continuous reconciliation

Separate analysis, NOT yet folded into FINAL_REPORT.md.

## TL;DR

1. **The discrepancy you spotted is a CP10k-normalisation artefact, not biology.**
   The binary `ExN_immature` label uses raw UMI thresholds (RBFOX3 = 0
   AND DCX ≥ 1). The continuous maturity score `RBFOX3_log1p_cp10k −
   DCX_log1p_cp10k` is depth-confounded: a shallow cell with raw
   RBFOX3 = 1 gets a high *normalised* RBFOX3 value, so binary-mature
   cells with low raw DCX but DCX-dominant CP10k normalisation slip
   into continuous Q0. Those contaminating cells carry the deep-mature
   anti-drop signal, dragging Q0's fuzzy d from +0.45 (true immature)
   to +0.09 (mixed pool).
2. **A multi-marker raw-count "maturation score" recovers the gradient.**
   When binned by the mean log1p_cp10k expression across NEUROD2, BCL11B,
   SATB2, MEF2C, NEFM (a "pure-mature module"), PsychAD-V3 fuzzy d
   goes from +0.19 in the below-median half to −0.51 in the above-median
   half — a 0.70-d swing. Velmeshev-V3 shows the same direction (+0.42 →
   +0.09). Vel-V2 stays saturated.
3. **Individual mature markers ALL show the same pattern:** cells
   expressing the mature marker → less drop (often anti-drop in PsychAD);
   cells lacking it → more drop. The most extreme is BCL11B (L5 ET marker):
   BCL11B+ cells in Velmeshev-V3 show fuzzy d = **−1.61** (the strongest
   single anti-drop signal anywhere in the analyses), and BCL11B+ in
   PsychAD-V3 d = −0.58. This is striking and worth investigating
   separately as a layer-specific finding.
4. **NEFL was undetected** under Ensembl ID ENSG00000277956 — the gene
   appears in the markers list with 0 expression in every cell of every
   group. Likely a wrong Ensembl ID for the assembly used; NEFM works
   correctly (high coverage). Exclude NEFL from the module.

## What this means for the depth-vs-maturity question (P5)

P5 said depth dominates over maturity in PsychAD-V3. That was using the
simple `RBFOX3 − DCX` CP10k score (which we've now shown is depth-
confounded). With the better multi-marker module score:

- Mature-module above/below median in PsychAD-V3: d = −0.51 vs +0.19 (0.70 d swing).
- Depth Q3 vs Q0 within ExN_mature in PsychAD-V3 (from P3): d = −0.42 vs +0.43 (0.85 d swing).

The two gradients are now of comparable magnitude. So with a properly-
specified maturity proxy, the maturity gradient is essentially as large
as the depth gradient. They're capturing largely the same biology, but
each captures some independent variance:

| | drop signal is carried by |
|---|---|
| binary ExN_immature (raw counts) | true immature cells |
| continuous-RBFOX3−DCX CP10k Q0 | mix of true immature + shallow mature artefact |
| mature-module above/below median | mature cells (high score) → anti-drop; lower-mature cells → drop |
| depth Q0 within mature subtype | shallow mature cells → drop; deep mature → anti-drop |
| per-cell UMI window 3–12k (L analysis) | partially overlaps with both depth- and maturity-low |

**Refined hypothesis (mostly your version):** C3+ developmental drop
lives in cells with less-mature transcriptional state. These cells
have lower mRNA content (smaller, less differentiated) → systematically
shallower libraries. PsychAD's FANS prep enriches large mature cells
(deep, high mature-module score), which both dilutes and counter-loads
the average. Depth-matching and maturity-binning are two largely-
equivalent ways of recovering the signal.

But: P5's residual point still holds — there's a deep-mature anti-drop
sub-population in PsychAD-V3 that isn't explained by maturity alone.
The mature_module quintile q4 is *less* negative than q3 in PsychAD-V3
(q3 d = −0.69, q4 d = −0.24), suggesting the strongest anti-drop is in
"middle-mature" cells, not the most-mature cells. Could be a specific
mature subtype (BCL11B+ L5 ET?) rather than "maturity per se".

## Part A — reconciliation of binary vs continuous

### Q-R1: crosstab of binary marker_annotation × continuous-CP10k Q

PsychAD-V3:

|                 |  Q0  |  Q1  |  Q2  |  Q3  |  Q4  |  All |
|-----------------|-----:|-----:|-----:|-----:|-----:|-----:|
| ExN_immature    | 3361 |    0 |    0 |    0 |    0 | 3361 |
| ExN_mature      | 2072 | 5692 | 6410 | 6411 | 6412 | 26997|
| ExN_weak        | 1699 |    0 |    0 |    0 |    0 | 1699 |
| **All**         | 7132 | 5692 | 6410 | 6411 | 6412 | 32057|

Q0 (the "most immature" continuous quintile) contains:
- 3361 binary-ExN_immature cells (47 %)
- 2072 binary-ExN_mature cells (29 %)
- 1699 binary-ExN_weak cells (24 %)

Velmeshev-V3:

|              | Q0   | Q1   | Q2   | Q3   | Q4   | All  |
|--------------|-----:|-----:|-----:|-----:|-----:|-----:|
| ExN_immature | 2677 |    0 |    0 |    0 |    0 | 2677 |
| ExN_mature   | 1426 | 4967 | 5931 | 5927 | 5930 | 24181|
| ExN_weak     | 2789 |    0 |    0 |    0 |    0 | 2789 |

Velmeshev-V2: only 4 effective quintiles because so many cells have
score = 0 (mostly the 10978 ExN_weak cells with RBFOX3 = 0 AND DCX = 0).

### Q-R3: fuzzy d in cells defined by different criteria

| definition | PsychAD-V3 | Vel-V2 | Vel-V3 |
|---|---:|---:|---:|
| binary ExN_immature                            | **+0.45** | +2.67 | +0.71 |
| binary_immature ∩ continuous-Q0 (subset)       | +0.45 | +2.67 | +0.71 |
| binary_mature ∩ continuous-Q0 (the contaminator) | **−0.06** | +2.82 | +0.71 |
| continuous-Q0 (mixed pool)                     | +0.09 | +3.02 | +0.89 |
| raw_immature_flag (raw RBFOX3 = 0 AND raw DCX ≥ 1) | +0.45 | +2.67 | +0.71 |
| raw_maturity = +1 (raw RBFOX3 ≥ 1)             | −0.04 | +2.42 | +0.51 |
| raw_maturity = 0 (raw RBFOX3 = 0 AND raw DCX = 0) | −0.13 | +3.07 | +0.82 |

The "binary_mature ∩ continuous-Q0" cells (2072 cells in PsychAD-V3)
have fuzzy d = **−0.06** — and these are the cells responsible for the
discrepancy. The continuous score lets them in (because DCX_log1p_cp10k
> RBFOX3_log1p_cp10k for these cells despite raw RBFOX3 ≥ 1); the
binary score correctly excludes them.

Decision implication: **the binary `ExN_immature` flag is a better
maturity-state proxy than the simple `RBFOX3_log1p_cp10k − DCX_log1p_cp10k`
score, because raw counts aren't depth-normalised.** The continuous
score as we defined it is a worse proxy specifically because of the
CP10k normalisation step.

## Part B — per-marker individual analyses

For each marker M:
- *Binary*: split cells by raw_M ≥ 1 vs raw_M = 0; fuzzy d in each half.
- *Continuous*: bin cells by log1p_cp10k_M quintile; fuzzy d per quintile.

Outputs: `qB_per_marker_binary_d.{csv,png}`, `qB_per_marker_quintile_d.{csv,png}`.

### Binary expressing-vs-not (selected rows)

| marker | group        | expressing+ d | not-expressing d |
|---|---|---:|---:|
| DCX     | PsychAD-V3   | **−0.31**   | −0.19  |
| DCX     | Velmeshev-V3 | +0.40   | +0.66  |
| RBFOX3  | PsychAD-V3   | −0.04   | +0.18  |
| RBFOX3  | Velmeshev-V3 | +0.51   | +0.85  |
| BCL11B  | PsychAD-V3   | **−0.58**   | −0.14  |
| BCL11B  | Velmeshev-V3 | **−1.61**   | +0.81  |
| SATB2   | PsychAD-V3   | −0.14   | +0.36  |
| SATB2   | Velmeshev-V3 | +0.35   | +0.68  |
| NEUROD2 | PsychAD-V3   | −0.31   | −0.10  |
| NEUROD2 | Velmeshev-V3 | −0.19   | +0.54  |
| MEF2C   | PsychAD-V3   | −0.33   | +0.44  |
| MEF2C   | Velmeshev-V3 | +0.15   | +0.44  |
| NEFM    | PsychAD-V3   | −0.30   | −0.10  |
| NEFM    | Velmeshev-V3 | −0.08   | +0.66  |

**Consistent direction**: in *both* PsychAD-V3 and Vel-V3, the
expressing+ population shows a *less* positive (more anti-drop) fuzzy
d than the not-expressing population, across DCX, RBFOX3, RBFOX1,
NEUROD2, BCL11B, SATB2, MEF2C, NEFM.

**Apparent paradox — DCX+ shows anti-drop?** Yes, but this is the same
contamination issue: any cell with raw_DCX ≥ 1 is "DCX+" in this
binary split — that includes mature cells with low ambient/trace DCX
detection, not just true immature ones. So DCX+ here is NOT the same
as binary ExN_immature.

**BCL11B is the most striking:** Vel-V3 BCL11B+ d = −1.61, by far the
most extreme anti-drop signal in any analysis. BCL11B is a L5-ET
marker. In Vel-V3, deep-layer ET cells specifically anti-correlate
with the developmental C3+ drop. This is a layer-resolved biological
result worth chasing separately.

### Continuous quintile (selected highlights)

- DCX: low-DCX quintiles all show small d in PsychAD; high-DCX quintile
  has only 3 effective levels (most cells DCX = 0). q4 (highest DCX)
  shows mild d in PsychAD-V3, big +d in Vel-V3 (+0.55). When DCX *is*
  detected and high, the cells are immature-state → drop.
- BCL11B q4 in Vel-V3: d = −1.61 (the standout anti-drop bar).
- MEF2C q0 (low MEF2C): +d in all 3 cohorts (+0.40 PsychAD, +2.5 V2,
  +0.35 V3). Higher MEF2C quintiles flip negative in PsychAD/V3.
- NEFL: undetected (likely wrong Ensembl ID; see note above).
- NEFM: dichotomous distribution (mostly 0 or large); q0 (low NEFM)
  shows +d (+2.46 V2, +0.56 V3, ~0 PsychAD). q2/high NEFM shows weak −d
  in PsychAD.

The consistent direction (more mature marker expression → less drop /
more anti-drop) is encouraging: maturity matters for the C3+
developmental gradient.

## Part C — multi-marker mature module

Define `mature_module = mean(log1p_cp10k_{NEUROD2, BCL11B, SATB2, MEF2C, NEFM})`
(5 markers — excludes DCX which is immature, RBFOX3/RBFOX1 already used
by the binary classifier, and NEFL which is undetected).

### Above/below median split

| group        | above-median d | below-median d | Δ |
|---|---:|---:|---:|
| PsychAD-V3   | **−0.51** | **+0.19** | 0.70 |
| Velmeshev-V2 | +2.26 | +2.44 | 0.18 |
| Velmeshev-V3 | +0.09 | +0.42 | 0.33 |

PsychAD-V3 shows a clean direction-reversing 0.70-d swing. Vel-V3 has
a positive-only gradient (both halves drop, less-mature drops more).
Vel-V2 is saturated (~2.3 everywhere; shallow-library d amplification
dominates over the maturity signal).

### Continuous quintile

| quintile | PsychAD-V3 | Vel-V2 | Vel-V3 |
|---|---:|---:|---:|
| q0 (least mature) | **+0.52** | +2.44 | +0.49 |
| q1                | −0.14 | +2.53 | +0.10 |
| q2                | −0.33 | +2.27 | −0.30 |
| q3                | **−0.69** | +2.34 | −0.25 |
| q4 (most mature)  | −0.24 | +1.93 | +0.37 |

Two observations:

1. **The gradient is monotonic from q0 → q3 in PsychAD-V3 and Vel-V3.**
   Less-mature cells carry the drop; more-mature cells anti-correlate
   (or simply don't carry it).
2. **Non-monotonic at q4** (most extreme mature cells). In PsychAD-V3,
   q3 (d = −0.69) is more negative than q4 (d = −0.24). In Vel-V3, q4
   is +0.37 (back into drop direction). The most-extreme-mature cells
   don't behave like the rest of the mature cells. This is consistent
   with the BCL11B observation — there may be a specific deeply-mature
   sub-population (perhaps L5-ET-like) that has its own non-developmental
   transcriptional dynamic.

### Variant: mature_module − DCX

Same shape as above; in PsychAD-V3 the q3 vs q0 swing is 0.86 d (q0
+0.21, q3 −0.65). Subtracting DCX makes the score slightly less
extreme at the immature end but doesn't qualitatively change the picture.

## Files (all in `outputs/`)

| | file |
|---|---|
| Extended cache with all 9 marker raw/log1p_cp10k | `m_per_cell_cache_v3.parquet` |
| Reconciliation crosstab | `qA_binary_x_cont_crosstab.csv` |
| Reconciliation median UMI | `qA_binary_x_cont_median_umi.csv` |
| Reconciliation fuzzy d table | `qA_reconciliation_d.csv` |
| Per-marker binary d | `qB_per_marker_binary_d.csv`, `qB_per_marker_binary_d.png` |
| Per-marker quintile d | `qB_per_marker_quintile_d.csv`, `qB_per_marker_quintile_d.png` |
| Module score d | `qC_module_d.csv`, `qC_module_quintile_d.png` |

## Loose ends to follow up before adding to FINAL_REPORT

1. **Fix the NEFL Ensembl ID** (current ENSG00000277956 gives 0
   detection across all 95k cells; likely wrong assembly version).
   Rebuild cache and include NEFL in the module — may shift the
   quintile pattern slightly.
2. **Layer × maturity-module joint analysis.** The BCL11B / NEFM /
   MEF2C signal directions suggest the "non-monotonic q4" behaviour
   may be specifically driven by L5 ET cells. Splitting the module-q
   d by cortical layer would test this.
3. **Repeat P5-style 2D analysis** but with `depth_q × mature_module_q`
   instead of `depth_q × (RBFOX3−DCX) score_q`. This will give a
   cleaner picture of which axis carries the developmental signal.
4. **Consider whether mature_module quartile/quintile binning should
   replace `marker_annotation` as the analytic primitive** for
   cross-cohort comparison — it's a continuous, biologically-motivated
   score and avoids the brittle thresholds of the rule-based classifier.
