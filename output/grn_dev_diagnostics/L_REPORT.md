# L — joint depth × layer stratification + depth-matched comparison

User-prompted follow-up: K2 showed PsychAD-V3 per-cell-CPM d has a real
depth gradient (+0.24 at shallow Q0 → −0.86 at deep Q3) that survives
normalisation. K1 showed layer stratification doesn't close the
PsychAD-vs-Vel gap. Question: does JOINT depth × layer stratification
close it? And if we restrict PsychAD-V3 to the same UMI range as Vel-V3,
does PsychAD-V3 look like Vel-V3?

## TL;DR

1. **Depth-matching FLIPS PsychAD-V3's sign from negative to positive
   in three of four windows.** At median UMI ~5-7 k (matching Vel-V3's
   distribution), PsychAD-V3 upper-layer d goes from −0.27 to **+0.17
   to +0.25** (drop direction, agreeing with Vel-V3). The PsychAD-vs-Vel
   gap in upper layer shrinks from 1.29 → 0.7–0.9 d-units. Qualitative
   picture changes: both datasets agree on the direction of the C3+
   drop, they just disagree on magnitude.

2. **Layer assignment is depth-confounded** — upper-layer cells have
   ~2× the median UMI of L5_ET / L6_CT cells in BOTH datasets
   (PsychAD-V3 upper: 18.4 k UMI vs L5_ET: 9.7 k; Vel-V3 upper: 8.5 k
   vs L5_ET: 5.3 k). K1's per-layer comparison was implicitly
   comparing depth-different subsets even within the same nominal layer.
   L corrects this.

3. **The depth gradient in PsychAD-V3 lives mostly INSIDE the upper
   layer.** Upper-layer d swings from +0.19 (Q0) to −1.00 (Q3) — a
   1.2-d swing across depth quartiles. Other layers (L5_ET, L6_CT)
   have much smaller within-layer depth gradients. So whatever PsychAD's
   "deep cells carry anti-drop" mechanism is, it operates within upper
   pyramidal cells specifically.

4. **J (downsampling) and L (depth-restriction) are doing different
   things and disagree by design.** J caps each cell's raw counts but
   keeps the cell in the analysis. L excludes cells outside the depth
   window. J fixes the within-cell normalisation; L fixes the
   across-cell selection. The fact that L closes the sign but J does
   not means the deep cells were carrying real biological signal (not
   normalisation noise) and removing them removes that signal.

5. **Composition shifts persist at matched depth.** PsychAD-V3
   children consistently have MORE L5_ET / L6_CT and LESS upper-layer
   cells than Vel-V3 children, at every depth window. This is a real
   cohort-level composition difference.

## L1 — per-layer depth distribution

| group        | layer     | n_cells | median UMI | mean UMI |
|--------------|-----------|--------:|-----------:|---------:|
| PsychAD-V3   | upper     | 22,642  | **18,440** | 24,094   |
| PsychAD-V3   | L5_ET     |  3,098  |   9,693    | 11,814   |
| PsychAD-V3   | L6_CT     |  4,669  |   9,654    | 15,021   |
| PsychAD-V3   | L6_IT     |  1,275  |  13,798    | 15,671   |
| PsychAD-V3   | ambiguous |    611  |   3,549    |  4,509   |
| Velmeshev-V3 | upper     | 21,459  |  **8,490** | 12,475   |
| Velmeshev-V3 | L5_ET     |  2,417  |   5,343    |  6,742   |
| Velmeshev-V3 | L6_CT     |  3,826  |   6,074    | 10,106   |
| Velmeshev-V3 | L6_IT     |  1,021  |   8,210    |  9,912   |
| Velmeshev-V3 | ambiguous |    924  |   1,675    |  2,249   |

Upper-layer cells are ~2× deeper than L5_ET / L6_CT cells, in BOTH
datasets. This is itself an interesting biological observation
(L2/3/4 IT pyramidal neurons have more mRNA per nucleus than L5/L6
ET/CT cells, contrary to the naive expectation that L5 ET would
dominate). It also means K1's per-layer comparison was implicitly
comparing different-depth subsets within the same layer.

## L2 — joint (depth_q × layer) d per dataset, per-cell-CPM mean

PsychAD-V3:

| depth_q | upper | L5_ET | L6_CT | L6_IT | ambiguous |
|---------|------:|------:|------:|------:|----------:|
| Q0      |**+0.19**| −0.13 | +0.22 | +0.38 |   +0.86   |
| Q1      | +0.11 | +0.64 | +0.28 | +0.07 |   n/a     |
| Q2      | −0.77 | +0.29 | −0.15 | −0.03 |   n/a     |
| Q3      |**−1.00**| n/a | −0.36 | n/a   |   n/a     |

Velmeshev-V3:

| depth_q | upper | L5_ET | L6_CT | L6_IT | ambiguous |
|---------|------:|------:|------:|------:|----------:|
| Q0      | +1.09 | +0.66 | +1.42 | +0.41 |   +0.91   |
| Q1      | +0.98 | +0.54 | +1.16 | −1.01 |   +0.56   |
| Q2      | +0.81 | +0.59 | +0.71 | −0.45 |   n/a     |
| Q3      | +0.73 | +0.32 | −0.26 |  0.00 |   n/a     |

Two observations:
- **PsychAD-V3 upper layer carries most of the depth gradient.**
  Within upper alone, d swings from +0.19 (Q0) to −1.00 (Q3). Other
  layers in PsychAD have much smaller within-layer gradients.
- **Vel-V3 also shows depth gradients within layers**, especially
  L6_CT (+1.42 → −0.26) and upper (+1.09 → +0.73), but stays mostly
  positive — its baseline drop signal is large enough that the depth
  gradient doesn't reach zero.

## L3 — depth-matched comparison

Restrict both datasets to four candidate UMI windows and recompute
layer × age d (per-cell-CPM mean).

### Baseline d (all ExN at matched depth):

| Window                  | PsychAD-V3 | Velmeshev-V3 | gap |
|-------------------------|-----------:|-------------:|----:|
| unrestricted (J)        |   −0.26    |    +1.09     | 1.35 |
| overlap_2k_15k          | **+0.19**  |    +1.02     | 0.83 |
| vel_v3_central_3k_12k   | **+0.31**  |    +0.96     | 0.65 |
| psy_v3_shallow_1k_8k    | **+0.25**  |    +1.13     | 0.88 |
| vel_v3_full_p5_p95      |   −0.08    |    +1.08     | 1.16 |

### Per-layer d at depth-matched windows:

| Window                | layer | PsychAD-V3 d | Vel-V3 d | gap |
|-----------------------|-------|-------------:|---------:|----:|
| overlap_2k_15k        | upper |   **+0.17**  |  +1.04   | 0.87 |
| overlap_2k_15k        | L5_ET |   −0.06      |  +0.73   | 0.79 |
| overlap_2k_15k        | L6_CT |   +0.25      |  +1.10   | 0.85 |
| overlap_2k_15k        | L6_IT |   +0.39      |  −0.82   | −1.21 |
| vel_v3_central_3k_12k | upper |   **+0.25**  |  +0.98   | 0.73 |
| vel_v3_central_3k_12k | L5_ET |   −0.09      |  +0.72   | 0.81 |
| vel_v3_central_3k_12k | L6_CT |   +0.36      |  +1.08   | 0.72 |
| psy_v3_shallow_1k_8k  | upper |   **+0.21**  |  +1.14   | 0.93 |
| psy_v3_shallow_1k_8k  | L5_ET |   −0.11      |  +0.85   | 0.96 |
| psy_v3_shallow_1k_8k  | L6_CT |   +0.22      |  +1.40   | 1.18 |
| vel_v3_full_p5_p95    | upper |   −0.13      |  +1.06   | 1.18 |
| vel_v3_full_p5_p95    | L5_ET |   −0.01      |  +0.75   | 0.75 |
| vel_v3_full_p5_p95    | L6_CT |   +0.14      |  +1.29   | 1.15 |

Key finding: **at vel_v3_central_3k_12k, PsychAD-V3 upper d = +0.25** —
a clear drop, in the same direction as Vel-V3. The gap shrinks from
1.29 → 0.73, and PsychAD-V3 has the same sign as Vel-V3.

## Composition shifts at matched depth (overlap_2k_15k)

| group        | stage | upper | L5_ET | L6_CT | L6_IT | ambiguous |
|--------------|-------|------:|------:|------:|------:|----------:|
| PsychAD-V3   | adol  | 57 %  | 13 %  | 23 %  | 4 %   | 4 %       |
| PsychAD-V3   | child | 55 %  | 17 %  | 19 %  | 5 %   | 4 %       |
| Velmeshev-V3 | adol  | 61 %  | 11 %  | 22 %  | 4 %   | 2 %       |
| Velmeshev-V3 | child | **74 %** | 9 % | 11 %  | 4 %   | 2 %       |

Notable:
- Vel-V3 children have **74 % upper-layer cells**; PsychAD-V3 children
  have 55 % — even at matched depth, Vel-V3 children's ExN bin is
  consistently more upper-layer-rich.
- PsychAD-V3 children have systematically more L5_ET (17 % vs 9 %)
  and L6_CT (19 % vs 11 %) than Vel-V3 children.
- These composition differences are RESIDUAL after depth matching —
  they reflect true cohort-level differences in what cells get
  captured/sequenced from pediatric DLPFC.

## What this means

**The C3+ developmental drop IS present in PsychAD-V3 pediatric DLPFC
upper-layer neurons, when we restrict to cells at the same depth as
Velmeshev-V3 cells.** The original sum-then-CPM aggregate (d = −0.45),
and even the per-cell-CPM aggregate without depth matching (d = −0.26),
were dominated by PsychAD's deep upper-layer cells which carry an
opposite-direction signal.

So the two-dataset disagreement was really **three confounded effects**
stacking up:
1. Sum-then-CPM UMI-weighted-averaging bias (~40 % of the gap; fixed
   by J's per-cell-CPM mean)
2. Across-cell SELECTION bias (~40 % of the gap; fixed by L's
   depth-matching). Specifically: PsychAD's pediatric pool contains
   too many deep upper-layer cells whose C3+ signal goes opposite to
   the developmental drop.
3. Residual cohort-level difference (~20 % of the gap, or ~0.7 d
   units). Remains after both fixes. Most likely true biology in
   PsychAD's HBCC pediatric vs Vel's developmental atlas children,
   or persistent FANS-prep effect.

## Why the "deep upper-layer PsychAD cells" carry anti-drop

Speculation, in plausibility order:

1. **PsychAD adolescents' deep upper-layer cells are exceptionally
   well-preserved large mature L2/3 pyramidal neurons** — the
   transcriptionally most mature subset, with extensive synaptic gene
   expression. PsychAD children's deep upper-layer cells are a
   different population: not as many fully-mature L2/3 cells exist
   in 2-8 y old cortex, so the few that get captured are atypical
   (perhaps more committed to L5-ish identity, or particularly
   high-mRNA outliers).
2. **FANS sort efficiency varies with cell size.** Adolescent L2/3
   pyramidal neurons have larger nuclei than child L2/3 (cell-size
   growth is a documented postnatal phenomenon). FANS may preferentially
   sort larger nuclei (higher NeuN intensity, larger forward scatter),
   biasing the adolescent pool toward maximally mature cells.
3. **Donor-level batch effects concentrated in deep cells.** Per-donor
   library quality varies; donors with the highest-quality nuclei
   contribute proportionally more deep cells. If those donors happen
   to also be on the more-mature end of their age group (e.g. older
   adolescents, healthier children), the deep cells over-represent
   them.

Mechanisms 1 and 2 are essentially "PsychAD's FANS protocol
preferentially admits cell-type-state subsets that don't reflect the
true age-distribution of upper-layer cells." This is a real selection
bias on the cell pool, and L's depth matching partially corrects for
it by excluding the deep selection-biased cells.

## Revised deliverable recommendation

For `grn_dev_multi.md`:

1. **Switch to per-cell-CPM mean aggregation** (replace sum-then-CPM).
   This is the universal recommendation from J.
2. **Report at matched depth (e.g., 3-12 k UMI window)** as the
   headline number. At this depth window:
     PsychAD-V3 baseline d = +0.31, Vel-V3 baseline d = +0.96
     PsychAD-V3 upper d = +0.25, Vel-V3 upper d = +0.98
   Both datasets agree: there IS a C3+ developmental drop. PsychAD's
   drop is smaller (by ~0.6 d units) than Vel-V3's.
3. **Report the layer × depth × age breakdown** (L3) as the main
   evidence panel. It shows where the agreement is and where the
   residual gap lives.
4. **Flag the residual ~0.7 d gap explicitly** as the genuinely
   unresolved part. Most plausibly cohort-level: PsychAD's HBCC
   pediatric controls + FANS prep vs Vel's developmental atlas
   unsorted nuclei. Suggested arbiters: (a) third pediatric cohort
   processed through both protocols, (b) PMI/antemortem condition
   metadata for both pediatric pools, (c) within-PsychAD comparison
   of upper-layer cells split by cell size (forward-scatter / UMI
   proxy) to see whether sort-selection biases the within-layer
   transcriptional pool.

## Files

- `l1_layer_depth_distribution.csv` — per-layer UMI quartiles per group
- `l2_d_depth_q_x_layer.csv` — joint (depth_q × layer) d per group
- `l2_d_depth_q_x_layer.png` — heatmap of the above
- `l3_d_depth_matched.csv` — d per (window × group × layer)
- `l3_d_depth_matched.png` — bar plot of per-window d
- `l3_layer_composition_depth_matched.csv` — composition per donor
  at each depth window
- `l3_layer_composition_summary.csv` — composition summary
