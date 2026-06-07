# K — layer-marker module stratification + depth × per-cell-CPM

User-requested follow-ups to J:
  K1: assign each ExN cell to a cortical layer via TF-module scoring,
      then compute C3+ d per layer × group with per-cell-CPM mean.
  K2: re-run the depth-quartile stratification using per-cell-CPM mean
      (replacing I2's sum-then-CPM).

## TL;DR

1. **The depth gradient inside PsychAD-V3 is real, not a normalization
   artifact.** K2 with per-cell-CPM mean produces virtually the same
   per-quartile d as I2 with sum-then-CPM did: PsychAD goes +0.24, +0.13,
   −0.36, −0.86 from shallow to deep quartiles (vs I2's +0.26, +0.10,
   −0.41, −0.84). So *within* a depth-narrow quartile, UMI weighting
   barely matters; per-cell-CPM only fixes the *across-quartile*
   weighting that drove the aggregate.

2. **Layer stratification does NOT close the PsychAD-vs-Vel gap.** The
   disagreement is present in EVERY layer at roughly the same magnitude:
   upper (gap 1.29), L5_ET (0.83), L6_CT (1.42). So cell-type subclass
   composition is NOT the residual explanation. We can largely
   eliminate mechanism (iii).

3. **Layer-internal pattern within PsychAD-V3 is informative:** the
   aggregate's anti-drop is entirely carried by the upper layer
   (d = −0.27), which is 66 % of all cells. L5_ET and L6_CT show
   d ≈ 0 (no signal in either direction); L6_IT shows a faint +0.15;
   the "ambiguous" (no TF detected) bin shows +0.65. So PsychAD's deep
   layers don't actively oppose the drop — they just don't show it.

4. **In Velmeshev-V3, the C3+ drop is strongest in L6_CT (+1.41), then
   upper (+1.02), then L5_ET (+0.81); L6_IT is weakly negative
   (−0.22).** The drop is a real per-layer signal across multiple
   layers, not a single-layer effect.

5. **Layer composition itself differs between datasets in opposite
   directions across stages.** PsychAD children have proportionally
   MORE L5_ET than PsychAD adolescents (15.3 % vs 9.7 %); Vel-V3
   children have proportionally MORE upper-layer cells than Vel-V3
   adolescents (74.2 % vs 65.7 %). This is itself a striking
   difference but, since the gap is present within every layer, the
   composition difference cannot explain the overall direction.

6. **The residual 1.3-d gap is now most likely a cohort-level
   biological/technical difference** between PsychAD's HBCC pediatric
   controls and Velmeshev's developmental-atlas children. PsychAD's
   pediatric upper-layer neurons in this dataset show a flat-or-
   declining C3+ trajectory across the developmental window; Vel's
   pediatric upper-layer neurons show a strong drop. Same cell type,
   same anatomy (DLPFC), same chemistry (V3), comparable depth.

## K2 — depth quartile with per-cell-CPM mean

Comparison to I2 (sum-then-CPM):

| group        | quartile | I2 (sum-CPM) | K2 (per-cell-CPM) | Δ      |
|--------------|---------|-------------:|------------------:|-------:|
| PsychAD-V3   | Q0      | +0.26        | +0.24             | −0.02  |
| PsychAD-V3   | Q1      | +0.10        | +0.13             | +0.03  |
| PsychAD-V3   | Q2      | −0.41        | −0.36             | +0.05  |
| PsychAD-V3   | Q3      | −0.84        | −0.86             | −0.02  |
| Velmeshev-V2 | Q0      | +1.66        | +3.05             | +1.39  |
| Velmeshev-V2 | Q1      | +1.30        | +2.27             | +0.97  |
| Velmeshev-V2 | Q2      | +1.16        | +2.27             | +1.11  |
| Velmeshev-V2 | Q3      | +1.01        | +2.06             | +1.05  |
| Velmeshev-V3 | Q0      | (not reported by dataset×chem)| +1.07 | — |
| Velmeshev-V3 | Q1      | "                              | +0.96 | — |
| Velmeshev-V3 | Q2      | "                              | +0.82 | — |
| Velmeshev-V3 | Q3      | "                              | +0.69 | — |

Key observations:

- **PsychAD-V3 per-quartile d is virtually unchanged** between
  sum-CPM and per-cell-CPM (|Δ| ≤ 0.05). Within a depth quartile, UMI
  variation across cells is small (e.g., Q0 = 1.2 k–7.7 k, spanning
  ~6× rather than the whole-ExN ~125×), so UMI-weighted vs equally-
  weighted averaging hardly differs.
- **Velmeshev-V2 per-quartile d INCREASES substantially** (Δ ≈ +1.0
  per quartile) under per-cell-CPM. The within-V2 UMI heterogeneity
  is larger (Q0 = 332-1098, Q3 = 8941-99k → ~100×), so per-cell-CPM
  matters more. Vel-V2 children at Q0 with per-cell-CPM show
  d = +3.05 — extraordinarily strong drop.
- **Per-cell-CPM mean aggregate ≈ uniform-cell-weight average of
  per-quartile d's:** PsychAD-V3 (0.24+0.13−0.36−0.86)/4 = −0.21,
  matching the J baseline of −0.26 (small discrepancy from per-donor
  cell counts varying across quartiles). So the per-cell-CPM
  aggregate IS the "depth-balanced" summary of the per-quartile d's.

The PsychAD depth gradient from +0.24 to −0.86 across quartiles is
**real and survives normalization correction**. This is consistent
with the deep cells of PsychAD's pediatric donors carrying systematically
lower synaptic transcripts than the deep cells of PsychAD's
adolescent donors — but this isn't a depth artifact, it's the actual
data structure. The aggregate negative d of −0.26 then comes from the
fact that PsychAD has cells across all depths so this gradient
averages to negative.

## K1 — child-vs-adol d per layer

Cells were assigned via per-cell layer-module argmax. Modules used
postnatally-stable TFs (established embryonically and persistent
in adult cortex): upper = {SATB2, CUX2, CUX1, RORB}; L5_ET = {FEZF2,
BCL11B, POU3F1}; L6_CT = {TBR1, FOXP2, TLE4, NXPH4, SYT6};
L6_IT = {SULF1, OPRK1}.

| Layer        | PsychAD-V3 d  | Vel-V2 d | Vel-V3 d | gap (V3 − PsychAD) |
|--------------|--------------:|---------:|---------:|-------------------:|
| upper        |       **−0.27** |   +2.25  |   +1.02  |       1.29       |
| L5_ET        |       −0.02   |   +2.80  |   +0.81  |       0.83       |
| L6_CT        |       −0.01   |   +2.35  |   +1.41  |       1.42       |
| L6_IT        |       +0.15   |   +2.44  |   −0.22  |      −0.37       |
| ambiguous    |       +0.65   |   +3.02  |   +1.00  |       0.35       |
| **baseline** |       **−0.26** | **+2.37** | **+1.09** |    1.35     |

The Vel-V3-vs-PsychAD-V3 disagreement is present in every meaningful
layer with similar magnitude as the whole-ExN baseline. Layer
stratification does NOT close the gap.

Layer-internal pattern in PsychAD-V3 (66 % of cells are "upper"):

- upper carries essentially all of PsychAD's aggregate anti-drop signal
  (d = −0.27, matching baseline −0.26).
- Deep layers (L5_ET, L6_CT) are flat (d ≈ 0). They are not actively
  opposing the drop; they just lack any signal.
- L6_IT and ambiguous bins are too small / too noisy to be informative.

In Velmeshev-V3, the C3+ drop is broadly distributed across layers,
strongest in L6_CT (+1.41) and upper (+1.02). Vel-V2 shows the drop
even more strongly (+2.0 to +2.8 in every layer).

## Layer composition shifts across stages

PsychAD-V3 child vs adol layer fractions (mean per donor):

| layer       | child | adol  | Δ (child − adol) |
|-------------|------:|------:|-----------------:|
| upper       | 58.7% | 66.0% |   **−7.3 pp**    |
| L5_ET       | 15.3% |  9.7% |   **+5.6 pp**    |
| L6_CT       | 17.9% | 18.1% |    −0.2 pp       |
| L6_IT       |  5.4% |  4.0% |    +1.4 pp       |
| ambiguous   |  3.0% |  2.4% |    +0.6 pp       |

Velmeshev-V3 child vs adol layer fractions:

| layer       | child | adol  | Δ (child − adol) |
|-------------|------:|------:|-----------------:|
| upper       | 74.2% | 65.7% |   **+8.5 pp**    |
| L5_ET       |  7.9% |  8.4% |    −0.5 pp       |
| L6_CT       | 11.6% | 18.8% |   **−7.2 pp**    |
| L6_IT       |  3.3% |  3.9% |    −0.6 pp       |
| ambiguous   |  3.0% |  3.3% |    −0.3 pp       |

**Striking opposite-direction composition shifts.** In PsychAD,
children have proportionally MORE deep-layer ET cells and FEWER
upper-layer cells than adolescents; in Velmeshev, the reverse —
children have MORE upper-layer and FEWER L6_CT than adolescents.

What does this mean?

- This could be **true biological cohort difference**: PsychAD's HBCC
  pediatric controls may represent a different developmental snapshot
  of cortical composition than Vel's developmental atlas.
- Or it could be **FANS-sorting bias**: PsychAD's NeuN+ sort may
  recover different layer ratios depending on developmental stage
  (e.g. younger brains' upper-layer neurons may have different NeuN
  protein levels or nuclear morphology that affects sort efficiency).
- Or **sampling bias within PFC**: PsychAD's pediatric DLPFC samples
  may have been taken from a slightly different sub-region than its
  adolescent samples (e.g., different gyri, different cortical depths
  in the gross dissection), which would affect layer composition.

The fact that this opposite-direction composition shift exists is a
flag — but it cannot explain the per-layer C3+ gap, since that gap
is present WITHIN each layer.

## What is the residual gap?

After all the controls, the remaining 1.3-d gap (PsychAD-V3 upper
−0.27 vs Vel-V3 upper +1.02) is:

- NOT depth (per-cell-CPM + downsampling don't move it)
- NOT marker-classifier subtype confounding
- NOT continuous-maturity (H+I1)
- NOT cell-type subclass composition (K1 — gap in every layer)
- NOT chemistry (Vel-V3 vs PsychAD-V3 both V3)
- NOT region (both DLPFC)
- NOT pediatric clinical pathology (F3 — all 11 HBCC normal)
- NOT V2 contamination of the Vel signal (V3 alone holds)

What remains is most parsimoniously **a cohort-level biological or
technical signal that we cannot further decompose without:**

1. **Donor-matched experimental replication** — running the same
   tissue (one of PsychAD's pediatric DLPFCs and one of Vel's
   pediatric DLPFCs of matched age, ideally from the same brain bank)
   through the same protocol. This is the only definitive arbiter.
2. **FANS vs unsorted side-by-side** on the same pediatric brain.
   If FANS sort biases the C3+ signal, this would show it.
3. **Third-party developmental cohort** (e.g., Wang+Lu et al. fetal-
   to-young-adult atlas, Allen Brain Cell Atlas adult-only) as an
   external arbiter. Whichever the new cohort agrees with is the
   informative answer.
4. **PMI and antemortem condition metadata** for both pediatric
   cohorts (sudden death vs prolonged illness, time on ICU, etc.) —
   these affect synaptic transcript stability differently and can
   produce systematic mature-marker depression in one cohort vs the
   other.

If none of those are available, the honest conclusion is:
**this dataset combination cannot resolve whether the C3+ child→adol
drop is present in PsychAD's pediatric DLPFC neurons. Velmeshev says
yes, PsychAD says no, and after all available technical controls the
disagreement is most likely a cohort-level systematic difference (FANS
prep, donor recruitment, sample handling) that the available metadata
doesn't reveal.**

## Recommended deliverable change (updated)

For `grn_dev_multi.md`:

1. Report PER-DATASET results separately, not as a "C3+ drops in our
   data" or "doesn't drop" headline. Both are partly true; they are
   measuring different things.

2. Use per-cell-CPM mean as the aggregation rule going forward (replace
   sum-then-CPM throughout). This is the depth-balanced, mathematically
   cleaner summary; it gives ~40 % less artifact than the current method.

3. Report per-layer d alongside the aggregate (K1's table). The
   per-layer breakdown is more interpretable than the marker_annotation
   subtypes (which mix layer and maturity).

4. Flag the residual disagreement explicitly: "PsychAD's pediatric
   DLPFC controls (n = 11 HBCC normals) show no developmental drop in
   AHBA C3+ aggregate expression across cortical layers; Velmeshev
   developmental atlas (n = 14 children, V2 + V3) shows a strong drop
   across all layers. After controlling for normalization, depth, chemistry,
   region, maturity bin, and clinical pathology, the disagreement
   persists. Most likely explanations are cohort-level biological or
   technical differences (FANS sort, donor recruitment, sample
   handling, PMI) that the metadata does not allow us to discriminate
   between. Cross-validation with a third developmental cohort is
   needed to determine which dataset's signal is normative."

## Files

- `k1_d_by_layer.csv` — per-layer C3+ d, per group
- `k1_layer_composition_per_donor.csv` — per-donor layer composition
- `k1_layer_composition_summary.csv` — mean fractions per group × stage × layer
- `k1_donor_scores_by_layer.csv` — per-donor C3+ scores per layer
- `k1_d_by_layer.png` — bar plot
- `k1_layer_composition.png` — stacked composition bars
- `k2_d_by_depth_q_perCellCPM.csv` — depth-quartile d under per-cell-CPM mean
- `k2_donor_scores_by_depth_q.csv` — per-donor scores by depth quartile
- `k2_d_by_depth_q_perCellCPM.png` — d vs depth quartile, 3 groups
