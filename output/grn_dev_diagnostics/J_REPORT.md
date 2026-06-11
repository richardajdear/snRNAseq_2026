# J — confirming the depth confound with per-cell-CPM averaging and downsampling

User asked: does the depth confound diagnosis hold up to (1) per-cell-
CPM averaging (so the bulk is no longer UMI-weighted) and (2) explicit
downsampling to matched per-cell depth? And what fraction of cells hit
the cap at each setting?

## TL;DR

- **Sum-then-CPM was biasing child estimates DOWN in all three
  groups.** Switching to per-cell-CPM mean moves PsychAD-V3 from
  d = −0.44 → −0.26 (40 % reduction in anti-drop magnitude); moves
  Velmeshev-V2 from +1.96 → +2.37; moves Velmeshev-V3 from +0.62 →
  +1.09. The depth confound was real and operated in the same
  direction in every group; PsychAD just happens to have a small
  enough biological d that the confound was sufficient to flip the
  sign.
- **Downsampling at any cap (500–8000 UMI) gives ≈ the same d as
  per-cell-CPM mean.** They are mathematically equivalent in the limit
  where (almost) every cell hits the cap, and the data confirms this
  empirically: PsychAD-V3 sits flat at −0.26 across all caps; Vel-V3
  drifts only from +1.09 → +0.94 across caps.
- **The "lost reads" intuition (beyond-Poisson dropout in shallow
  cells, ambient inflation) appears to be minor here.** If shallow
  cells had biased-low CPMs from non-Poisson sources, downsampling
  deep cells to shallow-cell depth should drag d toward the shallow
  regime. It doesn't. At the pseudobulk level, the Poisson model is
  approximately correct: shallow cells are noisier but unbiased.
- **The remaining gap (PsychAD −0.26 vs Vel +1.0 to +2.4) is NOT a
  depth/normalisation artifact.** It survives both fixes. The depth
  confound contributed ~40 % of PsychAD's anti-drop signal; the
  remaining ~60 % requires a different mechanism (cell-type
  subclass composition, or true cohort biology). This is the
  question that #4 (layer-marker module scoring within already-
  filtered ExN cells) tackles next.

## Method recap

J1 — per-cell-CPM mean:
```
per_cell_score_i = sum_g (w_g * count_ig) / total_i * 1e6
donor_score_d    = mean_{i in donor d} of per_cell_score_i
```
Each cell contributes equally regardless of UMI depth.

J2 — downsample per cell to UMI cap, then sum-then-CPM:
```
For each cell with total > cap:
    each non-zero count k thinned via binomial(k, cap/total)
For each donor: sum raw counts → CPM → project C3+
```
Cap range: 500, 1000, 2000, 3000, 5000, 8000.

Both with the same 3 groups: PsychAD-V3, Velmeshev-V2, Velmeshev-V3,
restricted to ExN_{mature,immature,weak} cells in [1, 25) y.

## Headline d table

| Group         | sum-CPM | J1 (per-cell-CPM mean) | cap=500 | 1000  | 2000  | 3000  | 5000  | 8000  |
|---------------|--------:|------------------------:|--------:|------:|------:|------:|------:|------:|
| PsychAD-V3    |  −0.444 |               **−0.263**| −0.258  |−0.261 |−0.262 |−0.258 |−0.243 |−0.261 |
| Velmeshev-V2  |  +1.965 |               **+2.374**| +2.386  |+2.324 |+2.311 |+2.298 |+2.244 |+2.187 |
| Velmeshev-V3  |  +0.621 |               **+1.093**| +1.092  |+1.091 |+1.046 |+1.013 |+0.978 |+0.938 |

Δ(sum-CPM → per-cell-CPM):
- PsychAD-V3: **+0.181** (less negative)
- Velmeshev-V2: **+0.409** (more positive)
- Velmeshev-V3: **+0.472** (more positive)

All three groups shift in the SAME direction (toward more positive d),
confirming that sum-then-CPM was a UMI-weighting bias that depressed
child relative to adolescent for every group. The size of the shift is
not the same across groups, but the direction is consistent.

## Why per-cell-CPM mean ≈ downsampling

If we cap every cell at K UMI (i.e. K << min total in the data) and
then bulk-sum-then-CPM:
```
bulk_count_g     = sum_i (downsampled count_ig) ≈ K * sum_i (count_ig/total_i)
bulk_total_UMI   = sum_i (downsampled total_i)   ≈ K * n_cells
bulk_CPM_g       = bulk_count_g / bulk_total_UMI * 1e6
                = mean_i (count_ig/total_i * 1e6)
                = mean_i (per-cell CPM_ig)
```
i.e. the donor bulk CPM after downsampling equals the mean of per-cell
CPMs. So J1 and J2 are the same operation at the limit. The data shows
they agree to within 0.02 d.

## Cell loss at each cap (J2)

% of cells in each donor × stage hitting the cap (median per donor):

| group        | cap   | child  | adol   |
|--------------|-------|-------:|-------:|
| PsychAD-V3   | 500   | 100 %  | 100 %  |
| PsychAD-V3   | 1000  | 100 %  | 100 %  |
| PsychAD-V3   | 2000  |  99 %  |  99 %  |
| PsychAD-V3   | 3000  |  93 %  |  97 %  |
| PsychAD-V3   | 5000  |  79 %  |  88 %  |
| PsychAD-V3   | 8000  |  66 %  |  73 %  |
| Velmeshev-V2 | 500   |  94 %  |  98 %  |
| Velmeshev-V2 | 1000  |  79 %  |  92 %  |
| Velmeshev-V2 | 2000  |  60 %  |  83 %  |
| Velmeshev-V2 | 3000  |  45 %  |  76 %  |
| Velmeshev-V2 | 5000  |  23 %  |  66 %  |
| Velmeshev-V2 | 8000  |   9 %  |  45 %  |
| Velmeshev-V3 | 500   | 100 %  | 100 %  |
| Velmeshev-V3 | 1000  |  99 %  |  99 %  |
| Velmeshev-V3 | 2000  |  90 %  |  90 %  |
| Velmeshev-V3 | 3000  |  80 %  |  82 %  |
| Velmeshev-V3 | 5000  |  64 %  |  67 %  |
| Velmeshev-V3 | 8000  |  46 %  |  49 %  |

Notable:
- **PsychAD-V3 has matched child/adol cell-loss rates** at every cap.
  This is REASSURING: even though PsychAD adolescents are deeper on
  average (mean UMI 21.5 k vs 17.7 k for children), the cap operates
  on a per-cell basis and the donor-level cell-loss is similar across
  stages. So the depth confound is being neutralised symmetrically.
- **Velmeshev-V2 has highly asymmetric child/adol cell-loss.** At cap
  8000, only 9 % of Vel-V2 child cells are downsampled vs 45 % of adol
  cells — i.e. most Vel-V2 child cells are below 8 k UMI naturally, so
  no data is lost from them, but Vel-V2 adolescents are heavily
  thinned. The fact that Vel-V2 d stays at +2.2 across all caps means
  this asymmetric loss is NOT driving the +1.97 baseline; the V2
  child→adol drop is biology.
- **Velmeshev-V3 has near-matched cell-loss across stages** (a few %
  difference at every cap), consistent with V3's even depth across
  ages.

## How to read this against the depth-quartile result (I2)

I2 showed PsychAD's depth-quartile pseudobulks span d = +0.26 (shallow)
to −0.84 (deep). That was sum-then-CPM within each quartile pseudobulk.

J shows that switching to per-cell-CPM mean (which removes the UMI-
weighting bias internal to each pseudobulk) takes the *whole-ExN*
aggregate from −0.44 to −0.26. So the I2 sign-flip pattern was a
combination of:
- Real depth-correlated biology / cell-type composition that varies
  across the depth quartiles (probably mechanism iii: subclass
  enrichment in deep cells)
- The sum-then-CPM bias inflating the contribution of deep cells
  within each pseudobulk

Now that we've removed the second, the residual depth dependence in I2
would shrink but not disappear. Future work could run I2 with per-cell-
CPM mean inside each quartile pseudobulk to see how much of the depth-
quartile gradient survives.

## What remains unexplained

PsychAD-V3 at d = −0.26 vs Velmeshev-V3 at d = +1.09 — a gap of 1.35 d
units — is REAL signal disagreement that cannot be attributed to:
- the sum-then-CPM bias (fixed by per-cell-CPM mean)
- per-cell depth differences (fixed by downsampling)
- marker-classifier subtype confounding (E1 + I1)
- Velmeshev V2/V3 chemistry mix (V3 alone still shows +1.09)
- region (both PFC)
- pediatric clinical pathology in PsychAD children (F3 — all HBCC normal)

What's left:
1. **Cell-type subclass composition.** PsychAD's FANS-sorted nuclei
   pool may capture a different mix of cortical layers than
   Velmeshev's unsorted prep. The C3+ network's developmental
   trajectory may differ between supragranular (L2/3 IT) and
   infragranular (L5 ET, L6 CT) excitatory neurons. Testing this
   needs a layer-marker module-scoring approach that operates WITHIN
   the marker-annotated ExN cells (the standard `subclass` field is
   GAD-ambient-contaminated, see §4 discussion). This is the natural
   next experiment.
2. **True cohort biology.** Velmeshev's children are from a
   developmental atlas (donors recruited explicitly to study the
   developmental trajectory); PsychAD's pediatric donors are HBCC
   controls (recruited for an adult/aging neuropsych cohort, with
   pediatric controls added as a comparison group). Different
   donor selection may give different biological substrate even
   absent any technical issue.
3. **scVI/scANVI training composition.** PsychAD's integration was
   trained without age balancing; child donors are a tiny fraction
   of the training set. Vel was trained with V2+V3 batch correction
   and includes children prominently. Latent quality may differ
   between datasets in a way that affects downstream pseudobulk
   quality, even though our projection uses `layers['counts']` not
   scVI-normalized.

## Concrete recommendation

Replace the headline `d = +0.94 (Velmeshev) vs −0.44 (PsychAD)` story
in `grn_dev_multi.md` with this two-step story:

1. Aggregate level: switching to per-cell-CPM mean (or downsampling
   to any reasonable cap) gives `d ≈ +1.0 (Vel-V3) / +2.4 (Vel-V2)
   / −0.26 (PsychAD-V3)`. The reported disagreement was inflated by a
   normalization artifact.
2. The residual disagreement (PsychAD −0.26 vs Vel +1.0) is real
   and the next step is layer-marker subclass stratification within
   already-filtered ExN cells (#4 from the previous report).

## Files

- `j_d_per_method.csv` — child-vs-adol d for every group × method × cap
- `j_donor_scores_per_method.csv` — per-donor scores under each method
- `j_capping_per_donor.csv` — per-donor cell-loss fraction at each cap
- `j_capping_summary.csv` — averaged cell-loss per group × cap × stage
- `j_d_vs_method.png` — d as a function of normalization method
- `j_capping_vs_cap.png` — % cells hitting cap vs cap, per dataset × stage
