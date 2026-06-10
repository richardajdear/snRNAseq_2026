# Does AHBA C3 encode a within-neuron maturation program, or just neuron-vs-glia composition?

**Investigation:** `c3_maturation` · scripts in `scripts/c3_maturation/`, outputs here in `output/c3_maturation/`
· _living document, updated as results come in._

---

## TL;DR (current state)

1. **A depth-robust C3 score exists and the prior artefact is understood.** The old C3+
   aggregate is depth-biased; its age trend disagrees between V2 and V3 — *that* disagreement
   is the artefact that produced the spurious adolescent "dip". A **signed, log-CPM score**
   (`signed_logcpm`) is depth-invariant and gives the **same age trend in V3-only as in V2+V3**.
2. **The strong C3↑-with-age signal is the prenatal→postnatal differentiation jump (composition).**
   Across the full age range C3 rises steeply with age, but this is the progenitor→neuron / glia
   emergence axis — i.e. the deflationary reading.
3. **Within a fixed mature excitatory-neuron (EN) type, there is no robust *positive* adolescent
   maturation program.** If anything there is a *weak postnatal decline* in the C3 score,
   consistent in direction across all three cohorts but individually non-significant
   (meta-analysis p≈0.09). It is **not** upper-layer-specific.
4. **A reproducible within-neuron developmental program exists — but it is AHBA *C1*, not C3.**
   Per-gene within-EN age slopes replicate across cohorts (ρ=0.24–0.40) and align with C1
   (ρ≈+0.33, robust to expression confound), while C3 weight is ~0 / slightly negative. So C3
   does **not** encode the within-neuron developmental axis.
5. **There IS a genuine synaptic-gene-set postnatal decline within EN** — synaptic genes decline
   more than background in all three cohorts, highly significant in clean V3 (Herring MWU
   p=3.5e-5; U01 p=4e-12; PsychAD p=0.018). This is pruning-consistent, but it is a property of
   synaptic genes broadly (only partially overlapping C3+), not a distinct "C3 program."
6. **The data are sound for this question.** PsychAD's notorious "5% EN in young donors" is an
   artefact of its aging-reference labels; using marker-based labels, young-donor EN fractions are
   normal (~30–50%), and the labeled-EN pseudobulks are EN-pure at all ages (Appendix A).

**Interpretation.** The evidence leans **deflationary**: C3 largely tracks the
neuron-vs-glia / differentiation (composition) axis. The genuine within-neuron developmental
program lives in **C1**, not C3. There *is* a robust synaptic-gene-set postnatal decline within
neurons (consistent with synaptic overproduction→pruning), and because C3+ is enriched for synaptic
genes this plausibly explains *why C3 hits psychiatric GWAS* — but the maturation **dynamics** are
not what C3 measures. In short: C3's GWAS relevance is inherited from the synaptic genes it weights,
not from C3 itself tracking an adolescent maturation program.

---

## Key results

### 1. Depth-robustness gate (Step 0)

Four candidate C3 scores were compared on the Velmeshev ExN-per-donor pseudobulk, which contains
both V2 (n=37) and V3 (n=39) donors.

| score | age-partialled depth ρ (want ≈0) | age trend V2+V3 vs V3-only (want equal) | verdict |
|---|---|---|---|
| `pos_cpm` (old C3+ aggregate) | **+0.33** | 0.75 vs 0.82 — **0.07 gap = the artefact** | depth-biased |
| `signed_cpm` | +0.24 | 0.76 vs 0.77 | ok |
| **`signed_logcpm`** ✅ | **+0.03** | **0.82 vs 0.83 — identical** | **adopted** |
| `rank_contrast` | −0.14 | 0.81 vs 0.82 | ok |

`signed_logcpm` = Σᵢ wᵢ · log1p(CPMᵢ) over signed C3 weights. Adopted for all downstream analyses.

![Step 0 depth harness](s00b_depth_harness.png)

*Left column: each score vs age (red=V2, blue=V3) — for `signed_logcpm`/`rank_contrast` the
chemistries interleave (no batch separation); for `pos_cpm` the V2 donors sit low, dragging the
aggregate. Right column: score vs depth.*

### 2. Composition vs within-cell-type program (Step 1)

Restricting to **postnatal mature EN subtypes** and conditioning on subtype identity + sequencing
depth (donor-clustered robust SE), the C3 score trend is reported **separately for three cohorts**
(per the design that V3-Herring and V3-PsychAD are different sources from V2-U01):

| cohort | layer | ρ(age \| depth) | slope p | donors |
|---|---|---|---|---|
| **PsychAD-V3** (cleanest, largest) | all mature EN | −0.17 | 0.44 | 71 |
| | upper | −0.09 | 0.26 | 68 |
| | deep | +0.02 | 0.18 | 18 |
| **Herring-V3** | all mature EN | −0.22 | 0.18 | 17 |
| | upper | −0.22 | 0.45 | 16 |
| | deep | −0.28 | **0.044** | 13 |
| **U01-V2** | all mature EN | −0.35 | 0.079 | 15 |
| | upper | −0.43 | 0.082 | 15 |
| | deep | −0.46 | 0.13 | 13 |

**Inverse-variance meta-analysis (all mature EN, 3 cohorts): slope = −0.94, p = 0.087.**

![Step 1 within-EN by cohort and layer](s04_within_en_cohorts.png)

Reading:
- All three cohorts decline in the same direction; none is individually significant; the meta is
  marginal (p≈0.09).
- The decline is **not upper-layer-specific** (upper ≈ deep, if anything deeper-biased) — so the
  simplest "upper-layer pruning" prediction is **not** supported by these data.
- Effect magnitude tracks cohort: largest in U01-V2 (steepest, also the V2/depth-prone cohort) and
  in the child-rich Velmeshev cohorts; weakest in the adult-heavy PsychAD-V3. This could reflect
  either residual V2-depth inflation **or** that the effect is a childhood phenomenon diluted by
  PsychAD's adult-heavy sampling — disentangling these is an open question (see below).

### 3. De novo within-EN developmental program, and its (non-)alignment to C3 (Step 2)

Rather than projecting C3 weights, we compute **per-gene** within-EN age slopes (log1p CPM
residualised on subtype + depth) in each cohort, and ask three questions.

**Q1 — does a within-neuron developmental program exist & replicate?** Yes. The per-gene slope
vectors correlate across cohorts (Spearman ρ): PsychAD-V3↔Herring-V3 **0.26**, PsychAD-V3↔U01-V2
**0.24**, Herring-V3↔U01-V2 **0.40**. A reproducible within-EN program is present even though the
aggregate C3 score barely moved.

**Q2 — is that program C3?** **No — it is C1.** Spearman(per-gene slope, AHBA weight):

| cohort | C1 | C2 | C3 |
|---|---|---|---|
| PsychAD-V3 | **+0.33** | −0.01 | −0.03 |
| Herring-V3 | **+0.22** | +0.07 | −0.08 |
| U01-V2 | **+0.18** | +0.05 | −0.14 |

The C1 alignment is **not** an expression-level artefact: partialling on mean expression leaves it
essentially unchanged (PsychAD C1: 0.326 → 0.322; C3 stays ≈ −0.04). So the within-neuron
developmental axis is captured by **C1**, and C3 weight does not predict within-neuron dynamics.

**Q3 — are the dynamic genes synaptic (pruning)?** Yes, at the set level. Canonical synaptic genes
(incl. the prior NRXN1/NLGN1/GRIK… set) decline within EN more than background in **all three
cohorts**: MWU p = 0.018 (PsychAD-V3), **3.5e-5 (Herring-V3, clean V3)**, 4e-12 (U01-V2). Individual
slopes are small (≈ −0.01 to −0.05 log1p-CPM/yr); the signal is robust at the set level.
Canonical maturation *switches* (GRIN2B→GRIN2A, KCC2/NKCC1) are **not** cleanly recovered — likely
because the switch is largely perinatal and our window is postnatal.

![Step 2 de novo within-EN](s06_denovo_within_en.png)

*Left: cross-cohort slope replication. Middle: within-EN slope vs C3 weight (flat). Right: synaptic
gene slopes — predominantly negative across cohorts.*

### Open items

- [x] **De novo within-EN axis (Step 2)** — done: program exists, is C1-aligned not C3, synaptic-set
      decline replicates.
- [ ] **Normalisation robustness** — current pseudobulk is *sum-counts-then-CPM* (deep-cell biased);
      repeat with *per-cell-CPM-then-mean*. _(sbatch job running)_
- [ ] **Characterise C1** — what is AHBA C1 biologically, and is the C1↔within-neuron-development
      link itself interesting?
- [ ] **Matched age windows** — separate "childhood effect diluted in adults" from "V2-depth
      inflation" across cohorts.

---

## Methods

### Datasets and cohorts

Three cohorts are treated independently (V3-Herring and V3-PsychAD are distinct sources from
V2-U01):

| cohort | source | chemistry | age coverage (this analysis) | role |
|---|---|---|---|---|
| PsychAD-V3 | PsychAD | V3 | 5–30 y (adult-heavy) | large, clean, but few children |
| Herring-V3 | Velmeshev "Herring" | V3 | 1–25 y | cleanest child coverage, small |
| U01-V2 | Velmeshev "U01" | V2 | 1–22 y | child coverage, but V2 depth-prone |

Velmeshev "Ramos" (V3) is ~all prenatal and is excluded from the postnatal test.
Pseudobulks are pre-computed per (donor × cell type) by the project pipeline
(`code/pipeline/pseudobulk.py`); we read the small pseudobulk h5ads (login-node safe).

### Defining "mature EN"

`cell_type_aligned` subtype labels, restricted to excitatory subtypes and excluding
immature/newborn/subplate/progenitor classes. Explicitly:
- **Velmeshev**: L2-3, L4, L5, L5-6-IT, L6 (upper = L2-3, L4).
- **PsychAD**: EN_L2_3_IT, EN_L3_5_IT_1/2/3, EN_L6_IT_1/2, EN_L6_CT, EN_L6B, EN_L5_6_NP
  (upper = L2_3_IT + L3_5_IT_*).

These labels are validated as genuinely EN at all ages in Appendix A (marker purity). **Subtype-
level** accuracy in young donors is not separately validated and is a caveat for the layer split.

### The C3 score

Signed AHBA C3 loadings (6641 genes mapped to Ensembl: 3456 positive "C3+", 3185 negative "C3−")
are applied as `signed_logcpm` = Σᵢ wᵢ · log1p(CPMᵢ). Inside a pure-EN population the C3− (glial)
genes are near-floor, so the score is effectively driven by the C3+ (synaptic/neuronal) pole.

### What "conditioning on depth" means here, and what it does/doesn't fix

- **CPM** puts pseudobulks of different total depth on a common per-million scale, removing the
  *linear library-size* effect. It does **not** remove *dropout / compositional* depth effects:
  shallower samples have more zeros and systematically under-detect mid/low-expression genes,
  distorting the relative profile. So CPM alone does **not** fully control depth.
- **"Conditioning on depth"** = adding log10(total pseudobulk counts) as a covariate in the
  regression (or partialling it in a Spearman correlation). This removes residual *linear*
  association between the score and remaining depth variation, but not nonlinear dropout effects.
- Therefore depth-covariate adjustment is **necessary but not sufficient**. The stronger controls
  are: (a) choosing a score that is empirically depth-insensitive (`signed_logcpm`, partialled
  ρ=0.03) and gives identical V2/V3 trends (Step 0); (b) a negative-control null band from
  expression-matched random gene sets; (c) **reporting V2 and V3 cohorts separately**, so a
  V2-specific age×depth confound cannot masquerade as biology in the V3 cohorts; (d) within-cohort
  replication, where depth is more uniform.
- **The specific V2 age×depth worry** (younger U01-V2 donors being shallower, so age and depth
  correlate within that cohort) is handled by separating cohorts and leaning on the two **V3**
  cohorts, where the same-direction decline appears without any V2 data.

### De novo within-EN slopes (Step 2)

Per cohort, on mature-EN postnatal pseudobulks, each gene's log1p(CPM) and age are residualised on
[subtype dummies + log10 depth] and the per-gene slope is cov(resid_expr, resid_age)/var(resid_age).
This is a within-subtype, depth-controlled developmental slope. Alignment to AHBA components uses
Spearman correlation over shared genes; the synaptic set is a curated list of canonical
synaptic/maturation genes (NRXN/NLGN/GRIK/GRM/DLG/SHANK/GRIN/GRIA/GABR/… incl. the prior drop set),
compared to all genes by Mann–Whitney U.

### Pseudobulk normalisation — current choice and caveat

The current scores use **sum-counts-then-CPM** (the standard pseudobulk library-size
normalisation used by edgeR/DESeq2), in which deeper cells contribute proportionally more to the
donor profile. An alternative, **per-cell-CPM-then-mean**, gives every cell equal weight and is
less sensitive to a few deep cells. Robustness of the within-EN decline to this choice is being
tested (open item above).

---

## Appendices

### Appendix A — Data sanity: PsychAD cell-type misclassification check

**Concern:** PsychAD native labels derive from an aging/dementia reference, so young-donor EN
subtypes might be mislabeled.

**A1. Composition vs age (marker-based labels).** Using PsychAD's independent *marker-based*
annotation (ExN_*/InN/glia/Unknown), young-donor EN fractions are normal — PsychAD-V3 EN fraction
median 0.51 (<2y), 0.30 (2–5y), 0.36 (5–10y) — **not** the ~5% seen with the native aging-reference
labels. The "5% EN in young donors" problem is therefore a **labeling** artefact, and the
marker-based annotation is the trustworthy one. (Velmeshev EN fraction *declines* with age, e.g.
Herring 0.46→0.08, but since all analyses are *within* EN this composition shift is conditioned out;
it does mean older Velmeshev EN pseudobulks rest on fewer cells.)

![EN fraction vs age](s03A_EN_fraction_vs_age.png)

**A2. Marker purity inside labeled-EN pseudobulks.** Across all cohorts and age bins, the labeled-EN
subtype pseudobulks express EN markers (SLC17A7/SATB2/RBFOX3) at log1p-CPM ≈5.5–6.1 and IN+glia
markers (GAD1/2, AQP4/GFAP, PLP1/MOBP, PDGFRA, CSF1R) at ≈1.7–2.0 — a clean ~4-log gap that is
**identical in young (<5y) and old (20+) donors**. So the within-EN analysis operates on genuine EN
cells at all ages.

![EN marker purity](s03B_en_marker_purity.png)

### Appendix B — Velmeshev per-subtype trajectories (postnatal)

![Velmeshev per-subtype](s01a_within_celltype_trajectory.png)

---

## Reproducibility

| script | purpose |
|---|---|
| `_lib_c3.py` | signed C3 weights (Ensembl), scorers, depth metrics, downsampling, null |
| `s00b_depth_harness.py` | Step 0 depth-robustness gate → `signed_logcpm` |
| `s01a/s01b/s01c…` | Step 1 development (per-subtype, confound check, PsychAD) |
| `s03_sanity_composition_markers.py` | Appendix A composition + marker purity |
| `s04_within_en_cohorts.py` | 3-cohort within-EN aggregate trajectory + layer split + meta |
| `s05_norm_robustness.py` | (sbatch) sum-CPM vs per-cell-CPM-then-mean robustness |
| `s06_denovo_within_en.py` | **Step 2** per-gene within-EN program, C3/C1 alignment, synaptic set |

Run pattern (login-safe, small pseudobulks):
`singularity exec --pwd $PWD <sif> micromamba run -n shortcake_default python3 -u scripts/c3_maturation/<script>.py`
