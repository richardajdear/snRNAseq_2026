# Is there a third independent pediatric-DLPFC cohort for the C3+ analysis?

> **⚠ CORRECTION (2026-06-08) — this document's central premise was wrong.**
> `velmeshev.h5ad` is **not** one Velmeshev-2023 study split by chemistry. It is
> a **composite atlas of four datasets** (`y3_diagnose.py`): **U01** (original
> Velmeshev, 10x **V2**), **Ramos** (V3), **Herring 2022** (V2+V3), **Trevino**
> (multiome). In the PFC developmental window **Ramos is entirely prenatal** and
> the postnatal "Velmeshev-V3" series **is Herring**. So: **Herring is already
> on disk** (it is postnatal Velmeshev-V3, present in every "Velmeshev-V3" number
> throughout); **PsychAD-V3 vs Velmeshev-V3 is genuinely independent** (PsychAD
> vs Herring, different labs) — the cross-cohort replication this doc said we
> lacked already exists; and the "V2/V3 = within-study chemistry control" framing
> below is **wrong** (V2 ≈ U01, V3 = Ramos+Herring are different studies; the V2
> confound is the U01 study). The Wang/Zhu/donor-recycling parts still hold. See
> FINAL_REPORT §0.5.

---

**Date:** 2026-06-07 · audit script `v_cohort_audit.py` ·
`v_cohort_age_audit.csv`. Investigates whether an independent
childhood→adolescence DLPFC snRNA-seq cohort exists to arbitrate the
residual PsychAD-vs-Velmeshev gap. **Conclusion: not really** — and the
reasons are structural, not just "we haven't downloaded it yet."

## 1. What our current "three cohorts" actually are

`Vel_prepost_noage_tuning5.yaml` has a **single** source — Velmeshev et al.
2023 (`velmeshev.h5ad`) — split by 10x chemistry into V2 and V3
(`transform_batch: VELMESHEV-V3`). So **Velmeshev-V2 and Velmeshev-V3 are
the same study**, and their agreement is a within-study chemistry control,
**not** independent replication. (They are *not* Herring, contrary to a
common assumption.)

We therefore have **two independent studies**:
- **PsychAD** — HBCC + Aging cohorts (PsychAD consortium; NIH NeuroBioBank /
  HBCC + Mount Sinai), FANS-sorted, aging-skewed.
- **Velmeshev 2023** — unsorted developmental atlas.

## 2. On-disk candidates audited (Wang, Zhu) — both fail

Using the pipeline's own backed readers, PFC-restricted, developmental
window 1–25 y:

| dataset | study / type | total donors | age range | PFC donors in 1–25 y | childhood 1–12 y | adolescence 12–25 y | verdict |
|---|---|---:|---|---:|---:|---:|---|
| **Velmeshev** | Velmeshev 2023 (in use) | 106 | −0.5 … 54 y | **32** | ~18 | ~14 | cohort 2 |
| **PsychAD** | PsychAD consortium (in use) | 413 | 0 … 100+ | ~11 children (HBCC) | ~11 | many | cohort 1 (FANS) |
| **Wang** | Wang — fetal multiome | 27 | **−0.6 … 13.9 y** | 4 (**all 12–18 y**) | **0** | 4 | **unusable — no childhood** |
| **Zhu** | Zhu 2023 — multiome | 12 | −0.4 … 39 y | 5 | ~2 | ~3 | **too small (1–2/bin)** |

- **Wang** is a fetal/perinatal atlas: 18 of 27 donors are prenatal, the
  rest infants, plus 4 adolescents (12–18 y) — and **not a single donor in
  the 1–12 y childhood window** (PFC or otherwise). It cannot support a
  childhood→adolescence contrast. (The old FINAL_REPORT "Wang+Lu" suggestion
  was wrong.)
- **Zhu** is genuinely independent but tiny: ~2 childhood + ~3 adolescent
  PFC donors, 1–2 per age bin. It could contribute only an anecdotal
  direction check, not a powered estimate.

## 3. External landscape (literature, to 2026)

| candidate | region | dev window | independent of ours? | usable? |
|---|---|---|---|---|
| **Lifespan DLPFC atlas, 284 donors 0–97 y** (medRxiv 2024–25) | DLPFC | dense, incl. childhood/adolescence | **NO — "a subset of the PsychAD Consortium", HBCC + Mt Sinai** | same donors as PsychAD |
| **PsychAD cross-disorder atlas** (Nat Sci Data 2025) | DLPFC | some pediatric | NO — PsychAD consortium | same source |
| **Herring et al. 2022** (Cell) | dlPFC | gestation→adult (childhood ≥1–<10, adol ≥10–<20) | **YES (different lab)** | best option, but **small (~2 dozen lifespan donors → a handful of children)**; not on disk |
| Velmeshev 2019 (ASD) | ACC/PFC | pediatric | same lab as Velmeshev 2023; donor overlap likely | weak |
| Trevino/Bhaduri/Polioudakis/Nowakowski | cortex | **fetal only** | yes | no postnatal window |
| Hodge 2019 (MTG), Siletti, Ma, Jorstad, ABCA/Siletti | MTG / whole-brain | **adult only** | yes | wrong age &/or region |

## 4. The structural reason this is hard

The bottleneck is **not** the number of datasets — it is that **pediatric
(especially 1–12 y) postmortem prefrontal cortex is intrinsically rare and
concentrated in a few NIH NeuroBioBank sites (HBCC, Maryland/UMBTB).**
Consequently:

- The largest, newest pediatric-DLPFC resource (the 284-donor lifespan
  atlas) **is our own consortium** (PsychAD/HBCC) — it re-uses, not
  replicates, our donors.
- Even a different-lab atlas like Herring draws pediatric tissue from the
  same NeuroBioBank pool, so **individual child donors may literally be
  shared** across "independent" studies.
- Every atlas — ours included — has only ~5–20 donors in the 1–25 y window,
  so "replication" would add a handful of donors, not statistical power.

True donor-level independence in this age window is therefore largely
illusory, which is why the earlier conclusion ("no good cohorts") holds and
is in fact stronger than it first appears.

## 5. Recommendations

1. **Treat Herring 2022 as a direction tie-breaker, not a powered cohort.**
   Process it through the marker-annotated, maturity-stratified pipeline and
   ask only: does the q0/immature C3+ child→adol drop replicate in sign?
   First **check Herring's donor manifest against HBCC** to quantify
   overlap; a positive result is only meaningful for non-shared donors.
2. **Enlarge the PsychAD pediatric sample via the 284-donor lifespan atlas
   (Synapse, controlled access).** Our current extraction has ~11 HBCC
   children; the lifespan atlas likely has more 1–19 y DLPFC donors from the
   same consortium. This buys **power within PsychAD**, not independence,
   and directly strengthens the within-PsychAD q0 result (which is the
   FANS-robust one).
3. **Use Zhu only as a weak supplementary direction check** (n ≈ 2–3 per
   stage); do not weight it.
4. **Recalibrate expectations.** The C3+ developmental question is near the
   ceiling of what postmortem human pediatric PFC snRNA-seq can currently
   arbitrate. The strongest remaining evidence is internal: the
   maturity-stratified q0 agreement between PsychAD-V3 and Velmeshev (two
   genuinely independent studies, different prep), the within-state
   decomposition, and — if pursued — an enlarged PsychAD pediatric sample.

### Artifacts
`v_cohort_audit.py`, `v_cohort_age_audit.csv` (age-bin × region × dataset
donor/cell counts).
