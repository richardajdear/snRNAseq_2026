# 02 — Composition trends · interpretation

## Key finding: PsychAD young donors are relabelled away from Excitatory in the joint scANVI integration

In `joint scANVI` labels, PsychAD donors <1 y have 0% Excitatory cells (only Inhibitory, OPC, Microglia). In `PsychAD-only` scANVI on the same donors, >66% are labelled Excitatory. See `psychad_joint_vs_original_excitatory_pct.csv` and `fig06_psychad_relabel_excitatory.png`.

## Bin-matched Mann–Whitney (Excitatory%, Vel vs PsychAD)

| label_source | age_bin | mean_vel | mean_psy | n_vel | n_psy | p |
|---|---|---|---|---|---|---|
| joint_scanvi | <1 | 0.565 | 0 | 38 | 10 | 1.35e-06 |
| joint_scanvi | 1-2 | 0.401 | 0.1 | 3 | 4 | 0.199 |
| joint_scanvi | 2-5 | 0.394 | 0.0738 | 8 | 4 | 0.0209 |
| joint_scanvi | 5-10 | 0.276 | 0.115 | 5 | 3 | 0.25 |
| joint_scanvi | 10-15 | 0.242 | 0.265 | 7 | 17 | 0.799 |
| joint_scanvi | 15-20 | 0.204 | 0.213 | 6 | 23 | 0.935 |
| joint_scanvi | 20-25 | 0.325 | 0.242 | 3 | 19 | 0.387 |
| joint_scanvi | 30-50 | 0.462 | 0.254 | 4 | 57 | 0.0134 |

Read this row by row:
- `<1` and `2-5` y bins: huge Excitatory% gap (joint scANVI gives PsychAD ≈ 0%).
- `10-15` and `15-20` y bins: datasets agree (p>0.7, means within a few percentage points).
- `30-50` y bin: Vel n=4 (only the few old Velmeshev donors), noisy.

## Within-dataset Spearman ρ of Excitatory% vs age

| label_source | cell_class | source_chem | n | rho_raw | p_raw | rho_clr | p_clr |
|---|---|---|---|---|---|---|---|
| joint_scanvi | Excitatory | PSYCHAD-V3 | 190 | 0.183 | 0.0114 | 0.307 | 1.6e-05 |
| joint_scanvi | Excitatory | VELMESHEV-V2 | 35 | -0.597 | 0.000155 | -0.734 | 5.31e-07 |
| joint_scanvi | Excitatory | VELMESHEV-V3 | 39 | -0.691 | 1.14e-06 | -0.759 | 2.14e-08 |
| vel_original | Excitatory | VELMESHEV-V2 | 37 | -0.661 | 8.36e-06 | -0.792 | 5.14e-09 |
| vel_original | Excitatory | VELMESHEV-V3 | 39 | -0.739 | 7.55e-08 | -0.872 | 4.64e-13 |
| psychad_original | Excitatory | PSYCHAD-V3 | 200 | 0.155 | 0.0281 | -0.0533 | 0.453 |

- Velmeshev: strongly negative under both raw% and CLR — robust developmental decrease.
- PsychAD: positive under raw% but ~0 under CLR for the original labels — its "increase" is compositional (Oligo gain shrinks others). Under joint scANVI labels CLR is +0.31, i.e. label-transfer artificially gives PsychAD a *positive* developmental signal.

## Reading the heatmap (`fig03_all_classes_heatmap.png`)

- Confirms that Excitatory loss across age is mathematically partnered with Oligo gain (composition is a simplex).
- PsychAD donors <5 y have anomalously low Excitatory and high OPC under joint labels.

## Donor counts per age bin

| label_source | source_chem | age_bin | n_donors |
|---|---|---|---|
| joint_scanvi | PSYCHAD-V3 | <1 | 10 |
| joint_scanvi | PSYCHAD-V3 | 1-2 | 4 |
| joint_scanvi | PSYCHAD-V3 | 2-5 | 4 |
| joint_scanvi | PSYCHAD-V3 | 5-10 | 3 |
| joint_scanvi | PSYCHAD-V3 | 10-15 | 17 |
| joint_scanvi | PSYCHAD-V3 | 15-20 | 23 |
| joint_scanvi | PSYCHAD-V3 | 20-25 | 19 |
| joint_scanvi | PSYCHAD-V3 | 25-30 | 20 |
| joint_scanvi | PSYCHAD-V3 | 30-50 | 57 |
| joint_scanvi | PSYCHAD-V3 | 50-70 | 22 |
| joint_scanvi | PSYCHAD-V3 | 70+ | 11 |
| joint_scanvi | VELMESHEV-V2 | <1 | 15 |
| joint_scanvi | VELMESHEV-V2 | 1-2 | 1 |
| joint_scanvi | VELMESHEV-V2 | 2-5 | 5 |
| joint_scanvi | VELMESHEV-V2 | 5-10 | 2 |
| joint_scanvi | VELMESHEV-V2 | 10-15 | 5 |
| joint_scanvi | VELMESHEV-V2 | 15-20 | 2 |
| joint_scanvi | VELMESHEV-V2 | 20-25 | 2 |
| joint_scanvi | VELMESHEV-V2 | 30-50 | 3 |
| joint_scanvi | VELMESHEV-V3 | <1 | 23 |
| joint_scanvi | VELMESHEV-V3 | 1-2 | 2 |
| joint_scanvi | VELMESHEV-V3 | 2-5 | 3 |
| joint_scanvi | VELMESHEV-V3 | 5-10 | 3 |
| joint_scanvi | VELMESHEV-V3 | 10-15 | 2 |
| joint_scanvi | VELMESHEV-V3 | 15-20 | 4 |
| joint_scanvi | VELMESHEV-V3 | 20-25 | 1 |
| joint_scanvi | VELMESHEV-V3 | 30-50 | 1 |
| psychad_original | PSYCHAD-V3 | <1 | 11 |
| psychad_original | PSYCHAD-V3 | 1-2 | 4 |
| psychad_original | PSYCHAD-V3 | 2-5 | 4 |
| psychad_original | PSYCHAD-V3 | 5-10 | 4 |
| psychad_original | PSYCHAD-V3 | 10-15 | 17 |
| psychad_original | PSYCHAD-V3 | 15-20 | 23 |
| psychad_original | PSYCHAD-V3 | 20-25 | 22 |
| psychad_original | PSYCHAD-V3 | 25-30 | 22 |
| psychad_original | PSYCHAD-V3 | 30-50 | 59 |
| psychad_original | PSYCHAD-V3 | 50-70 | 23 |
| psychad_original | PSYCHAD-V3 | 70+ | 11 |
| vel_original | VELMESHEV-V2 | <1 | 17 |
| vel_original | VELMESHEV-V2 | 1-2 | 1 |
| vel_original | VELMESHEV-V2 | 2-5 | 5 |
| vel_original | VELMESHEV-V2 | 5-10 | 2 |
| vel_original | VELMESHEV-V2 | 10-15 | 5 |
| vel_original | VELMESHEV-V2 | 15-20 | 2 |
| vel_original | VELMESHEV-V2 | 20-25 | 2 |
| vel_original | VELMESHEV-V2 | 30-50 | 3 |
| vel_original | VELMESHEV-V3 | <1 | 23 |
| vel_original | VELMESHEV-V3 | 1-2 | 2 |
| vel_original | VELMESHEV-V3 | 2-5 | 3 |
| vel_original | VELMESHEV-V3 | 5-10 | 3 |
| vel_original | VELMESHEV-V3 | 10-15 | 2 |
| vel_original | VELMESHEV-V3 | 15-20 | 4 |
| vel_original | VELMESHEV-V3 | 20-25 | 1 |
| vel_original | VELMESHEV-V3 | 30-50 | 1 |