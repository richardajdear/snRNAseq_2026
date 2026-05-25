# D11/D12/D13 — PsychAD <1y biological characterisation

Raw counts from base files: Wang/Vel via `.raw.X`, PsychAD via `.X`.

## D11 — Per-donor marker analysis (PsychAD <1y, sorted by age)

| individual | age_mo | devstage | n_cells | UMI/cell | SATB2_frac1 | SATB2_frac3 | SATB2_ret | SLC17A7_frac1 | GAD1_frac1 | GAD1_frac3 | EN/IN_cpm |
|-----------|-------:|---------|--------:|---------:|------------:|------------:|----------:|--------------:|-----------:|-----------:|----------:|
| Donor_122 | 1 | 1-month-old stage | 3,324 | 9,439 | 0.219 | 0.054 | 0.25 | 0.102 | 0.644 | 0.385 | 0.176 |
| Donor_1208 | 2 | 2-month-old stage | 3,140 | 20,452 | 0.181 | 0.039 | 0.21 | 0.088 | 0.696 | 0.492 | 0.092 |
| Donor_316 | 3 | 3-month-old stage | 3,103 | 14,873 | 0.197 | 0.056 | 0.28 | 0.131 | 0.593 | 0.361 | 0.148 |
| Donor_594 | 3 | 3-month-old stage | 2,707 | 14,014 | 0.189 | 0.059 | 0.32 | 0.113 | 0.608 | 0.427 | 0.141 |
| Donor_920 | 3 | 3-month-old stage | 392 | 21,215 | 0.217 | 0.097 | 0.45 | 0.166 | 0.620 | 0.472 | 0.175 |
| Donor_47 | 4 | 4-month-old stage | 2,729 | 17,131 | 0.164 | 0.064 | 0.39 | 0.126 | 0.601 | 0.470 | 0.141 |
| Donor_1326 | 4 | 4-month-old stage | 3,875 | 12,055 | 0.192 | 0.083 | 0.43 | 0.132 | 0.603 | 0.358 | 0.168 |
| Donor_1171 | 4 | 4-month-old stage | 2,803 | 20,543 | 0.156 | 0.064 | 0.41 | 0.095 | 0.542 | 0.413 | 0.127 |
| Donor_648 | 4 | 4-month-old stage | 3,847 | 11,969 | 0.159 | 0.037 | 0.23 | 0.058 | 0.517 | 0.251 | 0.135 |
| Donor_202 | 6 | 6-month-old stage | 1,772 | 8,598 | 0.115 | 0.014 | 0.12 | 0.050 | 0.414 | 0.131 | 0.122 |
| Donor_503 | 6 | 6-month-old stage | 4,565 | 18,545 | 0.294 | 0.097 | 0.33 | 0.188 | 0.598 | 0.421 | 0.216 |

SATB2_frac1 range: [0.115, 0.294]  SATB2_retention (frac3/frac1): mean=0.31  (Wang cohort=0.77, Vel=0.75)

## D12 — Fraction of cells with detectable marker expression

| group | n_cells | EN_frac(>=1) | IN_frac(>=1) | EN_frac(>=3) | IN_frac(>=3) | EN/IN_frac1 | EN/IN_frac3 |
|-------|--------:|-------------:|-------------:|-------------:|-------------:|------------:|------------:|
| PSYCHAD_under1y | 32,257 | 0.116 | 0.445 | 0.022 | 0.231 | 0.262 | 0.097 |
| PSYCHAD_1_5y | 31,343 | 0.146 | 0.277 | 0.048 | 0.118 | 0.528 | 0.408 |
| WANG_under1y | 31,914 | 0.244 | 0.202 | 0.090 | 0.063 | 1.207 | 1.427 |
| VEL_V3_under1y | 62,711 | 0.308 | 0.187 | 0.120 | 0.040 | 1.647 | 3.039 |

## D13 — Ambient RNA check (non-neuronal via scANVI cell_type_aligned)

| group | n_nonneuronal | GAD1_cpm | GAD2_cpm | SLC32A1_cpm | SLC17A7_cpm | IN/EN ratio |
|-------|-------------:|---------:|---------:|------------:|------------:|------------:|
| PSYCHAD_under1y | 15,030 | 0.629 | 0.121 | 0.022 | 0.112 | 5.36 |
| PSYCHAD_1_5y | 18,811 | 0.562 | 0.123 | 0.020 | 0.142 | 3.73 |
| WANG_under1y | 9,607 | 0.464 | 0.036 | 0.009 | 0.066 | 4.46 |
| VEL_V3_under1y | 15,572 | 0.833 | 0.118 | 0.025 | 0.288 | 1.40 |

## D13b — Ambient RNA check (source-native labels: HBCC class / cell_type_aligned)

| group | label_source | n_nonneuronal | GAD1_cpm | GAD2_cpm | SLC32A1_cpm | SLC17A7_cpm | IN/EN ratio |
|-------|-------------|-------------:|---------:|---------:|------------:|------------:|------------:|
| PSYCHAD_under1y | HBCC_class | 15,037 | 0.629 | 0.121 | 0.022 | 0.112 | 5.32 |
| PSYCHAD_1_5y | HBCC_class | 18,811 | 0.562 | 0.123 | 0.020 | 0.142 | 3.73 |
| WANG_under1y | cell_type_aligned | 9,607 | 0.464 | 0.036 | 0.009 | 0.066 | 4.46 |
| VEL_V3_under1y | cell_type_aligned | 15,572 | 0.833 | 0.118 | 0.025 | 0.288 | 1.40 |
