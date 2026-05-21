# 01 — Batch-correction audit · interpretation

Pseudobulk: `by_cell_class.h5ad`
GRN: `ahba_dme_hcp_top8kgenes_weights.csv`
Gene filter: no

## V3-pooled Excitatory child→adolescence (fixed window 1–9 / 10–20)
Cohen's d sign convention: positive ⇒ childhood > adolescence (downward trajectory).

| layer | d | p | n_child | n_adol |
|---|---|---|---|---|
| counts_cpm | -0.1998 | 0.8063 | 9 | 33 |
| scvi_normalized | -0.1373 | 0.6457 | 9 | 33 |
| scanvi_normalized | -0.3837 | 0.2833 | 9 | 33 |

## Excitatory best-of-grid (mirroring the notebook's 4D sensitivity grid)
Cohen's d with largest |d| over (CHILD_START × CHILD_END × ADOL_START × ADOL_END).

| cell_class | layer | stratum | best_d | best_p | n_child | n_adol | child_window | adol_window |
|---|---|---|---|---|---|---|---|---|
| Excitatory | counts_cpm | Vel-V2 | 3.018 | 0.001332 | 8 | 6 | (1, 8) | (12, 20) |
| Excitatory | counts_cpm | Vel-V3 | -1.208 | 0.4 | 4 | 3 | (3, 10) | (12, 18) |
| Excitatory | counts_cpm | Vel-all | 1.24 | 0.004529 | 14 | 13 | (1, 8) | (12, 22) |
| Excitatory | counts_cpm | PsychAD-V3 | -0.7186 | 0.2624 | 3 | 31 | (1, 9) | (11, 22) |
| Excitatory | counts_cpm | V3-pooled | -0.6471 | 0.2211 | 6 | 36 | (3, 10) | (12, 22) |
| Excitatory | scvi_normalized | Vel-V2 | 0.9417 | 0.3152 | 7 | 4 | (2, 8) | (12, 18) |
| Excitatory | scvi_normalized | Vel-V3 | -0.7846 | 0.4 | 4 | 3 | (3, 10) | (12, 18) |
| Excitatory | scvi_normalized | Vel-all | 0.6981 | 0.1486 | 11 | 11 | (2, 8) | (12, 20) |
| Excitatory | scvi_normalized | PsychAD-V3 | -1.226 | 0.1197 | 3 | 31 | (1, 9) | (11, 22) |
| Excitatory | scvi_normalized | V3-pooled | -0.4938 | 0.2498 | 6 | 36 | (3, 10) | (12, 22) |
| Excitatory | scanvi_normalized | Vel-V2 | 0.9449 | 0.2303 | 7 | 4 | (2, 8) | (12, 18) |
| Excitatory | scanvi_normalized | Vel-V3 | -0.6771 | 0.4 | 4 | 3 | (3, 10) | (12, 18) |
| Excitatory | scanvi_normalized | Vel-all | 0.5909 | 0.131 | 11 | 11 | (2, 8) | (12, 20) |
| Excitatory | scanvi_normalized | PsychAD-V3 | -1.36 | 0.1197 | 3 | 31 | (1, 9) | (11, 22) |
| Excitatory | scanvi_normalized | V3-pooled | -0.6694 | 0.1595 | 6 | 36 | (3, 10) | (12, 22) |

## Interpretation cues

- If `counts_cpm` shows d>0 (positive child→adol drop) while `scvi_normalized`/`scanvi_normalized` reverse it, batch correction is responsible for the flip observed between the non-integrated and joint integrations.
- Inspect `fig01` (C3+ vs age, Excitatory) — does the slope change sign across layers?
- Inspect `fig03` — are PsychAD adolescent donors pushed UP while Velmeshev adolescents are pushed DOWN by batch correction? If so, scVI is mistaking developmental signal for batch effect.
- Inspect `fig05` — do neuron markers and housekeeping genes show similar shifts, or only neuron markers? Targeted distortion of the regulon set is more concerning than uniform shrinkage.
- Inspect `fig04` — on raw `counts_cpm` we expect dataset clustering on PC1; on `scvi_normalized` we expect age to become the dominant axis. If age still does not drive PC1 after correction, the integration may be over-aggressive.

## Full effect-size table

| cell_class | layer | stratum | d | p | n_child | n_adol |
|---|---|---|---|---|---|---|
| Astrocytes | counts_cpm | Vel-V2 | 1.446 | 0.07343 | 6 | 7 |
| Astrocytes | counts_cpm | Vel-V3 | -0.01079 | 0.3939 | 6 | 6 |
| Astrocytes | counts_cpm | Vel-all | 0.8232 | 0.0276 | 12 | 13 |
| Astrocytes | counts_cpm | PsychAD-V3 | -1.07 | 0.2505 | 4 | 19 |
| Astrocytes | counts_cpm | V3-pooled | -0.4471 | 0.3153 | 10 | 25 |
| Astrocytes | scvi_normalized | Vel-V2 | 1.361 | 0.01399 | 6 | 7 |
| Astrocytes | scvi_normalized | Vel-V3 | 0.5956 | 0.06494 | 6 | 6 |
| Astrocytes | scvi_normalized | Vel-all | 1.046 | 0.002118 | 12 | 13 |
| Astrocytes | scvi_normalized | PsychAD-V3 | -0.3755 | 0.5574 | 4 | 19 |
| Astrocytes | scvi_normalized | V3-pooled | 0.03966 | 0.8124 | 10 | 25 |
| Astrocytes | scanvi_normalized | Vel-V2 | 1.549 | 0.02214 | 6 | 7 |
| Astrocytes | scanvi_normalized | Vel-V3 | 0.7048 | 0.06494 | 6 | 6 |
| Astrocytes | scanvi_normalized | Vel-all | 1.186 | 0.001211 | 12 | 13 |
| Astrocytes | scanvi_normalized | PsychAD-V3 | -0.3433 | 0.5574 | 4 | 19 |
| Astrocytes | scanvi_normalized | V3-pooled | 0.02483 | 0.6481 | 10 | 25 |
| Excitatory | counts_cpm | Vel-V2 | 2.278 | 0.002176 | 8 | 7 |
| Excitatory | counts_cpm | Vel-V3 | 0.3762 | 0.2403 | 6 | 6 |
| Excitatory | counts_cpm | Vel-all | 1.127 | 0.007077 | 14 | 13 |
| Excitatory | counts_cpm | PsychAD-V3 | -0.646 | 0.3498 | 3 | 27 |
| Excitatory | counts_cpm | V3-pooled | -0.1998 | 0.8063 | 9 | 33 |
| Excitatory | scvi_normalized | Vel-V2 | 0.6804 | 0.281 | 8 | 7 |
| Excitatory | scvi_normalized | Vel-V3 | 0.5654 | 0.3095 | 6 | 6 |
| Excitatory | scvi_normalized | Vel-all | 0.591 | 0.1523 | 14 | 13 |
| Excitatory | scvi_normalized | PsychAD-V3 | -1.166 | 0.135 | 3 | 27 |
| Excitatory | scvi_normalized | V3-pooled | -0.1373 | 0.6457 | 9 | 33 |
| Excitatory | scanvi_normalized | Vel-V2 | 0.5383 | 0.4634 | 8 | 7 |
| Excitatory | scanvi_normalized | Vel-V3 | 0.4481 | 0.3095 | 6 | 6 |
| Excitatory | scanvi_normalized | Vel-all | 0.4642 | 0.2159 | 14 | 13 |
| Excitatory | scanvi_normalized | PsychAD-V3 | -1.253 | 0.1547 | 3 | 27 |
| Excitatory | scanvi_normalized | V3-pooled | -0.3837 | 0.2833 | 9 | 33 |
| Glia | counts_cpm | Vel-V2 | nan | nan | 0 | 0 |
| Glia | counts_cpm | Vel-V3 | nan | nan | 0 | 0 |
| Glia | counts_cpm | Vel-all | nan | nan | 0 | 0 |
| Glia | counts_cpm | PsychAD-V3 | nan | nan | 0 | 0 |
| Glia | counts_cpm | V3-pooled | nan | nan | 0 | 0 |
| Glia | scvi_normalized | Vel-V2 | nan | nan | 0 | 0 |
| Glia | scvi_normalized | Vel-V3 | nan | nan | 0 | 0 |
| Glia | scvi_normalized | Vel-all | nan | nan | 0 | 0 |
| Glia | scvi_normalized | PsychAD-V3 | nan | nan | 0 | 0 |
| Glia | scvi_normalized | V3-pooled | nan | nan | 0 | 0 |
| Glia | scanvi_normalized | Vel-V2 | nan | nan | 0 | 0 |
| Glia | scanvi_normalized | Vel-V3 | nan | nan | 0 | 0 |
| Glia | scanvi_normalized | Vel-all | nan | nan | 0 | 0 |
| Glia | scanvi_normalized | PsychAD-V3 | nan | nan | 0 | 0 |
| Glia | scanvi_normalized | V3-pooled | nan | nan | 0 | 0 |
| Inhibitory | counts_cpm | Vel-V2 | 2.807 | 0.0006216 | 8 | 7 |
| Inhibitory | counts_cpm | Vel-V3 | -0.01279 | 0.9307 | 6 | 5 |
| Inhibitory | counts_cpm | Vel-all | 1.376 | 0.008076 | 14 | 12 |
| Inhibitory | counts_cpm | PsychAD-V3 | 0.4342 | 0.6509 | 7 | 26 |
| Inhibitory | counts_cpm | V3-pooled | 0.2064 | 0.918 | 13 | 31 |
| Inhibitory | scvi_normalized | Vel-V2 | 1.914 | 0.005905 | 8 | 7 |
| Inhibitory | scvi_normalized | Vel-V3 | 0.7024 | 0.6623 | 6 | 5 |
| Inhibitory | scvi_normalized | Vel-all | 1.335 | 0.00506 | 14 | 12 |
| Inhibitory | scvi_normalized | PsychAD-V3 | 0.5271 | 0.5315 | 7 | 26 |
| Inhibitory | scvi_normalized | V3-pooled | 0.5343 | 0.4252 | 13 | 31 |
| Inhibitory | scanvi_normalized | Vel-V2 | 1.62 | 0.009324 | 8 | 7 |
| Inhibitory | scanvi_normalized | Vel-V3 | 0.5544 | 1 | 6 | 5 |
| Inhibitory | scanvi_normalized | Vel-all | 1.11 | 0.02209 | 14 | 12 |
| Inhibitory | scanvi_normalized | PsychAD-V3 | 0.6715 | 0.4238 | 7 | 26 |
| Inhibitory | scanvi_normalized | V3-pooled | 0.5098 | 0.625 | 13 | 31 |
| Microglia | counts_cpm | Vel-V2 | 1.154 | 0.1167 | 3 | 7 |
| Microglia | counts_cpm | Vel-V3 | 0.07122 | 0.4762 | 4 | 6 |
| Microglia | counts_cpm | Vel-all | 0.6236 | 0.06749 | 7 | 13 |
| Microglia | counts_cpm | PsychAD-V3 | -0.1582 | 0.9578 | 6 | 10 |
| Microglia | counts_cpm | V3-pooled | -0.02458 | 0.4768 | 10 | 16 |
| Microglia | scvi_normalized | Vel-V2 | 0.9048 | 0.1833 | 3 | 7 |
| Microglia | scvi_normalized | Vel-V3 | 0.1095 | 0.6095 | 4 | 6 |
| Microglia | scvi_normalized | Vel-all | 0.5148 | 0.1348 | 7 | 13 |
| Microglia | scvi_normalized | PsychAD-V3 | 0.3197 | 0.6354 | 6 | 10 |
| Microglia | scvi_normalized | V3-pooled | 0.2169 | 0.3041 | 10 | 16 |
| Microglia | scanvi_normalized | Vel-V2 | 0.8246 | 0.1833 | 3 | 7 |
| Microglia | scanvi_normalized | Vel-V3 | 0.1647 | 0.4762 | 4 | 6 |
| Microglia | scanvi_normalized | Vel-all | 0.5011 | 0.1146 | 7 | 13 |
| Microglia | scanvi_normalized | PsychAD-V3 | 0.296 | 0.7128 | 6 | 10 |
| Microglia | scanvi_normalized | V3-pooled | 0.2326 | 0.3041 | 10 | 16 |
| OPC | counts_cpm | Vel-V2 | 1.755 | 0.01399 | 6 | 7 |
| OPC | counts_cpm | Vel-V3 | 0.2161 | 0.5887 | 6 | 6 |
| OPC | counts_cpm | Vel-all | 1.057 | 0.02079 | 12 | 13 |
| OPC | counts_cpm | PsychAD-V3 | -0.2038 | 0.9321 | 6 | 21 |
| OPC | counts_cpm | V3-pooled | -0.03448 | 0.9394 | 12 | 27 |
| OPC | scvi_normalized | Vel-V2 | 2.012 | 0.004662 | 6 | 7 |
| OPC | scvi_normalized | Vel-V3 | 0.5244 | 0.3939 | 6 | 6 |
| OPC | scvi_normalized | Vel-all | 1.19 | 0.02079 | 12 | 13 |
| OPC | scvi_normalized | PsychAD-V3 | -0.214 | 0.7547 | 6 | 21 |
| OPC | scvi_normalized | V3-pooled | 0.1015 | 0.4378 | 12 | 27 |
| OPC | scanvi_normalized | Vel-V2 | 1.802 | 0.01399 | 6 | 7 |
| OPC | scanvi_normalized | Vel-V3 | 0.6336 | 0.3939 | 6 | 6 |
| OPC | scanvi_normalized | Vel-all | 1.171 | 0.0276 | 12 | 13 |
| OPC | scanvi_normalized | PsychAD-V3 | -0.1639 | 0.7983 | 6 | 21 |
| OPC | scanvi_normalized | V3-pooled | 0.145 | 0.456 | 12 | 27 |
| Other | counts_cpm | Vel-V2 | nan | nan | 3 | 2 |
| Other | counts_cpm | Vel-V3 | nan | nan | 1 | 1 |
| Other | counts_cpm | Vel-all | 1.453 | 0.2286 | 4 | 3 |
| Other | counts_cpm | PsychAD-V3 | nan | nan | 0 | 0 |
| Other | counts_cpm | V3-pooled | nan | nan | 1 | 1 |
| Other | scvi_normalized | Vel-V2 | nan | nan | 3 | 2 |
| Other | scvi_normalized | Vel-V3 | nan | nan | 1 | 1 |
| Other | scvi_normalized | Vel-all | 1.821 | 0.05714 | 4 | 3 |
| Other | scvi_normalized | PsychAD-V3 | nan | nan | 0 | 0 |
| Other | scvi_normalized | V3-pooled | nan | nan | 1 | 1 |
| Other | scanvi_normalized | Vel-V2 | nan | nan | 3 | 2 |
| Other | scanvi_normalized | Vel-V3 | nan | nan | 1 | 1 |
| Other | scanvi_normalized | Vel-all | 1.637 | 0.1143 | 4 | 3 |
| Other | scanvi_normalized | PsychAD-V3 | nan | nan | 0 | 0 |
| Other | scanvi_normalized | V3-pooled | nan | nan | 1 | 1 |
| Oligos | counts_cpm | Vel-V2 | 1.365 | 0.07273 | 4 | 7 |
| Oligos | counts_cpm | Vel-V3 | 0.706 | 0.1255 | 5 | 6 |
| Oligos | counts_cpm | Vel-all | 1.01 | 0.01943 | 9 | 13 |
| Oligos | counts_cpm | PsychAD-V3 | 0.5697 | 0.8134 | 3 | 26 |
| Oligos | counts_cpm | V3-pooled | 1.004 | 0.03574 | 8 | 32 |
| Oligos | scvi_normalized | Vel-V2 | 1.423 | 0.07273 | 4 | 7 |
| Oligos | scvi_normalized | Vel-V3 | 1.052 | 0.05195 | 5 | 6 |
| Oligos | scvi_normalized | Vel-all | 1.236 | 0.00756 | 9 | 13 |
| Oligos | scvi_normalized | PsychAD-V3 | 0.7515 | 0.6596 | 3 | 26 |
| Oligos | scvi_normalized | V3-pooled | 1.292 | 0.01496 | 8 | 32 |
| Oligos | scanvi_normalized | Vel-V2 | 1.463 | 0.07273 | 4 | 7 |
| Oligos | scanvi_normalized | Vel-V3 | 1.154 | 0.05195 | 5 | 6 |
| Oligos | scanvi_normalized | Vel-all | 1.3 | 0.00756 | 9 | 13 |
| Oligos | scanvi_normalized | PsychAD-V3 | 0.8713 | 0.5161 | 3 | 26 |
| Oligos | scanvi_normalized | V3-pooled | 1.402 | 0.01347 | 8 | 32 |
| Endothelial | counts_cpm | Vel-V2 | nan | nan | 0 | 0 |
| Endothelial | counts_cpm | Vel-V3 | nan | nan | 0 | 0 |
| Endothelial | counts_cpm | Vel-all | nan | nan | 0 | 0 |
| Endothelial | counts_cpm | PsychAD-V3 | 0.7505 | 0.2857 | 4 | 5 |
| Endothelial | counts_cpm | V3-pooled | 0.7505 | 0.2857 | 4 | 5 |
| Endothelial | scvi_normalized | Vel-V2 | nan | nan | 0 | 0 |
| Endothelial | scvi_normalized | Vel-V3 | nan | nan | 0 | 0 |
| Endothelial | scvi_normalized | Vel-all | nan | nan | 0 | 0 |
| Endothelial | scvi_normalized | PsychAD-V3 | 0.34 | 0.2857 | 4 | 5 |
| Endothelial | scvi_normalized | V3-pooled | 0.34 | 0.2857 | 4 | 5 |
| Endothelial | scanvi_normalized | Vel-V2 | nan | nan | 0 | 0 |
| Endothelial | scanvi_normalized | Vel-V3 | nan | nan | 0 | 0 |
| Endothelial | scanvi_normalized | Vel-all | nan | nan | 0 | 0 |
| Endothelial | scanvi_normalized | PsychAD-V3 | 0.3579 | 0.4127 | 4 | 5 |
| Endothelial | scanvi_normalized | V3-pooled | 0.3579 | 0.4127 | 4 | 5 |