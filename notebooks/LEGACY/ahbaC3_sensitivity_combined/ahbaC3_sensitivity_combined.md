# AHBA C3 Age Sensitivity (Combined)


## 1. Setup

### 1.1 Environment

``` python
import os
import sys
import warnings
from IPython import get_ipython
_ip = get_ipython()
if _ip:
    _ip.run_line_magic('load_ext', 'autoreload')
    _ip.run_line_magic('autoreload', '2')
```

``` python
def _find_repo_root(marker='.git'):
    current = os.path.abspath(os.getcwd())
    while True:
        if os.path.exists(os.path.join(current, marker)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(f"Could not find '{marker}' above {os.getcwd()}")
        current = parent

_repo_root = _find_repo_root()
if os.path.join(_repo_root, 'code') not in sys.path:
    sys.path.insert(0, os.path.join(_repo_root, 'code'))

from environment import get_environment
_env = get_environment()
rds_dir  = _env['rds_dir']
code_dir = _env['code_dir']
ref_dir  = _env['ref_dir']

print(f"Environment : {_env['name']}")
print(f"  rds_dir  : {rds_dir}")
print(f"  code_dir : {code_dir}")
print(f"  ref_dir  : {ref_dir}")
```

    Environment : local
      rds_dir  : /Users/richard/Git/snRNAseq_2026/rds-cam-psych-transc-Pb9UGUlrwWc
      code_dir : /Users/richard/Git/snRNAseq_2026/code
      ref_dir  : /Users/richard/Git/snRNAseq_2026/reference

### 1.2 Libraries

``` python
import scanpy as sc
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

if _env['name'] == 'local':
    os.environ['R_HOME'] = '/Users/richard/mambaforge/envs/scanpy/lib/R'

%load_ext rpy2.ipython
```

``` python
%%R

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(ggpubr)
library(pwr)
library(ggvenn)
```

    R[write to console]: 
    Attaching package: ‘dplyr’


    R[write to console]: The following objects are masked from ‘package:stats’:

        filter, lag


    R[write to console]: The following objects are masked from ‘package:base’:

        intersect, setdiff, setequal, union

``` python
from regulons import get_ahba_GRN, project_GRN
from gene_mapping import map_grn_symbols_to_ensembl
from hvg_investigation import (build_conditions, run_hvg_conditions,
                                prepare_for_r, save_cache, load_cache)
```

## 2. Data & HVG Projections

``` python
DATA_FILE = rds_dir + "/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld_normal.h5ad"
N_VALUES = [1000, 2000, 4000, 6000, 8000, 10000]
CACHE_DIR = os.path.join(_repo_root, 'notebooks', 'ahbaC3_sensitivity_combined', '_cache')

cached = load_cache(CACHE_DIR)
```

    Loaded cache from /Users/richard/Git/snRNAseq_2026/notebooks/ahbaC3_sensitivity_combined/_cache

``` python
if cached is not None:
    scores_df, stats_df, final_df, hvg_df = cached
    print(f"scores: {len(scores_df)} rows, stats: {len(stats_df)} rows, final: {len(final_df)} rows, hvg_df: {len(hvg_df)} rows")
else:
    # ── Load and normalize ──
    adata = sc.read_h5ad(DATA_FILE)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6)
    if 'source' not in adata.obs.columns:
        adata.obs['source'] = 'combined'
    print(f"Shape: {adata.shape}")

    adata_log = adata.copy()
    sc.pp.log1p(adata_log)

    # ── Load and remap GRN ──
    grn_file = os.path.join(ref_dir, "ahba_dme_hcp_top8kgenes_weights.csv")
    ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file, use_weights=True)
    ahba_GRN = map_grn_symbols_to_ensembl(ahba_GRN, adata)
    grn_pivot = ahba_GRN.pivot_table(index='Network', columns='Gene', values='Importance', fill_value=0)
    total_grn_genes = len(np.intersect1d(grn_pivot.columns, adata.var_names))
    print(f"GRN genes in adata: {total_grn_genes} / {grn_pivot.shape[1]}")

    # ── Run HVG conditions ──
    conditions = build_conditions(N_VALUES)
    scores_df, stats_df, hvg_df = run_hvg_conditions(
        adata, adata_log, ahba_GRN, conditions, total_grn_genes)
    del adata_log

    # ── Prepare for R ──
    final_df = prepare_for_r(scores_df, adata, N_VALUES)
    del adata
    import gc; gc.collect()

    # ── Save cache ──
    save_cache(CACHE_DIR, scores_df, stats_df, final_df, hvg_df)

print(f"scores: {len(scores_df)}, stats: {len(stats_df)}, final_df (excitatory): {len(final_df)}, hvg_df: {len(hvg_df)}")
```

    scores: 7543912 rows, stats: 19 rows, final: 2857638 rows, hvg_df: 93000 rows
    scores: 7543912, stats: 19, final_df (excitatory): 2857638, hvg_df: 93000

### Gene Overlap Summary

``` python
print(stats_df.to_string(index=False))
```

          condition  n_hvg  n_grn_genes_used  pct_grn_retained
          all_genes  15540              6397             100.0
     seurat_v3_1000   1000               575               9.0
        seurat_1000   1000               523               8.2
       pearson_1000   1000               618               9.7
     seurat_v3_2000   2000              1145              17.9
        seurat_2000   2000               995              15.6
       pearson_2000   2000              1170              18.3
     seurat_v3_4000   4000              2181              34.1
        seurat_4000   4000              1800              28.1
       pearson_4000   4000              2148              33.6
     seurat_v3_6000   6000              2981              46.6
        seurat_6000   6000              2626              41.1
       pearson_6000   6000              3053              47.7
     seurat_v3_8000   8000              3578              55.9
        seurat_8000   8000              3405              53.2
       pearson_8000   8000              3863              60.4
    seurat_v3_10000  10000              4292              67.1
       seurat_10000  10000              4214              65.9
      pearson_10000  10000              4630              72.4

## 3. Age Range Sensitivity (Gap Model)

We fix the childhood lower bound at 1 year and independently vary:

- **Childhood upper bound**: 6, 7, 8, 9 (row facets)
- **Adolescence lower bound**: 12, 13, 14, 15 (x-axis)
- **Adolescence upper bound**: 22, 23, 24, 25 (column facets)

This leaves a gap between childhood and adolescence to account for
uncertainty about when children enter adolescence.

### 3.1 Compute Sensitivity Grid

``` python
%%R -i final_df -i code_dir -i N_VALUES

source(file.path(code_dir, 'hvg_plots.r'))
source(file.path(code_dir, 'sensitivity_gap_plots.r'))

df <- prepare_r_data(final_df, N_VALUES)

selected_conds <- c('seurat_v3_2000', 'seurat_2000', 'pearson_2000',
                     'seurat_v3_8000', 'seurat_8000', 'pearson_8000', 'all_genes')

CHILD_START <- 1

sens_all <- compute_sensitivity_gap(df, selected_conds,
                                     child_start = CHILD_START,
                                     child_ends  = c(7, 8, 9, 10),
                                     adol_starts = c(10, 11, 12, 13, 14, 15),
                                     adol_ends   = c(19, 21, 23, 25))

best <- select_best_gap(sens_all)
best_ce <- best$child_end
best_as <- best$adol_start
best_ae <- best$adol_end

cat(sprintf("Best age range (lowest p for all_genes): Childhood [%d, %d), Adolescence [%d, %d)  (p = %.4f, d = %.2f)\n",
            CHILD_START, best_ce, best_as, best_ae, best$p_value, best$cohens_d))
```

    Best age range (lowest p for all_genes): Childhood [1, 9), Adolescence [14, 21)  (p = 0.1827, d = 0.45)

### 3.2 Cohen’s d

``` python
%%R -w 280 -h 220 -u mm -r 300

plot_gap_cohens_d(sens_all)
```

![](ahbaC3_sensitivity_combined_files/figure-markdown_strict/cell-11-output-1.png)

### 3.3 P-value

``` python
%%R -w 280 -h 220 -u mm -r 300

plot_gap_pvalue(sens_all)
```

![](ahbaC3_sensitivity_combined_files/figure-markdown_strict/cell-12-output-1.png)

### 3.4 Minimum Detectable Effect Size

``` python
%%R -w 280 -h 220 -u mm -r 300

plot_gap_power(sens_all)
```

![](ahbaC3_sensitivity_combined_files/figure-markdown_strict/cell-13-output-1.png)

## 4. HVG Comparison (best age range)

All subsequent analyses use the age range with the lowest p-value from
the `all_genes` baseline above.

``` python
%%R

cat(sprintf("Using: Childhood = [%d, %d), Gap = [%d, %d), Adolescence = [%d, %d)\n",
            CHILD_START, best_ce, best_ce, best_as, best_as, best_ae))
```

    Using: Childhood = [1, 9), Gap = [9, 14), Adolescence = [14, 21)

### 4.1 GRN Gene Retention

``` python
%%R -i stats_df -w 220 -h 80 -u mm -r 300

plot_gene_retention(stats_df, N_VALUES)
```

![](ahbaC3_sensitivity_combined_files/figure-markdown_strict/cell-15-output-1.png)

### 4.1b HVG Gene Set Overlap (Euler diagrams)

``` python
%%R -i hvg_df -w 280 -h 110 -u mm -r 300

plot_hvg_euler(hvg_df)
```

![](ahbaC3_sensitivity_combined_files/figure-markdown_strict/cell-16-output-1.png)

### 4.2 Age Trajectories & Developmental Stage Scores

``` python
%%R -w 360 -h 280 -u mm -r 300

p_a <- plot_gap_trajectories(df, CHILD_START, best_ce, best_as, best_ae)
p_b <- plot_gap_pseudobulk(df, CHILD_START, best_ce, best_as, best_ae)
df_boxes <- make_boxes_gap_df(df, CHILD_START, best_ce, best_as, best_ae)
p_c <- plot_gap_boxes(df_boxes, CHILD_START, best_ce, best_as, best_ae)

(p_a | p_b | p_c) +
  plot_layout(guides = 'collect') +
  plot_annotation(tag_levels = 'a',
                  theme = theme(legend.position = 'right'))
```

    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'
    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](ahbaC3_sensitivity_combined_files/figure-markdown_strict/cell-17-output-2.png)

### 4.3 Z-scored

``` python
%%R -w 360 -h 280 -u mm -r 300

p_a <- plot_gap_trajectories(df, CHILD_START, best_ce, best_as, best_ae, zscore = TRUE)
p_b <- plot_gap_pseudobulk(df, CHILD_START, best_ce, best_as, best_ae, zscore = TRUE)
p_c <- plot_gap_boxes(df_boxes, CHILD_START, best_ce, best_as, best_ae, zscore = TRUE)

(p_a | p_b | p_c) +
  plot_layout(guides = 'collect') +
  plot_annotation(tag_levels = 'a',
                  theme = theme(legend.position = 'right'))
```

    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'
    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](ahbaC3_sensitivity_combined_files/figure-markdown_strict/cell-18-output-2.png)

### 4.4 C3+ Effect Summary

``` python
%%R -w 340 -h 160 -u mm -r 300

plot_effect_summary(df_boxes, 'Childhood', 'Adolescence') /
plot_effect_summary(df_boxes, 'Adolescence', 'Adulthood')
```

![](ahbaC3_sensitivity_combined_files/figure-markdown_strict/cell-19-output-1.png)

### 4.5 Signal vs Noise Decomposition

Why does power drop when adding more genes? The GRN score is a weighted
sum across genes. Adding non-HVG genes contributes noise
(individual-level SD) without contributing developmental signal
(childhood-adolescence delta).

``` python
%%R -w 280 -h 100 -u mm -r 300

# Decompose Cohen's d = delta / pooled_sd into its components
decomp <- df_boxes %>%
  filter(network == 'C3+') %>%
  group_by(condition) %>%
  summarize(
    mean_child = mean(value[age_range == 'Childhood']),
    mean_adol  = mean(value[age_range == 'Adolescence']),
    n_child    = sum(age_range == 'Childhood'),
    n_adol     = sum(age_range == 'Adolescence'),
    sd_child   = sd(value[age_range == 'Childhood']),
    sd_adol    = sd(value[age_range == 'Adolescence']),
    .groups = 'drop'
  ) %>%
  mutate(
    delta     = mean_child - mean_adol,
    pooled_sd = sqrt(((pmax(n_child,1)-1)*sd_child^2 +
                      (pmax(n_adol,1)-1)*sd_adol^2) /
                     pmax(n_child + n_adol - 2, 1)),
    cohens_d  = ifelse(pooled_sd > 0, delta / pooled_sd, NA_real_)
  ) %>%
  add_hvg_columns()

ref <- decomp %>% filter(condition == 'all_genes')
hvg <- decomp %>% filter(condition != 'all_genes')
x_breaks <- sort(unique(hvg$n_genes))
ct <- theme_classic() + theme(axis.text.x = element_text(size = 8), legend.position = 'none')

p1 <- hvg %>%
  ggplot(aes(x = n_genes, y = delta / 1e3, color = flavor, group = flavor)) +
  geom_hline(yintercept = ref$delta / 1e3, linetype = 'dotted', color = 'grey30') +
  geom_point(size = 2) + geom_line(linewidth = 0.5) +
  scale_x_continuous(breaks = x_breaks) +
  scale_color_brewer(palette = 'Set2') +
  labs(x = 'n_top_genes', y = 'Signal\n(delta, K)') + ct

p2 <- hvg %>%
  ggplot(aes(x = n_genes, y = pooled_sd / 1e3, color = flavor, group = flavor)) +
  geom_hline(yintercept = ref$pooled_sd / 1e3, linetype = 'dotted', color = 'grey30') +
  geom_point(size = 2) + geom_line(linewidth = 0.5) +
  scale_x_continuous(breaks = x_breaks) +
  scale_color_brewer(palette = 'Set2') +
  labs(x = 'n_top_genes', y = 'Noise\n(pooled SD, K)') + ct

p3 <- hvg %>%
  ggplot(aes(x = n_genes, y = cohens_d, color = flavor, group = flavor)) +
  geom_hline(yintercept = ref$cohens_d, linetype = 'dotted', color = 'grey30') +
  geom_point(size = 2) + geom_line(linewidth = 0.5) +
  scale_x_continuous(breaks = x_breaks) +
  scale_color_brewer(palette = 'Set2') +
  labs(x = 'n_top_genes', y = "Signal/Noise\n(Cohen's d)") +
  theme_classic() + theme(axis.text.x = element_text(size = 8))

(p1 | p2 | p3) +
  plot_layout(guides = 'collect') +
  plot_annotation(
    title = "Signal vs noise decomposition: why power drops with more genes",
    subtitle = "Signal (delta) plateaus while noise (pooled SD) keeps rising. d = delta / SD.",
    tag_levels = 'a'
  )
```

![](ahbaC3_sensitivity_combined_files/figure-markdown_strict/cell-20-output-1.png)
