# AHBA C3 Age Sensitivity (Velmeshev)


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
DATA_FILE = rds_dir + "/Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad"
N_VALUES = [1000, 2000, 4000, 6000, 8000, 10000]
CACHE_DIR = os.path.join(_repo_root, 'notebooks', 'ahbaC3_sensitivity_velmeshev', '_cache')

cached = load_cache(CACHE_DIR)
```

    Loaded cache from /Users/richard/Git/snRNAseq_2026/notebooks/ahbaC3_sensitivity_velmeshev/_cache

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
        adata.obs['source'] = 'VELMESHEV'
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

    scores: 2441500 rows, stats: 19 rows, final: 1284970 rows, hvg_df: 93000 rows
    scores: 2441500, stats: 19, final_df (excitatory): 1284970, hvg_df: 93000

### Gene Overlap Summary

``` python
print(stats_df.to_string(index=False))
```

          condition  n_hvg  n_grn_genes_used  pct_grn_retained
          all_genes  17663              6650             100.0
     seurat_v3_1000   1000               626               9.4
        seurat_1000   1000               495               7.4
       pearson_1000   1000               626               9.4
     seurat_v3_2000   2000              1220              18.3
        seurat_2000   2000               918              13.8
       pearson_2000   2000              1131              17.0
     seurat_v3_4000   4000              2237              33.6
        seurat_4000   4000              1653              24.9
       pearson_4000   4000              2018              30.3
     seurat_v3_6000   6000              2861              43.0
        seurat_6000   6000              2373              35.7
       pearson_6000   6000              2896              43.5
     seurat_v3_8000   8000              3312              49.8
        seurat_8000   8000              3127              47.0
       pearson_8000   8000              3666              55.1
    seurat_v3_10000  10000              3956              59.5
       seurat_10000  10000              3923              59.0
      pearson_10000  10000              4438              66.7

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

    Best age range (lowest p for all_genes): Childhood [1, 8), Adolescence [12, 23)  (p = 0.0107, d = 1.21)

### 3.2 Cohen’s d

``` python
%%R -w 280 -h 220 -u mm -r 300

plot_gap_cohens_d(sens_all)
```

![](ahbaC3_sensitivity_velmeshev_files/figure-markdown_strict/cell-11-output-1.png)

### 3.3 P-value

``` python
%%R -w 280 -h 220 -u mm -r 300

plot_gap_pvalue(sens_all)
```

![](ahbaC3_sensitivity_velmeshev_files/figure-markdown_strict/cell-12-output-1.png)

### 3.4 Minimum Detectable Effect Size

``` python
%%R -w 280 -h 220 -u mm -r 300

plot_gap_power(sens_all)
```

![](ahbaC3_sensitivity_velmeshev_files/figure-markdown_strict/cell-13-output-1.png)

## 4. HVG Comparison (best age range)

All subsequent analyses use the age range with the lowest p-value from
the `all_genes` baseline above.

``` python
%%R

cat(sprintf("Using: Childhood = [%d, %d), Gap = [%d, %d), Adolescence = [%d, %d)\n",
            CHILD_START, best_ce, best_ce, best_as, best_as, best_ae))
```

    Using: Childhood = [1, 8), Gap = [8, 12), Adolescence = [12, 23)

### 4.1 GRN Gene Retention

``` python
%%R -i stats_df -w 220 -h 80 -u mm -r 300

plot_gene_retention(stats_df, N_VALUES)
```

![](ahbaC3_sensitivity_velmeshev_files/figure-markdown_strict/cell-15-output-1.png)

### 4.1b HVG Gene Set Overlap (Euler diagrams)

``` python
%%R -i hvg_df -w 280 -h 110 -u mm -r 300

plot_hvg_euler(hvg_df)
```

![](ahbaC3_sensitivity_velmeshev_files/figure-markdown_strict/cell-16-output-1.png)

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

![](ahbaC3_sensitivity_velmeshev_files/figure-markdown_strict/cell-17-output-2.png)

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

![](ahbaC3_sensitivity_velmeshev_files/figure-markdown_strict/cell-18-output-2.png)

### 4.4 C3+ Effect Summary

``` python
%%R -w 300 -h 220 -u mm -r 300

plot_effect_summary(df_boxes, 'Childhood', 'Adolescence') /
plot_effect_summary(df_boxes, 'Adolescence', 'Adulthood')
```

![](ahbaC3_sensitivity_velmeshev_files/figure-markdown_strict/cell-19-output-1.png)
