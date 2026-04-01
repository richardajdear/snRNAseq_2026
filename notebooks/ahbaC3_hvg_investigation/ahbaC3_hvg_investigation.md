# AHBA C3 HVG Investigation (Velmeshev)


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
from hvg_investigation import (load_single_raw, setup_grn,
                                run_projection_pipeline, load_cache)
```

## 2. Data & HVG Projections

``` python
DATA_FILE = rds_dir + "/Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad"
N_VALUES = [1000, 2000, 4000, 6000, 8000, 10000]
CACHE_DIR = os.path.join(_repo_root, 'notebooks', 'ahbaC3_hvg_investigation', '_cache')

cached = load_cache(CACHE_DIR)
```

    Loaded cache from /Users/richard/Git/snRNAseq_2026/notebooks/ahbaC3_hvg_investigation/_cache

``` python
if cached is not None:
    scores_df, stats_df, final_df, hvg_df = cached
    print(f"scores: {len(scores_df)} rows, stats: {len(stats_df)} rows, final: {len(final_df)} rows, hvg_df: {len(hvg_df)} rows")
else:
    adata, adata_log = load_single_raw(DATA_FILE, source_label='VELMESHEV')
    ahba_GRN, total_grn_genes = setup_grn(ref_dir, adata)
    scores_df, stats_df, final_df, hvg_df = run_projection_pipeline(
        adata, adata_log, ahba_GRN, total_grn_genes, N_VALUES, CACHE_DIR)
    del adata, adata_log
    import gc; gc.collect()

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

## 3. Age Range Sensitivity

Before comparing HVG methods, we identify the age range definitions that
best capture the childhood-adolescence C3+ difference.

### 3.1 Compute Sensitivity Grid

``` python
%%R -i final_df -i code_dir -i N_VALUES

source(file.path(code_dir, 'hvg_plots.r'))
df <- prepare_r_data(final_df, N_VALUES)

selected_conds <- c('seurat_v3_2000', 'seurat_2000', 'pearson_2000',
                     'seurat_v3_8000', 'seurat_8000', 'pearson_8000', 'all_genes')

sens_all <- compute_sensitivity(df, selected_conds)

best <- select_best_age_range(sens_all)
best_cs <- best$child_start
best_bd <- best$boundary
best_ae <- best$adol_end

cat(sprintf("Best age range (lowest p for all_genes): childhood >= %.1fy, boundary = %dy, adolescence < %dy (p = %.4f, d = %.2f)\n",
            best_cs, best_bd, best_ae, best$p_value, best$cohens_d))
```

    Best age range (lowest p for all_genes): childhood >= 1.0y, boundary = 12y, adolescence < 21y (p = 0.0089, d = 1.27)

### 3.2 Cohen’s d

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_cohens_d(sens_all)
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-11-output-1.png)

### 3.3 P-value

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_pvalue(sens_all)
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-12-output-1.png)

### 3.4 Minimum Detectable Effect Size

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_power(sens_all)
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-13-output-1.png)

## 4. HVG Comparison (best age range)

All subsequent analyses use the age range with the lowest p-value from
the `all_genes` baseline above.

``` python
%%R

cat(sprintf("Using: Childhood = [%.1f, %d), Adolescence = [%d, %d)\n",
            best_cs, best_bd, best_bd, best_ae))
```

    Using: Childhood = [1.0, 12), Adolescence = [12, 21)

### 4.1 GRN Gene Retention

``` python
%%R -i stats_df -w 220 -h 80 -u mm -r 300

plot_gene_retention(stats_df, N_VALUES)
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-15-output-1.png)

### 4.1b HVG Gene Set Overlap (Euler diagrams)

``` python
%%R -i hvg_df -w 280 -h 110 -u mm -r 300

plot_hvg_euler(hvg_df)
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-16-output-1.png)

### 4.2 Age Trajectories & Developmental Stage Scores

``` python
%%R -w 360 -h 280 -u mm -r 300

p_a <- plot_age_trajectories(df, best_cs, best_bd, best_ae)
p_b <- plot_pseudobulk_trajectories(df, best_cs, best_bd, best_ae)
df_boxes <- make_boxes_df(df, best_cs, best_bd, best_ae)
p_c <- plot_boxes(df_boxes, best_cs, best_bd, best_ae)

(p_a | p_b | p_c) +
  plot_layout(guides = 'collect') +
  plot_annotation(tag_levels = 'a',
                  theme = theme(legend.position = 'right'))
```

    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'
    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-17-output-2.png)

### 4.3 Z-scored

``` python
%%R -w 360 -h 280 -u mm -r 300

p_a <- plot_age_trajectories(df, best_cs, best_bd, best_ae, zscore = TRUE)
p_b <- plot_pseudobulk_trajectories(df, best_cs, best_bd, best_ae, zscore = TRUE)
p_c <- plot_boxes(df_boxes, best_cs, best_bd, best_ae, zscore = TRUE)

(p_a | p_b | p_c) +
  plot_layout(guides = 'collect') +
  plot_annotation(tag_levels = 'a',
                  theme = theme(legend.position = 'right'))
```

    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'
    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-18-output-2.png)

### 4.4 C3+ Effect Summary

``` python
%%R -w 300 -h 220 -u mm -r 300

plot_effect_summary(df_boxes, 'Childhood', 'Adolescence') /
plot_effect_summary(df_boxes, 'Adolescence', 'Adulthood')
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-19-output-1.png)
