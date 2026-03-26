# AHBA C3 HVG Investigation (PsychAD)


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

    Environment : hpc
      rds_dir  : /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc
      code_dir : /home/rajd2/rds/hpc-work/snRNAseq_2026/code
      ref_dir  : /home/rajd2/rds/hpc-work/snRNAseq_2026/reference

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

    /opt/micromamba/envs/shortcake_default/lib/python3.10/site-packages/anndata/utils.py:429: FutureWarning:

    Importing read_csv from `anndata` is deprecated. Import anndata.io.read_csv instead.

    /opt/micromamba/envs/shortcake_default/lib/python3.10/site-packages/anndata/utils.py:429: FutureWarning:

    Importing read_text from `anndata` is deprecated. Import anndata.io.read_text instead.

    /opt/micromamba/envs/shortcake_default/lib/python3.10/site-packages/anndata/utils.py:429: FutureWarning:

    Importing read_excel from `anndata` is deprecated. Import anndata.io.read_excel instead.

    /opt/micromamba/envs/shortcake_default/lib/python3.10/site-packages/anndata/utils.py:429: FutureWarning:

    Importing read_mtx from `anndata` is deprecated. Import anndata.io.read_mtx instead.

    /opt/micromamba/envs/shortcake_default/lib/python3.10/site-packages/anndata/utils.py:429: FutureWarning:

    Importing read_loom from `anndata` is deprecated. Import anndata.io.read_loom instead.

    /opt/micromamba/envs/shortcake_default/lib/python3.10/site-packages/anndata/utils.py:429: FutureWarning:

    Importing read_hdf from `anndata` is deprecated. Import anndata.io.read_hdf instead.

    /opt/micromamba/envs/shortcake_default/lib/python3.10/site-packages/anndata/utils.py:429: FutureWarning:

    Importing read_umi_tools from `anndata` is deprecated. Import anndata.io.read_umi_tools instead.

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


    Attaching package: ‘dplyr’

    The following objects are masked from ‘package:stats’:

        filter, lag

    The following objects are masked from ‘package:base’:

        intersect, setdiff, setequal, union

    In addition: Warning message:
    In (function (package, help, pos = 2, lib.loc = NULL, character.only = FALSE,  :
      library ‘/usr/lib/R/site-library’ contains no packages

``` python
from regulons import get_ahba_GRN, project_GRN
from gene_mapping import map_grn_symbols_to_ensembl
from hvg_investigation import (build_conditions, run_hvg_conditions,
                                prepare_for_r, save_cache, load_cache)
```

## 2. Data & HVG Projections

``` python
AGING_FILE = rds_dir + "/Cam_PsychAD/RNAseq/Aging_Cohort_100k_PFC_lessOld.h5ad"
HBCC_FILE  = rds_dir + "/Cam_PsychAD/RNAseq/HBCC_Cohort_100k_PFC_lessOld.h5ad"
N_VALUES = [1000, 2000, 4000, 6000, 8000, 10000]
CACHE_DIR = os.path.join(_repo_root, 'notebooks', 'ahbaC3_hvg_investigation_psychAD', '_cache')

cached = load_cache(CACHE_DIR)
```

``` python
if cached is not None:
    scores_df, stats_df, final_df, hvg_df = cached
    print(f"scores: {len(scores_df)} rows, stats: {len(stats_df)} rows, final: {len(final_df)} rows, hvg_df: {len(hvg_df)} rows")
else:
    # ── Load and combine Aging + HBCC cohorts ──
    aging = sc.read_h5ad(AGING_FILE)
    hbcc  = sc.read_h5ad(HBCC_FILE)
    aging.obs['source'] = 'Aging'
    hbcc.obs['source']  = 'HBCC'
    print(f"Aging shape: {aging.shape}, HBCC shape: {hbcc.shape}")

    # Concatenate; index_unique='-' disambiguates the 27 shared cell barcodes
    # sc.concat drops var columns, so save them first and restore afterwards
    var_df = aging.var.copy()
    adata = sc.concat([aging, hbcc], keys=['Aging', 'HBCC'], index_unique='-',
                      join='outer', fill_value=0)
    adata.var = var_df.loc[adata.var_names]
    del aging, hbcc
    print(f"Combined shape: {adata.shape}")

    # ── Normalize ──
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e6)

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

    Aging shape: (52713, 34176), HBCC shape: (56696, 34176)
    Combined shape: (109409, 34176)

    Input sequence provided is already in string format. No operation performed
    Input sequence provided is already in string format. No operation performed

    Mapped 7214/7973 symbols via adata.var
    Querying mygene for 759 unmapped symbols...

    41 input query terms found dup hits:    [('ACTG1P4', 2), ('ADAM20P1', 2), ('ANKRD19P', 2), ('ARHGAP27P2', 2), ('BMS1P2', 2), ('BMS1P20', 2),
    353 input query terms found no hit: ['AAED1', 'AARS', 'ADAL', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARNTL', 'ARNTL2', 'AZIN1-AS1',

    After mygene: 7228/7973 mapped, 745 dropped
    GRN genes in adata: 7228 / 7228

    ============================================================
    Condition: all_genes
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 7228 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 34176, GRN genes used: 7228/7228 (100.0%)

    ============================================================
    Condition: seurat_v3_1000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 367 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 367/7228 (5.1%)

    ============================================================
    Condition: seurat_1000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 362 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 362/7228 (5.0%)

    ============================================================
    Condition: pearson_1000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 512 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 512/7228 (7.1%)

    ============================================================
    Condition: seurat_v3_2000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 720 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 720/7228 (10.0%)

    ============================================================
    Condition: seurat_2000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 641 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 641/7228 (8.9%)

    ============================================================
    Condition: pearson_2000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 979 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 979/7228 (13.5%)

    ============================================================
    Condition: seurat_v3_4000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 1398 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 1398/7228 (19.3%)

    ============================================================
    Condition: seurat_4000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 1057 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 1057/7228 (14.6%)

    ============================================================
    Condition: pearson_4000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 1645 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 1645/7228 (22.8%)

    ============================================================
    Condition: seurat_v3_6000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 1947 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 6000, GRN genes used: 1947/7228 (26.9%)

    ============================================================
    Condition: seurat_6000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 1410 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 6000, GRN genes used: 1410/7228 (19.5%)

    ============================================================
    Condition: pearson_6000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 2211 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 6000, GRN genes used: 2211/7228 (30.6%)

    ============================================================
    Condition: seurat_v3_8000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 2348 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 8000, GRN genes used: 2348/7228 (32.5%)

    ============================================================
    Condition: seurat_8000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 1721 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 8000, GRN genes used: 1721/7228 (23.8%)

    ============================================================
    Condition: pearson_8000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 2755 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 8000, GRN genes used: 2755/7228 (38.1%)

    ============================================================
    Condition: seurat_v3_10000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 2639 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 10000, GRN genes used: 2639/7228 (36.5%)

    ============================================================
    Condition: seurat_10000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 2034 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 10000, GRN genes used: 2034/7228 (28.1%)

    ============================================================
    Condition: pearson_10000
    Found 7228 matching genes in var_names.
    Aligning GRN weights to 3230 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 10000, GRN genes used: 3230/7228 (44.7%)
    Cache saved to /rds/user/rajd2/hpc-work/snRNAseq_2026/notebooks/ahbaC3_hvg_investigation_psychAD/_cache
    scores: 4157542, stats: 19, final_df (excitatory): 944642, hvg_df: 93000

### Gene Overlap Summary

``` python
print(stats_df.to_string(index=False))
```

          condition  n_hvg  n_grn_genes_used  pct_grn_retained
          all_genes  34176              7228             100.0
     seurat_v3_1000   1000               367               5.1
        seurat_1000   1000               362               5.0
       pearson_1000   1000               512               7.1
     seurat_v3_2000   2000               720              10.0
        seurat_2000   2000               641               8.9
       pearson_2000   2000               979              13.5
     seurat_v3_4000   4000              1398              19.3
        seurat_4000   4000              1057              14.6
       pearson_4000   4000              1645              22.8
     seurat_v3_6000   6000              1947              26.9
        seurat_6000   6000              1410              19.5
       pearson_6000   6000              2211              30.6
     seurat_v3_8000   8000              2348              32.5
        seurat_8000   8000              1721              23.8
       pearson_8000   8000              2755              38.1
    seurat_v3_10000  10000              2639              36.5
       seurat_10000  10000              2034              28.1
      pearson_10000  10000              3230              44.7

## 4. Age Range Sensitivity

Before comparing HVG methods, we identify the age range definitions that
best capture the childhood-adolescence C3+ difference.

### 4.1 Compute Sensitivity Grid

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

    Best age range (lowest p for all_genes): childhood >= 1.0y, boundary = 14y, adolescence < 21y (p = 0.0598, d = 0.64)

### 4.2 Cohen’s d

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_cohens_d(sens_all)
```

![](ahbaC3_hvg_investigation_psychAD_files/figure-markdown_strict/cell-11-output-1.png)

### 4.3 P-value

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_pvalue(sens_all)
```

![](ahbaC3_hvg_investigation_psychAD_files/figure-markdown_strict/cell-12-output-1.png)

### 4.4 Power

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_power(sens_all)
```

![](ahbaC3_hvg_investigation_psychAD_files/figure-markdown_strict/cell-13-output-1.png)

## 5. HVG Comparison (best age range)

All subsequent analyses use the age range with the lowest p-value from
the `all_genes` baseline above.

``` python
%%R

cat(sprintf("Using: Childhood = [%.1f, %d), Adolescence = [%d, %d)\n",
            best_cs, best_bd, best_bd, best_ae))
```

    Using: Childhood = [1.0, 14), Adolescence = [14, 21)

### 5.1 GRN Gene Retention

``` python
%%R -i stats_df -w 220 -h 80 -u mm -r 300

plot_gene_retention(stats_df, N_VALUES)
```

![](ahbaC3_hvg_investigation_psychAD_files/figure-markdown_strict/cell-15-output-1.png)

### 5.1b HVG Gene Set Overlap (Euler diagrams)

``` python
%%R -i hvg_df -w 280 -h 110 -u mm -r 300

plot_hvg_euler(hvg_df)
```

![](ahbaC3_hvg_investigation_psychAD_files/figure-markdown_strict/cell-16-output-1.png)

### 5.2 Age Trajectories & Developmental Stage Scores

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

    R[write to console]: In addition: 
    R[write to console]: Warning messages:

    R[write to console]: 1: Removed 262968 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 2: Removed 262968 rows containing missing values or values outside the scale range
    (`geom_point()`). 

    R[write to console]: 3: Removed 2772 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 4: Removed 2772 rows containing missing values or values outside the scale range
    (`geom_point()`). 

    In addition: Warning message:
    The `panel.margin` argument of `theme()` is deprecated as of ggplot2 2.2.0.
    ℹ Please use the `panel.spacing` argument instead.
    This warning is displayed once every 8 hours.
    Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    generated. 

![](ahbaC3_hvg_investigation_psychAD_files/figure-markdown_strict/cell-17-output-4.png)

### 5.3 Z-scored

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

    R[write to console]: In addition: 
    R[write to console]: Warning messages:

    R[write to console]: 1: Removed 262968 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 2: Removed 262968 rows containing missing values or values outside the scale range
    (`geom_point()`). 

    R[write to console]: 3: Removed 2772 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 4: Removed 2772 rows containing missing values or values outside the scale range
    (`geom_point()`). 

![](ahbaC3_hvg_investigation_psychAD_files/figure-markdown_strict/cell-18-output-3.png)

### 5.4 C3+ Effect Summary

``` python
%%R -w 300 -h 220 -u mm -r 300

plot_effect_summary(df_boxes, 'Childhood', 'Adolescence') /
plot_effect_summary(df_boxes, 'Adolescence', 'Adulthood')
```

    In addition: Warning messages:
    1: There was 1 warning in `mutate()`.
    ℹ In argument: `n_genes = ifelse(...)`.
    Caused by warning in `ifelse()`:
    ! NAs introduced by coercion 
    2: There was 1 warning in `mutate()`.
    ℹ In argument: `n_genes = ifelse(...)`.
    Caused by warning in `ifelse()`:
    ! NAs introduced by coercion 

![](ahbaC3_hvg_investigation_psychAD_files/figure-markdown_strict/cell-19-output-2.png)
