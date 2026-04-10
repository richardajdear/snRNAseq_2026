# AHBA C3 Age Sensitivity (Integrated, scANVI)


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
from hvg_investigation import (load_single_scvi, setup_grn,
                                run_projection_pipeline, load_cache)
```

## 2. Data & HVG Projections

``` python
DATA_FILE = rds_dir + "/Cam_snRNAseq/integrated/VelWangPsychAD_100k_dataset/scvi_output/integrated.h5ad"
SCVI_LAYER = 'scanvi_normalized'
N_VALUES = [1000, 2000, 4000, 6000, 8000, 10000]
CACHE_DIR = os.path.join(_repo_root, 'notebooks', 'ahbaC3_sensitivity_combined_scANVI', '_cache')

FILTER_CELL_TYPES = None
# FILTER_CELL_TYPES = ['EN-L2_3-IT', 'EN-Newborn', 'EN-IT-Immature', 'EN-Non-IT-Immature']

cached = load_cache(CACHE_DIR)
```

``` python
if cached is not None:
    scores_df, stats_df, final_df, hvg_df = cached
    print(f"scores: {len(scores_df)} rows, stats: {len(stats_df)} rows, final: {len(final_df)} rows, hvg_df: {len(hvg_df)} rows")
else:
    adata, adata_log = load_single_scvi(DATA_FILE, scvi_layer=SCVI_LAYER)
    ahba_GRN, total_grn_genes = setup_grn(ref_dir, adata)
    scores_df, stats_df, final_df, hvg_df = run_projection_pipeline(
        adata, adata_log, ahba_GRN, total_grn_genes, N_VALUES, CACHE_DIR)
    del adata, adata_log
    import gc; gc.collect()

print(f"scores: {len(scores_df)}, stats: {len(stats_df)}, final_df (excitatory): {len(final_df)}, hvg_df: {len(hvg_df)}")
```

    [MEM] load_single_scvi: start: 0.8 GB RSS
    [MEM] load_single_scvi: backed-mode open done (shape=(400000, 15540), layers=['counts', 'scanvi_normalized'], obsm=['X_pca_raw', 'X_scANVI', 'X_scVI', 'X_umap', 'X_umap_raw', 'X_umap_scanvi', 'X_umap_scvi']): 34.0 GB RSS
    [MEM] load_single_scvi: after extracting target layer; other layers never loaded: 25.2 GB RSS
    [MEM] load_single_scvi: after building AnnData: 25.2 GB RSS
    [MEM] load_single_scvi: after normalize_total: 49.5 GB RSS
    Shape: (400000, 15540)
    [MEM] load_single_scvi: after adata_log creation (X-only, no layers): 73.8 GB RSS

    Input sequence provided is already in string format. No operation performed
    Input sequence provided is already in string format. No operation performed

    Mapped 122/7973 symbols via adata.var
    Querying mygene for 7851 unmapped symbols...

    136 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    376 input query terms found no hit: ['AAED1', 'AARS', 'ADAL', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZ

    After mygene: 6348/7973 mapped, 1625 dropped
    GRN genes in adata: 6348 / 6348
    [MEM] run_projection_pipeline: start: 73.8 GB RSS
    [MEM] run_hvg_conditions: start 'all_genes': 73.8 GB RSS

    ============================================================
    Condition: all_genes
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 15540, GRN genes used: 6348/6348 (100.0%)
    [MEM] run_hvg_conditions: start 'seurat_v3_1000': 73.9 GB RSS

    ============================================================
    Condition: seurat_v3_1000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 578 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 578/6348 (9.1%)
    [MEM] run_hvg_conditions: start 'seurat_1000': 73.9 GB RSS

    ============================================================
    Condition: seurat_1000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 597 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 597/6348 (9.4%)
    [MEM] run_hvg_conditions: start 'pearson_1000': 73.9 GB RSS

    ============================================================
    Condition: pearson_1000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 686 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 686/6348 (10.8%)
    [MEM] run_hvg_conditions: start 'seurat_v3_2000': 74.0 GB RSS

    ============================================================
    Condition: seurat_v3_2000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 1115 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 1115/6348 (17.6%)
    [MEM] run_hvg_conditions: start 'seurat_2000': 74.0 GB RSS

    ============================================================
    Condition: seurat_2000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 1143 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 1143/6348 (18.0%)
    [MEM] run_hvg_conditions: start 'pearson_2000': 74.0 GB RSS

    ============================================================
    Condition: pearson_2000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 1308 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 1308/6348 (20.6%)
    [MEM] run_hvg_conditions: start 'seurat_v3_4000': 74.0 GB RSS

    ============================================================
    Condition: seurat_v3_4000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 2180 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 2180/6348 (34.3%)
    [MEM] run_hvg_conditions: start 'seurat_4000': 74.1 GB RSS

    ============================================================
    Condition: seurat_4000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 2196 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 2196/6348 (34.6%)
    [MEM] run_hvg_conditions: start 'pearson_4000': 74.1 GB RSS

    ============================================================
    Condition: pearson_4000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 2433 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 2433/6348 (38.3%)
    [MEM] run_hvg_conditions: start 'seurat_v3_6000': 74.1 GB RSS

    ============================================================
    Condition: seurat_v3_6000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 3118 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 6000, GRN genes used: 3118/6348 (49.1%)
    [MEM] run_hvg_conditions: start 'seurat_6000': 74.1 GB RSS

    ============================================================
    Condition: seurat_6000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 4879 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 11540, GRN genes used: 4879/6348 (76.9%)
    [MEM] run_hvg_conditions: start 'pearson_6000': 74.2 GB RSS

    ============================================================
    Condition: pearson_6000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 3319 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 6000, GRN genes used: 3319/6348 (52.3%)
    [MEM] run_hvg_conditions: start 'seurat_v3_8000': 74.2 GB RSS

    ============================================================
    Condition: seurat_v3_8000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 3927 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 8000, GRN genes used: 3927/6348 (61.9%)
    [MEM] run_hvg_conditions: start 'seurat_8000': 74.2 GB RSS

    ============================================================
    Condition: seurat_8000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 5672 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 13540, GRN genes used: 5672/6348 (89.4%)
    [MEM] run_hvg_conditions: start 'pearson_8000': 74.2 GB RSS

    ============================================================
    Condition: pearson_8000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 4092 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 8000, GRN genes used: 4092/6348 (64.5%)
    [MEM] run_hvg_conditions: start 'seurat_v3_10000': 74.3 GB RSS

    ============================================================
    Condition: seurat_v3_10000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 4589 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 10000, GRN genes used: 4589/6348 (72.3%)
    [MEM] run_hvg_conditions: start 'seurat_10000': 74.3 GB RSS

    ============================================================
    Condition: seurat_10000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 15540, GRN genes used: 6348/6348 (100.0%)
    [MEM] run_hvg_conditions: start 'pearson_10000': 74.3 GB RSS

    ============================================================
    Condition: pearson_10000
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 4589 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 10000, GRN genes used: 4589/6348 (72.3%)
    [MEM] run_projection_pipeline: after run_hvg_conditions: 74.8 GB RSS
    Cache saved to /rds/user/rajd2/hpc-work/snRNAseq_2026/notebooks/ahbaC3_sensitivity_combined_scANVI/_cache
    [MEM] run_projection_pipeline: done: 74.9 GB RSS
    scores: 15200000, stats: 19, final_df (excitatory): 5584936, hvg_df: 109620

``` python
# Filter to mature L2-3 IT neurons and immature/newborn EN neurons (scANVI labels)
if 'cell_type_aligned' in final_df.columns and FILTER_CELL_TYPES is not None:
    final_df = final_df[final_df['cell_type_aligned'].isin(FILTER_CELL_TYPES)].copy()
    print(f"Filtered final_df to EN subtypes {FILTER_CELL_TYPES}")
    print(f"  {len(final_df)} rows remaining")
    print(final_df['cell_type_aligned'].value_counts().to_string())
else:
    print("No cell type filtering applied.")
```

    No cell type filtering applied.

### Gene Overlap Summary

``` python
print(stats_df.to_string(index=False))
```

          condition  n_hvg  n_grn_genes_used  pct_grn_retained
          all_genes  15540              6348             100.0
     seurat_v3_1000   1000               578               9.1
        seurat_1000   1000               597               9.4
       pearson_1000   1000               686              10.8
     seurat_v3_2000   2000              1115              17.6
        seurat_2000   2000              1143              18.0
       pearson_2000   2000              1308              20.6
     seurat_v3_4000   4000              2180              34.3
        seurat_4000   4000              2196              34.6
       pearson_4000   4000              2433              38.3
     seurat_v3_6000   6000              3118              49.1
        seurat_6000  11540              4879              76.9
       pearson_6000   6000              3319              52.3
     seurat_v3_8000   8000              3927              61.9
        seurat_8000  13540              5672              89.4
       pearson_8000   8000              4092              64.5
    seurat_v3_10000  10000              4589              72.3
       seurat_10000  15540              6348             100.0
      pearson_10000  10000              4589              72.3

## 3. Age Range Sensitivity (Gap Model)

We fix the childhood lower bound at 1 year and independently vary:

-   **Childhood upper bound**: 6, 7, 8, 9 (row facets)
-   **Adolescence lower bound**: 12, 13, 14, 15 (x-axis)
-   **Adolescence upper bound**: 22, 23, 24, 25 (column facets)

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

    Best age range (lowest p for all_genes): Childhood [1, 7), Adolescence [15, 25)  (p = 0.0242, d = -0.62)

### 3.2 Cohen’s d

``` python
%%R -w 280 -h 220 -u mm -r 300

plot_gap_cohens_d(sens_all)
```

![](ahbaC3_sensitivity_dataset_scANVI_files/figure-markdown_strict/cell-12-output-1.png)

### 3.3 P-value

``` python
%%R -w 280 -h 220 -u mm -r 300

plot_gap_pvalue(sens_all)
```

![](ahbaC3_sensitivity_dataset_scANVI_files/figure-markdown_strict/cell-13-output-1.png)

### 3.4 Minimum Detectable Effect Size

``` python
%%R -w 280 -h 220 -u mm -r 300

plot_gap_power(sens_all)
```

![](ahbaC3_sensitivity_dataset_scANVI_files/figure-markdown_strict/cell-14-output-1.png)

## 4. HVG Comparison (best age range)

All subsequent analyses use the age range with the lowest p-value from
the `all_genes` baseline above.

``` python
%%R

cat(sprintf("Using: Childhood = [%d, %d), Gap = [%d, %d), Adolescence = [%d, %d)\n",
            CHILD_START, best_ce, best_ce, best_as, best_as, best_ae))
```

    Using: Childhood = [1, 7), Gap = [7, 15), Adolescence = [15, 25)

### 4.1 GRN Gene Retention

``` python
%%R -i stats_df -w 220 -h 80 -u mm -r 300

plot_gene_retention(stats_df, N_VALUES)
```

![](ahbaC3_sensitivity_dataset_scANVI_files/figure-markdown_strict/cell-16-output-1.png)

### 4.1b HVG Gene Set Overlap (Euler diagrams)

``` python
%%R -i hvg_df -w 280 -h 110 -u mm -r 300

plot_hvg_euler(hvg_df)
```

![](ahbaC3_sensitivity_dataset_scANVI_files/figure-markdown_strict/cell-17-output-1.png)

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

    R[write to console]: In addition: 
    R[write to console]: Warning messages:

    R[write to console]: 1: Removed 1443926 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 2: Removed 1443926 rows containing missing values or values outside the scale
    range (`geom_point()`). 

    R[write to console]: 3: Removed 2898 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 4: Removed 2898 rows containing missing values or values outside the scale range
    (`geom_point()`). 

    In addition: Warning message:
    The `panel.margin` argument of `theme()` is deprecated as of ggplot2 2.2.0.
    ℹ Please use the `panel.spacing` argument instead.
    This warning is displayed once every 8 hours.
    Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    generated. 

![](ahbaC3_sensitivity_dataset_scANVI_files/figure-markdown_strict/cell-18-output-4.png)

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

    R[write to console]: In addition: 
    R[write to console]: Warning messages:

    R[write to console]: 1: Removed 1443926 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 2: Removed 1443926 rows containing missing values or values outside the scale
    range (`geom_point()`). 

    R[write to console]: 3: Removed 2898 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 4: Removed 2898 rows containing missing values or values outside the scale range
    (`geom_point()`). 

![](ahbaC3_sensitivity_dataset_scANVI_files/figure-markdown_strict/cell-19-output-3.png)

### 4.4 C3+ Effect Summary

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

![](ahbaC3_sensitivity_dataset_scANVI_files/figure-markdown_strict/cell-20-output-2.png)
