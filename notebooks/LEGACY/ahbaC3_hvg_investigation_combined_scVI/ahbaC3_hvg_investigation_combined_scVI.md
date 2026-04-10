# AHBA C3 HVG Investigation (Combined)


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
DATA_FILE = rds_dir + "/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld/scvi_output/integrated.h5ad"
N_VALUES = [1000, 2000, 4000, 6000, 8000, 10000]
CACHE_DIR = os.path.join(_repo_root, 'notebooks', 'ahbaC3_hvg_investigation_combined_scVI', '_cache')

cached = load_cache(CACHE_DIR)
```

``` python
if cached is not None:
    scores_df, stats_df, final_df, hvg_df = cached
    print(f"scores: {len(scores_df)} rows, stats: {len(stats_df)} rows, final: {len(final_df)} rows, hvg_df: {len(hvg_df)} rows")
else:
    adata, adata_log = load_single_scvi(DATA_FILE, source_label='combined')
    ahba_GRN, total_grn_genes = setup_grn(ref_dir, adata)
    scores_df, stats_df, final_df, hvg_df = run_projection_pipeline(
        adata, adata_log, ahba_GRN, total_grn_genes, N_VALUES, CACHE_DIR)
    del adata, adata_log
    import gc; gc.collect()

print(f"scores: {len(scores_df)}, stats: {len(stats_df)}, final_df (excitatory): {len(final_df)}, hvg_df: {len(hvg_df)}")
```

    Shape: (221567, 15540)
    WARNING: adata.X seems to be already log-transformed.

    Input sequence provided is already in string format. No operation performed
    Input sequence provided is already in string format. No operation performed

    Mapped 6391/7973 symbols via adata.var
    Querying mygene for 1582 unmapped symbols...

    134 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    342 input query terms found no hit: ['AAED1', 'AARS', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZIN1-AS1'

    After mygene: 6397/7973 mapped, 1576 dropped
    GRN genes in adata: 6397 / 6397

    ============================================================
    Condition: all_genes
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 6397 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 15540, GRN genes used: 6397/6397 (100.0%)

    ============================================================
    Condition: seurat_v3_1000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 512 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 512/6397 (8.0%)

    ============================================================
    Condition: seurat_1000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 543 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 543/6397 (8.5%)

    ============================================================
    Condition: pearson_1000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 675 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 675/6397 (10.6%)

    ============================================================
    Condition: seurat_v3_2000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 1028 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 1028/6397 (16.1%)

    ============================================================
    Condition: seurat_2000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 1055 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 1055/6397 (16.5%)

    ============================================================
    Condition: pearson_2000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 1261 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 1261/6397 (19.7%)

    ============================================================
    Condition: seurat_v3_4000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 2028 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 2028/6397 (31.7%)

    ============================================================
    Condition: seurat_4000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 2043 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 2043/6397 (31.9%)

    ============================================================
    Condition: pearson_4000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 2347 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 2347/6397 (36.7%)

    ============================================================
    Condition: seurat_v3_6000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 2972 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 6000, GRN genes used: 2972/6397 (46.5%)

    ============================================================
    Condition: seurat_6000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 4982 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 11540, GRN genes used: 4982/6397 (77.9%)

    ============================================================
    Condition: pearson_6000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 3189 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 6000, GRN genes used: 3189/6397 (49.9%)

    ============================================================
    Condition: seurat_v3_8000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 3701 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 8000, GRN genes used: 3701/6397 (57.9%)

    ============================================================
    Condition: seurat_8000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 5700 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 13540, GRN genes used: 5700/6397 (89.1%)

    ============================================================
    Condition: pearson_8000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 3870 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 8000, GRN genes used: 3870/6397 (60.5%)

    ============================================================
    Condition: seurat_v3_10000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 4363 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 10000, GRN genes used: 4363/6397 (68.2%)

    ============================================================
    Condition: seurat_10000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 6397 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 15540, GRN genes used: 6397/6397 (100.0%)

    ============================================================
    Condition: pearson_10000
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 4363 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 10000, GRN genes used: 4363/6397 (68.2%)
    Cache saved to /rds/user/rajd2/hpc-work/snRNAseq_2026/notebooks/ahbaC3_hvg_investigation_combined_scVI/_cache
    scores: 8419546, stats: 19, final_df (excitatory): 3101978, hvg_df: 109620

### Gene Overlap Summary

``` python
print(stats_df.to_string(index=False))
```

          condition  n_hvg  n_grn_genes_used  pct_grn_retained
          all_genes  15540              6397             100.0
     seurat_v3_1000   1000               512               8.0
        seurat_1000   1000               543               8.5
       pearson_1000   1000               675              10.6
     seurat_v3_2000   2000              1028              16.1
        seurat_2000   2000              1055              16.5
       pearson_2000   2000              1261              19.7
     seurat_v3_4000   4000              2028              31.7
        seurat_4000   4000              2043              31.9
       pearson_4000   4000              2347              36.7
     seurat_v3_6000   6000              2972              46.5
        seurat_6000  11540              4982              77.9
       pearson_6000   6000              3189              49.9
     seurat_v3_8000   8000              3701              57.9
        seurat_8000  13540              5700              89.1
       pearson_8000   8000              3870              60.5
    seurat_v3_10000  10000              4363              68.2
       seurat_10000  15540              6397             100.0
      pearson_10000  10000              4363              68.2

### Diagnostic Comparison

Numbers directly comparable to `diagnose_grn_batch_effect.py` (scVI
condition). Both pipelines compute `CPM(scvi_normalized) @ C3+_weights`;
`all_genes` uses the full gene set, matching the diagnostic script
exactly.

``` python
_AGE_BINS   = [-1, 0, 1, 3, 6, 10, 15, 20, 30, 40, 100]
_AGE_LABELS = ["prenatal", "0-1", "1-3", "3-6", "6-10",
               "10-15", "15-20", "20-30", "30-40", "40+"]
_CHILDHOOD  = (1.0, 10.0)

_ag = final_df[(final_df["condition"] == "all_genes") & (final_df["C"] == "C3+")].copy()
_ag["age_bin"] = pd.cut(_ag["age_years"], bins=_AGE_BINS, labels=_AGE_LABELS, right=False)
_sources = sorted(_ag["source"].unique())

print("=" * 68)
print("Notebook C3+ scores  (all_genes, Excitatory)  cf. diagnose_grn_*.out")
print("=" * 68)

print("\n--- All Excitatory: mean C3+ by source ---")
print(f"  {'Source':12}  {'N cells':>8}  {'mean':>12}")
for _src in _sources:
    _d = _ag[_ag["source"] == _src]
    print(f"  {_src:12}  {len(_d):>8,}  {_d['value'].mean():>12.1f}")

print("\n--- Excitatory: mean C3+ by source x age bin ---")
_pivot = _ag.groupby(["age_bin", "source"], observed=True)["value"].mean().unstack("source")
print(_pivot.round(0).to_string())

_ag_child = _ag[(_ag["age_years"] >= _CHILDHOOD[0]) & (_ag["age_years"] < _CHILDHOOD[1])]
print(f"\n--- Childhood ({_CHILDHOOD[0]}-{_CHILDHOOD[1]}y): N cells and mean C3+ by source ---")
print(f"  {'Source':12}  {'N cells':>8}  {'mean':>12}")
for _src in _sources:
    _d = _ag_child[_ag_child["source"] == _src]
    print(f"  {_src:12}  {len(_d):>8,}  "
          f"{_d['value'].mean() if len(_d) else float('nan'):>12.1f}")

_ind_col = next((c for c in ["individual", "donor_id"] if c in _ag_child.columns), None)
if _ind_col is not None:
    _pb = (_ag_child.groupby([_ind_col, "source"], observed=True)["value"]
           .mean().reset_index(name="c3_mean"))
    _pb = _pb.dropna(subset=["c3_mean"])
    print(f"\n--- Childhood: pseudobulk C3+ by source (per donor) ---")
    print(f"  {'Source':12}  {'N donors':>9}  {'mean':>12}  {'sem':>8}")
    for _src in _sources:
        _d = _pb[_pb["source"] == _src]["c3_mean"]
        if len(_d) == 0:
            print(f"  {_src:12}  {'0':>9}  {'nan':>12}  {'nan':>8}")
            continue
        print(f"  {_src:12}  {len(_d):>9,}  {_d.mean():>12.1f}  {_d.sem():>8.1f}"
              f"  (range {_d.min():.0f}–{_d.max():.0f})")
else:
    print("\n  (no donor ID column found for pseudobulk)")
```

    ====================================================================
    Notebook C3+ scores  (all_genes, Excitatory)  cf. diagnose_grn_*.out
    ====================================================================

    --- All Excitatory: mean C3+ by source ---
      Source         N cells          mean
      AGING           12,109      123210.1
      HBCC            14,036      123494.5
      VELMESHEV       33,815      101928.0
      WANG            21,671      106523.1

    --- Excitatory: mean C3+ by source x age bin ---
    source    VELMESHEV      WANG     AGING      HBCC
    age_bin                                          
    prenatal    94569.0   97725.0       NaN       NaN
    0-1        105880.0  115967.0  109040.0  108401.0
    1-3        123249.0       NaN  116304.0  116013.0
    3-6        117609.0       NaN  125457.0  126162.0
    6-10       131117.0       NaN  123076.0  121387.0
    10-15      115147.0  125224.0  123736.0  123793.0
    15-20      123815.0       NaN  123656.0  123327.0
    20-30      124910.0       NaN  122690.0  123682.0
    30-40      129869.0       NaN  122772.0  122242.0
    40+        127858.0       NaN  124988.0  125666.0

    --- Childhood (1.0-10.0y): N cells and mean C3+ by source ---
      Source         N cells          mean
      AGING              509      121105.3
      HBCC               489      120659.9
      VELMESHEV        3,241      123537.1
      WANG                 0           nan

    --- Childhood: pseudobulk C3+ by source (per donor) ---
      Source         N donors          mean       sem
      AGING                10      121279.0    2302.7  (range 108350–131902)
      HBCC                 10      121616.6    2452.3  (range 114310–139955)
      VELMESHEV            16      124717.5    1481.1  (range 113212–135591)
      WANG                  0           nan       nan

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

    Best age range (lowest p for all_genes): childhood >= 1.0y, boundary = 14y, adolescence < 21y (p = 0.0533, d = 0.50)

### 3.2 Cohen’s d

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_cohens_d(sens_all)
```

![](ahbaC3_hvg_investigation_combined_scVI_files/figure-markdown_strict/cell-12-output-1.png)

### 3.3 P-value

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_pvalue(sens_all)
```

![](ahbaC3_hvg_investigation_combined_scVI_files/figure-markdown_strict/cell-13-output-1.png)

### 3.4 Minimum Detectable Effect Size

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_power(sens_all)
```

![](ahbaC3_hvg_investigation_combined_scVI_files/figure-markdown_strict/cell-14-output-1.png)

## 4. HVG Comparison (best age range)

All subsequent analyses use the age range with the lowest p-value from
the `all_genes` baseline above.

``` python
%%R

cat(sprintf("Using: Childhood = [%.1f, %d), Adolescence = [%d, %d)\n",
            best_cs, best_bd, best_bd, best_ae))
```

    Using: Childhood = [1.0, 14), Adolescence = [14, 21)

### 4.1 GRN Gene Retention

``` python
%%R -i stats_df -w 220 -h 80 -u mm -r 300

plot_gene_retention(stats_df, N_VALUES)
```

![](ahbaC3_hvg_investigation_combined_scVI_files/figure-markdown_strict/cell-16-output-1.png)

### 4.1b HVG Gene Set Overlap (Euler diagrams)

``` python
%%R -i hvg_df -w 280 -h 110 -u mm -r 300

plot_hvg_euler(hvg_df)
```

![](ahbaC3_hvg_investigation_combined_scVI_files/figure-markdown_strict/cell-17-output-1.png)

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

    R[write to console]: In addition: 
    R[write to console]: Warning messages:

    R[write to console]: 1: Removed 895424 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 2: Removed 895424 rows containing missing values or values outside the scale range
    (`geom_point()`). 

    R[write to console]: 3: Removed 3492 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 4: Removed 3492 rows containing missing values or values outside the scale range
    (`geom_point()`). 

    In addition: Warning message:
    The `panel.margin` argument of `theme()` is deprecated as of ggplot2 2.2.0.
    ℹ Please use the `panel.spacing` argument instead.
    This warning is displayed once every 8 hours.
    Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    generated. 

![](ahbaC3_hvg_investigation_combined_scVI_files/figure-markdown_strict/cell-18-output-4.png)

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

    R[write to console]: In addition: 
    R[write to console]: Warning messages:

    R[write to console]: 1: Removed 895424 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 2: Removed 895424 rows containing missing values or values outside the scale range
    (`geom_point()`). 

    R[write to console]: 3: Removed 3492 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 4: Removed 3492 rows containing missing values or values outside the scale range
    (`geom_point()`). 

![](ahbaC3_hvg_investigation_combined_scVI_files/figure-markdown_strict/cell-19-output-3.png)

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

![](ahbaC3_hvg_investigation_combined_scVI_files/figure-markdown_strict/cell-20-output-2.png)
