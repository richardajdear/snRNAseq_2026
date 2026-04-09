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

    Need help getting started? Try the R Graphics Cookbook:
    https://r-graphics.org

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
DATA_FILE = rds_dir + "/Cam_snRNAseq/integrated/VelWangPsychAD_100k_pearson/scvi_output/integrated.h5ad"
SCVI_LAYER = 'scanvi_normalized'
N_VALUES = [1000, 2000, 4000, 6000, 8000, 10000]
CACHE_DIR = os.path.join(_repo_root, 'notebooks', 'ahbaC3_hvg_investigation_combined_scANVI', '_cache')

cached = load_cache(CACHE_DIR)
```

    Loaded cache from /rds/user/rajd2/hpc-work/snRNAseq_2026/notebooks/ahbaC3_hvg_investigation_combined_scANVI/_cache

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

    scores: 15200000 rows, stats: 19 rows, final: 5584936 rows, hvg_df: 109620 rows
    scores: 15200000, stats: 19, final_df (excitatory): 5584936, hvg_df: 109620

``` python
# Filter to mature L2-3 IT neurons and immature/newborn EN neurons (scANVI labels)
_EN_SUBTYPES = ['EN-L2_3-IT', 'EN-Newborn', 'EN-IT-Immature', 'EN-Non-IT-Immature']
if 'cell_type_aligned' in final_df.columns:
    final_df = final_df[final_df['cell_type_aligned'].isin(_EN_SUBTYPES)].copy()
    print(f"Filtered final_df to EN subtypes {_EN_SUBTYPES}")
    print(f"  {len(final_df)} rows remaining")
    print(final_df['cell_type_aligned'].value_counts().to_string())
else:
    print("WARNING: cell_type_aligned not found in final_df — no subtype filter applied")
```

    Filtered final_df to EN subtypes ['EN-L2_3-IT', 'EN-Newborn', 'EN-IT-Immature', 'EN-Non-IT-Immature']
      2597870 rows remaining
    cell_type_aligned
    EN-L2_3-IT                  1109752
    EN-IT-Immature               705280
    EN-Newborn                   611838
    EN-Non-IT-Immature           171000
    Astrocyte-Fibrous                 0
    Cajal-Retzius cell                0
    EN-L4-IT                          0
    Astrocyte-Protoplasmic            0
    Astrocyte-Immature                0
    EN-L5-IT                          0
    EN-L5-ET                          0
    EN-L5_6-NP                        0
    EN-L6-CT                          0
    EN-L6b                            0
    EN-L6-IT                          0
    IN-CGE-Immature                   0
    IN-CGE-SNCG                       0
    IN-CGE-VIP                        0
    IN-MGE-Immature                   0
    IN-MGE-PV                         0
    IN-MGE-SST                        0
    IN-Mix-LAMP5                      0
    IN-NCx_dGE-Immature               0
    IPC-EN                            0
    Microglia                         0
    OPC                               0
    Oligodendrocyte                   0
    Oligodendrocyte-Immature          0
    RG-oRG                            0
    RG-tRG                            0
    RG-vRG                            0
    Tri-IPC                           0
    Vascular                          0

### Gene Overlap Summary

``` python
print(stats_df.to_string(index=False))
```

          condition  n_hvg  n_grn_genes_used  pct_grn_retained
          all_genes  15540              6348             100.0
     seurat_v3_1000   1000               573               9.0
        seurat_1000   1000               579               9.1
       pearson_1000   1000               687              10.8
     seurat_v3_2000   2000              1118              17.6
        seurat_2000   2000              1129              17.8
       pearson_2000   2000              1313              20.7
     seurat_v3_4000   4000              2188              34.5
        seurat_4000   4000              2180              34.3
       pearson_4000   4000              2423              38.2
     seurat_v3_6000   6000              3118              49.1
        seurat_6000  11540              4880              76.9
       pearson_6000   6000              3309              52.1
     seurat_v3_8000   8000              3926              61.8
        seurat_8000  13540              5677              89.4
       pearson_8000   8000              4086              64.4
    seurat_v3_10000  10000              4589              72.3
       seurat_10000  15540              6348             100.0
      pearson_10000  10000              4589              72.3

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
      AGING            9,590      126776.6
      HBCC             9,997      126963.5
      VELMESHEV       26,099      101147.6
      WANG            22,679      102416.9

    --- Excitatory: mean C3+ by source x age bin ---
    source       AGING      HBCC  VELMESHEV      WANG
    age_bin                                          
    prenatal       NaN       NaN    94811.0   95110.0
    0-1       108940.0  108843.0   104271.0  114960.0
    1-3       120061.0  119139.0   122148.0       NaN
    3-6       126679.0  126604.0   121444.0       NaN
    6-10      118445.0  117890.0   131765.0       NaN
    10-15     126368.0  126638.0   124908.0  129300.0
    15-20     125830.0  126152.0   128241.0       NaN
    20-30     127365.0  127924.0   130199.0       NaN
    30-40     127268.0  126552.0   130804.0       NaN
    40+       127788.0  127613.0   130213.0       NaN

    --- Childhood (1.0-10.0y): N cells and mean C3+ by source ---
      Source         N cells          mean
      AGING              359      123348.8
      HBCC               293      122768.2
      VELMESHEV        2,136      123899.1
      WANG                 0           nan

    --- Childhood: pseudobulk C3+ by source (per donor) ---
      Source         N donors          mean       sem
      AGING                 9      121148.3    1639.9  (range 113790–127924)
      HBCC                  9      120363.9    1466.4  (range 116008–127052)
      VELMESHEV            16      125656.5    1161.0  (range 116599–132666)
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

    Best age range (lowest p for all_genes): childhood >= 0.5y, boundary = 9y, adolescence < 25y (p = 0.0018, d = -0.75)

### 3.2 Cohen’s d

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_cohens_d(sens_all)
```

![](ahbaC3_hvg_investigation_combined_scANVI_files/figure-markdown_strict/cell-13-output-1.png)

### 3.3 P-value

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_pvalue(sens_all)
```

![](ahbaC3_hvg_investigation_combined_scANVI_files/figure-markdown_strict/cell-14-output-1.png)

### 3.4 Minimum Detectable Effect Size

``` python
%%R -w 260 -h 200 -u mm -r 300

plot_sensitivity_power(sens_all)
```

![](ahbaC3_hvg_investigation_combined_scANVI_files/figure-markdown_strict/cell-15-output-1.png)

## 4. HVG Comparison (best age range)

All subsequent analyses use the age range with the lowest p-value from
the `all_genes` baseline above.

``` python
%%R

cat(sprintf("Using: Childhood = [%.1f, %d), Adolescence = [%d, %d)\n",
            best_cs, best_bd, best_bd, best_ae))
```

    Using: Childhood = [0.5, 9), Adolescence = [9, 25)

### 4.1 GRN Gene Retention

``` python
%%R -i stats_df -w 220 -h 80 -u mm -r 300

plot_gene_retention(stats_df, N_VALUES)
```

![](ahbaC3_hvg_investigation_combined_scANVI_files/figure-markdown_strict/cell-17-output-1.png)

### 4.1b HVG Gene Set Overlap (Euler diagrams)

``` python
%%R -i hvg_df -w 280 -h 110 -u mm -r 300

plot_hvg_euler(hvg_df)
```

![](ahbaC3_hvg_investigation_combined_scANVI_files/figure-markdown_strict/cell-18-output-1.png)

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

    R[write to console]: 1: Removed 769261 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 2: Removed 769261 rows containing missing values or values outside the scale range
    (`geom_point()`). 

    R[write to console]: 3: Removed 3402 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 4: Removed 3402 rows containing missing values or values outside the scale range
    (`geom_point()`). 

    In addition: Warning message:
    The `panel.margin` argument of `theme()` is deprecated as of ggplot2 2.2.0.
    ℹ Please use the `panel.spacing` argument instead.
    This warning is displayed once every 8 hours.
    Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    generated. 

![](ahbaC3_hvg_investigation_combined_scANVI_files/figure-markdown_strict/cell-19-output-4.png)

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

    R[write to console]: 1: Removed 769261 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 2: Removed 769261 rows containing missing values or values outside the scale range
    (`geom_point()`). 

    R[write to console]: 3: Removed 3402 rows containing non-finite outside the scale range
    (`stat_smooth()`). 

    R[write to console]: 4: Removed 3402 rows containing missing values or values outside the scale range
    (`geom_point()`). 

![](ahbaC3_hvg_investigation_combined_scANVI_files/figure-markdown_strict/cell-20-output-3.png)

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

![](ahbaC3_hvg_investigation_combined_scANVI_files/figure-markdown_strict/cell-21-output-2.png)
