# AHBA C3 HVG Investigation


## Setup

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
# ── Environment Configuration ────────────────────────────────────────────────
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
library(ggbeeswarm)
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
```

## Load and Normalize Data

``` python
adata = sc.read_h5ad(rds_dir + "/Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad")

# Back up raw counts BEFORE normalization (needed for seurat_v3 HVG)
adata.layers['counts'] = adata.X.copy()

# CPM normalize for projection
sc.pp.normalize_total(adata, target_sum=1e6)

# Fill source if missing
if 'source' not in adata.obs.columns:
    adata.obs['source'] = 'velmeshev'

print(f"Shape: {adata.shape}")
print(f"Layers: {list(adata.layers.keys())}")
```

    Shape: (64250, 17663)
    Layers: ['counts']

``` python
# seurat flavor needs log-normalized data in .X
# Create a copy just for HVG selection
adata_log = adata.copy()
sc.pp.log1p(adata_log)
print("Log-normalized copy ready for seurat HVG selection.")
```

    Log-normalized copy ready for seurat HVG selection.

## Load and Remap GRN

``` python
grn_file = os.path.join(ref_dir, "ahba_dme_hcp_top8kgenes_weights.csv")

print(f"Loading GRN from {grn_file}...")
ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file, use_weights=True)
ahba_GRN = map_grn_symbols_to_ensembl(ahba_GRN, adata)

# Count total GRN genes available in adata (baseline)
grn_pivot = ahba_GRN.pivot_table(index='Network', columns='Gene', values='Importance', fill_value=0)
total_grn_genes_in_adata = len(np.intersect1d(grn_pivot.columns, adata.var_names))
print(f"GRN genes in adata: {total_grn_genes_in_adata} / {grn_pivot.shape[1]}")
```

    Input sequence provided is already in string format. No operation performed
    Input sequence provided is already in string format. No operation performed

    Loading GRN from /Users/richard/Git/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv...
    Mapped 6641/7973 symbols via adata.var
    Querying mygene for 1332 unmapped symbols...

    134 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    337 input query terms found no hit: ['AAED1', 'AARS', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZIN1-AS1'

    After mygene: 6650/7973 mapped, 1323 dropped
    GRN genes in adata: 6650 / 6650

## Run HVG Conditions and Project

``` python
N_VALUES = [500, 1000, 2000, 4000, 8000]

conditions = [{'label': 'all_genes', 'flavor': None, 'n_top_genes': None}]
for n in N_VALUES:
    conditions.append({'label': f'seurat_v3_{n}', 'flavor': 'seurat_v3', 'n_top_genes': n})
    conditions.append({'label': f'seurat_{n}', 'flavor': 'seurat', 'n_top_genes': n})
    conditions.append({'label': f'pearson_{n}', 'flavor': 'pearson_residuals', 'n_top_genes': n})

all_scores = []
gene_stats = []

for cond in conditions:
    label = cond['label']
    print(f"\n{'='*60}\nCondition: {label}")

    if cond['flavor'] is None:
        # Baseline: all genes
        if 'highly_variable' in adata.var.columns:
            del adata.var['highly_variable']
        project_GRN(adata, ahba_GRN, 'X_ahba', use_highly_variable=False, log_transform=False)
        n_hvg = adata.shape[1]
        n_grn_used = total_grn_genes_in_adata
    else:
        # Select HVGs
        if cond['flavor'] == 'seurat_v3':
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3',
                                        n_top_genes=cond['n_top_genes'], layer='counts')
        elif cond['flavor'] == 'seurat':
            sc.pp.highly_variable_genes(adata_log, flavor='seurat',
                                        n_top_genes=cond['n_top_genes'])
            adata.var['highly_variable'] = adata_log.var['highly_variable']
        elif cond['flavor'] == 'pearson_residuals':
            sc.experimental.pp.highly_variable_genes(adata,
                                                      n_top_genes=cond['n_top_genes'],
                                                      layer='counts')

        n_hvg = int(adata.var['highly_variable'].sum())
        hvg_genes = adata.var_names[adata.var['highly_variable']]
        n_grn_used = len(np.intersect1d(grn_pivot.columns, hvg_genes))

        project_GRN(adata, ahba_GRN, 'X_ahba', use_highly_variable=True, log_transform=False)

    gene_stats.append({
        'condition': label,
        'n_hvg': n_hvg,
        'n_grn_genes_used': n_grn_used,
        'pct_grn_retained': round(100 * n_grn_used / total_grn_genes_in_adata, 1)
    })

    # Extract C3+/C3- scores
    proj = pd.DataFrame(adata.obsm['X_ahba'], index=adata.obs_names,
                        columns=adata.uns['X_ahba_names'])
    proj = proj[['C3+', 'C3-']].copy()
    proj['obs_names'] = proj.index
    proj['condition'] = label
    melted = proj.melt(id_vars=['obs_names', 'condition'], var_name='C', value_name='value')
    all_scores.append(melted)

    print(f"  HVGs: {n_hvg}, GRN genes used: {n_grn_used}/{total_grn_genes_in_adata} "
          f"({gene_stats[-1]['pct_grn_retained']}%)")

# Free the log-normalized copy
del adata_log
import gc; gc.collect()

scores_df = pd.concat(all_scores, ignore_index=True)
stats_df = pd.DataFrame(gene_stats)
print(f"\nTotal score rows: {len(scores_df)}")
```


    ============================================================
    Condition: all_genes
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 17663, GRN genes used: 6650/6650 (100.0%)

    ============================================================
    Condition: seurat_v3_500
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 316 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 500, GRN genes used: 316/6650 (4.8%)

    ============================================================
    Condition: seurat_500
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 270 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 500, GRN genes used: 270/6650 (4.1%)

    ============================================================
    Condition: pearson_500
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 326 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 500, GRN genes used: 326/6650 (4.9%)

    ============================================================
    Condition: seurat_v3_1000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 626 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 626/6650 (9.4%)

    ============================================================
    Condition: seurat_1000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 495 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 495/6650 (7.4%)

    ============================================================
    Condition: pearson_1000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 626 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 1000, GRN genes used: 626/6650 (9.4%)

    ============================================================
    Condition: seurat_v3_2000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 1220 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 1220/6650 (18.3%)

    ============================================================
    Condition: seurat_2000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 918 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 918/6650 (13.8%)

    ============================================================
    Condition: pearson_2000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 1131 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 2000, GRN genes used: 1131/6650 (17.0%)

    ============================================================
    Condition: seurat_v3_4000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 2237 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 2237/6650 (33.6%)

    ============================================================
    Condition: seurat_4000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 1653 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 1653/6650 (24.9%)

    ============================================================
    Condition: pearson_4000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 2018 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 4000, GRN genes used: 2018/6650 (30.3%)

    ============================================================
    Condition: seurat_v3_8000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 3312 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 8000, GRN genes used: 3312/6650 (49.8%)

    ============================================================
    Condition: seurat_8000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 3127 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 8000, GRN genes used: 3127/6650 (47.0%)

    ============================================================
    Condition: pearson_8000
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 3666 matched genes for projection...
    Computing sparse-dense dot product...
      HVGs: 8000, GRN genes used: 3666/6650 (55.1%)

    Total score rows: 2056000

## Gene Overlap Summary

``` python
print(stats_df.to_string(index=False))
```

         condition  n_hvg  n_grn_genes_used  pct_grn_retained
         all_genes  17663              6650             100.0
     seurat_v3_500    500               316               4.8
        seurat_500    500               270               4.1
       pearson_500    500               326               4.9
    seurat_v3_1000   1000               626               9.4
       seurat_1000   1000               495               7.4
      pearson_1000   1000               626               9.4
    seurat_v3_2000   2000              1220              18.3
       seurat_2000   2000               918              13.8
      pearson_2000   2000              1131              17.0
    seurat_v3_4000   4000              2237              33.6
       seurat_4000   4000              1653              24.9
      pearson_4000   4000              2018              30.3
    seurat_v3_8000   8000              3312              49.8
       seurat_8000   8000              3127              47.0
      pearson_8000   8000              3666              55.1

## Prepare Data for R

``` python
cols_to_keep = ['individual', 'age_years', 'cell_class', 'cell_subclass', 'cell_type']
cols_to_keep = [c for c in cols_to_keep if c in adata.obs.columns]

meta = adata.obs[cols_to_keep].copy()
meta['obs_names'] = meta.index

final_df = pd.merge(scores_df, meta, on='obs_names')
final_df = final_df[final_df['cell_class'] == 'Excitatory']

# Define condition ordering for plots
condition_order = ['all_genes'] + [f'seurat_v3_{n}' for n in N_VALUES] + [f'seurat_{n}' for n in N_VALUES] + [f'pearson_{n}' for n in N_VALUES]
final_df['condition'] = pd.Categorical(final_df['condition'], categories=condition_order, ordered=True)

print(f"Excitatory C3 rows: {len(final_df)}")
```

    Excitatory C3 rows: 1082080

## Plotting

### GRN Gene Retention by HVG Condition

``` python
%%R -i stats_df -w 180 -h 80 -u mm -r 300

stats_df <- stats_df %>%
  mutate(
    flavor = case_when(
      condition == 'all_genes' ~ 'none',
      grepl('seurat_v3', condition) ~ 'seurat_v3',
      grepl('pearson', condition) ~ 'pearson_residuals',
      TRUE ~ 'seurat'
    )
  ) %>%
  mutate(condition = factor(condition, levels = c(
    'all_genes',
    paste0('seurat_v3_', c(500, 1000, 2000, 4000, 8000)),
    paste0('seurat_', c(500, 1000, 2000, 4000, 8000)),
    paste0('pearson_', c(500, 1000, 2000, 4000, 8000))
  )))

baseline_val <- stats_df %>% filter(condition == 'all_genes') %>% pull(n_grn_genes_used)

ggplot(stats_df, aes(x = condition, y = n_grn_genes_used, fill = flavor)) +
  geom_col() +
  geom_hline(yintercept = baseline_val, linetype = 'dashed', color = 'red') +
  geom_text(aes(label = n_grn_genes_used), vjust = -0.3, size = 2.5) +
  scale_fill_brewer(palette = 'Set2') +
  labs(x = NULL, y = 'GRN genes used in projection',
       title = 'GRN gene retention by HVG condition') +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-13-output-1.png)

### Age Trajectories by HVG Condition

``` python
%%R -i final_df -w 300 -h 130 -u mm -r 300

df <- final_df %>%
  mutate(Age_log2 = log2(age_years + 1)) %>%
  mutate(condition = factor(condition, levels = c(
    'all_genes',
    paste0('seurat_v3_', c(500, 1000, 2000, 4000, 8000)),
    paste0('seurat_', c(500, 1000, 2000, 4000, 8000)),
    paste0('pearson_', c(500, 1000, 2000, 4000, 8000))
  )))

ylim_max <- quantile(df$value, .999)
ylim_min <- quantile(df$value, .001)

df %>%
  filter(value >= ylim_min & value <= ylim_max) %>%
  ggplot(aes(x = Age_log2, y = value)) +
  facet_grid(C ~ condition, scales = 'free_y') +
  geom_point(size = 0.05, alpha = 0.1, color = 'grey50') +
  geom_smooth(se = FALSE, color = 'black', linewidth = 0.5) +
  scale_x_continuous(
    name = 'Donor Age',
    breaks = log2(1 + c(0, 1, 9, 25, 60)),
    labels = function(x) round(2^x - 1, 1)
  ) +
  scale_y_continuous(
    name = 'Score (CPM)',
    labels = function(y) paste0(round(y / 1e3, 1), 'K')
  ) +
  theme_classic() +
  theme(
    text = element_text(size = 8),
    strip.text = element_text(size = 7),
    strip.text.y.right = element_text(angle = 0),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
    panel.spacing = unit(0.3, 'lines')
  ) +
  ggtitle("C3 age trajectories by HVG condition")
```

    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-14-output-2.png)

### Childhood vs Adolescence Box Plots

``` python
%%R -w 300 -h 130 -u mm -r 300

comparisons <- list(c('Childhood', 'Adolescence'))

df_boxes <- df %>%
  rename(Individual = individual, network = C) %>%
  mutate(age_range = case_when(
    age_years < 1 ~ "Infancy",
    age_years >= 1 & age_years < 9 ~ "Childhood",
    age_years >= 9 & age_years < 25 ~ "Adolescence",
    age_years >= 25 ~ "Adulthood"
  )) %>%
  mutate(age_range = factor(age_range, ordered = TRUE,
    levels = c("Infancy", "Childhood", "Adolescence", "Adulthood"))) %>%
  filter(age_range %in% c('Childhood', 'Adolescence')) %>%
  group_by(condition, network, Individual, age_range) %>%
  summarize(value = mean(value), .groups = 'drop')

df_boxes %>%
  ggplot(aes(x = age_range, y = value)) +
  facet_grid(network ~ condition, scales = 'free_y') +
  geom_boxplot(outlier.shape = NA, alpha = 0.4) +
  geom_jitter(width = 0.15, size = 0.5, alpha = 0.6) +
  stat_compare_means(comparisons = comparisons, label = 'p.signif', size = 3) +
  scale_y_continuous(
    name = 'Pseudobulked Score (CPM)',
    labels = function(y) paste0(round(y / 1e3, 1), 'K')
  ) +
  theme_classic() +
  theme(
    text = element_text(size = 8),
    strip.text = element_text(size = 7),
    strip.text.y.right = element_text(angle = 0),
    axis.text.x = element_text(angle = 30, hjust = 1, size = 8),
    axis.title.x = element_blank(),
    panel.spacing = unit(0.3, 'lines')
  ) +
  ggtitle("Childhood vs Adolescence: C3+/C3- by HVG condition")
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-15-output-1.png)

### Summary: C3+ Effect Size by Condition

``` python
%%R -w 180 -h 90 -u mm -r 300

delta_df <- df_boxes %>%
  group_by(condition, network, age_range) %>%
  summarize(median_score = median(value), .groups = 'drop') %>%
  pivot_wider(names_from = age_range, values_from = median_score) %>%
  mutate(delta = Childhood - Adolescence) %>%
  mutate(
    flavor = case_when(
      condition == 'all_genes' ~ 'none',
      grepl('seurat_v3', condition) ~ 'seurat_v3',
      grepl('pearson', condition) ~ 'pearson_residuals',
      TRUE ~ 'seurat'
    ),
    n_genes = case_when(
      condition == 'all_genes' ~ 99999,
      TRUE ~ as.numeric(gsub('.*_(\\d+)$', '\\1', condition))
    )
  )

delta_df %>%
  filter(network == 'C3+') %>%
  ggplot(aes(x = reorder(condition, n_genes), y = delta, color = flavor, group = flavor)) +
  geom_point(size = 3) +
  geom_line(linewidth = 0.5) +
  geom_hline(yintercept = 0, linetype = 'dashed', color = 'grey50') +
  scale_color_brewer(palette = 'Set2') +
  labs(
    x = NULL,
    y = 'Delta (Childhood - Adolescence median)',
    title = 'C3+ effect size: Childhood vs Adolescence',
    subtitle = 'Pseudobulked scores by individual'
  ) +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9))
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-16-output-1.png)

### P-value Summary

``` python
%%R -w 180 -h 65 -u mm -r 300

pval_df <- df_boxes %>%
  filter(network == 'C3+') %>%
  group_by(condition) %>%
  summarize(
    p_value = wilcox.test(
      value[age_range == 'Childhood'],
      value[age_range == 'Adolescence']
    )$p.value,
    n_childhood = sum(age_range == 'Childhood'),
    n_adolescence = sum(age_range == 'Adolescence'),
    .groups = 'drop'
  ) %>%
  mutate(signif = case_when(
    p_value < 0.001 ~ '***',
    p_value < 0.01 ~ '**',
    p_value < 0.05 ~ '*',
    TRUE ~ 'ns'
  ))

print(pval_df)
```

    # A tibble: 16 × 5
       condition       p_value n_childhood n_adolescence signif
       <ord>             <dbl>       <int>         <int> <chr> 
     1 all_genes      0.0371            14            18 *     
     2 seurat_v3_500  0.779             14            18 ns    
     3 seurat_v3_1000 0.357             14            18 ns    
     4 seurat_v3_2000 0.145             14            18 ns    
     5 seurat_v3_4000 0.135             14            18 ns    
     6 seurat_v3_8000 0.135             14            18 ns    
     7 seurat_500     0.667             14            18 ns    
     8 seurat_1000    0.722             14            18 ns    
     9 seurat_2000    0.985             14            18 ns    
    10 seurat_4000    0.0593            14            18 ns    
    11 seurat_8000    0.0450            14            18 *     
    12 pearson_500    0.000802          14            18 ***   
    13 pearson_1000   0.0143            14            18 *     
    14 pearson_2000   0.0160            14            18 *     
    15 pearson_4000   0.0160            14            18 *     
    16 pearson_8000   0.0143            14            18 *     

### Sensitivity to Age Range Definitions

How does the childhood-adolescence C3+ effect depend on how we define
the age boundaries? We sweep three parameters — childhood lower bound,
the childhood/adolescence boundary, and adolescence upper bound — across
a representative subset of HVG conditions.

``` python
%%R

# Compute all sensitivity stats in one pass: Cohen's d, p-values, sample sizes, power
library(pwr)

selected_conds <- c('seurat_v3_2000', 'seurat_2000', 'pearson_2000', 'seurat_v3_8000', 'seurat_8000', 'pearson_8000', 'all_genes')

params <- expand.grid(
  child_start = c(0.5, 1, 2, 3),
  boundary    = seq(8, 14),
  adol_end    = c(18, 21, 23, 25),
  stringsAsFactors = FALSE
)

all_stats <- list()
for (i in seq_len(nrow(params))) {
  cs <- params$child_start[i]
  bd <- params$boundary[i]
  ae <- params$adol_end[i]

  # Pseudobulk per individual, C3+ only
  pb <- df %>%
    filter(condition %in% selected_conds, C == 'C3+') %>%
    mutate(age_range = case_when(
      age_years >= cs & age_years < bd ~ "Childhood",
      age_years >= bd & age_years < ae ~ "Adolescence",
      TRUE ~ NA_character_
    )) %>%
    filter(!is.na(age_range)) %>%
    group_by(condition, individual, age_range) %>%
    summarize(value = mean(value), .groups = 'drop')

  tmp <- pb %>%
    group_by(condition) %>%
    summarize(
      n_child  = sum(age_range == 'Childhood'),
      n_adol   = sum(age_range == 'Adolescence'),
      mean_child = mean(value[age_range == 'Childhood']),
      mean_adol  = mean(value[age_range == 'Adolescence']),
      sd_child   = sd(value[age_range == 'Childhood']),
      sd_adol    = sd(value[age_range == 'Adolescence']),
      p_value = tryCatch(
        wilcox.test(value[age_range == 'Childhood'],
                    value[age_range == 'Adolescence'])$p.value,
        error = function(e) NA_real_
      ),
      .groups = 'drop'
    ) %>%
    mutate(
      pooled_sd = sqrt(((pmax(n_child,1) - 1) * sd_child^2 +
                        (pmax(n_adol,1) - 1) * sd_adol^2) /
                       pmax(n_child + n_adol - 2, 1)),
      cohens_d = ifelse(pooled_sd > 0, (mean_child - mean_adol) / pooled_sd, NA_real_),
      power = mapply(function(n1, n2, d) {
        if (is.na(d) || n1 < 2 || n2 < 2) return(NA_real_)
        tryCatch(
          pwr.t2n.test(n1 = n1, n2 = n2, d = abs(d), sig.level = 0.05)$power,
          error = function(e) NA_real_
        )
      }, n_child, n_adol, cohens_d),
      child_start = cs, boundary = bd, adol_end = ae
    )
  all_stats[[i]] <- tmp
}

sens_all <- bind_rows(all_stats) %>%
  mutate(
    condition = factor(condition, levels = selected_conds),
    log10p = -log10(p_value),
    signif = ifelse(p_value < 0.05, TRUE, FALSE),
    p_star = case_when(
      p_value < 0.001 ~ '***',
      p_value < 0.01 ~ '**',
      p_value < 0.05 ~ '*',
      TRUE ~ 'ns'
    ),
    p_label = ifelse(signif, paste0(p_star, '\n', sprintf("%.2f", round(p_value, 2))), ''),
    n_label = paste0(n_child, '/', n_adol, '\n', sprintf("%.2f", round(power, 2)))
  )

cat("Stats computed:", nrow(sens_all), "rows\n")
```

    Stats computed: 784 rows

``` python
%%R -w 260 -h 180 -u mm -r 300

# Heatmap of Cohen's d
sens_all %>%
  ggplot(aes(x = factor(boundary), y = condition, fill = cohens_d)) +
  facet_grid(
    paste0("child \u2265 ", child_start, "y") ~ paste0("adol < ", adol_end, "y")
  ) +
  geom_tile(color = 'white', linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.2f", cohens_d)), size = 2) +
  scale_fill_gradient2(
    low = '#2166AC', mid = 'white', high = '#B2182B', midpoint = 0,
    name = "Cohen's d"
  ) +
  labs(
    x = 'Childhood / Adolescence boundary (years)',
    y = 'HVG condition',
    title = "C3+ effect size (Cohen's d): Childhood vs Adolescence",
    subtitle = 'd = (mean_child - mean_adol) / pooled_sd, pseudobulked by individual'
  ) +
  theme_minimal(base_size = 9) +
  theme(
    strip.text = element_text(size = 8),
    axis.text.x = element_text(size = 8),
    panel.spacing = unit(0.4, 'lines'),
    plot.title = element_text(size = 11)
  )
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-19-output-1.png)

``` python
%%R -w 260 -h 180 -u mm -r 300

# Heatmap of -log10(p)
sens_all %>%
  ggplot(aes(x = factor(boundary), y = condition, fill = log10p)) +
  facet_grid(
    paste0("child \u2265 ", child_start, "y") ~ paste0("adol < ", adol_end, "y")
  ) +
  geom_tile(color = 'white', linewidth = 0.3) +
  geom_text(aes(label = p_label,
                fontface = ifelse(signif, 'bold', 'plain')),
            size = 2, color = 'white') +
  scale_fill_gradient(
    low = 'grey90', high = '#B2182B',
    name = expression(-log[10](p))
  ) +
  labs(
    x = 'Childhood / Adolescence boundary (years)',
    y = 'HVG condition',
    title = 'C3+ Wilcoxon p-value sensitivity to age range definitions',
    subtitle = 'Bold values: p < 0.05'
  ) +
  theme_minimal(base_size = 9) +
  theme(
    strip.text = element_text(size = 8),
    axis.text.x = element_text(size = 8),
    panel.spacing = unit(0.4, 'lines'),
    plot.title = element_text(size = 11)
  )
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-20-output-1.png)

``` python
%%R -w 260 -h 180 -u mm -r 300

# Power and sample size heatmap
# Color = power (from pwr.t2n.test with observed Cohen's d), text = n_child/n_adol
sens_all %>%
  ggplot(aes(x = factor(boundary), y = condition, fill = power)) +
  facet_grid(
    paste0("child \u2265 ", child_start, "y") ~ paste0("adol < ", adol_end, "y")
  ) +
  geom_tile(color = 'white', linewidth = 0.3) +
  geom_text(aes(label = n_label), size = 2) +
  scale_fill_gradient(
    low = 'grey95', high = '#1A9850', limits = c(0, 1),
    name = 'Power'
  ) +
  labs(
    x = 'Childhood / Adolescence boundary (years)',
    y = 'HVG condition',
    title = "Power analysis: ability to detect observed C3+ effect at p < 0.05",
    subtitle = "Cell labels: n_childhood / n_adolescence donors"
  ) +
  theme_minimal(base_size = 9) +
  theme(
    strip.text = element_text(size = 8),
    axis.text.x = element_text(size = 8),
    panel.spacing = unit(0.4, 'lines'),
    plot.title = element_text(size = 11)
  )
```

![](ahbaC3_hvg_investigation_files/figure-markdown_strict/cell-21-output-1.png)
