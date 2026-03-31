# AHBA C3 Projection Analysis v2


## Setup

``` python
import os
import sys
import re
import warnings
from IPython import get_ipython
_ip = get_ipython()
if _ip:
    _ip.run_line_magic('load_ext', 'autoreload')
    _ip.run_line_magic('autoreload', '2')
```

``` python
# ── Environment Configuration ────────────────────────────────────────────────
# Bootstrap: find repo root so we can add code/ to sys.path before importing.
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
    # Point rpy2 at the conda env's R (not the system R)
    os.environ['R_HOME'] = '/Users/richard/mambaforge/envs/scanpy/lib/R'

%load_ext rpy2.ipython
```

``` python
%%R

library(ggplot2)
library(dplyr)
library(readr)
library(patchwork)
library(ggpubr)
library(stringr)
```

    R[write to console]: 
    Attaching package: ‘dplyr’


    R[write to console]: The following objects are masked from ‘package:stats’:

        filter, lag


    R[write to console]: The following objects are masked from ‘package:base’:

        intersect, setdiff, setequal, union

``` python
sys.path.append(code_dir)
try:
    from regulons import get_ahba_GRN, project_GRN
    from metadata_utils import get_original_metadata
    from gene_mapping import map_grn_symbols_to_ensembl
    from batch_correction import correct_projection_scores
except ImportError as e:
    print(f"Error importing modules: {e}")
```

``` python
# adata=sc.read_h5ad(rds_dir + "/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld.h5ad")
adata=sc.read_h5ad(rds_dir + "/Cam_snRNAseq/velmeshev/velmeshev_100k_PFC_lessOld.h5ad")
```

``` python
sc.pp.normalize_total(adata, target_sum=1e6)
```

## Projection

``` python
grn_file = os.path.join(ref_dir, "ahba_dme_hcp_top8kgenes_weights.csv")

print(f"Loading GRN from {grn_file}...")
ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file, use_weights=True)

# Remap GRN gene symbols to Ensembl IDs to match adata.var_names directly
ahba_GRN = map_grn_symbols_to_ensembl(ahba_GRN, adata)

print("Projecting GRN...")
project_GRN(adata, ahba_GRN, 'X_ahba', use_highly_variable=False, log_transform=False)
print(f"Projected shape: {adata.obsm['X_ahba'].shape}")
```

    Input sequence provided is already in string format. No operation performed
    Input sequence provided is already in string format. No operation performed

    Loading GRN from /Users/richard/Git/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv...
    Mapped 6641/7973 symbols via adata.var
    Querying mygene for 1332 unmapped symbols...

    134 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    337 input query terms found no hit: ['AAED1', 'AARS', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZIN1-AS1'

    After mygene: 6650/7973 mapped, 1323 dropped
    Projecting GRN...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
    Projected shape: (64250, 6)

``` python
# Fill 'source' if missing (e.g. velmeshev-only data)
if 'source' not in adata.obs.columns:
    adata.obs['source'] = 'velmeshev'

# Correct for batch (source dataset), preserving age signal
correct_projection_scores(adata, batch_key='source', covariates=['age_years', 'cell_class'])
```

    Stored corrected scores in adata.obsm['X_ahba_corrected'] (batch_key='source', covariates=['age_years', 'cell_class'])

## Prepare Data for R Plotting

``` python
# Extract projection (raw + corrected)
for key, label in [('X_ahba', 'raw'), ('X_ahba_corrected', 'corrected')]:
    proj = pd.DataFrame(adata.obsm[key], index=adata.obs_names, columns=adata.uns['X_ahba_names'])
    proj['obs_names'] = proj.index
    proj['correction'] = label

    melted = proj.melt(id_vars=['obs_names', 'correction'], var_name='C', value_name='value')
    if key == 'X_ahba':
        all_melted = melted
    else:
        all_melted = pd.concat([all_melted, melted], ignore_index=True)

# Merge with metadata
cols_to_keep = ['individual', 'age_years', 'cell_class', 'cell_subclass', 'cell_type', 'source']
cols_to_keep = [c for c in cols_to_keep if c in adata.obs.columns]

meta = adata.obs[cols_to_keep].copy()
meta['obs_names'] = meta.   index

final_df = pd.merge(all_melted, meta, on='obs_names')
final_df = final_df[final_df['C'].isin(['C3+', 'C3-'])]

final_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">obs_names</th>
<th data-quarto-table-cell-role="th">correction</th>
<th data-quarto-table-cell-role="th">C</th>
<th data-quarto-table-cell-role="th">value</th>
<th data-quarto-table-cell-role="th">individual</th>
<th data-quarto-table-cell-role="th">age_years</th>
<th data-quarto-table-cell-role="th">cell_class</th>
<th data-quarto-table-cell-role="th">cell_subclass</th>
<th data-quarto-table-cell-role="th">cell_type</th>
<th data-quarto-table-cell-role="th">source</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">257000</td>
<td>Ramos_34C_AACACACCATAATCCG-1</td>
<td>raw</td>
<td>C3+</td>
<td>104119.886084</td>
<td>34</td>
<td>-0.379452</td>
<td>Excitatory</td>
<td>Inhibitory</td>
<td>Interneurons</td>
<td>velmeshev</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">257001</td>
<td>Ramos_64G_CAACGGCAGACATAGT-1</td>
<td>raw</td>
<td>C3+</td>
<td>87456.653320</td>
<td>64</td>
<td>-0.312329</td>
<td>Excitatory</td>
<td>Inhibitory</td>
<td>Interneurons</td>
<td>velmeshev</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">257002</td>
<td>U01_TAAGTGCTCGGAGGTA-1_4369_BA9</td>
<td>raw</td>
<td>C3+</td>
<td>103390.332177</td>
<td>4369</td>
<td>2.720548</td>
<td>Excitatory</td>
<td>EN_L5_IT</td>
<td>L5</td>
<td>velmeshev</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">257003</td>
<td>Ramos_23C_CTCAAGACAGATAAAC-1</td>
<td>raw</td>
<td>C3+</td>
<td>79516.136862</td>
<td>23</td>
<td>-0.312329</td>
<td>Inhibitory</td>
<td>Inhibitory</td>
<td>INT</td>
<td>velmeshev</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">257004</td>
<td>U01_CAAGAAATCCAATGGT-1_GW16-2-2-20_PFC</td>
<td>raw</td>
<td>C3+</td>
<td>67368.422990</td>
<td>GW16-2-2-20</td>
<td>-0.465753</td>
<td>Excitatory</td>
<td>Progenitors</td>
<td>Progenitors</td>
<td>velmeshev</td>
</tr>
</tbody>
</table>

</div>

## Plotting with R (rpy2)

``` python
%%R -i final_df -i code_dir -w 220 -h 220 -u mm -r 300

source_path <- file.path(code_dir, "age_plots.r")
message(paste("Sourcing functions from", source_path))
source(source_path)

message("Processing data in R...")

# Clean/Process Data for R
df <- final_df %>%
  filter(cell_class == 'Excitatory') %>%
  rename(Individual = individual) %>%
  mutate(Age_log2 = log2(age_years+1)) %>%
  mutate(age_range = case_when(
    age_years < 0 ~ "Prenatal",
    age_years < 1 ~ "Infancy",
    age_years >= 1 & age_years < 9 ~ "Childhood",
    age_years >= 9 & age_years < 25 ~ "Adolescence",
    age_years >= 25 ~ "Adulthood"
  )) %>%
  mutate(age_range = factor(age_range, ordered=T, levels=c("Prenatal","Infancy", "Childhood", "Adolescence", "Adulthood"))) %>%
  filter(!is.na(age_range)) %>%
  mutate(C = factor(C, levels=c('C3+', 'C3-')))

# --- Raw plots ---
df_raw <- df %>% filter(correction == 'raw')

color_col <- "source"
p_age_raw <- df_raw %>% plot_age(color_var = color_col) + ggtitle("Raw")

comparisons <- list(
    c('Adolescence', 'Adulthood'),
    c('Adolescence', 'Childhood')
)
df_box_raw <- df_raw %>% rename(network = C)
p_boxes_raw <- df_box_raw %>% plot_boxes(color_var='source') + stat_compare_means(comparisons = comparisons, color='blue', label='p.signif')

# --- Corrected plots ---
df_corr <- df %>% filter(correction == 'corrected')

p_age_corr <- df_corr %>% plot_age(color_var = color_col) + ggtitle("Batch-corrected")

df_box_corr <- df_corr %>% rename(network = C)
p_boxes_corr <- df_box_corr %>% plot_boxes(color_var='source') + stat_compare_means(comparisons = comparisons, color='blue', label='p.signif')

# Combine: raw on top, corrected on bottom
p_final <- (p_age_raw | p_boxes_raw) / (p_age_corr | p_boxes_corr) +
    plot_annotation(tag_levels='a', title="AHBA C3: raw vs batch-corrected scores")

p_final
```

    R[write to console]: Sourcing functions from /Users/richard/Git/snRNAseq_2026/code/age_plots.r

    R[write to console]: 
    Attaching package: ‘scales’


    R[write to console]: The following object is masked from ‘package:readr’:

        col_factor

    ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ✔ forcats   1.0.0     ✔ tibble    3.3.0
    ✔ lubridate 1.9.4     ✔ tidyr     1.3.1
    ✔ purrr     1.1.0     
    ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ✖ scales::col_factor() masks readr::col_factor()
    ✖ purrr::discard()     masks scales::discard()
    ✖ dplyr::filter()      masks stats::filter()
    ✖ dplyr::lag()         masks stats::lag()
    ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

    R[write to console]: Processing data in R...

    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'
    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'

![](ahbaC3_projection_v2_files/figure-markdown_strict/cell-12-output-5.png)

``` python
%%R -w 220 -h 115 -u mm -r 300

# Same data, colour by cell subclass instead of source
p_age_raw_sub  <- df_raw  %>% plot_age(color_var = 'cell_subclass') + ggtitle("Raw") +
    guides(color=guide_legend(ncol=1, override.aes = list(size = 3, alpha=.8)))
p_age_corr_sub <- df_corr %>% plot_age(color_var = 'cell_subclass') + ggtitle("Batch-corrected") +
    guides(color=guide_legend(ncol=1, override.aes = list(size = 3, alpha=.8)))

(p_age_raw_sub | p_age_corr_sub) +
    plot_layout(guides='collect') +
    plot_annotation(title="AHBA C3: coloured by cell subclass")
```

    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'
    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'

![](ahbaC3_projection_v2_files/figure-markdown_strict/cell-13-output-2.png)
