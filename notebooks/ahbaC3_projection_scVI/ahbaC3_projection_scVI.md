# AHBA C3 Projection on scVI Batch-Corrected Data


## Setup

``` python
import os
import sys
import warnings
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

# ── Input: scVI-integrated h5ad ──────────────────────────────────────────────
# Change this path to switch between datasets (10k test / 100k / full)
SCVI_INPUT = os.path.join(
    rds_dir,
    "Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld/scvi_output/integrated.h5ad"
)
SCVI_LAYER = 'scvi_normalized'   # or 'scanvi_normalized'

print(f"Environment : {_env['name']}")
print(f"  input    : {SCVI_INPUT}")
print(f"  layer    : {SCVI_LAYER}")
```

    Environment : local
      input    : /Users/richard/Git/snRNAseq_2026/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/VelWangPsychad_10k_PFC_lessOld/scvi_output/integrated.h5ad
      layer    : scvi_normalized

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
library(readr)
library(patchwork)
library(ggpubr)
library(stringr)
```

    R[write to console]: Use suppressPackageStartupMessages() to eliminate package startup
    messages

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
    from gene_mapping import map_grn_symbols_to_ensembl
    from batch_correction import correct_projection_scores
except ImportError as e:
    print(f"Error importing modules: {e}")
```

## Load Data

``` python
adata = sc.read_h5ad(SCVI_INPUT)
print(adata)
print(f"\nLayers: {list(adata.layers.keys())}")
print(f"obsm:   {list(adata.obsm.keys())}")
```

    AnnData object with n_obs × n_vars = 22282 × 15540
        obs: 'age_years', 'assay', 'assay_ontology_term_id', 'cell_class', 'cell_type', 'cell_type_ontology_term_id', 'chemistry', 'dataset', 'development_stage', 'development_stage_ontology_term_id', 'disease', 'disease_ontology_term_id', 'donor_id', 'is_primary_data', 'observation_joinid', 'region', 'self_reported_ethnicity', 'self_reported_ethnicity_ontology_term_id', 'sex', 'sex_ontology_term_id', 'suspension_type', 'tissue', 'tissue_ontology_term_id', 'tissue_type', 'source', 'cell_subclass'
        var: 'gene_symbol', 'feature_length', 'highly_variable', 'highly_variable_rank', 'means', 'variances', 'variances_norm'
        uns: 'hvg', 'neighbors_raw', 'neighbors_scanvi', 'neighbors_scvi', 'umap'
        obsm: 'X_pca_raw', 'X_scANVI', 'X_scVI', 'X_umap', 'X_umap_raw', 'X_umap_scanvi', 'X_umap_scvi'
        layers: 'counts', 'scanvi_normalized', 'scvi_normalized'
        obsp: 'neighbors_raw_connectivities', 'neighbors_raw_distances', 'neighbors_scanvi_connectivities', 'neighbors_scanvi_distances', 'neighbors_scvi_connectivities', 'neighbors_scvi_distances'

    Layers: ['counts', 'scanvi_normalized', 'scvi_normalized']
    obsm:   ['X_pca_raw', 'X_scANVI', 'X_scVI', 'X_umap', 'X_umap_raw', 'X_umap_scanvi', 'X_umap_scvi']

``` python
# Use scVI batch-corrected expression as adata.X
# scvi_normalized sums to ~1.0 per cell (proportions); scale to CPM for
# comparability with the unnormalized pipeline
import scipy.sparse as sp
layer = adata.layers[SCVI_LAYER]
adata.X = layer * 1e6
print(f"Set adata.X = layers['{SCVI_LAYER}'] × 1e6  (CPM-equivalent)")
print(f"  per-cell sum sample: {adata.X[:3].sum(axis=1).A1 if sp.issparse(adata.X) else adata.X[:3].sum(axis=1)}")
```

    Set adata.X = layers['scvi_normalized'] × 1e6  (CPM-equivalent)
      per-cell sum sample: [1000000.2  1000000.25 1000000.25]

## Projection

``` python
grn_file = os.path.join(ref_dir, "ahba_dme_hcp_top8kgenes_weights.csv")

print(f"Loading GRN from {grn_file}...")
ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file, use_weights=True)

# adata.var_names are Ensembl IDs; remap GRN symbols to Ensembl
ahba_GRN = map_grn_symbols_to_ensembl(ahba_GRN, adata)

print("Projecting GRN...")
project_GRN(adata, ahba_GRN, 'X_ahba', use_highly_variable=False, log_transform=False)
print(f"Projected shape: {adata.obsm['X_ahba'].shape}")
```

    Input sequence provided is already in string format. No operation performed
    Input sequence provided is already in string format. No operation performed

    Loading GRN from /Users/richard/Git/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv...
    Mapped 6391/7973 symbols via adata.var
    Querying mygene for 1582 unmapped symbols...

    134 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    340 input query terms found no hit: ['AAED1', 'AARS', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZIN1-AS1'

    After mygene: 6397/7973 mapped, 1576 dropped
    Projecting GRN...
    Found 6397 matching genes in var_names.
    Aligning GRN weights to 6397 matched genes for projection...
    Computing sparse-dense dot product...
    Projected shape: (22282, 6)

``` python
# Optional: OLS batch correction on projection scores.
# scVI already corrects batch at the expression level, so this primarily
# removes any residual score-level batch shifts not captured by scVI.
correct_projection_scores(adata, batch_key='source', covariates=['age_years', 'cell_class'])
```

    Stored corrected scores in adata.obsm['X_ahba_corrected'] (batch_key='source', covariates=['age_years', 'cell_class'])

## Prepare Data for R Plotting

``` python
# Extract projection scores (raw scVI-based + score-level corrected)
for key, label in [('X_ahba', 'scvi'), ('X_ahba_corrected', 'scvi+ols')]:
    proj = pd.DataFrame(adata.obsm[key], index=adata.obs_names, columns=adata.uns['X_ahba_names'])
    proj['obs_names'] = proj.index
    proj['correction'] = label

    melted = proj.melt(id_vars=['obs_names', 'correction'], var_name='C', value_name='value')
    if key == 'X_ahba':
        all_melted = melted
    else:
        all_melted = pd.concat([all_melted, melted], ignore_index=True)

# Merge with metadata
cols_to_keep = ['donor_id', 'age_years', 'cell_class', 'cell_subclass', 'cell_type', 'source']
cols_to_keep = [c for c in cols_to_keep if c in adata.obs.columns]

meta = adata.obs[cols_to_keep].copy()
meta['obs_names'] = meta.index

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
<th data-quarto-table-cell-role="th">donor_id</th>
<th data-quarto-table-cell-role="th">age_years</th>
<th data-quarto-table-cell-role="th">cell_class</th>
<th data-quarto-table-cell-role="th">cell_subclass</th>
<th data-quarto-table-cell-role="th">cell_type</th>
<th data-quarto-table-cell-role="th">source</th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">89128</td>
<td>Ramos_34C_AACACACCATAATCCG-1-VELMESHEV</td>
<td>scvi</td>
<td>C3+</td>
<td>106217.160085</td>
<td>34</td>
<td>-0.379452</td>
<td>Excitatory</td>
<td>Inhibitory</td>
<td>Interneurons</td>
<td>VELMESHEV</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">89129</td>
<td>Ramos_64G_CAACGGCAGACATAGT-1-VELMESHEV</td>
<td>scvi</td>
<td>C3+</td>
<td>96796.756920</td>
<td>64</td>
<td>-0.312329</td>
<td>Excitatory</td>
<td>Inhibitory</td>
<td>Interneurons</td>
<td>VELMESHEV</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">89130</td>
<td>U01_TAAGTGCTCGGAGGTA-1_4369_BA9-VELMESHEV</td>
<td>scvi</td>
<td>C3+</td>
<td>108140.620028</td>
<td>4369</td>
<td>2.720548</td>
<td>Excitatory</td>
<td>EN_L5_IT</td>
<td>L5</td>
<td>VELMESHEV</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">89131</td>
<td>Ramos_23C_CTCAAGACAGATAAAC-1-VELMESHEV</td>
<td>scvi</td>
<td>C3+</td>
<td>90613.795632</td>
<td>23</td>
<td>-0.312329</td>
<td>Inhibitory</td>
<td>Inhibitory</td>
<td>INT</td>
<td>VELMESHEV</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">89132</td>
<td>U01_CAAGAAATCCAATGGT-1_GW16-2-2-20_PFC-VELMESHEV</td>
<td>scvi</td>
<td>C3+</td>
<td>79503.562423</td>
<td>GW16-2-2-20</td>
<td>-0.465753</td>
<td>Excitatory</td>
<td>Progenitors</td>
<td>Progenitors</td>
<td>VELMESHEV</td>
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

df <- final_df %>%
  rename(Individual = donor_id) %>%
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

# --- scVI-corrected plots ---
df_scvi <- df %>% filter(correction == 'scvi')

color_col <- "source"
p_age_scvi <- df_scvi %>% plot_age(color_var = color_col) + ggtitle("scVI batch-corrected")

comparisons <- list(
    c('Adolescence', 'Adulthood'),
    c('Adolescence', 'Childhood')
)
df_box_scvi <- df_scvi %>% rename(network = C)
p_boxes_scvi <- df_box_scvi %>% plot_boxes(color_var='source') + stat_compare_means(comparisons = comparisons, color='blue', label='p.signif')

# --- scVI + OLS plots ---
df_ols <- df %>% filter(correction == 'scvi+ols')

p_age_ols <- df_ols %>% plot_age(color_var = color_col) + ggtitle("scVI + OLS score correction")

df_box_ols <- df_ols %>% rename(network = C)
p_boxes_ols <- df_box_ols %>% plot_boxes(color_var='source') + stat_compare_means(comparisons = comparisons, color='blue', label='p.signif')

# Combine
p_final <- (p_age_scvi | p_boxes_scvi) / (p_age_ols | p_boxes_ols) +
    plot_annotation(tag_levels='a', title="AHBA C3: scVI expression vs scVI+OLS score correction")

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

![](ahbaC3_projection_scVI_files/figure-markdown_strict/cell-12-output-5.png)

``` python
%%R -w 220 -h 115 -u mm -r 300

# Same data, colour by cell subclass instead of source
p_age_scvi_sub <- df_scvi %>% plot_age(color_var = 'cell_subclass') + ggtitle("scVI batch-corrected")
p_age_ols_sub  <- df_ols  %>% plot_age(color_var = 'cell_subclass') + ggtitle("scVI + OLS score correction")

(p_age_scvi_sub | p_age_ols_sub) +
    plot_annotation(title="AHBA C3: coloured by cell subclass")
```

    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'
    `geom_smooth()` using method = 'gam' and formula = 'y ~ s(x, bs = "cs")'

![](ahbaC3_projection_scVI_files/figure-markdown_strict/cell-13-output-2.png)
