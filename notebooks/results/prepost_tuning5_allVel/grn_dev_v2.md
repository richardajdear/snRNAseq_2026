# AHBA C3 Developmental Trends — GRN Projection (Pseudobulk)


## 1. Setup

### 1.1 Environment

    Environment : local
      rds_dir  : /Users/richard/Git/snRNAseq_2026/rds-cam-psych-transc-Pb9UGUlrwWc
      code_dir : /Users/richard/Git/snRNAseq_2026/code
      ref_dir  : /Users/richard/Git/snRNAseq_2026/reference

### 1.2 Parameters

Parameters are loaded from a YAML config file whose path is passed via
the `NOTEBOOK_PARAMS` environment variable (set automatically by
`render_single.sh`). Defaults below are used when no config is provided
(e.g. for interactive use).

    Loading params from: /Users/richard/Git/snRNAseq_2026/notebooks/results/prepost_tuning5_allVel/grn_dev_v2_params.yaml
    EXPERIMENT_NAME  : prepost_tuning5_allVel
    PSEUDOBULK_FILE  : /Users/richard/Git/snRNAseq_2026/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel/pseudobulk_output/by_cell_class.h5ad
    Filter           : cell_class == 'Excitatory'
    N_VALUES         : [4000]
    CHILD_STARTS     : [1, 2, 3, 4, 5]
    GAP_STARTS       : [9, 10, 11, 12, 13, 14, 15]
    GAP_LENGTHS      : [0, 1, 2, 3]
    ADOL_ENDS        : [18, 20, 22, 24, 26]
    SENSITIVITY_CPM  : True

### 1.3 Libraries

## 2. Data & GRN Projection

### 2.1 Load Pseudobulk Data

    Loading: /Users/richard/Git/snRNAseq_2026/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel/pseudobulk_output/by_cell_class.h5ad
    Full shape: (1101, 15540)
    Cell classes: {'Inhibitory': np.int64(228), 'Excitatory': np.int64(224), 'Oligos': np.int64(175), 'Astrocytes': np.int64(156), 'OPC': np.int64(151), 'Microglia': np.int64(80), 'Other': np.int64(39), 'Glia': np.int64(25), 'Endothelial': np.int64(23)}

    Subset shape : (224, 15540)
    Donors       : 224
    Age range    : -0.47 – 89.00 years
    Sources      : {'PSYCHAD': np.int64(133), 'VELMESHEV': np.int64(74), 'WANG': np.int64(17)}

### 2.2 GRN Setup

    Input sequence provided is already in string format. No operation performed
    Input sequence provided is already in string format. No operation performed

    Mapped 122/7973 symbols via adata.var
    Querying mygene for 7851 unmapped symbols...

    136 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    376 input query terms found no hit: ['AAED1', 'AARS', 'ADAL', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZ

    After mygene: 6348/7973 mapped, 1625 dropped
    GRN genes in adata: 6348 / 6348

### 2.3 GRN Projection — CPM vs scANVI-Normalized

Two normalization methods × (all genes + top-N HVGs) = 10 conditions:

<table>
<thead>
<tr>
<th>condition</th>
<th>description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>cpm_all</code></td>
<td>CPM-normalized raw counts, all genes</td>
</tr>
<tr>
<td><code>cpm_{n}</code></td>
<td>CPM-normalized, top-<em>n</em> genes (by Pearson HVG method)</td>
</tr>
<tr>
<td><code>scanvi_all</code></td>
<td>scANVI batch-corrected expression, all genes</td>
</tr>
<tr>
<td><code>scanvi_{n}</code></td>
<td>scANVI batch-corrected, top-<em>n</em> genes by variance</td>
</tr>
</tbody>
</table>

    CPM: 15540 genes, 224 donors.
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    cpm_all : 15540 genes, 6348 GRN genes
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 1988 matched genes for projection...
    Computing sparse-dense dot product...
    cpm_4000 : 4000 HVGs, 1988 GRN genes
    scanvi_normalized: 15540 genes, 224 donors.
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    scanvi_all  : 15540 genes, 6348 GRN genes
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 2391 matched genes for projection...
    Computing sparse-dense dot product...
    scanvi_4000 : 4000 HVGs, 2391 GRN genes

    17536

### 2.4 Build Scores DataFrame

    final_df: 1792 rows × 10 cols
    obs_names             object
    condition             object
    C                     object
    value                float64
    individual          category
    age_years            float64
    source              category
    source_chemistry      object
    sex                 category
    dataset             category
    dtype: object

## 3. Age Range Sensitivity (4D Grid)

We independently vary four age-range parameters:

<table>
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr>
<th>Parameter</th>
<th>Values</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>child_start</code></td>
<td><code>CHILD_STARTS</code></td>
<td>Childhood lower bound</td>
</tr>
<tr>
<td><code>gap_start</code></td>
<td><code>GAP_STARTS</code></td>
<td>Childhood upper bound (= gap start)</td>
</tr>
<tr>
<td><code>gap_length</code></td>
<td><code>GAP_LENGTHS</code></td>
<td>Gap duration; <code>adol_start = gap_start + gap_length</code></td>
</tr>
<tr>
<td><code>adol_end</code></td>
<td><code>ADOL_ENDS</code></td>
<td>Adolescence upper bound</td>
</tr>
</tbody>
</table>

Childhood = `[child_start, gap_start)`, Adolescence =
`[gap_start + gap_length, adol_end)`.

### 3.1 Compute Sensitivity Grid

    scANVI grid: 700 combinations × 2 conditions
      Significant (p<0.05): 62 / 1400
      Valid cohens_d: 1370 / 1400  (NA rate: 2.1%)

    scANVI best HVG condition : scanvi_all
    scANVI best age range     : child [1, 13)  gap [13, 16)  adol [16, 26)
      p = 0.0070  d = -0.74

    CPM grid: 700 combinations × 2 conditions
      Significant (p<0.05): 1 / 1400
      Valid cohens_d: 1370 / 1400  (NA rate: 2.1%)

    CPM best HVG condition    : cpm_all
    CPM best age range        : child [3, 13)  gap [13, 16)  adol [16, 26)
      p = 0.0878  d = -0.39

### 3.2 Cohen’s d

![](grn_dev_v2_files/figure-markdown_strict/cell-12-output-1.png)

![](grn_dev_v2_files/figure-markdown_strict/cell-13-output-1.png)

### 3.3 P-value

![](grn_dev_v2_files/figure-markdown_strict/cell-14-output-1.png)

![](grn_dev_v2_files/figure-markdown_strict/cell-15-output-1.png)

### 3.4 Minimum Detectable Effect Size

![](grn_dev_v2_files/figure-markdown_strict/cell-16-output-1.png)

![](grn_dev_v2_files/figure-markdown_strict/cell-17-output-1.png)

## 4. Developmental Trends (Best Age Range)

All subsequent sections use the sensitivity-selected age boundaries for
each normalization method. When `SENSITIVITY_CPM` is disabled, CPM
panels fall back to fixed boundaries (childhood 1–9, adolescence 9–25).

    scANVI best range : Childhood = [1, 13)  Gap = [13, 16)  Adolescence = [16, 26)
    CPM best range    : Childhood = [3, 13)  Gap = [13, 16)  Adolescence = [16, 26)

### 4.1 Pseudobulk Trajectories + Stage Boxes

    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](grn_dev_v2_files/figure-markdown_strict/cell-20-output-2.png)

### 4.2 Unweighted Sum of Top-1000 C3+ Genes

    Top-1000 C3+ (all genes): 1000
    Top-1000 C3+ ∩ HVG4000: 395

    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](grn_dev_v2_files/figure-markdown_strict/cell-22-output-2.png)

### 4.3 V2 vs V3 C3+ sanity check (CPM, age 5–25)

    === V2 vs V3 CPM C3+ in age 5–25 (pseudobulk donors) ===
      Cell class filter in this notebook: CELL_CLASS_COL='cell_class' CELL_CLASS_VALUE='Excitatory'

      V2: n_donors=11
        C3+ CPM — mean=108328.7  median=109502.4
        age     — mean=14.9  range [6.0, 22.0]

      V3: n_donors=55
        C3+ CPM — mean=119192.3  median=118899.4
        age     — mean=17.3  range [6.5, 25.0]

      V2/V3 mean ratio = 0.9089  (V2 < V3)
