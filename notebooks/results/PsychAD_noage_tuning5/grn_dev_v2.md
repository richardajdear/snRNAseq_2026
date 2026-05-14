# AHBA C3 Developmental Trends — GRN Projection (Pseudobulk)


## 1. Setup

### 1.1 Environment

    Environment : hpc
      rds_dir  : /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc
      code_dir : /home/rajd2/rds/hpc-work/snRNAseq_2026/code
      ref_dir  : /home/rajd2/rds/hpc-work/snRNAseq_2026/reference

### 1.2 Parameters

Parameters are loaded from a YAML config file whose path is passed via
the `NOTEBOOK_PARAMS` environment variable (set automatically by
`render_single.sh`). Defaults below are used when no config is provided
(e.g. for interactive use).

    Loading params from: /home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/results/PsychAD_noage_tuning5/grn_dev_v2_params.yaml
    EXPERIMENT_NAME  : PsychAD_noage_tuning5
    PSEUDOBULK_FILE  : /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/by_cell_class.h5ad
    Filter           : cell_class == 'Excitatory'
    N_VALUES         : [4000]
    CHILD_STARTS     : [1, 2, 3, 4, 5]
    GAP_STARTS       : [10, 11, 12, 13, 14]
    GAP_LENGTHS      : [0, 1, 2, 3, 4]
    ADOL_ENDS        : [18, 20, 22, 24, 26]

### 1.3 Libraries

## 2. Data & GRN Projection

### 2.1 Load Pseudobulk Data

    Loading: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/by_cell_class.h5ad
    Full shape: (1265, 34176)
    Cell classes: {'OPC': 191, 'Inhibitory': 190, 'Excitatory': 187, 'Astrocytes': 181, 'Microglia': 176, 'Oligos': 176, 'Endothelial': 164}

    Subset shape : (187, 34176)
    Donors       : 187
    Age range    : 0.08 – 89.00 years
    Sources      : {'PSYCHAD': 187}

### 2.2 GRN Setup

    Input sequence provided is already in string format. No operation performed
    Input sequence provided is already in string format. No operation performed

    No gene_symbol/feature_name column found — resolving all symbols via mygene
    Querying mygene for 7973 unmapped symbols...

    152 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    389 input query terms found no hit: ['AAED1', 'AARS', 'ADAL', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZ

    After mygene: 6955/7973 mapped, 1018 dropped
    GRN genes in adata: 6955 / 6955

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

    CPM: 34176 genes, 187 donors.
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    cpm_all : 34176 genes, 6955 GRN genes
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 1077 matched genes for projection...
    Computing sparse-dense dot product...
    cpm_4000 : 4000 HVGs, 1077 GRN genes
    scanvi_normalized: 34176 genes, 187 donors.
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    scanvi_all  : 34176 genes, 6955 GRN genes
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 1891 matched genes for projection...
    Computing sparse-dense dot product...
    scanvi_4000 : 4000 HVGs, 1891 GRN genes

    0

### 2.4 Build Scores DataFrame

    final_df: 1496 rows × 10 cols
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

    Grid size: 625 combinations × 4 conditions
    Significant (p<0.05): 117 / 2500
    Valid cohens_d: 2480 / 2500  (NA rate: 0.8%)

    Best HVG condition : scanvi_all
    Best age range     : child [5, 14)  gap [14, 14)  adol [14, 18)
      p = 0.0267  d = 0.92

### 3.2 Cohen’s d

![](grn_dev_v2_files/figure-markdown_strict/cell-12-output-1.png)

### 3.3 P-value

    R[write to console]: In addition: 
    R[write to console]: Warning message:

    R[write to console]: Removed 5 rows containing missing values or values outside the scale range
    (`geom_text()`). 

![](grn_dev_v2_files/figure-markdown_strict/cell-13-output-2.png)

### 3.4 Minimum Detectable Effect Size

![](grn_dev_v2_files/figure-markdown_strict/cell-14-output-1.png)

## 4. Developmental Trends (Best Age Range)

All subsequent sections use the scANVI sensitivity-selected age
boundaries for scANVI panels, and fixed boundaries (childhood 1–9,
adolescence 9–25) for CPM panels.

    scANVI best range: Childhood = [5, 14)  Gap = [14, 14)  Adolescence = [14, 18)
    CPM fixed range:  Childhood = [1, 9)  Adolescence = [9, 25)

### 4.1 Pseudobulk Trajectories + Stage Boxes

    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](grn_dev_v2_files/figure-markdown_strict/cell-17-output-2.png)

### 4.2 Unweighted Sum of Top-1000 C3+ Genes

    Top-1000 C3+ (all genes): 1000
    Top-1000 C3+ ∩ HVG4000: 184

    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](grn_dev_v2_files/figure-markdown_strict/cell-19-output-2.png)

### 4.3 V2 vs V3 C3+ sanity check (CPM, age 5–25)

    === V2 vs V3 CPM C3+ in age 5–25 (pseudobulk donors) ===
      Cell class filter in this notebook: CELL_CLASS_COL='cell_class' CELL_CLASS_VALUE='Excitatory'

      V3: n_donors=62
        C3+ CPM — mean=102915.1  median=102299.7
        age     — mean=17.8  range [5.0, 25.0]
