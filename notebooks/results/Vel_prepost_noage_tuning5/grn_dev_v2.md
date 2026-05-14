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

    Loading params from: /home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/results/Vel_prepost_noage_tuning5/grn_dev_v2_params.yaml
    EXPERIMENT_NAME  : Vel_prepost_noage_tuning5
    PSEUDOBULK_FILE  : /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class.h5ad
    Filter           : cell_class_original == 'Excitatory'
    N_VALUES         : [4000]
    CHILD_STARTS     : [1, 2, 3, 4, 5]
    GAP_STARTS       : [10, 11, 12, 13, 14]
    GAP_LENGTHS      : [0, 1, 2, 3, 4]
    ADOL_ENDS        : [18, 20, 22, 24, 26]

### 1.3 Libraries

## 2. Data & GRN Projection

### 2.1 Load Pseudobulk Data

    Loading: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class.h5ad
    Full shape: (428, 17663)
    Cell classes: {'Excitatory': 76, 'Inhibitory': 75, 'OPC': 73, 'Microglia': 67, 'Astrocytes': 67, 'Glia': 36, 'Oligos': 34}

    Subset shape : (76, 17663)
    Donors       : 76
    Age range    : -0.47 – 43.99 years
    Sources      : {'VELMESHEV': 76}

### 2.2 GRN Setup

    Input sequence provided is already in string format. No operation performed
    Input sequence provided is already in string format. No operation performed

    Mapped 6641/7973 symbols via adata.var
    Querying mygene for 1332 unmapped symbols...

    134 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    338 input query terms found no hit: ['AAED1', 'AARS', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZIN1-AS1'

    After mygene: 6650/7973 mapped, 1323 dropped
    GRN genes in adata: 6650 / 6650

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

    CPM: 17663 genes, 76 donors.
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
    cpm_all : 17663 genes, 6650 GRN genes
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 2059 matched genes for projection...
    Computing sparse-dense dot product...
    cpm_4000 : 4000 HVGs, 2059 GRN genes
    scanvi_normalized: 17663 genes, 76 donors.
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
    scanvi_all  : 17663 genes, 6650 GRN genes
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 2359 matched genes for projection...
    Computing sparse-dense dot product...
    scanvi_4000 : 4000 HVGs, 2359 GRN genes

    0

### 2.4 Build Scores DataFrame

    final_df: 608 rows × 10 cols
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
    Significant (p<0.05): 274 / 2500
    Valid cohens_d: 2380 / 2500  (NA rate: 4.8%)

    Best HVG condition : scanvi_all
    Best age range     : child [1, 14)  gap [14, 17)  adol [17, 20)
      p = 0.0245  d = 1.09

### 3.2 Cohen’s d

![](grn_dev_v2_files/figure-markdown_strict/cell-12-output-1.png)

### 3.3 P-value

    R[write to console]: In addition: 
    R[write to console]: Warning message:

    R[write to console]: Removed 15 rows containing missing values or values outside the scale range
    (`geom_text()`). 

![](grn_dev_v2_files/figure-markdown_strict/cell-13-output-2.png)

### 3.4 Minimum Detectable Effect Size

![](grn_dev_v2_files/figure-markdown_strict/cell-14-output-1.png)

## 4. Developmental Trends (Best Age Range)

All subsequent sections use the scANVI sensitivity-selected age
boundaries for scANVI panels, and fixed boundaries (childhood 1–9,
adolescence 9–25) for CPM panels.

    scANVI best range: Childhood = [1, 14)  Gap = [14, 17)  Adolescence = [17, 20)
    CPM fixed range:  Childhood = [1, 9)  Adolescence = [9, 25)

### 4.1 Pseudobulk Trajectories + Stage Boxes

    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](grn_dev_v2_files/figure-markdown_strict/cell-17-output-2.png)

### 4.2 Unweighted Sum of Top-1000 C3+ Genes

    Top-1000 C3+ (all genes): 1000
    Top-1000 C3+ ∩ HVG4000: 429

    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'
    `geom_smooth()` using method = 'loess' and formula = 'y ~ x'

![](grn_dev_v2_files/figure-markdown_strict/cell-19-output-2.png)

### 4.3 V2 vs V3 C3+ sanity check (CPM, age 5–25)

    === V2 vs V3 CPM C3+ in age 5–25 (pseudobulk donors) ===
      Cell class filter in this notebook: CELL_CLASS_COL='cell_class_original' CELL_CLASS_VALUE='Excitatory'

      V2: n_donors=11
        C3+ CPM — mean=108034.2  median=109133.7
        age     — mean=14.9  range [6.0, 22.0]

      V3: n_donors=10
        C3+ CPM — mean=116951.1  median=115788.9
        age     — mean=15.0  range [6.5, 25.0]

      V2/V3 mean ratio = 0.9238  (V2 < V3)
