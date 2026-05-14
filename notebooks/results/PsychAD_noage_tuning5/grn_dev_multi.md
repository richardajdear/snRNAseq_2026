# AHBA C3 Developmental Trends — GRN Projection (Multi-Pseudobulk)


## 1. Setup

### 1.1 Environment

    Environment : hpc
      rds_dir  : /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc
      code_dir : /home/rajd2/rds/hpc-work/snRNAseq_2026/code
      ref_dir  : /home/rajd2/rds/hpc-work/snRNAseq_2026/reference

### 1.2 Parameters

Parameters are loaded from a YAML config file whose path is passed via
the `NOTEBOOK_PARAMS` environment variable (set automatically by
`render_single.sh` or `step5_notebook.sh`).

When `PSEUDOBULK_INPUTS` is present, the analysis runs for each listed
input and results are grouped by plot type below. When only
`PSEUDOBULK_FILE` is present (legacy / single-run mode), it is wrapped
into a one-entry list automatically.

Defaults below are used when no config is provided (interactive mode).

    Loading params from: /home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/results/PsychAD_noage_tuning5/grn_dev_multi_params.yaml

    EXPERIMENT_NAME : PsychAD_noage_tuning5
    N inputs        : 2
      [by_cell_class_Excitatory]  filter=cell_class == 'Excitatory'
        /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/by_cell_class.h5ad
      [all_cells_by_donor]  filter=none
        /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/all_cells_by_donor.h5ad
    N_VALUES        : [4000]
    CHILD_STARTS    : [1, 2, 3, 4, 5]
    GAP_STARTS      : [8, 9, 10, 11, 12, 13, 14]
    GAP_LENGTHS     : [0, 1, 2, 3, 4]
    ADOL_ENDS       : [18, 20, 22, 24, 26]

### 1.3 Libraries

## 2. Data & GRN Projection

### 2.1 GRN Setup (shared across all inputs)

    Input sequence provided is already in string format. No operation performed
    Input sequence provided is already in string format. No operation performed

    No gene_symbol/feature_name column found — resolving all symbols via mygene
    Querying mygene for 7973 unmapped symbols...

    152 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    389 input query terms found no hit: ['AAED1', 'AARS', 'ADAL', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZ

    After mygene: 6955/7973 mapped, 1018 dropped
    GRN genes in adata: 6955 / 6955

    0

### 2.2 Load & Project All Inputs


    ============================================================
    Input: by_cell_class_Excitatory
      File: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/by_cell_class.h5ad
      Full shape: (1265, 34176)
      Cell classes: {'OPC': 191, 'Inhibitory': 190, 'Excitatory': 187, 'Astrocytes': 181, 'Microglia': 176, 'Oligos': 176, 'Endothelial': 164}
      Subset shape : (187, 34176)
      Donors       : 187
      Age range    : 0.08 – 89.00 years
      Sources      : {'PSYCHAD': 187}
      CPM: 34176 genes, 187 donors.
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 1077 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 1891 matched genes for projection...
    Computing sparse-dense dot product...
      final_df: 1496 rows × 10 cols

    ============================================================
    Input: all_cells_by_donor
      File: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/all_cells_by_donor.h5ad
      Full shape: (201, 34176)
      Subset shape : (201, 34176)
      Donors       : 201
      Age range    : 0.08 – 89.00 years
      Sources      : {'PSYCHAD': 201}
      CPM: 34176 genes, 201 donors.
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 1151 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 1953 matched genes for projection...
    Computing sparse-dense dot product...
      final_df: 1608 rows × 10 cols

    Combined: 3104 rows across 2 inputs

## 3. Age Range Sensitivity (4D Grid)

We independently vary four age-range parameters across all pseudobulk
inputs.

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

### 3.1 Compute Sensitivity Grids


    [by_cell_class_Excitatory]
      Grid: 875 combos | significant: 117
      Best: scanvi_all  p=0.0267  d=0.92
      Age: child[5,14) gap[14,14) adol[14,18)

    [all_cells_by_donor]
      Grid: 875 combos | significant: 71
      Best: scanvi_all  p=0.1197  d=0.55
      Age: child[1,14) gap[14,14) adol[14,18)

### 3.3 P-value (blue = negative Cohen’s d, red = positive)

#### Input 1: by_cell_class_Excitatory

    In addition: Warning message:
    Removed 5 rows containing missing values or values outside the scale range
    (`geom_text()`). 

![](grn_dev_multi_files/figure-markdown_strict/cell-11-output-2.png)

#### Input 2: all_cells_by_donor

    In addition: Warning message:
    Removed 5 rows containing missing values or values outside the scale range
    (`geom_text()`). 

![](grn_dev_multi_files/figure-markdown_strict/cell-13-output-2.png)

## 4. Developmental Trends (Best Age Range per Input)

All plots use scANVI sensitivity-selected age boundaries for scANVI
panels and fixed boundaries (childhood 1–9, adolescence 9–25) for CPM
panels.

### 4.1 Pseudobulk Trajectories + Stage Boxes

#### Input 1: by_cell_class_Excitatory

![](grn_dev_multi_files/figure-markdown_strict/cell-20-output-1.png)

#### Input 2: all_cells_by_donor

![](grn_dev_multi_files/figure-markdown_strict/cell-22-output-1.png)

### 4.2 Unweighted Sum of Top-1000 C3+ Genes

    [by_cell_class_Excitatory] Top-1000 C3+ (all): 1000, ∩HVG4000: 184
    [all_cells_by_donor] Top-1000 C3+ (all): 1000, ∩HVG4000: 115

#### Input 1: by_cell_class_Excitatory

![](grn_dev_multi_files/figure-markdown_strict/cell-30-output-1.png)

#### Input 2: all_cells_by_donor

![](grn_dev_multi_files/figure-markdown_strict/cell-32-output-1.png)

### 4.3 V2 vs V3 C3+ sanity check (CPM, age 5–25)


    ============================================================
    === by_cell_class_Excitatory ===
    ============================================================
      Cell class filter: cell_class_col='cell_class'  cell_class_value='Excitatory'
      CPM C3+ in age 5–25: n_donors=62

      V3: n_donors=62
        C3+ CPM — mean=102915.1  median=102299.7
        age     — mean=17.8  range [5.0, 25.0]

    ============================================================
    === all_cells_by_donor ===
    ============================================================
      Cell class filter: cell_class_col=''  cell_class_value=''
      CPM C3+ in age 5–25: n_donors=67

      V3: n_donors=67
        C3+ CPM — mean=79833.2  median=83158.4
        age     — mean=17.7  range [5.0, 25.0]

## 5. Composition & per-cell-class diagnostics

Diagnostic outputs to investigate **why** Excitatory-only vs all-cells
pseudobulks give different developmental signals. Runs only when at
least one `PSEUDOBULK_INPUT` has `cell_class_col` set (i.e. is a
by-cell-class file).

### 5.1 Donor-level cell-class composition

    by_cell_class reference: by_cell_class_Excitatory
      Path: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/by_cell_class.h5ad
      Class column: cell_class
      Loaded shape: (1265, 34176)
      Classes present: ['Astrocytes', 'Endothelial', 'Excitatory', 'Inhibitory', 'Microglia', 'OPC', 'Oligos']

    N donors with any cell-class pseudobulk: 201

    Mean cell-class composition across donors:
                 mean_frac  sd_frac  median_cells_per_donor  n_donors_with_class
    cell_class                                                                  
    Oligos           0.270    0.230                     209                  176
    Excitatory       0.206    0.138                     184                  187
    Inhibitory       0.190    0.124                     168                  190
    Astrocytes       0.133    0.142                      79                  181
    OPC              0.098    0.077                      79                  191
    Microglia        0.060    0.053                      48                  176
    Endothelial      0.044    0.051                      32                  164

### 5.2 Cell-class fraction vs age

    Spearman correlation of cell-class fraction vs age (n=201 donors):
      Excitatory    rho=+0.155  p=0.0281
      Inhibitory    rho=-0.238  p=0.000708
      Astrocytes    rho=+0.195  p=0.00555
      Microglia     rho=-0.439  p=7.54e-11
      OPC           rho=-0.413  p=1.21e-09
      Oligos        rho=+0.340  p=8.5e-07

    frac_long_df: (1206, 5)

![](grn_dev_multi_files/figure-markdown_strict/cell-40-output-1.png)

### 5.3 Per-cell-class C3+ developmental trend

For each major cell class, run the same CPM + scANVI projection and the
same 4D sensitivity grid used in Section 3. Output: one row per class
showing best scANVI p-value, Cohen’s d, and how many age-window combos
pass p\<0.05.

    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
      Excitatory: n_donors=187
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
      Inhibitory: n_donors=190
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
      Astrocytes: n_donors=181
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
      Microglia: n_donors=176
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
      OPC: n_donors=191
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6955 matching genes in var_names.
    Aligning GRN weights to 6955 matched genes for projection...
    Computing sparse-dense dot product...
      Oligos: n_donors=176

    Combined per-class df: (4404, 9), classes: ['Astrocytes', 'Excitatory', 'Inhibitory', 'Microglia', 'OPC', 'Oligos']


    Per-cell-class C3+ best scANVI sensitivity:
     cell_class n_donors     best_p   cohens_d n_sig_combos n_total_combos
     Excitatory      187 0.02669277  0.9238366           37           1750
     Inhibitory      190 0.04068940  0.9194390            8           1750
     Astrocytes      181 0.05079365 -1.5002028            0           1750
      Microglia      176 0.00284495 -1.4616123          149           1750
            OPC      191 0.02220167 -1.8282489           67           1750
         Oligos      176 0.03172080  0.8220859           12           1750
     child_start gap_start adol_start adol_end p_label
               5        14         14       18       *
               1        13         15       18       *
               5         8          9       20      ns
               5         8          8       20      **
               5         8          8       22       *
               1        13         16       22       *

![](grn_dev_multi_files/figure-markdown_strict/cell-43-output-1.png)

![](grn_dev_multi_files/figure-markdown_strict/cell-44-output-1.png)

### 5.4 Donor inclusion / cell-count comparison

    Excitatory-filter input: by_cell_class_Excitatory (file=/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/by_cell_class.h5ad)
    All-cells input:         all_cells_by_donor (file=/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/all_cells_by_donor.h5ad)

    Donor sets:
      Excitatory (Excitatory): n=187
      All-cells              : n=201
      In both                : n=187
      Only in Excitatory     : n=0
      Only in all-cells      : n=14

    Donors only in all-cells (not in Excitatory):
      ages — min=0.50  median=24.50  max=64.00
      Donor_202  age=0.50  chem=V3
      Donor_1400  age=3.00  chem=V3
      Donor_1472  age=8.00  chem=V3
      Donor_1454  age=14.00  chem=V3
      Donor_183  age=16.00  chem=V3
      Donor_460  age=21.00  chem=V3
      Donor_751  age=23.00  chem=V3
      Donor_1375  age=26.00  chem=V3
      Donor_1096  age=32.00  chem=V3
      Donor_754  age=36.00  chem=V3
      Donor_161  age=39.00  chem=V3
      Donor_1435  age=44.00  chem=V3
      Donor_812  age=62.00  chem=V3
      Donor_1351  age=64.00  chem=V3

### 5.5 V2/V3 chemistry sensitivity (Velmeshev only)

If the dataset mixes V2 and V3 chemistries, re-run the 4D sensitivity
grid on the V3-only subset of `all_cells_by_donor` to test whether the
apparent all-cells signal depends on the chemistry mix.

    Skipping 5.5: chemistry column absent or only one chemistry present.

    Section 5.5 skipped (no chemistry variation).
