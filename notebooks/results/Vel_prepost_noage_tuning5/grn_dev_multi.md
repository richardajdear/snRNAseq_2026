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

    Loading params from: /home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/results/Vel_prepost_noage_tuning5/grn_dev_multi_params.yaml

    EXPERIMENT_NAME : Vel_prepost_noage_tuning5
    N inputs        : 3
      [by_cell_class_L23]  filter=cell_class_original == 'Excitatory'
        /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class_L23.h5ad
      [by_cell_class_Excitatory]  filter=cell_class_original == 'Excitatory'
        /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class.h5ad
      [all_cells_by_donor]  filter=none
        /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/all_cells_by_donor.h5ad
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

    Mapped 6641/7973 symbols via adata.var
    Querying mygene for 1332 unmapped symbols...

    134 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    338 input query terms found no hit: ['AAED1', 'AARS', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZIN1-AS1'

    After mygene: 6650/7973 mapped, 1323 dropped
    GRN genes in adata: 6650 / 6650

    0

### 2.2 Load & Project All Inputs


    ============================================================
    Input: by_cell_class_L23
      File: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class_L23.h5ad
      Full shape: (75, 17663)
      Cell classes: {'Excitatory': 75, 'Astrocytes': 0, 'Glia': 0, 'Inhibitory': 0, 'Microglia': 0, 'OPC': 0, 'Oligos': 0}
      Subset shape : (75, 17663)
      Donors       : 75
      Age range    : -0.47 – 43.99 years
      Sources      : {'VELMESHEV': 75}
      CPM: 17663 genes, 75 donors.
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 2275 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 2341 matched genes for projection...
    Computing sparse-dense dot product...
      final_df: 600 rows × 10 cols

    ============================================================
    Input: by_cell_class_Excitatory
      File: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class.h5ad
      Full shape: (428, 17663)
      Cell classes: {'Excitatory': 76, 'Inhibitory': 75, 'OPC': 73, 'Microglia': 67, 'Astrocytes': 67, 'Glia': 36, 'Oligos': 34}
      Subset shape : (76, 17663)
      Donors       : 76
      Age range    : -0.47 – 43.99 years
      Sources      : {'VELMESHEV': 76}
      CPM: 17663 genes, 76 donors.
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 2059 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 2359 matched genes for projection...
    Computing sparse-dense dot product...
      final_df: 608 rows × 10 cols

    ============================================================
    Input: all_cells_by_donor
      File: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/all_cells_by_donor.h5ad
      Full shape: (76, 17663)
      Subset shape : (76, 17663)
      Donors       : 76
      Age range    : -0.47 – 43.99 years
      Sources      : {'VELMESHEV': 76}
      CPM: 17663 genes, 76 donors.
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 1983 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 2331 matched genes for projection...
    Computing sparse-dense dot product...
      final_df: 608 rows × 10 cols

    Combined: 1816 rows across 3 inputs

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


    [by_cell_class_L23]
      Grid: 875 combos | significant: 273
      Best: scanvi_all  p=0.0711  d=-0.91
      Age: child[1,14) gap[14,15) adol[15,18)

    [by_cell_class_Excitatory]
      Grid: 875 combos | significant: 416
      Best: scanvi_all  p=0.0245  d=1.09
      Age: child[1,14) gap[14,17) adol[17,20)

    [all_cells_by_donor]
      Grid: 875 combos | significant: 913
      Best: scanvi_all  p=0.0003  d=1.65
      Age: child[1,8) gap[8,8) adol[8,20)

### 3.3 P-value (blue = negative Cohen’s d, red = positive)

#### Input 1: by_cell_class_L23

    In addition: Warning message:
    Removed 15 rows containing missing values or values outside the scale range
    (`geom_text()`). 

![](grn_dev_multi_files/figure-markdown_strict/cell-11-output-2.png)

#### Input 2: by_cell_class_Excitatory

    In addition: Warning message:
    Removed 15 rows containing missing values or values outside the scale range
    (`geom_text()`). 

![](grn_dev_multi_files/figure-markdown_strict/cell-13-output-2.png)

#### Input 3: all_cells_by_donor

    In addition: Warning message:
    Removed 15 rows containing missing values or values outside the scale range
    (`geom_text()`). 

![](grn_dev_multi_files/figure-markdown_strict/cell-15-output-2.png)

## 4. Developmental Trends (Best Age Range per Input)

All plots use scANVI sensitivity-selected age boundaries for scANVI
panels and fixed boundaries (childhood 1–9, adolescence 9–25) for CPM
panels.

### 4.1 Pseudobulk Trajectories + Stage Boxes

#### Input 1: by_cell_class_L23

![](grn_dev_multi_files/figure-markdown_strict/cell-20-output-1.png)

#### Input 2: by_cell_class_Excitatory

![](grn_dev_multi_files/figure-markdown_strict/cell-22-output-1.png)

#### Input 3: all_cells_by_donor

![](grn_dev_multi_files/figure-markdown_strict/cell-24-output-1.png)

### 4.2 Unweighted Sum of Top-1000 C3+ Genes

    [by_cell_class_L23] Top-1000 C3+ (all): 1000, ∩HVG4000: 476
    [by_cell_class_Excitatory] Top-1000 C3+ (all): 1000, ∩HVG4000: 429
    [all_cells_by_donor] Top-1000 C3+ (all): 1000, ∩HVG4000: 391

#### Input 1: by_cell_class_L23

![](grn_dev_multi_files/figure-markdown_strict/cell-30-output-1.png)

#### Input 2: by_cell_class_Excitatory

![](grn_dev_multi_files/figure-markdown_strict/cell-32-output-1.png)

#### Input 3: all_cells_by_donor

![](grn_dev_multi_files/figure-markdown_strict/cell-34-output-1.png)

### 4.3 V2 vs V3 C3+ sanity check (CPM, age 5–25)


    ============================================================
    === by_cell_class_L23 ===
    ============================================================
      Cell class filter: cell_class_col='cell_class_original'  cell_class_value='Excitatory'
      CPM C3+ in age 5–25: n_donors=21

      V2: n_donors=11
        C3+ CPM — mean=110230.4  median=110565.1
        age     — mean=14.9  range [6.0, 22.0]

      V3: n_donors=10
        C3+ CPM — mean=121671.7  median=121337.1
        age     — mean=15.0  range [6.5, 25.0]

      V2/V3 mean ratio = 0.9060  (V2 < V3)

    ============================================================
    === by_cell_class_Excitatory ===
    ============================================================
      Cell class filter: cell_class_col='cell_class_original'  cell_class_value='Excitatory'
      CPM C3+ in age 5–25: n_donors=21

      V2: n_donors=11
        C3+ CPM — mean=108034.2  median=109133.7
        age     — mean=14.9  range [6.0, 22.0]

      V3: n_donors=10
        C3+ CPM — mean=116951.1  median=115788.9
        age     — mean=15.0  range [6.5, 25.0]

      V2/V3 mean ratio = 0.9238  (V2 < V3)

    ============================================================
    === all_cells_by_donor ===
    ============================================================
      Cell class filter: cell_class_col=''  cell_class_value=''
      CPM C3+ in age 5–25: n_donors=21

      V2: n_donors=11
        C3+ CPM — mean=97175.4  median=94572.2
        age     — mean=14.9  range [6.0, 22.0]

      V3: n_donors=10
        C3+ CPM — mean=88238.1  median=83600.3
        age     — mean=15.0  range [6.5, 25.0]

      V2/V3 mean ratio = 1.1013  (V2 > V3)

## 5. Composition & per-cell-class diagnostics

Diagnostic outputs to investigate **why** Excitatory-only vs all-cells
pseudobulks give different developmental signals. Runs only when at
least one `PSEUDOBULK_INPUT` has `cell_class_col` set (i.e. is a
by-cell-class file).

### 5.1 Donor-level cell-class composition

    by_cell_class reference: by_cell_class_L23
      Path: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class_L23.h5ad
      Class column: cell_class_original
      Loaded shape: (75, 17663)
      Classes present: ['Excitatory']

    N donors with any cell-class pseudobulk: 76

    Mean cell-class composition across donors:
                         mean_frac  sd_frac  median_cells_per_donor  n_donors_with_class
    cell_class_original                                                                 
    Excitatory                 1.0      0.0                     337                   75
    Astrocytes                 0.0      0.0                       0                    0
    Glia                       0.0      0.0                       0                    0
    Inhibitory                 0.0      0.0                       0                    0
    Microglia                  0.0      0.0                       0                    0
    OPC                        0.0      0.0                       0                    0
    Oligos                     0.0      0.0                       0                    0

    Mean fraction × chemistry:
      V2 (n=36): Astrocytes=0.00, Excitatory=1.00, Glia=0.00, Inhibitory=0.00, Microglia=0.00, OPC=0.00, Oligos=0.00
      V3 (n=39): Astrocytes=0.00, Excitatory=1.00, Glia=0.00, Inhibitory=0.00, Microglia=0.00, OPC=0.00, Oligos=0.00

### 5.2 Cell-class fraction vs age

    Spearman correlation of cell-class fraction vs age (n=76 donors):
      Excitatory    rho=+0.000  p=1
      Inhibitory    rho=+0.000  p=1
      Astrocytes    rho=+0.000  p=1
      Microglia     rho=+0.000  p=1
      OPC           rho=+0.000  p=1
      Oligos        rho=+0.000  p=1

    Spearman by chemistry:
      Excitatory   [V2, n=36]  rho=+nan  p=nan
      Excitatory   [V3, n=39]  rho=+nan  p=nan
      Inhibitory   [V2, n=36]  rho=+nan  p=nan
      Inhibitory   [V3, n=39]  rho=+nan  p=nan
      Astrocytes   [V2, n=36]  rho=+nan  p=nan
      Astrocytes   [V3, n=39]  rho=+nan  p=nan
      Microglia    [V2, n=36]  rho=+nan  p=nan
      Microglia    [V3, n=39]  rho=+nan  p=nan
      OPC          [V2, n=36]  rho=+nan  p=nan
      OPC          [V3, n=39]  rho=+nan  p=nan
      Oligos       [V2, n=36]  rho=+nan  p=nan
      Oligos       [V3, n=39]  rho=+nan  p=nan

    frac_long_df: (456, 5)

![](grn_dev_multi_files/figure-markdown_strict/cell-40-output-1.png)

### 5.3 Per-cell-class C3+ developmental trend

For each major cell class, run the same CPM + scANVI projection and the
same 4D sensitivity grid used in Section 3. Output: one row per class
showing best scANVI p-value, Cohen’s d, and how many age-window combos
pass p\<0.05.

    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6650 matching genes in var_names.
    Aligning GRN weights to 6650 matched genes for projection...
    Computing sparse-dense dot product...
      Excitatory: n_donors=75
      Skipping Inhibitory: not present in dataset
      Skipping Astrocytes: not present in dataset
      Skipping Microglia: not present in dataset
      Skipping OPC: not present in dataset
      Skipping Oligos: not present in dataset

    Combined per-class df: (300, 9), classes: ['Excitatory']


    Per-cell-class C3+ best scANVI sensitivity:
     cell_class n_donors     best_p   cohens_d n_sig_combos n_total_combos
     Excitatory       75 0.07114625 -0.9135074          161           1750
     child_start gap_start adol_start adol_end p_label
               1        14         15       18      ns

![](grn_dev_multi_files/figure-markdown_strict/cell-43-output-1.png)

![](grn_dev_multi_files/figure-markdown_strict/cell-44-output-1.png)

### 5.4 Donor inclusion / cell-count comparison

    Excitatory-filter input: by_cell_class_L23 (file=/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class_L23.h5ad)
    All-cells input:         all_cells_by_donor (file=/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/all_cells_by_donor.h5ad)

    Donor sets:
      Excitatory (Excitatory): n=75
      All-cells              : n=76
      In both                : n=75
      Only in Excitatory     : n=0
      Only in all-cells      : n=1

    Donors only in all-cells (not in Excitatory):
      ages — min=-0.05  median=-0.05  max=-0.05
      779  age=-0.05  chem=V2

### 5.5 V2/V3 chemistry sensitivity (Velmeshev only)

If the dataset mixes V2 and V3 chemistries, re-run the 4D sensitivity
grid on the V3-only subset of `all_cells_by_donor` to test whether the
apparent all-cells signal depends on the chemistry mix.

    Chemistry values in by_cell_class: ['V2', 'V3']

    v3_compare_df: (1832, 13)
      subset counts: {('all_cells', 'V3_only'): 39, ('all_cells', 'all_chemistries'): 76, ('excitatory', 'V3_only'): 39, ('excitatory', 'all_chemistries'): 75}


    V3-only vs all-chemistries best scANVI sensitivity:
        pb_kind          subset n_donors       best_p   cohens_d n_sig_combos
      all_cells all_chemistries       76 0.0002756877  1.6503381          531
      all_cells         V3_only       39 0.0293040293  1.4262794           27
     excitatory all_chemistries       75 0.0711462451 -0.9135074          161
     excitatory         V3_only       39 0.0714285714 -1.2706103            0
     n_total_combos child_start gap_start adol_start adol_end p_label
               1750           1         8          8       20     ***
               1750           1         8          8       20       *
               1750           1        14         15       18      ns
               1750           3        14         14       18      ns
