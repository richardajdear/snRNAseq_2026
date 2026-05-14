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

    Loading params from: /home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/results/prepost_tuning5_allVel/grn_dev_multi_params.yaml

    EXPERIMENT_NAME : prepost_tuning5_allVel
    N inputs        : 2
      [by_cell_class_Excitatory]  filter=cell_class == 'Excitatory'
        /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel/pseudobulk_output/by_cell_class.h5ad
      [all_cells_by_donor]  filter=none
        /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel/pseudobulk_output/all_cells_by_donor.h5ad
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

    Mapped 122/7973 symbols via adata.var
    Querying mygene for 7851 unmapped symbols...

    136 input query terms found dup hits:   [('ACTG1P4', 2), ('ADAM20P1', 2), ('AKR7A2P1', 3), ('AMZ2P1', 2), ('ANKRD18CP', 2), ('ANKRD19P', 2),
    376 input query terms found no hit: ['AAED1', 'AARS', 'ADAL', 'ADPRHL2', 'ADSSL1', 'ALS2CR12', 'APOPT1', 'ARMT1', 'ARNTL', 'ARNTL2', 'AZ

    After mygene: 6348/7973 mapped, 1625 dropped
    GRN genes in adata: 6348 / 6348

    0

### 2.2 Load & Project All Inputs


    ============================================================
    Input: by_cell_class_Excitatory
      File: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel/pseudobulk_output/by_cell_class.h5ad
      Full shape: (1101, 15540)
      Cell classes: {'Inhibitory': 228, 'Excitatory': 224, 'Oligos': 175, 'Astrocytes': 156, 'OPC': 151, 'Microglia': 80, 'Other': 39, 'Glia': 25, 'Endothelial': 23}
      Subset shape : (224, 15540)
      Donors       : 224
      Age range    : -0.47 – 89.00 years
      Sources      : {'PSYCHAD': 133, 'VELMESHEV': 74, 'WANG': 17}
      CPM: 15540 genes, 224 donors.
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 1988 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 2391 matched genes for projection...
    Computing sparse-dense dot product...
      final_df: 1792 rows × 10 cols

    ============================================================
    Input: all_cells_by_donor
      File: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel/pseudobulk_output/all_cells_by_donor.h5ad
      Full shape: (287, 15540)
      Subset shape : (287, 15540)
      Donors       : 287
      Age range    : -0.47 – 89.00 years
      Sources      : {'PSYCHAD': 195, 'VELMESHEV': 75, 'WANG': 17}
      CPM: 15540 genes, 287 donors.
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 1924 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 2352 matched genes for projection...
    Computing sparse-dense dot product...
      final_df: 2296 rows × 10 cols

    Combined: 4088 rows across 2 inputs

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
      Grid: 875 combos | significant: 71
      Best: scanvi_all  p=0.0064  d=-0.74
      Age: child[1,12) gap[12,16) adol[16,26)

    [all_cells_by_donor]
      Grid: 875 combos | significant: 499
      Best: scanvi_all  p=0.0012  d=0.85
      Age: child[1,8) gap[8,8) adol[8,20)

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

    [by_cell_class_Excitatory] Top-1000 C3+ (all): 1000, ∩HVG4000: 395
    [all_cells_by_donor] Top-1000 C3+ (all): 1000, ∩HVG4000: 309

#### Input 1: by_cell_class_Excitatory

![](grn_dev_multi_files/figure-markdown_strict/cell-30-output-1.png)

#### Input 2: all_cells_by_donor

![](grn_dev_multi_files/figure-markdown_strict/cell-32-output-1.png)

### 4.3 V2 vs V3 C3+ sanity check (CPM, age 5–25)


    ============================================================
    === by_cell_class_Excitatory ===
    ============================================================
      Cell class filter: cell_class_col='cell_class'  cell_class_value='Excitatory'
      CPM C3+ in age 5–25: n_donors=70

      V2: n_donors=11
        C3+ CPM — mean=108328.7  median=109502.4
        age     — mean=14.9  range [6.0, 22.0]

      V3: n_donors=55
        C3+ CPM — mean=119192.3  median=118899.4
        age     — mean=17.3  range [6.5, 25.0]

      V2/V3 mean ratio = 0.9089  (V2 < V3)

    ============================================================
    === all_cells_by_donor ===
    ============================================================
      Cell class filter: cell_class_col=''  cell_class_value=''
      CPM C3+ in age 5–25: n_donors=92

      V2: n_donors=11
        C3+ CPM — mean=97366.3  median=94738.6
        age     — mean=14.9  range [6.0, 22.0]

      V3: n_donors=77
        C3+ CPM — mean=89404.9  median=92625.3
        age     — mean=17.3  range [5.0, 25.0]

      V2/V3 mean ratio = 1.0890  (V2 > V3)

## 5. Composition & per-cell-class diagnostics

Diagnostic outputs to investigate **why** Excitatory-only vs all-cells
pseudobulks give different developmental signals. Runs only when at
least one `PSEUDOBULK_INPUT` has `cell_class_col` set (i.e. is a
by-cell-class file).

### 5.1 Donor-level cell-class composition

    by_cell_class reference: by_cell_class_Excitatory
      Path: /home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel/pseudobulk_output/by_cell_class.h5ad
      Class column: cell_class
      Loaded shape: (1101, 15540)
      Classes present: ['Astrocytes', 'Endothelial', 'Excitatory', 'Glia', 'Inhibitory', 'Microglia', 'OPC', 'Oligos', 'Other']

    N donors with any cell-class pseudobulk: 293

    Mean cell-class composition across donors:
                 mean_frac  sd_frac  median_cells_per_donor  n_donors_with_class
    cell_class                                                                  
    Excitatory       0.305    0.234                     298                  224
    Oligos           0.273    0.320                     156                  175
    Inhibitory       0.202    0.154                     247                  228
    Astrocytes       0.105    0.159                     115                  156
    OPC              0.063    0.097                     101                  151
    Microglia        0.024    0.049                       0                   80
    Glia             0.010    0.046                       0                   25
    Other            0.010    0.049                       0                   39
    Endothelial      0.009    0.036                       0                   23

    Mean fraction × chemistry:
      V2 (n=35): Astrocytes=0.10, Endothelial=0.00, Excitatory=0.50, Glia=0.01, Inhibitory=0.19, Microglia=0.03, OPC=0.06, Oligos=0.08, Other=0.04
      V3 (n=229): Astrocytes=0.11, Endothelial=0.01, Excitatory=0.27, Glia=0.01, Inhibitory=0.20, Microglia=0.02, OPC=0.06, Oligos=0.32, Other=0.00
      multiome (n=17): Astrocytes=0.07, Endothelial=0.01, Excitatory=0.45, Glia=0.03, Inhibitory=0.22, Microglia=0.04, OPC=0.07, Oligos=0.10, Other=0.02

### 5.2 Cell-class fraction vs age

    Spearman correlation of cell-class fraction vs age (n=293 donors):
      Excitatory    rho=-0.254  p=1.68e-05
      Inhibitory    rho=-0.214  p=0.000307
      Astrocytes    rho=+0.011  p=0.848
      Microglia     rho=-0.276  p=2.71e-06
      OPC           rho=-0.356  p=8.43e-10
      Oligos        rho=+0.530  p=9.7e-22

    Spearman by chemistry:
      Excitatory   [V2, n=35]  rho=-0.597  p=0.000155
      Excitatory   [V3, n=229]  rho=-0.046  p=0.489
      Excitatory   [multiome, n=17]  rho=-0.801  p=0.00011
      Inhibitory   [V2, n=35]  rho=+0.180  p=0.3
      Inhibitory   [V3, n=229]  rho=-0.245  p=0.000179
      Inhibitory   [multiome, n=17]  rho=-0.400  p=0.112
      Astrocytes   [V2, n=35]  rho=+0.364  p=0.0318
      Astrocytes   [V3, n=229]  rho=-0.007  p=0.914
      Astrocytes   [multiome, n=17]  rho=+0.150  p=0.566
      Microglia    [V2, n=35]  rho=+0.485  p=0.00313
      Microglia    [V3, n=229]  rho=-0.345  p=8.26e-08
      Microglia    [multiome, n=17]  rho=+0.884  p=2.54e-06
      OPC          [V2, n=35]  rho=+0.679  p=7.4e-06
      OPC          [V3, n=229]  rho=-0.457  p=3.25e-13
      OPC          [multiome, n=17]  rho=+0.843  p=2.17e-05
      Oligos       [V2, n=35]  rho=+0.764  p=9.03e-08
      Oligos       [V3, n=229]  rho=+0.423  p=2.22e-11
      Oligos       [multiome, n=17]  rho=+0.835  p=3.1e-05

    frac_long_df: (1758, 5)

![](grn_dev_multi_files/figure-markdown_strict/cell-40-output-1.png)

### 5.3 Per-cell-class C3+ developmental trend

For each major cell class, run the same CPM + scANVI projection and the
same 4D sensitivity grid used in Section 3. Output: one row per class
showing best scANVI p-value, Cohen’s d, and how many age-window combos
pass p\<0.05.

    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
      Excitatory: n_donors=224
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
      Inhibitory: n_donors=228
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
      Astrocytes: n_donors=156
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
      Microglia: n_donors=80
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
      OPC: n_donors=151
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
    Found 6348 matching genes in var_names.
    Aligning GRN weights to 6348 matched genes for projection...
    Computing sparse-dense dot product...
      Oligos: n_donors=175

    Combined per-class df: (4056, 9), classes: ['Astrocytes', 'Excitatory', 'Inhibitory', 'Microglia', 'OPC', 'Oligos']


    Per-cell-class C3+ best scANVI sensitivity:
     cell_class n_donors       best_p   cohens_d n_sig_combos n_total_combos
     Excitatory      224 6.400565e-03 -0.7367101           48           1750
     Inhibitory      228 2.416299e-02  0.6323759          101           1750
     Astrocytes      156 2.569194e-02 -0.9537636            5           1750
      Microglia       80 8.946377e-03  1.0379542          131           1750
            OPC      151 7.847418e-02  0.4772065           79           1750
         Oligos      175 3.306508e-05  1.8580949          758           1750
     child_start gap_start adol_start adol_end p_label
               1        12         16       26      **
               2        11         11       22       *
               5        14         15       18       *
               2        12         16       26      **
               1        11         11       22      ns
               1         9         12       22     ***

![](grn_dev_multi_files/figure-markdown_strict/cell-43-output-1.png)

![](grn_dev_multi_files/figure-markdown_strict/cell-44-output-1.png)

### 5.4 Donor inclusion / cell-count comparison

    Excitatory-filter input: by_cell_class_Excitatory (file=/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel/pseudobulk_output/by_cell_class.h5ad)
    All-cells input:         all_cells_by_donor (file=/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_noage_tuning5_allVel/pseudobulk_output/all_cells_by_donor.h5ad)

    Donor sets:
      Excitatory (Excitatory): n=224
      All-cells              : n=287
      In both                : n=224
      Only in Excitatory     : n=0
      Only in all-cells      : n=63

    Donors only in all-cells (not in Excitatory):
      ages — min=-0.05  median=21.00  max=83.00
      779  age=-0.05  chem=V2
      Donor_122  age=0.08  chem=V3
      Donor_1208  age=0.17  chem=V3
      Donor_594  age=0.25  chem=V3
      Donor_316  age=0.25  chem=V3
      Donor_47  age=0.33  chem=V3
      Donor_1171  age=0.33  chem=V3
      Donor_1326  age=0.33  chem=V3
      Donor_648  age=0.33  chem=V3
      Donor_503  age=0.50  chem=V3
      Donor_202  age=0.50  chem=V3
      Donor_734  age=2.00  chem=V3
      Donor_1319  age=2.00  chem=V3
      Donor_701  age=2.00  chem=V3
      Donor_1400  age=3.00  chem=V3
      Donor_28  age=4.00  chem=V3
      Donor_1341  age=5.00  chem=V3
      Donor_83  age=6.00  chem=V3
      Donor_1472  age=8.00  chem=V3
      Donor_1342  age=13.00  chem=V3
      Donor_1454  age=14.00  chem=V3
      Donor_822  age=14.00  chem=V3
      Donor_571  age=15.00  chem=V3
      Donor_183  age=16.00  chem=V3
      Donor_248  age=16.00  chem=V3
      Donor_577  age=17.00  chem=V3
      Donor_240  age=18.00  chem=V3
      Donor_1159  age=19.00  chem=V3
      Donor_537  age=20.00  chem=V3
      Donor_833  age=20.00  chem=V3
      … (33 more)

### 5.5 V2/V3 chemistry sensitivity (Velmeshev only)

If the dataset mixes V2 and V3 chemistries, re-run the 4D sensitivity
grid on the V3-only subset of `all_cells_by_donor` to test whether the
apparent all-cells signal depends on the chemistry mix.

    Chemistry values in by_cell_class: ['V2', 'V3', 'multiome']

    v3_compare_df: (7336, 13)
      subset counts: {('all_cells', 'V3_only'): 234, ('all_cells', 'all_chemistries'): 287, ('excitatory', 'V3_only'): 172, ('excitatory', 'all_chemistries'): 224}


    V3-only vs all-chemistries best scANVI sensitivity:
        pb_kind          subset n_donors      best_p   cohens_d n_sig_combos
      all_cells all_chemistries      287 0.001185896  0.8512370          255
      all_cells         V3_only      234 0.076880743  0.5453257            0
     excitatory all_chemistries      224 0.006400565 -0.7367101           48
     excitatory         V3_only      172 0.012512156 -0.8108196           50
     n_total_combos child_start gap_start adol_start adol_end p_label
               1750           1         8          8       20      **
               1750           1        14         17       20      ns
               1750           1        12         16       26      **
               1750           1        12         16       26       *
