
import scanpy as sc
import argparse
import os
import sys
import numpy as np
import psutil
import time
import pandas as pd

from pipeline import read_data

START_TIME = time.time()

def log_mem(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    elapsed = time.time() - START_TIME
    print(f"\n[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB ({process.memory_percent():.2f}%) - Elapsed: {elapsed/60:.2f} min")
    sys.stdout.flush()


def _materialize_psychad(aging_backed, hbcc_backed, hbcc_unique_mask,
                          meta_df, mask, n_aging_cells):
    """Materialize a filtered PSYCHAD subset from two backed h5ad files.

    meta_df is laid out as [AGING rows (first n_aging_cells), HBCC-unique rows].
    mask is a boolean Series over that same index. This function splits the mask,
    loads the selected cells from each backed file, concatenates them, and applies
    the standardised metadata from meta_df.
    """
    aging_keep = mask.iloc[:n_aging_cells]
    hbcc_keep  = mask.iloc[n_aging_cells:]

    adatas = []

    if aging_keep.any():
        aging_idx = np.where(aging_keep.values)[0]
        print(f"  Loading {len(aging_idx):,} AGING cells into memory...")
        sub = aging_backed[aging_idx].to_memory()
        sub.X = read_data.get_raw_counts(sub)
        adatas.append(sub)

    if hbcc_keep.any():
        # Map from "positions among HBCC-unique cells" back to positions in full HBCC
        hbcc_unique_pos  = np.where(hbcc_unique_mask)[0]
        hbcc_final_pos   = hbcc_unique_pos[hbcc_keep.values]
        print(f"  Loading {len(hbcc_final_pos):,} HBCC-unique cells into memory...")
        sub = hbcc_backed[hbcc_final_pos].to_memory()
        sub.X = read_data.get_raw_counts(sub)
        adatas.append(sub)

    if not adatas:
        return None

    adata = sc.concat(adatas, join='inner') if len(adatas) > 1 else adatas[0]

    # Apply standardised metadata, aligned by barcode index
    sub_meta = meta_df[mask.values]
    for col in sub_meta.columns:
        adata.obs[col] = sub_meta.loc[adata.obs_names, col].values

    return adata


def main():
    parser = argparse.ArgumentParser(description="Filter and optionally downsample AnnData (fully backed mode).")

    # Input: single file for all dataset types except PsychAD; two files for PsychAD
    parser.add_argument("--input", default=None,
                        help="Path to input .h5ad file (all dataset types except PsychAD).")
    parser.add_argument("--inputs", nargs=2, default=None,
                        metavar='PATH',
                        help="Paths to Aging_Cohort.h5ad and HBCC_Cohort.h5ad (PsychAD only).")
    parser.add_argument("--output", required=True, help="Path to output .h5ad file")

    parser.add_argument("--dataset_type", choices=['Velmeshev', 'Wang', 'PsychAD', 'Generic'],
                        default='Generic', help="Dataset type to determine reading logic.")

    parser.add_argument("--cell_type_field", default=None,
                        help="Source obs/meta column to store as cell_type_raw. "
                             "Defaults: Velmeshev='Cell_Type', Wang='Type-updated', PsychAD='subclass'.")
    parser.add_argument("--pfc_only", action='store_true', help="Keep only 'prefrontal cortex' regions.")
    parser.add_argument("--age_downsample", action='store_true', help="Keep all donors <40; keep 20%% of donors >=40.")
    parser.add_argument("--postnatal_only", action='store_true',
                        help="Keep only postnatal cells (age_years >= 0). Applied before age_downsample.")
    parser.add_argument("--min_age", type=float, default=None,
                        help="Keep only cells from donors with age_years >= this value.")
    parser.add_argument("--cell_class_filter", nargs='+', default=None,
                        help="Keep only cells whose cell_class is in this list (e.g. Excitatory Glia).")
    parser.add_argument("--n_cells", type=int, default=None,
                        help="Target number of cells (random downsample). "
                             "Omit or set to null in config to use all cells.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    log_mem("Start")

    # =========================================================================
    # Step 1: Load data in BACKED mode (expression matrix stays on disk)
    # =========================================================================
    ctf = args.cell_type_field  # None → each reader uses its own default

    # PsychAD uses --inputs (two files); all other types use --input (one file)
    if args.dataset_type == 'PsychAD':
        if args.inputs is None:
            print("Error: --inputs AGING_PATH HBCC_PATH is required for PsychAD.")
            sys.exit(1)
        print(f"Loading PsychAD from:\n  AGING: {args.inputs[0]}\n  HBCC:  {args.inputs[1]}")
    else:
        if args.input is None:
            print(f"Error: --input is required for dataset_type={args.dataset_type}.")
            sys.exit(1)
        print(f"Loading {args.dataset_type} data from {args.input} (backed mode)...")

    adata_backed     = None
    meta_df          = None
    aging_backed     = None
    hbcc_backed      = None
    hbcc_unique_mask = None
    n_aging_cells    = 0

    if args.dataset_type == 'Velmeshev':
        kw = {'cell_type_field': ctf} if ctf else {}
        adata_backed, meta_df = read_data.read_velmeshev_backed(h5ad_path=args.input, **kw)
    elif args.dataset_type == 'Wang':
        kw = {'cell_type_field': ctf} if ctf else {}
        adata_backed, meta_df = read_data.read_wang_backed(h5ad_path=args.input, **kw)
    elif args.dataset_type == 'PsychAD':
        kw = {'cell_type_field': ctf} if ctf else {}
        aging_backed, hbcc_backed, hbcc_unique_mask, meta_df = \
            read_data.read_psychad_backed(args.inputs[0], args.inputs[1], **kw)
        if meta_df is None:
            print("Error: Failed to load PsychAD data.")
            sys.exit(1)
        n_aging_cells = len(aging_backed)
    else:
        # Generic: load backed, use obs as metadata
        adata_backed = sc.read_h5ad(args.input, backed='r')
        meta_df = adata_backed.obs.copy()

    if meta_df is None or (args.dataset_type != 'PsychAD' and adata_backed is None):
        print(f"Error: Failed to load data.")
        sys.exit(1)

    log_mem("Data Loaded (backed)")
    if args.dataset_type == 'PsychAD':
        print(f"Total cells available (deduplicated): {len(meta_df):,}")
    else:
        print(f"Total cells in file: {adata_backed.shape[0]}")
    print(f"Cells with metadata: {len(meta_df)}")

    # =========================================================================
    # Step 2: Compute ALL filter masks on the lightweight metadata DataFrame
    #         (no expression matrix access needed)
    # =========================================================================

    # Start with all cells that have metadata
    mask = pd.Series(True, index=meta_df.index)

    # --- Region filter ---
    if args.pfc_only:
        if 'region' not in meta_df.columns:
            print("Error: --pfc_only requested but 'region' column missing.")
            sys.exit(1)

        pfc_mask = meta_df['region'] == 'prefrontal cortex'
        n_before = mask.sum()
        mask = mask & pfc_mask
        print(f"PFC filter: {n_before} -> {mask.sum()} cells")

    # --- Postnatal-only filter ---
    if args.postnatal_only:
        if 'age_years' not in meta_df.columns:
            print("Error: --postnatal_only requested but 'age_years' column missing.")
            sys.exit(1)
        n_before = mask.sum()
        mask = mask & (meta_df['age_years'] >= 0)
        print(f"Postnatal filter (age_years >= 0): {n_before} -> {mask.sum()} cells")

    # --- Minimum age filter ---
    if args.min_age is not None:
        if 'age_years' not in meta_df.columns:
            print("Error: --min_age requested but 'age_years' column missing.")
            sys.exit(1)
        n_before = mask.sum()
        mask = mask & (meta_df['age_years'] >= args.min_age)
        print(f"Min age filter (age_years >= {args.min_age}): {n_before} -> {mask.sum()} cells")

    # --- Cell class filter ---
    if args.cell_class_filter:
        if 'cell_class' not in meta_df.columns:
            print("Error: --cell_class_filter requested but 'cell_class' column missing.")
            sys.exit(1)
        n_before = mask.sum()
        mask = mask & meta_df['cell_class'].isin(args.cell_class_filter)
        print(f"Cell class filter {args.cell_class_filter}: {n_before} -> {mask.sum()} cells")

    # --- Age-based donor downsampling ---
    if args.age_downsample:
        if 'age_years' not in meta_df.columns:
            print("Error: --age_downsample requested but 'age_years' column missing.")
            sys.exit(1)

        donor_col = None
        for col in ['individual', 'individualID', 'donor_id']:
            if col in meta_df.columns:
                donor_col = col
                break

        if donor_col is None:
            print("Error: --age_downsample requires 'individual', 'individualID', or 'donor_id' column.")
            print(f"Available columns: {list(meta_df.columns)}")
            sys.exit(1)

        print(f"Donor-based downsampling (using '{donor_col}')...")
        print(f"  Policy: Keep all donors < 40. Keep 20% of donors >= 40.")

        active_meta = meta_df[mask]
        donor_ages  = active_meta.groupby(donor_col)['age_years'].mean()

        donors_young = donor_ages[donor_ages < 40].index.tolist()
        donors_old   = donor_ages[donor_ages >= 40].index.tolist()

        print(f"  Young Donors (<40): {len(donors_young)}")
        print(f"  Old Donors (>=40): {len(donors_old)}")

        n_keep = max(1, int(np.ceil(len(donors_old) * 0.20))) if donors_old else 0
        np.random.seed(args.seed)
        donors_old_keep = np.random.choice(donors_old, size=n_keep, replace=False).tolist() if donors_old else []

        print(f"  Keeping {len(donors_young)} young + {len(donors_old_keep)} old donors")

        donors_keep = set(donors_young + donors_old_keep)
        donor_mask  = meta_df[donor_col].isin(donors_keep)

        n_before = mask.sum()
        mask = mask & donor_mask
        print(f"  Donor filter: {n_before} -> {mask.sum()} cells")

    # --- Random cell downsampling ---
    if args.n_cells and mask.sum() > args.n_cells:
        print(f"Random downsampling: {mask.sum()} -> {args.n_cells} cells...")
        np.random.seed(args.seed)
        keep_idx = np.random.choice(np.where(mask.values)[0], size=args.n_cells, replace=False)
        new_mask = pd.Series(False, index=meta_df.index)
        new_mask.iloc[keep_idx] = True
        mask = new_mask
        print(f"  After random downsample: {mask.sum()} cells")
    elif not args.n_cells:
        print(f"No cell-count cap — using all {mask.sum():,} cells that pass filters.")

    log_mem("All filters computed")
    print(f"\nFinal cell count: {mask.sum()} / {len(meta_df)}")

    # =========================================================================
    # Step 3: Materialize ONLY the final subset into memory
    # =========================================================================
    print(f"\nLoading {mask.sum()} cells into memory...")
    if args.dataset_type == 'PsychAD':
        adata = _materialize_psychad(
            aging_backed, hbcc_backed, hbcc_unique_mask, meta_df, mask, n_aging_cells)
        del aging_backed, hbcc_backed
    else:
        adata = read_data.materialize_subset(adata_backed, meta_df, mask)
        del adata_backed

    if adata is None:
        print("Error: no cells remain after filtering.")
        sys.exit(1)

    log_mem("Subset materialized")
    print(f"In-memory shape: {adata.shape}")

    # =========================================================================
    # Step 3b: Build cell_type_for_scanvi column
    #   WANG cells: keep their fine-grained cell_type_raw as the reference label
    #   All other datasets: "Unknown" (treated as unlabelled by scANVI)
    # =========================================================================
    adata.obs['cell_type_for_scanvi'] = 'Unknown'
    if args.dataset_type == 'Wang':
        broad = {'Unknown', 'unknown', 'Excitatory', 'Inhibitory', 'Glia', 'Other'}
        labeled_mask = ~adata.obs['cell_type_raw'].isin(broad)
        adata.obs.loc[labeled_mask, 'cell_type_for_scanvi'] = \
            adata.obs.loc[labeled_mask, 'cell_type_raw']
        counts = adata.obs.loc[labeled_mask, 'cell_type_for_scanvi'].value_counts()
        print(f"\ncell_type_for_scanvi: {labeled_mask.sum()} labelled WANG cells "
              f"across {len(counts)} types")
        for lbl, n in counts.items():
            flag = "  *** LOW (<15)" if n < 15 else ""
            print(f"  {lbl:35s}  {n:5d}{flag}")
        n_broad = (~labeled_mask).sum()
        if n_broad:
            print(f"  {'(broad/excluded → Unknown)':35s}  {n_broad:5d}")

    # =========================================================================
    # Step 3c: Add log-postconceptional-age column for scVI covariate
    #   age_log_pc = log(age_years + 268/365)
    #   Both Wang and Velmeshev readers use 268 days as the birth offset,
    #   so adding 268/365 converts age_years back to postconceptional age in years.
    # =========================================================================
    if 'age_years' in adata.obs.columns:
        adata.obs['age_log_pc'] = np.log(adata.obs['age_years'] + 268 / 365)
        n_valid = adata.obs['age_log_pc'].notna().sum()
        print(f"\nage_log_pc: {n_valid} valid values "
              f"(range: {adata.obs['age_log_pc'].min():.2f} to {adata.obs['age_log_pc'].max():.2f})")

    # Combined batch key for joint correction of source and chemistry.
    if 'source' in adata.obs.columns and 'chemistry' in adata.obs.columns:
        source = adata.obs['source'].fillna('unknown').astype(str)
        chemistry = adata.obs['chemistry'].fillna('unknown').astype(str)
        adata.obs['source-chemistry'] = source + '-' + chemistry
        n_combo = adata.obs['source-chemistry'].nunique()
        print(f"source-chemistry: {n_combo} unique combinations")
    else:
        print("Warning: could not create source-chemistry (missing source and/or chemistry columns)")

    # =========================================================================
    # Step 4: Save
    # =========================================================================
    print(f"Saving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    adata.write_h5ad(args.output, compression='gzip')
    log_mem("Saved")
    print("Done.")

if __name__ == "__main__":
    main()
