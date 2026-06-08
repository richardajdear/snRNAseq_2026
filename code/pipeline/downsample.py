
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

    parser.add_argument("--dataset_type", choices=['Velmeshev', 'Wang', 'PsychAD', 'Zhu', 'Generic'],
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
    parser.add_argument("--max_age", type=float, default=None,
                        help="Drop cells from donors with age_years > this value "
                             "(applied before n_cells random downsample).")
    parser.add_argument("--unlabel_below_age", type=float, default=None,
                        help="After shared-label assignment, override cell_type_for_scanvi "
                             "to 'Unknown' for cells with age_years < this value. "
                             "Use for datasets whose annotation is unreliable for "
                             "developmental cells (e.g. PsychAD < 5 y).")
    parser.add_argument("--cell_class_filter", nargs='+', default=None,
                        help="Keep only cells whose cell_class is in this list (e.g. Excitatory Glia).")
    parser.add_argument("--cell_id_filter", default=None,
                        help="Path to a parquet/csv whose index (or first column) lists "
                             "cell barcodes to KEEP. Use to inject a custom cell selection "
                             "(e.g. a cluster/marker-based ExN set) that is not expressible "
                             "via the native cell_class labels.")
    parser.add_argument("--chemistry_filter", nargs='+', default=None,
                        help="Keep only cells whose chemistry is in this list (e.g. V3).")
    parser.add_argument("--n_cells", type=int, default=None,
                        help="Target number of cells (random downsample). "
                             "Omit or set to null in config to use all cells.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--use_shared_labels", action='store_true',
                        help="Apply reference/shared_fine_labels.csv to set "
                             "cell_type_for_scanvi for ALL dataset types "
                             "(semi-supervised scANVI). Default off preserves "
                             "the legacy Wang-only-supervised behaviour.")
    parser.add_argument("--shared_labels_csv", type=str,
                        default='reference/shared_fine_labels.csv',
                        help="Path to the shared label CSV. Relative paths are "
                             "resolved against the repo root.")
    parser.add_argument("--use_raw_labels", action='store_true',
                        help="Use cell_type_raw directly as cell_type_for_scanvi "
                             "without mapping through the shared vocabulary. "
                             "Intended for diagnostic runs where native source labels "
                             "are the supervision target (e.g. Vel 'Interneurons').")
    parser.add_argument("--n_cells_per_age_bin", type=int, default=None,
                        help="After all other filters, cap to this many cells per "
                             "age bin (<1y, 1-5y, 5-18y, 18+y). Applied before "
                             "the existing --n_cells random downsample.")

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
    elif args.dataset_type == 'Zhu':
        kw = {'cell_type_field': ctf} if ctf else {}
        adata_backed, meta_df = read_data.read_zhu_backed(h5ad_path=args.input, **kw)
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

    # --- Maximum age filter (Phase D) ---
    if args.max_age is not None:
        if 'age_years' not in meta_df.columns:
            print("Error: --max_age requested but 'age_years' column missing.")
            sys.exit(1)
        n_before = mask.sum()
        mask = mask & (meta_df['age_years'] <= args.max_age)
        print(f"Max age filter (age_years <= {args.max_age}): {n_before} -> {mask.sum()} cells")

    # --- Cell class filter ---
    if args.cell_class_filter:
        if 'cell_class' not in meta_df.columns:
            print("Error: --cell_class_filter requested but 'cell_class' column missing.")
            sys.exit(1)
        n_before = mask.sum()
        mask = mask & meta_df['cell_class'].isin(args.cell_class_filter)
        print(f"Cell class filter {args.cell_class_filter}: {n_before} -> {mask.sum()} cells")

    # --- Explicit cell-ID filter (custom selection, e.g. cluster-based ExN) ---
    if args.cell_id_filter:
        if args.cell_id_filter.endswith('.parquet'):
            ids_df = pd.read_parquet(args.cell_id_filter)
        else:
            ids_df = pd.read_csv(args.cell_id_filter)
        # Accept either an index of barcodes or a first column of barcodes.
        keep_ids = set(ids_df.index.astype(str)) | set(ids_df.iloc[:, 0].astype(str))
        n_before = mask.sum()
        mask = mask & meta_df.index.astype(str).isin(keep_ids)
        print(f"Cell-ID filter ({args.cell_id_filter}): {n_before} -> {mask.sum()} cells "
              f"({len(keep_ids):,} ids in file)")

    # --- Chemistry filter ---
    if args.chemistry_filter:
        if 'chemistry' not in meta_df.columns:
            print("Error: --chemistry_filter requested but 'chemistry' column missing.")
            sys.exit(1)
        n_before = mask.sum()
        mask = mask & meta_df['chemistry'].isin(args.chemistry_filter)
        print(f"Chemistry filter {args.chemistry_filter}: {n_before} -> {mask.sum()} cells")

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

    # --- Age-bin stratified downsampling ---
    if args.n_cells_per_age_bin is not None:
        if 'age_years' not in meta_df.columns:
            print("Error: --n_cells_per_age_bin requested but 'age_years' column missing.")
            sys.exit(1)
        bins = [(-np.inf, 1), (1, 5), (5, 18), (18, np.inf)]
        bin_labels = ['<1y', '1-5y', '5-18y', '18+y']
        active_idx = np.where(mask.values)[0]
        ages = meta_df['age_years'].values[active_idx]
        keep_positions = []
        print(f"Age-bin stratified sampling (cap={args.n_cells_per_age_bin} per bin):")
        for (lo, hi), label in zip(bins, bin_labels):
            bin_pos = active_idx[(ages >= lo) & (ages < hi)]
            if len(bin_pos) == 0:
                continue
            n_before_bin = len(bin_pos)
            if len(bin_pos) > args.n_cells_per_age_bin:
                np.random.seed(args.seed)
                bin_pos = np.random.choice(bin_pos, size=args.n_cells_per_age_bin, replace=False)
            print(f"  {label}: {n_before_bin:,} → {len(bin_pos):,} cells")
            keep_positions.extend(bin_pos)
        new_mask = pd.Series(False, index=meta_df.index)
        new_mask.iloc[sorted(keep_positions)] = True
        n_before = mask.sum()
        mask = new_mask
        print(f"  Total after age-bin stratification: {n_before:,} → {mask.sum():,} cells")

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
    # =========================================================================
    if args.use_shared_labels:
        # Semi-supervised mode: translate native fine labels into a shared
        # vocabulary using reference/shared_fine_labels.csv, so scANVI's
        # classifier is supervised by examples from all three datasets.
        # See code/pipeline/shared_labels.py for the API.
        from pipeline.shared_labels import (
            load_shared_label_map, apply_shared_labels)

        csv_path = args.shared_labels_csv
        if not os.path.isabs(csv_path):
            repo_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            csv_path = os.path.join(repo_root, csv_path)

        mapping = load_shared_label_map(csv_path)
        # cell_type_raw is set by read_data.py for every dataset_type and holds
        # the native fine label (Vel cell_type, Wang Type-updated, PsychAD subclass).
        if args.dataset_type == 'Generic':
            raise ValueError(
                "--use_shared_labels requires --dataset_type in "
                "{Wang, Velmeshev, PsychAD, Zhu}")
        labels, summary = apply_shared_labels(
            adata, args.dataset_type, 'cell_type_raw', mapping)
        adata.obs['cell_type_for_scanvi'] = labels.values
        # Persistent marker so anyone inspecting the .h5ad downstream can verify
        # the run used shared labels (no hidden fallback).
        adata.uns['cell_type_for_scanvi_source'] = 'shared_labels'
        adata.uns['shared_labels_csv'] = os.path.basename(csv_path)
        adata.uns['shared_labels_coverage'] = summary['coverage_fraction']

        print(f"\ncell_type_for_scanvi ({args.dataset_type}, SHARED VOCABULARY): "
              f"{summary['n_mapped']}/{summary['n_cells']} cells mapped "
              f"({summary['coverage_fraction']:.1%}) across "
              f"{summary['n_shared_labels']} shared labels")
        print(f"  CSV: {csv_path}")
        print(f"  Full value_counts (sorted, label → n_cells):")
        for lbl, n in sorted(summary['value_counts'].items(),
                             key=lambda kv: -kv[1]):
            print(f"    {lbl:30s}  {n:7d}")
        if summary['n_unmapped']:
            print(f"  Top unmapped native labels: {summary['unmapped_top10']}")

        # --- Withhold labels from young cells (Phase B) ---
        # Applied AFTER apply_shared_labels so we can see what labels would have
        # been assigned, but override them to Unknown so the scANVI classifier
        # is not taught potentially wrong developmental labels.
        if args.unlabel_below_age is not None and 'age_years' in adata.obs.columns:
            young_mask = adata.obs['age_years'] < args.unlabel_below_age
            n_unlabeled = int(young_mask.sum())
            adata.obs.loc[young_mask, 'cell_type_for_scanvi'] = 'Unknown'
            adata.uns['unlabel_below_age'] = args.unlabel_below_age
            print(f"  Withheld supervised labels from {n_unlabeled} cells with "
                  f"age_years < {args.unlabel_below_age} (now 'Unknown').")
            # Re-emit value_counts after withhold
            vc = adata.obs['cell_type_for_scanvi'].value_counts()
            print(f"  Post-withhold value_counts:")
            for lbl, n in vc.items():
                print(f"    {lbl:30s}  {n:7d}")
    elif args.use_raw_labels:
        # Diagnostic mode: native cell_type_raw labels directly as scANVI supervision.
        # No mapping through shared vocabulary — the scANVI class vocabulary IS the
        # native label set (e.g. Vel "Interneurons", "L4", "SST", ...).
        adata.obs['cell_type_for_scanvi'] = adata.obs['cell_type_raw'].astype(str)
        adata.uns['cell_type_for_scanvi_source'] = 'raw_labels'
        vc = adata.obs['cell_type_for_scanvi'].value_counts()
        print(f"\ncell_type_for_scanvi ({args.dataset_type}, RAW LABELS): "
              f"{len(vc)} classes across {len(adata):,} cells")
        for lbl, n in vc.items():
            print(f"  {lbl:35s}  {n:7d}")
        if args.unlabel_below_age is not None and 'age_years' in adata.obs.columns:
            young_mask = adata.obs['age_years'] < args.unlabel_below_age
            n_unlabeled = int(young_mask.sum())
            adata.obs.loc[young_mask, 'cell_type_for_scanvi'] = 'Unknown'
            print(f"  Withheld labels from {n_unlabeled:,} cells with "
                  f"age_years < {args.unlabel_below_age} (now 'Unknown').")
    else:
        #   WANG cells: keep their fine-grained cell_type_raw as the reference label
        #   All other datasets: "Unknown" (treated as unlabelled by scANVI)
        adata.uns['cell_type_for_scanvi_source'] = 'legacy_wang_only'
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
