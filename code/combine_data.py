
import scanpy as sc
import pandas as pd
import numpy as np
import os
import sys
import argparse
import warnings
import psutil
import time
import gc

# Import our new module
try:
    import read_data
except ImportError:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    import read_data

OUTPUT_DIR = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/"

START_TIME = time.time()

def log_mem(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    elapsed = time.time() - START_TIME
    print(f"\n[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB ({process.memory_percent():.2f}%) - Elapsed: {elapsed/60:.2f} min")
    sys.stdout.flush()

def check_structure_from_files(file_dict):
    """
    Print diagnostic crosstabs from backed files (no full matrix load needed).
    """
    print("\n" + "="*40)
    print("      DIAGNOSTIC STRUCTURE CHECK")
    print("="*40)
    
    col_sets = []
    combined_meta = []
    
    for name, path in file_dict.items():
        ad = sc.read_h5ad(path, backed='r')
        obs = ad.obs
        col_sets.append(set(obs.columns))
        
        df = pd.DataFrame({'Dataset_Key': name}, index=obs.index)
        df['region'] = obs['region'] if 'region' in obs.columns else 'MISSING'
        df['cell_class'] = obs['cell_class'] if 'cell_class' in obs.columns else 'MISSING'
        combined_meta.append(df)
        
        print(f"  {name}: {ad.shape[0]} cells x {ad.shape[1]} genes")
        del ad
    
    common_cols = sorted(list(set.intersection(*col_sets)))
    print(f"\n[Common .obs Columns] Count: {len(common_cols)}")
    print(", ".join(common_cols))
    
    full_df = pd.concat(combined_meta)
    print("\n[Crosstab: Dataset x Region]")
    print(pd.crosstab(full_df['region'], full_df['Dataset_Key']))
    print("\n[Crosstab: Dataset x Cell Class]")
    print(pd.crosstab(full_df['cell_class'], full_df['Dataset_Key']))
    print("\n" + "="*40 + "\n")
    
    del full_df, combined_meta
    gc.collect()
    return common_cols

def combine_on_disk(file_dict, output_path):
    """
    Use anndata.experimental.concat_on_disk to combine datasets without loading into memory.
    This streams data from input files and writes directly to the output file.
    """
    from anndata.experimental import concat_on_disk
    
    in_files = list(file_dict.values())
    
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir): 
        os.makedirs(out_dir)
    
    print(f"Combining {len(in_files)} files on disk...")
    print(f"  Input files: {in_files}")
    print(f"  Output: {output_path}")
    
    log_mem("Before concat_on_disk")
    
    concat_on_disk(
        in_files,
        output_path,
        label='source',
        keys=list(file_dict.keys()),
        join='inner'
    )
    
    log_mem("After concat_on_disk")
    
    # Restore gene metadata using backed access to reference and output
    print("Restoring gene metadata...")
    ref_path = in_files[0]
    ref = sc.read_h5ad(ref_path, backed='r')
    out = sc.read_h5ad(output_path, backed='r+')
    
    common_vars = out.var_names
    
    if 'feature_name' in ref.var.columns:
        out.var['gene_symbol'] = ref.var.loc[common_vars, 'feature_name'].values
        print(f"  Restored 'gene_symbol' for {len(out.var)} genes.")
    elif 'gene_name' in ref.var.columns:
        out.var['gene_symbol'] = ref.var.loc[common_vars, 'gene_name'].values
        print(f"  Restored 'gene_symbol' for {len(out.var)} genes.")
        
    if 'feature_length' in ref.var.columns:
        out.var['feature_length'] = ref.var.loc[common_vars, 'feature_length'].values
        print(f"  Restored 'feature_length' for {len(out.var)} genes.")
    
    # Write updated var back
    out.file.close()
    del ref, out
    gc.collect()
    
    log_mem("After metadata restore")
    print(f"Combined file saved to: {output_path}")


def combine_in_memory(file_dict, output_path, common_cols):
    """Fallback: load all datasets into memory and combine with sc.concat."""
    adatas = {}
    for name, path in file_dict.items():
        print(f"Loading {name} from {path} (backed)...")
        ad = sc.read_h5ad(path, backed='r')
        ad = ad[:].to_memory()
        adatas[name] = ad
        log_mem(f"After loading {name}")
    
    # Standardize gene symbols and subset obs
    for name in adatas:
        adData = adatas[name]
        if 'feature_name' in adData.var.columns:
            adData.var['gene_symbol'] = adData.var['feature_name']
        elif 'gene_name' in adData.var.columns:
            adData.var['gene_symbol'] = adData.var['gene_name']
        else:
            adData.var['gene_symbol'] = adData.var.index
        
        valid_cols = [c for c in common_cols if c in adData.obs.columns]
        adData.obs = adData.obs[valid_cols]
        adatas[name] = adData
    
    log_mem("Before concat")
    combined = sc.concat(adatas, label='source', index_unique='-')
    print(f"Combined Shape: {combined.shape}")
    log_mem("After concat")
    
    del adatas
    gc.collect()
    log_mem("After freeing datasets")
    
    # Restore gene metadata
    ref_path = list(file_dict.values())[0]
    ref = sc.read_h5ad(ref_path, backed='r')
    common_vars = combined.var_names
    if 'feature_name' in ref.var.columns:
        combined.var['gene_symbol'] = ref.var.loc[common_vars, 'feature_name'].values
        print(f"Restored 'gene_symbol' for {len(combined.var)} genes.")
    if 'feature_length' in ref.var.columns:
        combined.var['feature_length'] = ref.var.loc[common_vars, 'feature_length'].values
        print(f"Restored 'feature_length' for {len(combined.var)} genes.")
    del ref
    
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
    
    log_mem("Before save")
    print(f"Saving to {output_path}...")
    combined.write_h5ad(output_path, compression='gzip')
    log_mem("After save")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Combine snRNAseq datasets.")
    parser.add_argument("--postnatal", action='store_true', help="(Deprecated) Filter for age >= 0")
    parser.add_argument("--diagnose_only", action='store_true', help="Only run diagnostics, do not save")
    parser.add_argument("--output", default=f"{OUTPUT_DIR}/combined_postnatal_full.h5ad")
    
    parser.add_argument("--aging_path", default=read_data.AGING_PATH)
    parser.add_argument("--hbcc_path", default=read_data.HBCC_PATH)
    parser.add_argument("--velmeshev_path", default=read_data.VELMESHEV_PATH)
    parser.add_argument("--wang_path", default=read_data.WANG_PATH)
    
    parser.add_argument("--direct_load", action='store_true', 
                        help="Load h5ad files directly (skip read_data processing). Use for pre-processed/downsampled inputs.")
    
    args = parser.parse_args()
    
    log_mem("Start")
    
    # Build file dict from paths that exist
    file_dict = {}
    for name, path in [('VELMESHEV', args.velmeshev_path), ('WANG', args.wang_path),
                       ('AGING', args.aging_path), ('HBCC', args.hbcc_path)]:
        if path and os.path.exists(path):
            file_dict[name] = path
        elif path:
            print(f"Warning: {name} path {path} does not exist.")
    
    if not file_dict:
        print("No datasets found.")
        sys.exit(1)
        
    if args.postnatal:
        print("Warning: --postnatal flag is deprecated. Assuming inputs are already filtered.")
    
    # Diagnostics (uses backed mode, minimal memory)
    common_cols = check_structure_from_files(file_dict)
    log_mem("After diagnostics")
    
    if args.diagnose_only:
        print("Diagnostics complete. Exiting.")
        return

    # Combine
    if args.direct_load:
        # On-disk concatenation: no expression matrices loaded into memory
        print("Using on-disk concatenation (anndata.experimental.concat_on_disk)...")
        combine_on_disk(file_dict, args.output)
    else:
        # In-memory fallback (for when reader functions need to process metadata)
        print("Using in-memory concatenation...")
        # Load via reader functions
        adatas = {}
        for name, path in file_dict.items():
            if name == 'VELMESHEV':
                ad = read_data.read_velmeshev(h5ad_path=path)
            elif name == 'WANG':
                ad = read_data.read_wang(h5ad_path=path)
            elif name in ('AGING', 'HBCC'):
                ad = read_data.read_psychad(path, name, min_age=None, max_age=None)
            else:
                ad = sc.read_h5ad(path, backed='r')
                ad = ad[:].to_memory()
            
            if ad is not None and ad.n_obs > 0:
                adatas[name] = ad
                print(f"  {name}: {ad.shape}")
            log_mem(f"After loading {name}")
        
        # Standardize and combine
        for name, adData in adatas.items():
            if 'feature_name' in adData.var.columns:
                adData.var['gene_symbol'] = adData.var['feature_name']
            elif 'gene_name' in adData.var.columns:
                adData.var['gene_symbol'] = adData.var['gene_name']
            else:
                adData.var['gene_symbol'] = adData.var.index
            valid_cols = [c for c in common_cols if c in adData.obs.columns]
            adData.obs = adData.obs[valid_cols]
        
        log_mem("Before concat")
        combined = sc.concat(adatas, label='source', index_unique='-')
        print(f"Combined Shape: {combined.shape}")
        log_mem("After concat")
        del adatas
        gc.collect()
        
        # Restore metadata
        ref_path = list(file_dict.values())[0]
        ref = sc.read_h5ad(ref_path, backed='r')
        common_vars = combined.var_names
        if 'feature_name' in ref.var.columns:
            combined.var['gene_symbol'] = ref.var.loc[common_vars, 'feature_name'].values
            print(f"Restored 'gene_symbol' for {len(combined.var)} genes.")
        if 'feature_length' in ref.var.columns:
            combined.var['feature_length'] = ref.var.loc[common_vars, 'feature_length'].values
            print(f"Restored 'feature_length' for {len(combined.var)} genes.")
        del ref
        
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir): os.makedirs(out_dir)
        
        log_mem("Before save")
        print(f"Saving to {args.output}...")
        combined.write_h5ad(args.output, compression='gzip')
        log_mem("After save")
        print("Done.")


if __name__ == "__main__":
    main()
