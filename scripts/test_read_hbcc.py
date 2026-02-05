import sys
import os
import scanpy as sc
import psutil
import gc
import numpy as np
import pandas as pd

# Add current dir to path to allow import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from read_and_combine.py
# Note: read_and_combine imports might trigger top-level code? 
# No, main is protected. Constants validation? No.
from read_and_combine import read_psychad, filter_age_thresholds

def print_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[Memory] {step_name}: {mem_info.rss / 1024 ** 3:.2f} GB")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Test reading HBCC dataset")
    parser.add_argument("path", help="Path to h5ad file", nargs='?', default="/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort_10k.h5ad")
    args = parser.parse_args()
    
    file_path = args.path

    print(f"Starting test loading of HBCC from {file_path}...")
    print_memory_usage("Start")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist yet.")
        sys.exit(1)

    try:
        ad = read_psychad(file_path, "psychAD", "HBCC")
        print_memory_usage("Loaded")
        
        # Test filtering
        ad = filter_age_thresholds(ad, "HBCC", min_age=0, max_age=40)
        print_memory_usage("Filtered")
        
        print(f"Final shape: {ad.shape}")
        if 'age_years' in ad.obs.columns:
            print(f"Age range: {ad.obs['age_years'].min()} - {ad.obs['age_years'].max()}")
            
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
