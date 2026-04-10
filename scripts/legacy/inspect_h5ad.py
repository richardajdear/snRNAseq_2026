import scanpy as sc
import pandas as pd
import argparse
import os
import sys

def inspect(input_file, show_obs=False, show_var=False, column_patterns=None):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        sys.exit(1)

    print(f"--- Inspecting {input_file} ---")
    adata = sc.read_h5ad(input_file)
    print(adata)
    
    if show_obs:
        print("\n--- Obs Columns ---")
        print(adata.obs.columns.tolist())
        print("\n--- Obs Head ---")
        print(adata.obs.head())

    if show_var:
        print("\n--- Var Columns ---")
        print(adata.var.columns.tolist())
        print("\n--- Var Head ---")
        print(adata.var.head())

    if column_patterns:
        print(f"\n--- Checking Columns matching {column_patterns} ---")
        obs_matches = [c for c in adata.obs.columns if any(p.lower() in c.lower() for p in column_patterns)]
        if obs_matches:
            print(f"Found obs columns: {obs_matches}")
            for c in obs_matches:
                print(f"\nColumn: {c}")
                print(f"Unique count: {len(adata.obs[c].unique())}")
                print(f"Top 10 values:\n{adata.obs[c].value_counts().head(10)}")
        else:
            print("No matching obs columns found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect h5ad file contents.")
    parser.add_argument("--input", required=True, help="Path to h5ad file")
    parser.add_argument("--obs", action="store_true", help="Show obs columns and head")
    parser.add_argument("--var", action="store_true", help="Show var columns and head")
    parser.add_argument("--find", nargs="+", help="Patterns to search for in column names (e.g. age class)")
    
    args = parser.parse_args()
    
    inspect(args.input, args.obs, args.var, args.find)
