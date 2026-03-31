
import scanpy as sc
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    # Use backed mode if large, but 10k combined is small
    adata = sc.read_h5ad(args.input)
    
    print("Extracting obs...")
    obs = adata.obs.copy()
    
    # Ensure source/dataset columns are clear
    if 'source' in obs.columns:
        print("Dataset counts by source:")
        print(obs['source'].value_counts())
    
    print("\nAge Summary per Source:")
    print(obs.groupby('source')['age_years'].agg(['min', 'max', 'count']))
    
    # Check for donor column
    donor_col = 'individual'
    if donor_col in obs.columns:
        print("\nDonor Counts per Source:")
        print(obs.groupby('source')[donor_col].nunique())
    elif 'individualID' in obs.columns:
         print("\nDonor Counts per Source (individualID):")
         print(obs.groupby('source')['individualID'].nunique())
    elif 'donor_id' in obs.columns:
         print("\nDonor Counts per Source (donor_id):")
         print(obs.groupby('source')['donor_id'].nunique())
    
    print(f"Saving metadata to {args.output}...")
    obs.to_csv(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
