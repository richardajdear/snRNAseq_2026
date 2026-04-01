import h5py
import pandas as pd
import numpy as np
import os

# Settings
h5ad_path = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_postnatal_full_harmony.h5ad"
out_dir = "/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/combined_full"
out_csv = os.path.join(out_dir, "combined_obs.csv")

print(f"Opening {h5ad_path}...")
f = h5py.File(h5ad_path, 'r')

obs = f['obs']
print(f"Obs keys found: {list(obs.keys())}")

# Collect columns to proper dictionary
data = {}
n_cells = None

# Helper to decode
def decode_bytes(arr):
    if arr.dtype.kind == 'S':
        return arr.astype(str)
    return arr

if '_index' in obs:
    index = obs['_index'][:]
    data['cell_id'] = decode_bytes(index)
    n_cells = len(index)

for key in obs.keys():
    if key == '_index':
        continue
    
    # Handle columns
    item = obs[key]
    print(f"Processing {key}...")
    if isinstance(item, h5py.Group):
        # Categorical
        if 'codes' in item and 'categories' in item:
            codes = item['codes'][:]
            cats = decode_bytes(item['categories'][:])
            
            # Use pandas efficient categorical creation
            # Handle potential codes that are out of bounds (e.g. -1 for nan) if any
            # Generally scanpy stores valid codes.
            # We can create a Categorical from codes directly.
            data[key] = pd.Categorical.from_codes(codes, categories=cats)
            
    elif isinstance(item, h5py.Dataset):
        # Numerical or simple string
        vals = item[:]
        data[key] = decode_bytes(vals)

f.close()

print("Creating DataFrame...")
df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")

print(f"Saving to {out_csv}...")
df.to_csv(out_csv, index=False)
print("Done.")
