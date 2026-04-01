import scanpy as sc
import pandas as pd
import numpy as np
import os
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr

# Activate automatic conversion
pandas2ri.activate()

# Load R libraries
ggplot2 = importr('ggplot2')
dplyr = importr('dplyr')
readr = importr('readr')
stringr = importr('stringr')

print("Libraries loaded successfully.")

# Load Data
file_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_40k_test.h5ad'
if not os.path.exists(file_path):
    print(f"Error: File {file_path} not found.")
    exit(1)

print(f"Loading {file_path}...")
adata = sc.read_h5ad(file_path, backed='r')
obs = adata.obs.copy()
print(f"Loaded {len(obs)} observations.")

# Clean Data
def clean_bytes(x):
    if isinstance(x, bytes):
        return x.decode('utf-8')
    if isinstance(x, str):
        if x.startswith("b'") and x.endswith("'"):
            return x[2:-1]
        if x.startswith('b"') and x.endswith('"'):
            return x[2:-1]
    return x

cols_to_clean = ['source', 'dataset', 'individual', 'age_years']
for col in cols_to_clean:
    if col in obs.columns:
        obs[col] = obs[col].apply(clean_bytes)

# Ensure age_years is numeric
obs['age_years'] = pd.to_numeric(obs['age_years'], errors='coerce')

print("Data cleaning complete.")
print("Columns:", obs.columns.tolist())
print("Index name:", obs.index.name)
if 'source' in obs.columns:
    print(obs[['source', 'age_years']].head())
else:
    print("'source' column missing!")
    print(obs.head())

# Test R capabilities (simple plot object creation, not rendering)
print("Testing R transfer...")
r_obs = pandas2ri.py2rpy(obs)
r.assign('obs', r_obs)

print("R transfer complete. Script logic verified.")

# Debug: Print column names in R
print("R column names:")
print(r('colnames(obs)'))
print("R head:")
print(r('head(obs)'))
