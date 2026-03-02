import pandas as pd
import numpy as np
import sys
import os

sys.path.append('/home/rajd2/rds/hpc-work/snRNAseq_2026/code')
import read_data

csv_path = '/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_10k_raw.csv'
out_csv_path = '/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_10k_raw_3.csv'
h5ad_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_10k.h5ad'

print("Loading original raw projection csv...")
df = pd.read_csv(csv_path, index_col=0)

print("Reading velmeshev backed metadata via read_data.py...")
import contextlib
import io
with contextlib.redirect_stdout(io.StringIO()):
    _, meta_df = read_data.read_velmeshev_backed(h5ad_path=h5ad_path)

common = df.index.intersection(meta_df.index)
print(f"Found metadata for {len(common)} out of {len(df)} cells.")

# Add age_years, cell_type, and cell_class from meta_df
df['age_years'] = np.nan
df.loc[common, 'age_years'] = meta_df.loc[common, 'age_years']

# Update cell_type and cell_class if they exist in meta_df
if 'Cell_Type' in meta_df.columns:
    df.loc[common, 'cell_type'] = meta_df.loc[common, 'Cell_Type'].astype(str)
if 'cell_class' in meta_df.columns:
    df.loc[common, 'cell_class'] = meta_df.loc[common, 'cell_class'].astype(str)

def categorize_age(age_years):
    if pd.isna(age_years): return "Unknown"
    if age_years < 0: return "Prenatal"
    if age_years <= 1: return "Infant"
    if age_years <= 9: return "Childhood"
    if age_years <= 25: return "Adolescence"
    return "Adulthood"

df['age_category'] = df['age_years'].apply(categorize_age)
df['age_log2'] = np.log2(np.maximum(df['age_years'] + 1, 1e-5))

print(f"Assigning results to {out_csv_path}")
df.to_csv(out_csv_path)
print("Done!")
