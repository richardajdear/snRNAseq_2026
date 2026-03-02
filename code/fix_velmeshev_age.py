import pandas as pd
import scanpy as sc
import numpy as np
import re

csv_path = '/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/ahbaC3_projection/projection_results_velmeshev_10k_raw.csv'
h5ad_path = '/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/velmeshev/velmeshev_10k.h5ad'

print("Loading CSV...")
df = pd.read_csv(csv_path, index_col=0)

print("Loading h5ad obs...")
adata = sc.read_h5ad(h5ad_path, backed='r')
obs = adata.obs

print("Mapping development_stage to age_years...")
# Extract development_stage
df['development_stage'] = obs.loc[df.index, 'development_stage'].values

word_to_num = {'first':1, 'second':2, 'third':3, 'fourth':4, 'fifth':5, 'sixth':6, 'seventh':7, 'eighth':8, 'ninth':9, 'tenth':10}

def parse_age(x):
    x = str(x).lower().replace(' human stage', '')
    if 'lmp month' in x:
        words = x.split(' lmp month')[0].strip()
        num = word_to_num.get(words, None)
        if num is not None:
             # approximation: gestation is 40 weeks, LMP month is 4 weeks. 
             # 40 weeks is term, so age_years is negative.
             # e.g. 5th LMP month is ~20 weeks. Age = (20 - 40) / 52 = -20/52 = -0.38
             weeks = num * 4.33
             age = (weeks - 40) / 52.0
             return age
    if 'under-1-year-old' in x:
        return 0.5
    if 'year-old' in x:
        num = int(x.split('-year-old')[0])
        return float(num)
    if 'month-old' in x:
        num = float(x.split('-month-old')[0])
        return num / 12.0
    if 'newborn' in x:
        return 0.0
    return np.nan

df['age_years'] = df['development_stage'].astype(str).apply(parse_age)

def categorize_age(age_years):
    if pd.isna(age_years): return "Unknown"
    if age_years < 0: return "Prenatal"
    if age_years <= 1: return "Infant"
    if age_years <= 9: return "Childhood"
    if age_years <= 25: return "Adolescence"
    return "Adulthood"

df['age_category'] = df['age_years'].apply(categorize_age)
df['age_log2'] = np.log2(df['age_years'] + 1)

print("Saving updated CSV...")
df.to_csv(csv_path)
print("Done!")
