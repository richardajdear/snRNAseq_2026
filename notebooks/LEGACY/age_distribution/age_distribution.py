import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# File
file_path = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_postnatal_full_harmony.h5ad"
out_dir = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined"

print(f"Loading {file_path} using h5py...")
f = h5py.File(file_path, 'r')

# 1. Dimensions
n_cells = f['obs']['_index'].shape[0]
n_vars = f['var']['_index'].shape[0]
print(f"Dimensions: ({n_cells}, {n_vars})")

# Helper to read obs columns
def read_obs_col(col_name):
    if col_name in f['obs']:
        data = f['obs'][col_name]
        if isinstance(data, h5py.Group): # Categorical
            codes = data['codes'][:]
            cats = data['categories'][:]
            # Decode categories
            cats = [c.decode('utf-8') for c in cats]
            return [cats[c] for c in codes]
        else: # Standard array
            vals = data[:]
            # Decode if bytes
            if vals.dtype.kind == 'S':
                 return [v.decode('utf-8') for v in vals]
            return vals
    return None

# Load required columns
print("Reading metadata...")
source = read_obs_col('source')
age_cat = read_obs_col('age category')
lineage = read_obs_col('lineage')
age_years = read_obs_col('age_years')

df = pd.DataFrame({
    'source': source,
    'age category': age_cat,
    'lineage': lineage
})
if age_years is not None:
    df['age_years'] = age_years

print("Metadata loaded. Shape:", df.shape)

# 2. Count table
print("\n--- Source x Age Category ---")
ct = pd.crosstab(df['source'], df['age category'])
print(ct)

# 3. Plot
print("\nGenerating plot...")
plt.figure(figsize=(10, 6))
if 'age_years' in df.columns:
    sns.violinplot(data=df, x='source', y='age_years')
    plt.title("Age Distribution by Source")
else:
    sns.countplot(data=df, x='source', hue='age category')
    plt.title("Age Category Counts by Source")
plot_path = os.path.join(out_dir, "age_distribution.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

# 4. Cell Class Split
glia_types = ['Astrocytes', 'Glia', 'Microglia', 'OPC', 'Oligos']
def get_broad_class(l):
    l = str(l)
    if l in glia_types:
        return 'All Glia'
    elif l == 'Excitatory':
        return 'Excitatory'
    elif l == 'Inhibitory':
        return 'Inhibitory'
    else:
        return 'Other'

df['broad_class'] = df['lineage'].apply(get_broad_class)

for cls in ['Excitatory', 'Inhibitory', 'All Glia']:
    print(f"\n--- {cls} ---")
    subset = df[df['broad_class'] == cls]
    print(pd.crosstab(subset['source'], subset['age category']))

f.close()
