import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Title
nb.cells.append(nbf.v4.new_markdown_cell("""
# snRNAseq Data Diagnostics
**File:** `/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_postnatal_full_harmony.h5ad`

This notebook analyzes the dataset dimensions and cell distributions.
"""))

# Imports
nb.cells.append(nbf.v4.new_code_cell("""
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File path
file_path = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/combined_postnatal_full_harmony.h5ad"
"""))

# Load Data
nb.cells.append(nbf.v4.new_code_cell("""
print(f"Loading {file_path} in backed mode...")
adata = sc.read_h5ad(file_path, backed='r')
"""))

# 1. Dimensions
nb.cells.append(nbf.v4.new_markdown_cell("## 1. Total Dimensions"))
nb.cells.append(nbf.v4.new_code_cell("""
print(f"Dimensions (Calls x Genes): {adata.shape}")
n_cells = adata.shape[0]
n_vars = adata.shape[1]
"""))

# 2. Count table
nb.cells.append(nbf.v4.new_markdown_cell("## 2. Cell Counts by Source and Age Category"))
nb.cells.append(nbf.v4.new_code_cell("""
# Extract obs for easier handling (it's small enough to fit in memory unlike X)
# We only fetch columns we need to speed up
obs_subset = adata.obs[['source', 'age category', 'lineage']].copy()

ct = pd.crosstab(obs_subset['source'], obs_subset['age category'])
display(ct)
"""))

# 3. Plot Age Distributions
nb.cells.append(nbf.v4.new_markdown_cell("## 3. Age Distributions by Data Source"))
nb.cells.append(nbf.v4.new_code_cell("""
# Create plot
plt.figure(figsize=(10, 6))

# Check if age_category is suitable for plotting or if we have numeric age
# We see 'age_years' in the file inspection, let's try to use that if available, else age_category
if 'age_years' in adata.obs.keys():
    age_col = 'age_years'
    obs_subset['age_years'] = adata.obs['age_years']
    sns.violinplot(data=obs_subset, x='source', y='age_years')
    plt.title("Age Distribution by Source")
else:
    # Categorical plot
    sns.countplot(data=obs_subset, x='source', hue='age category')
    plt.title("Age Category Counts by Source")

plt.show()
"""))

# 4. Split by Major Cell Class
nb.cells.append(nbf.v4.new_markdown_cell("""
## 4. Count Tables by Major Cell Class
Classes: Excitatory, Inhibitory, and All Glia.
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# Define basic classes
# Lineage categories found: ['Astrocytes', 'Excitatory', 'Glia', 'Inhibitory', 'Microglia', 'OPC', 'Oligos', 'Other']
glia_types = ['Astrocytes', 'Glia', 'Microglia', 'OPC', 'Oligos']

def get_broad_class(l):
    if l in glia_types:
        return 'All Glia'
    elif l == 'Excitatory':
        return 'Excitatory'
    elif l == 'Inhibitory':
        return 'Inhibitory'
    else:
        return 'Other'

obs_subset['broad_class'] = obs_subset['lineage'].apply(get_broad_class)

# Generate tables
for cls in ['Excitatory', 'Inhibitory', 'All Glia']:
    print(f"\\n--- {cls} ---")
    subset = obs_subset[obs_subset['broad_class'] == cls]
    ct_cls = pd.crosstab(subset['source'], subset['age category'])
    display(ct_cls)
"""))

# Write to file
with open('/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/diagnostics.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created.")
