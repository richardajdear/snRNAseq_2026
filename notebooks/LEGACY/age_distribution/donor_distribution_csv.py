import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Settings
csv_path = "/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/combined_full/combined_obs.csv"
out_dir = "/home/rajd2/rds/hpc-work/snRNAseq_2026/notebooks/combined_full"

print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Data shape: {df.shape}")

# Clean byte strings if present
def clean_bytes(s):
    try:
        if isinstance(s, str):
            # Match b'...' pattern
            m = re.match(r"b['\"](.*)['\"]", s)
            if m:
                return m.group(1)
        return str(s)
    except:
        return str(s)

# Apply to string columns
str_cols = ['source', 'age_category', 'lineage', 'dataset', 'region', 'sex', 'chemistry', 'individual']
for col in str_cols:
    if col in df.columns:
        print(f"Cleaning {col}...")
        df[col] = df[col].apply(clean_bytes)

# Enforce Age Category Order
# Definitions: Infant (0-1), Childhood (1-9), Adolescence (9-25), Adulthood (25+)
age_order = ['Infant', 'Childhood', 'Adolescence', 'Adulthood']
df['age_category'] = pd.Categorical(df['age_category'], categories=age_order, ordered=True)

# Define Broad Class
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

df['broad_class'] = df['lineage'].apply(get_broad_class)

# --- CELL COUNTS ---
print("\n=== CELL COUNTS ===")
print("\n--- Source x Age Category (Cells) ---")
ct_cells = pd.crosstab(df['source'], df['age_category'])
print(ct_cells)

for cls in ['Excitatory', 'Inhibitory', 'All Glia']:
    print(f"\n--- {cls} (Cells) ---")
    subset = df[df['broad_class'] == cls]
    print(pd.crosstab(subset['source'], subset['age_category']))

# --- DONOR COUNTS ---
print("\n=== DONOR COUNTS ===")
# Helper for unique count pivot
def pivot_unique(data, index_col, col_col, count_col):
    return data.groupby([index_col, col_col])[count_col].nunique().unstack(fill_value=0)

print("\n--- Source x Age Category (Donors) ---")
ct_donors = pivot_unique(df, 'source', 'age_category', 'individual')
print(ct_donors)

# Plotting age distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='source', hue='age_category', palette='viridis')
plt.title("Age Category Counts by Source")
plt.xlabel("Source")
plt.ylabel("Number of Cells")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plot_path = os.path.join(out_dir, "age_distribution.png")
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
plt.close() # Close the plot to free memory

for cls in ['Excitatory', 'Inhibitory', 'All Glia']:
    print(f"\n--- {cls} (Donors) ---")
    subset = df[df['broad_class'] == cls]
    print(pivot_unique(subset, 'source', 'age_category', 'individual'))

print("Analysis complete.")
