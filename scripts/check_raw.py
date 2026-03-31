import scanpy as sc
import numpy as np
import pandas as pd

from sys import path
path.insert(0, "/home/rajd2/rds/hpc-work/snRNAseq_2026/code")
from regulons import get_ahba_GRN, project_GRN
from gene_mapping import map_grn_symbols_to_ensembl

print("Loading data...")
DATA_FILE = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld/scvi_output/integrated.h5ad"
adata = sc.read_h5ad(DATA_FILE)

# Keep only Excitatory and childhood
adata = adata[adata.obs["cell_class"] == "Excitatory"].copy()
adata = adata[(adata.obs["age_years"] >= 1.0) & (adata.obs["age_years"] < 10.0)].copy()

# Follow notebook exactly
adata.layers['counts'] = adata.layers['scvi_normalized']
sc.pp.normalize_total(adata, target_sum=1e6)

# Load GRN
grn_file = "/home/rajd2/rds/hpc-work/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv"
ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file, use_weights=True)
ahba_GRN = map_grn_symbols_to_ensembl(ahba_GRN, adata)

# Project as notebook does
project_GRN(adata, ahba_GRN, 'X_ahba', use_highly_variable=False, log_transform=False)

proj = pd.DataFrame(adata.obsm['X_ahba'], index=adata.obs_names, columns=adata.uns['X_ahba_names'])
adata.obs['nb_C3+'] = proj['C3+']

# Compare to scvi CPM via diagnostic logic
w_c3pos = ahba_GRN[ahba_GRN["Network"] == "C3+"].set_index("Gene")["Importance"]
w_c3pos = w_c3pos.reindex(adata.var_names).fillna(0).values

scvi_expr = adata.layers['scvi_normalized']
def cpm(counts):
    counts = counts.toarray() if hasattr(counts, 'toarray') else np.asarray(counts)
    s = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(s, 1) * 1e6

score_scvi = cpm(scvi_expr) @ w_c3pos
adata.obs['diag_scvi_C3+'] = score_scvi

print("\n--- Childhood Means ---")
for src in sorted(adata.obs["source"].unique()):
    d = adata.obs[adata.obs["source"] == src]
    print(f"{src:12}  N={len(d):<4}  Notebook_C3: {d['nb_C3+'].mean():.1f}   Diag_scVI_C3: {d['diag_scvi_C3+'].mean():.1f}")
