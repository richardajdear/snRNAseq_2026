"""
Analyse cell type distribution by source in the combined 100k h5ad.

Loads in backed mode (no expression matrix in memory).
Checks whether cell_class and cell_subclass mappings are present; if
cell_subclass is absent, derives it from the cell_type (CellxGene ontology)
column using map_cellxgene_subclass from code/read_data.py.

Outputs (scripts/outputs/):
  celltype_by_source.csv       — cell_class × source counts + row/col totals
  celltype_pct_by_source.csv   — same, normalised to % within each source
  cellsubclass_by_source.csv   — cell_subclass × source counts (if available)
  celltype_mapping_check.txt   — unmapped cell_type labels
"""

import os
import sys

# Allow `from read_data import ...` without installing the package
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'code'))

import scanpy as sc
from read_data import map_cellxgene_subclass, map_velmeshev_subclass

# ── paths ────────────────────────────────────────────────────────────────────
from environment import get_environment
_rds = get_environment()['rds_dir']
H5AD = os.path.join(_rds, 'Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld.h5ad')
OUT  = os.path.join(REPO_ROOT, 'scripts', 'outputs')
os.makedirs(OUT, exist_ok=True)

# ── load (backed = no matrix in RAM) ─────────────────────────────────────────
print(f"Loading (backed): {H5AD}")
adata = sc.read_h5ad(H5AD, backed='r')
print(f"  Shape: {adata.shape}")
obs = adata.obs.copy()   # small: just the metadata DataFrame

# ── report available cell-type columns ───────────────────────────────────────
CT_COLS = ['source', 'cell_class', 'cell_subclass', 'cell_type', 'dataset', 'class', 'subclass']
present = [c for c in CT_COLS if c in obs.columns]
missing = [c for c in CT_COLS if c not in obs.columns]
print(f"\nPresent obs columns of interest: {present}")
print(f"Missing obs columns:              {missing}")

# ── cell_class × source ───────────────────────────────────────────────────────
print("\n--- cell_class × source ---")
ct_counts = (
    obs.groupby(['source', 'cell_class'], observed=True)
       .size()
       .unstack(fill_value=0)
)
ct_counts['TOTAL'] = ct_counts.sum(axis=1)
ct_counts.loc['TOTAL'] = ct_counts.sum(axis=0)

print(ct_counts.to_string())
ct_counts.to_csv(os.path.join(OUT, 'celltype_by_source.csv'))
print(f"  Saved → {OUT}/celltype_by_source.csv")

# percentage within each source (excluding the TOTAL row/col)
src_rows = ct_counts.index[ct_counts.index != 'TOTAL']
cls_cols = [c for c in ct_counts.columns if c != 'TOTAL']
pct = ct_counts.loc[src_rows, cls_cols].div(ct_counts.loc[src_rows, 'TOTAL'], axis=0) * 100
pct = pct.round(1)
pct['TOTAL'] = pct.sum(axis=1).round(1)
pct.to_csv(os.path.join(OUT, 'celltype_pct_by_source.csv'))
print(f"  Saved → {OUT}/celltype_pct_by_source.csv")

# ── cell_subclass: derive if missing ─────────────────────────────────────────
if 'cell_subclass' in obs.columns:
    print("\ncell_subclass already present in file.")
    cs_col = 'cell_subclass'
else:
    if 'cell_type' not in obs.columns:
        print("\nNeither cell_subclass nor cell_type found — cannot derive subclass.")
        cs_col = None
    else:
        print("\ncell_subclass missing; deriving from cell_type (source-specific mappers)...")
        # VELMESHEV uses source-specific labels; all others use CellxGene ontology
        vel_mask = obs['source'] == 'VELMESHEV'
        obs['cell_subclass'] = obs['cell_type'].map(map_cellxgene_subclass)
        obs.loc[vel_mask, 'cell_subclass'] = obs.loc[vel_mask, 'cell_type'].map(map_velmeshev_subclass)
        cs_col = 'cell_subclass'

        # Report labels that passed through unchanged (potential gaps in mapping dicts)
        passthrough = obs.loc[obs['cell_type'] == obs['cell_subclass'], 'cell_type'].unique()
        mapping_report = [
            "=== cell_type → cell_subclass mapping check ===\n",
            f"Total unique cell_type labels: {obs['cell_type'].nunique()}",
            f"Labels passed through unchanged (check if intentional): {len(passthrough)}\n",
            "--- Passed-through labels ---",
        ]
        for lbl in sorted(passthrough):
            src = obs.loc[obs['cell_type'] == lbl, 'source'].unique()
            n = (obs['cell_type'] == lbl).sum()
            mapping_report.append(f"  {lbl!r:55s}  n={n:6d}  sources={list(src)}")
        rpt_path = os.path.join(OUT, 'celltype_mapping_check.txt')
        with open(rpt_path, 'w') as f:
            f.write('\n'.join(mapping_report))
        print(f"  Saved mapping check → {rpt_path}")

if cs_col:
    print("\n--- cell_subclass × source ---")
    cs_counts = (
        obs.groupby(['source', cs_col], observed=True)
           .size()
           .unstack(fill_value=0)
    )
    cs_counts['TOTAL'] = cs_counts.sum(axis=1)
    cs_counts.loc['TOTAL'] = cs_counts.sum(axis=0)
    print(cs_counts.to_string())
    cs_counts.to_csv(os.path.join(OUT, 'cellsubclass_by_source.csv'))
    print(f"  Saved → {OUT}/cellsubclass_by_source.csv")

print("\nDone.")
