"""Quick sanity check: PC1 loadings vs AHBA C3+ weights.

For each pca_scanvi_loadings.csv given as argument, prints the Pearson r
between PC1 (and PC2/PC3) and the AHBA C3+ network's gene weights on the
shared gene IDs. Expects PC1↔C3+ |r| ≥ 0.4 if the GRN-PC1 hypothesis holds.

Usage:
    python -m scripts.pc1_vs_ahba.sanity_check_loadings <loadings.csv> [<loadings.csv> ...]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# Make sibling code importable when run as a script.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "code"))
from environment import get_environment  # noqa: E402
from hvg_investigation import setup_grn  # noqa: E402


def _load_ahba_c3pos_for_genes(ref_dir, gene_ids):
    """Build AHBA C3+ weights indexed by Ensembl ID, mapped via a stub adata."""
    import anndata as ad
    # Construct minimal adata with the requested var_names so map_grn_symbols_to_ensembl
    # has something to map against.
    stub = ad.AnnData(X=np.zeros((1, len(gene_ids)), dtype=np.float32))
    stub.var_names = list(gene_ids)
    ahba_full, _ = setup_grn(ref_dir, stub)
    c3pos = (ahba_full[ahba_full["Network"] == "C3+"]
             .set_index("Gene")["Importance"]
             .groupby(level=0).first())
    return c3pos


def check(csv_path: Path, ref_dir: str, pcs=("PC1", "PC2", "PC3")):
    print(f"\n=== {csv_path} ===")
    loadings = pd.read_csv(csv_path, index_col=0)
    print(f"  shape={loadings.shape}, columns={list(loadings.columns)[:5]}...")
    c3pos = _load_ahba_c3pos_for_genes(ref_dir, loadings.index)
    shared = loadings.index.intersection(c3pos.index)
    print(f"  shared genes with AHBA C3+: {len(shared):,}")
    if len(shared) < 50:
        print("  WARN: too few shared genes for a meaningful correlation.")
        return
    ref = c3pos.loc[shared].values
    for pc in pcs:
        if pc not in loadings.columns:
            continue
        x = loadings.loc[shared, pc].values
        mask = np.isfinite(x) & np.isfinite(ref)
        r = np.corrcoef(x[mask], ref[mask])[0, 1]
        print(f"  {pc} vs AHBA-C3+ : r = {r:+.3f}  (n={mask.sum():,})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("loadings", type=Path, nargs="+")
    args = p.parse_args()
    env = get_environment()
    ref_dir = env["ref_dir"]
    print(f"AHBA ref_dir : {ref_dir}")
    for csv in args.loadings:
        if not csv.exists():
            print(f"Missing: {csv}", file=sys.stderr)
            continue
        check(csv, ref_dir)


if __name__ == "__main__":
    main()
