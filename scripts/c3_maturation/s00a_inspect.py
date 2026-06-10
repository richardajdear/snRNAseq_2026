#!/usr/bin/env python3
"""C3-maturation Step 0 — quick inventory of the pseudobulk objects we will use.

Confirms which obs columns carry chemistry / age / depth and what layers exist,
so the depth-robustness harness (s00b) can be built against real columns.

Inline-safe: only reads small pseudobulk h5ads (<200 MB) per CLAUDE.md.
"""
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

B = Path("/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated")
FILES = {
    "Vel_ExN_by_donor":   B / "Vel_prepost_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
    "Vel_by_cell_class":   B / "Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class.h5ad",
    "Vel_exc_by_celltype": B / "Vel_prepost_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad",
}

INTEREST = ["age_years", "individual", "donor_id", "source", "dataset",
            "source-chemistry", "chemistry", "cell_class", "cell_type_aligned",
            "cell_type_raw", "n_cells", "n_counts", "total_counts", "n_genes",
            "region", "sex"]


def main():
    for name, path in FILES.items():
        print(f"\n{'='*70}\n{name}\n  {path}")
        if not path.exists():
            print("  MISSING"); continue
        a = ad.read_h5ad(path)
        print(f"  shape: {a.shape}   layers: {list(a.layers.keys())}")
        print(f"  var_names[:3]: {list(a.var_names[:3])}")
        print(f"  var cols: {list(a.var.columns)}")
        print(f"  obs cols ({len(a.obs.columns)}): {list(a.obs.columns)}")
        for c in INTEREST:
            if c in a.obs.columns:
                s = a.obs[c]
                if s.dtype.kind in "Ob" or str(s.dtype) == "category":
                    vc = s.astype(str).value_counts().head(8)
                    print(f"  [{c}] {dict(vc)}")
                else:
                    sn = pd.to_numeric(s, errors="coerce")
                    print(f"  [{c}] num: min={sn.min():.3g} med={sn.median():.3g} "
                          f"max={sn.max():.3g} nan={sn.isna().sum()}")
        # depth proxy from counts layer
        if "counts" in a.layers:
            X = a.layers["counts"]
            tot = np.asarray(X.sum(axis=1)).ravel()
            print(f"  counts total/sample: min={tot.min():.3g} med={np.median(tot):.3g} max={tot.max():.3g}")


if __name__ == "__main__":
    main()
