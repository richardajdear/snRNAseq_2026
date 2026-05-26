"""Extract PC1 (and other PCs) gene loadings from an existing integrated.h5ad.

Use this when you want PC1 gene weights but the integration was run before
the scVI pipeline started saving loadings automatically. Computes a
randomised SVD on the log1p of the scanvi_normalized layer (HVG subset)
and writes pca_scanvi_loadings.csv next to integrated.h5ad.

Uses sklearn.utils.extmath.randomized_svd instead of IncrementalPCA — the
latter was unexpectedly slow on float32 (30k × 5k) data; randomised SVD
finishes in <10 s for typical integrated.h5ad sizes.

Usage:
    python -m scripts.pc1_vs_ahba.extract_loadings <path/to/integrated.h5ad> [--n-pcs 20]
"""

import argparse
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd


def extract(h5ad_path: Path, n_pcs: int = 20,
            layer: str = "scanvi_normalized"):
    t0 = time.time()
    print(f"Loading {h5ad_path} (backed, layer={layer})", flush=True)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    if layer not in adata.layers:
        raise ValueError(
            f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}"
        )

    hvg_mask = adata.var.get("highly_variable", None)
    if hvg_mask is not None and hvg_mask.any() and not hvg_mask.all():
        hvg_idx = np.where(np.asarray(hvg_mask))[0]
        use_all_cols = False
    else:
        hvg_idx = None
        use_all_cols = True
    n_cols = adata.n_vars if use_all_cols else len(hvg_idx)
    print(f"  n_obs={adata.n_obs:,} n_features_used={n_cols:,} n_pcs={n_pcs}",
          flush=True)

    # Materialise the layer in memory once. For typical pipeline outputs the
    # scanvi_normalized layer is dense float32 — ~1.8 GB at (30k, 15k).
    # Prefer the .npy backup next to integrated.h5ad if present (faster than
    # round-tripping through HDF5 for the dense case).
    npy_path = h5ad_path.parent / f"{layer}.npy"
    if npy_path.exists():
        print(f"Loading {npy_path} (faster path)...", flush=True)
        mat = np.load(npy_path)
    else:
        print(f"Loading {layer} layer from h5ad...", flush=True)
        block = adata.layers[layer][:]
        if sp.issparse(block):
            block = block.toarray()
        mat = np.asarray(block, dtype=np.float32)
    print(f"  shape={mat.shape} dtype={mat.dtype}  [{time.time() - t0:.1f}s]",
          flush=True)

    print("Slicing HVGs and log1p...", flush=True)
    if use_all_cols:
        X = np.log1p(mat).astype(np.float32, copy=True)
    else:
        X = np.log1p(mat[:, hvg_idx]).astype(np.float32, copy=True)
    print(f"  shape={X.shape}  [{time.time() - t0:.1f}s]", flush=True)
    del mat

    print("Centering...", flush=True)
    X -= X.mean(axis=0, dtype=np.float64).astype(np.float32)

    print(f"randomized_svd n_pcs={n_pcs}...", flush=True)
    _, S, Vt = randomized_svd(X, n_components=n_pcs, random_state=42, n_iter=4)
    n_samples = X.shape[0]
    total_var = X.var(axis=0).sum()
    explained = (S ** 2) / ((n_samples - 1) * total_var)
    print(f"  done  [{time.time() - t0:.1f}s]  top-5 explained_var_ratio="
          f"{explained[:5].round(4).tolist()}", flush=True)

    columns = [f"PC{i + 1}" for i in range(n_pcs)]
    comps = Vt.astype(np.float32)  # (n_pcs, n_features)
    if use_all_cols:
        loadings = comps.T
    else:
        loadings = np.full((adata.n_vars, n_pcs), np.nan, dtype=np.float32)
        loadings[hvg_idx, :] = comps.T

    df = pd.DataFrame(loadings, index=adata.var_names, columns=columns).dropna(how="all")
    out_path = h5ad_path.parent / "pca_scanvi_loadings.csv"
    df.to_csv(out_path, index_label="gene_id")
    print(f"Wrote {out_path} (shape={df.shape})  [{time.time() - t0:.1f}s total]",
          flush=True)

    adata.file.close()
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5ad", type=Path)
    p.add_argument("--n-pcs", type=int, default=20)
    p.add_argument("--layer", default="scanvi_normalized")
    args = p.parse_args()
    if not args.h5ad.exists():
        print(f"File not found: {args.h5ad}", file=sys.stderr)
        sys.exit(1)
    extract(args.h5ad, n_pcs=args.n_pcs, layer=args.layer)


if __name__ == "__main__":
    main()
