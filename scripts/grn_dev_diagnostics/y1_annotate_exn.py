#!/usr/bin/env python3
"""
Y1 — call ExN cells by clustering the Stage-1 joint V3 embedding.

Cluster-level marker annotation (ambient/label-robust; no scANVI label transfer):
  1. Leiden on the unsupervised scVI latent X_scVI.
  2. Score each cluster on ExN / InN / glia marker panels (mean log1p-CP10k +
     detection rate over HVG-space counts).
  3. Label clusters; ExN = high SLC17A7/SATB2/RBFOX3, low GAD1/2/SLC32A1 & glia.
  4. Write per-source ExN cell-ID parquets for the Stage-3 cell_id_filter,
     a per-cluster annotation table, and a QC figure.

Cross-tabs each cluster against the native cell_class and the existing
marker_annotation for transparency (flags PsychAD clusters a reference would
call InN).

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=06:00:00 --mem=300G \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y1_annotate_exn.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("scripts/grn_dev_diagnostics/outputs")
INTEGRATED = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"
LEIDEN_RES = 1.0

PANELS = {
    "ExN":  ["SLC17A7", "SATB2", "RBFOX3", "NEUROD2", "NEUROD6", "TBR1", "SLC17A6"],
    "InN":  ["GAD1", "GAD2", "SLC32A1", "LHX6", "ADARB2", "DLX1", "DLX2"],
    "Astro": ["AQP4", "GFAP", "SLC1A3", "ALDH1L1"],
    "Oligo": ["MBP", "PLP1", "MOBP", "MOG"],
    "OPC":   ["PDGFRA", "VCAN", "OLIG1", "OLIG2"],
    "Micro": ["CX3CR1", "P2RY12", "CSF1R"],
}
NONEXN = ["InN", "Astro", "Oligo", "OPC", "Micro"]


def resolve(adata, symbols):
    """Map gene symbols -> var positions via var['gene_symbol'] (fallback var_names)."""
    sym = adata.var["gene_symbol"].astype(str) if "gene_symbol" in adata.var.columns \
        else pd.Series(adata.var_names, index=adata.var_names)
    lut = {}
    for pos, s in enumerate(sym.values):
        lut.setdefault(s, pos)
    for pos, vn in enumerate(adata.var_names.astype(str)):
        lut.setdefault(vn, pos)
    out = {}
    for g in symbols:
        if g in lut:
            out[g] = lut[g]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Loading {INTEGRATED} (backed) ...")
    a = sc.read_h5ad(INTEGRATED, backed="r")
    obs = a.obs.copy()
    Z = np.asarray(a.obsm["X_scVI"][:])
    print(f"  {Z.shape[0]:,} cells, latent dim {Z.shape[1]}")

    # X is HVG-space counts; load just X (not the dense normalized layers)
    Xc = a.X[:]
    Xc = sp.csr_matrix(Xc)
    var = a.var.copy()
    adata = ad.AnnData(X=Xc, obs=obs, var=var)
    adata.obsm["X_scVI"] = Z
    del a

    # CP10k log1p for marker scoring
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ---- cluster on the latent ----
    print("Neighbors + Leiden on X_scVI ...")
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
    try:
        sc.tl.leiden(adata, resolution=LEIDEN_RES, flavor="igraph",
                     n_iterations=2, directed=False)
    except TypeError:
        sc.tl.leiden(adata, resolution=LEIDEN_RES)
    nclust = adata.obs["leiden"].nunique()
    print(f"  {nclust} Leiden clusters")

    # ---- per-cluster marker scores ----
    panel_pos = {p: resolve(adata, gs) for p, gs in PANELS.items()}
    for p, d in panel_pos.items():
        print(f"  panel {p}: {len(d)}/{len(PANELS[p])} markers found "
              f"({sorted(d.keys())})")
    Xlog = adata.X  # CP10k log1p, sparse
    cl = adata.obs["leiden"].values
    clusters = sorted(adata.obs["leiden"].cat.categories, key=lambda x: int(x))

    rows = []
    panel_score = {p: np.zeros(len(clusters)) for p in PANELS}
    for ci, c in enumerate(clusters):
        idx = np.where(cl == c)[0]
        row = {"leiden": c, "n_cells": len(idx)}
        for p, d in panel_pos.items():
            if not d:
                row[f"{p}_mean"] = np.nan
                continue
            sub = Xlog[idx][:, list(d.values())]
            m = float(np.asarray(sub.mean()))
            row[f"{p}_mean"] = m
            panel_score[p][ci] = m
        rows.append(row)
    cdf = pd.DataFrame(rows).set_index("leiden")

    # z-score each panel across clusters, then call
    zs = {}
    for p in PANELS:
        v = cdf[f"{p}_mean"].values.astype(float)
        zs[p] = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
        cdf[f"{p}_z"] = zs[p]
    exn_z = zs["ExN"]
    nonexn_z = np.nanmax(np.vstack([zs[p] for p in NONEXN]), axis=0)
    cdf["nonExN_max_z"] = nonexn_z
    # ExN cluster: ExN panel clearly highest and above the strongest non-ExN panel
    is_exn = (exn_z > 0.25) & (exn_z - nonexn_z > 0.25)
    cdf["call"] = np.where(is_exn, "ExN",
                  np.array([NONEXN[i] for i in np.nanargmax(
                      np.vstack([zs[p] for p in NONEXN]), axis=0)]))
    cdf.loc[is_exn, "call"] = "ExN"

    # cross-tab vs native labels for transparency
    if "cell_class" in obs.columns:
        ct = pd.crosstab(adata.obs["leiden"], obs["cell_class"])
        ct.to_csv(OUT / "y1_cluster_vs_cellclass.csv")
        cdf["top_native_cell_class"] = ct.idxmax(axis=1).reindex(cdf.index)
    if "marker_annotation" in obs.columns:
        cm = pd.crosstab(adata.obs["leiden"], obs["marker_annotation"])
        cdf["top_marker_annotation"] = cm.idxmax(axis=1).reindex(cdf.index)

    cdf.to_csv(OUT / "y1_cluster_annotation.csv")
    print("\nPer-cluster calls:")
    print(cdf[["n_cells", "ExN_z", "nonExN_max_z", "call"]
              + [c for c in ["top_native_cell_class"] if c in cdf.columns]].to_string())

    # ---- ExN cell IDs, split by source ----
    exn_clusters = cdf.index[cdf["call"] == "ExN"].tolist()
    adata.obs["y1_call"] = adata.obs["leiden"].map(cdf["call"]).astype(str)
    exn_mask = adata.obs["leiden"].isin(exn_clusters).values
    print(f"\nExN clusters: {exn_clusters}")
    print(f"ExN cells: {exn_mask.sum():,} / {len(adata):,} "
          f"({100*exn_mask.sum()/len(adata):.1f}%)")

    src = obs["source"].astype(str).values if "source" in obs.columns else \
        np.array(["ALL"] * len(adata))
    for s in np.unique(src):
        ids = adata.obs_names[exn_mask & (src == s)]
        df = pd.DataFrame(index=pd.Index(ids, name="cell_id"))
        df.to_parquet(OUT / f"y1_exn_cellids_{s}.parquet")
        print(f"  {s}: {len(ids):,} ExN cells -> y1_exn_cellids_{s}.parquet")
    # combined too
    pd.DataFrame(index=pd.Index(adata.obs_names[exn_mask], name="cell_id")).to_parquet(
        OUT / "y1_exn_cellids_ALL.parquet")

    # composition by source x age for sanity
    if "age_years" in obs.columns:
        comp = (adata.obs.assign(is_exn=exn_mask)
                .groupby([obs["source"].astype(str)])["is_exn"]
                .agg(["mean", "size"]))
        comp.to_csv(OUT / "y1_exn_fraction_by_source.csv")
        print("\nExN fraction by source:\n", comp.to_string())

    # ---- QC figure: cluster x panel heatmap + UMAP subsample ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    hm = cdf[[f"{p}_z" for p in PANELS]].copy()
    hm.columns = list(PANELS.keys())
    im = axes[0].imshow(hm.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axes[0].set_xticks(range(len(PANELS)))
    axes[0].set_xticklabels(list(PANELS.keys()))
    axes[0].set_yticks(range(len(cdf)))
    axes[0].set_yticklabels([f"{i} [{cdf.loc[i,'call']}] n={cdf.loc[i,'n_cells']}"
                             for i in cdf.index], fontsize=7)
    axes[0].set_title("cluster × marker-panel z-score (ExN call in brackets)")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="z-score")

    # UMAP on a subsample for the QC scatter
    n_sub = min(120000, len(adata))
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(adata), n_sub, replace=False)
    sub = adata[sub_idx].copy()
    sc.pp.neighbors(sub, use_rep="X_scVI", n_neighbors=15)
    sc.tl.umap(sub)
    U = sub.obsm["X_umap"]
    callcolor = {"ExN": "#C0392B"}
    cols = [callcolor.get(c, "#BDC3C7") for c in sub.obs["y1_call"]]
    axes[1].scatter(U[:, 0], U[:, 1], s=2, c=cols, alpha=0.5, linewidths=0)
    axes[1].set_title(f"UMAP (subsample {n_sub:,}) — ExN red, other grey")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.suptitle("Y1 — cluster-based ExN annotation of the joint V3 embedding",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "y1_exn_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved y1_cluster_annotation.csv, y1_exn_cellids_*.parquet, "
          f"y1_exn_clusters.png in {OUT}")


if __name__ == "__main__":
    main()
