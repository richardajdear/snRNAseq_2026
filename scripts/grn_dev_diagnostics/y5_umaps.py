#!/usr/bin/env python3
"""
Y5 — UMAPs for the new embeddings, for visual inspection. Uses the in-file
obsm['X_umap'] if present, else computes UMAP on X_scVI (subsample). Colours by
source, dataset, age, native cell_class, chemistry, and principled-ExN call.

Edit EMBEDDINGS to add the 4-dataset run once it lands.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=02:00:00 --mem=200G --cpus-per-task=16 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y5_umaps.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("scripts/grn_dev_diagnostics/outputs")
BASE = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated"
EMBEDDINGS = {
    "allcell": f"{BASE}/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad",
    "ExN":     f"{BASE}/VelPsychAD_V3_ExN_dev30/scvi_output/integrated.h5ad",
    "4dataset": f"{BASE}/VelWangZhuPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad",
}
N_PLOT = 150000


def exn_ids():
    s = set()
    for src in ["VELMESHEV", "PSYCHAD"]:
        f = OUT / f"y4_principledexn_cellids_{src}.parquet"
        if f.exists():
            s |= set(pd.read_parquet(f).index.astype(str))
    return s


def cat_scatter(ax, U, labels, title, palette=None, legend=True):
    cats = pd.Series(labels).astype(str)
    uniq = sorted(cats.unique())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 3)))
    pal = palette or {c: cmap[i] for i, c in enumerate(uniq)}
    for c in uniq:
        m = (cats == c).values
        ax.scatter(U[m, 0], U[m, 1], s=2, color=pal.get(c, "#999999"), alpha=0.5,
                   label=f"{c} ({m.sum()})", linewidths=0)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(markerscale=4, fontsize=6, loc="best")


def run_one(name, path):
    if not Path(path).exists():
        print(f"[{name}] missing {path}; skip"); return
    print(f"[{name}] loading {path}")
    a = ad.read_h5ad(path, backed="r")
    n = a.n_obs
    idx = np.random.default_rng(0).choice(n, min(N_PLOT, n), replace=False)
    idx.sort()
    obs = a.obs.iloc[idx].copy()
    if "X_umap" in a.obsm:
        U = np.asarray(a.obsm["X_umap"][:])[idx]
        print(f"[{name}] using in-file X_umap")
    else:
        Z = np.asarray(a.obsm["X_scVI"][:])[idx]
        s = ad.AnnData(np.zeros((len(idx), 1))); s.obsm["X_scVI"] = Z
        sc.pp.neighbors(s, use_rep="X_scVI", n_neighbors=15); sc.tl.umap(s)
        U = s.obsm["X_umap"]; print(f"[{name}] computed UMAP on subsample")
    eids = exn_ids()
    is_exn = obs.index.astype(str).isin(eids)
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    logage = np.log2(np.clip(age, 0.01, None) + 0.74)

    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    cat_scatter(axes[0, 0], U, obs["source"], "source")
    if "dataset" in obs:
        cat_scatter(axes[0, 1], U, obs["dataset"], "dataset (Herring/Ramos/U01/Wang/Zhu/PsychAD)")
    else:
        axes[0, 1].axis("off")
    sccol = axes[0, 2].scatter(U[:, 0], U[:, 1], s=2, c=logage, cmap="viridis", alpha=0.6, linewidths=0)
    axes[0, 2].set_title("age (log2 post-conception)"); axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])
    fig.colorbar(sccol, ax=axes[0, 2], fraction=0.046)
    cat_scatter(axes[1, 0], U, obs["cell_class"], "native cell_class")
    schem = "source-chemistry" if "source-chemistry" in obs else "chemistry"
    cat_scatter(axes[1, 1], U, obs[schem], schem)
    axes[1, 2].scatter(U[~is_exn, 0], U[~is_exn, 1], s=2, c="#D5DBDB", alpha=0.4, linewidths=0, label="other")
    axes[1, 2].scatter(U[is_exn, 0], U[is_exn, 1], s=2, c="#C0392B", alpha=0.5, linewidths=0,
                       label=f"principled ExN ({is_exn.sum()})")
    axes[1, 2].set_title("principled-ExN call"); axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
    axes[1, 2].legend(markerscale=4, fontsize=7)
    fig.suptitle(f"UMAP — {name}  ({n:,} cells, plotting {len(idx):,})", fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / f"y5_umap_{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[{name}] saved y5_umap_{name}.png")


def main():
    for name, path in EMBEDDINGS.items():
        try:
            run_one(name, path)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[{name}] failed: {e}")


if __name__ == "__main__":
    main()
