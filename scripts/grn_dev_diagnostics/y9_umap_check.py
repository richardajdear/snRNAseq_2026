#!/usr/bin/env python3
"""
Y9 — is the bad-looking integration real (in X_scVI) or a UMAP-projection
artefact? The dev30 pipeline never ran the umap step, yet the file has an
X_umap of unknown provenance — y5 plotted THAT. Here we recompute a proper
UMAP FROM X_scVI (sc.pp.neighbors use_rep=X_scVI -> sc.tl.umap) and show it
beside the in-file X_umap, coloured by batch (source-chemistry) and age, for
the dev30 all-cell run. If the recomputed UMAP mixes batches but the in-file
one does not, the embedding is fine and the earlier picture was a projection
artefact.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=01:00:00 --mem=200G --cpus-per-task=16 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y9_umap_check.py
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
PATH = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"
N = 150000


def colscatter(ax, U, lab, title, cat=True):
    if cat:
        s = pd.Series(lab).astype(str); u = sorted(s.unique())
        cm = plt.cm.tab10(np.linspace(0, 1, max(len(u), 3)))
        for i, c in enumerate(u):
            m = (s == c).values
            ax.scatter(U[m, 0], U[m, 1], s=2, color=cm[i], alpha=0.5, label=f"{c} ({m.sum()})", linewidths=0)
        ax.legend(markerscale=4, fontsize=7)
    else:
        sc_ = ax.scatter(U[:, 0], U[:, 1], s=2, c=lab, cmap="viridis", alpha=0.6, linewidths=0)
        plt.colorbar(sc_, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def main():
    a = ad.read_h5ad(PATH, backed="r")
    n = a.n_obs
    idx = np.sort(np.random.default_rng(0).choice(n, min(N, n), replace=False))
    obs = a.obs.iloc[idx]
    Z = np.asarray(a.obsm["X_scVI"][:])[idx]
    Uin = np.asarray(a.obsm["X_umap"][:])[idx] if "X_umap" in a.obsm else None
    batch = obs["source-chemistry"].astype(str).values if "source-chemistry" in obs else obs["source"].astype(str).values
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    logage = np.log2(np.clip(age, 0.01, None) + 0.74)

    # proper UMAP recomputed FROM X_scVI
    s = ad.AnnData(np.zeros((len(idx), 1))); s.obsm["X_scVI"] = Z
    sc.pp.neighbors(s, use_rep="X_scVI", n_neighbors=30)
    sc.tl.umap(s)
    Urec = s.obsm["X_umap"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    if Uin is not None:
        colscatter(axes[0, 0], Uin, batch, "IN-FILE X_umap — batch")
        colscatter(axes[0, 1], Uin, logage, "IN-FILE X_umap — age", cat=False)
    else:
        axes[0, 0].set_title("no in-file X_umap"); axes[0, 1].axis("off")
    colscatter(axes[1, 0], Urec, batch, "RECOMPUTED from X_scVI — batch")
    colscatter(axes[1, 1], Urec, logage, "RECOMPUTED from X_scVI — age", cat=False)
    fig.suptitle("Y9 — dev30 all-cell: in-file X_umap vs proper X_scVI UMAP (is bad mixing real or a projection artefact?)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "y9_umap_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # correlation between in-file and recomputed (Procrustes-free: just report if in-file looks like X_scVI structure)
    if Uin is not None:
        # quick: does in-file X_umap separate batches more than recomputed? batch kNN purity in 2D
        from sklearn.neighbors import NearestNeighbors
        b = pd.factorize(batch)[0]
        def purity(U):
            nn = NearestNeighbors(n_neighbors=16).fit(U); _, ind = nn.kneighbors(U)
            return float((b[ind[:, 1:]] == b[:, None]).mean())
        print(f"in-file X_umap 2D same-batch-kNN purity = {purity(Uin):.3f}")
        print(f"recomputed   X_umap 2D same-batch-kNN purity = {purity(Urec):.3f}")
        print("(lower purity = better mixed; if in-file >> recomputed, the in-file UMAP misrepresents X_scVI)")
    print("saved y9_umap_check.png")


if __name__ == "__main__":
    main()
