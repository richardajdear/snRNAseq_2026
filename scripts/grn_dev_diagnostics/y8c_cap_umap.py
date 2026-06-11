#!/usr/bin/env python3
"""
Y8c — render a batch/age UMAP for the controlled cap run
(VelPsychAD_V3_allcell_dev30_psyCAP200k), recomputed FROM X_scVI exactly like
y9, and save it into the run's own plots/ folder so it sits beside the good
run's plots/all/umaps_latent.png for visual review. Also drops a copy in our
diagnostics outputs/.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30_psyCAP200k"
PATH = f"{RUN}/scvi_output/integrated.h5ad"
PLOTS = Path(RUN) / "plots" / "all"; PLOTS.mkdir(parents=True, exist_ok=True)
OUT = Path("scripts/grn_dev_diagnostics/outputs")
N = 150000


def colscatter(ax, U, lab, title, cat=True):
    if cat:
        s = pd.Series(lab).astype(str); u = sorted(s.unique())
        cm = plt.cm.tab10(np.linspace(0, 1, max(len(u), 3)))
        for i, c in enumerate(u):
            m = (s == c).values
            ax.scatter(U[m, 0], U[m, 1], s=2, color=cm[i], alpha=0.5,
                       label=f"{c} ({m.sum()})", linewidths=0)
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
    batch = obs["source-chemistry"].astype(str).values if "source-chemistry" in obs else obs["source"].astype(str).values
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    logage = np.log2(np.clip(age, 0.01, None) + 0.74)
    s = ad.AnnData(np.zeros((len(idx), 1))); s.obsm["X_scVI"] = Z
    sc.pp.neighbors(s, use_rep="X_scVI", n_neighbors=30); sc.tl.umap(s)
    U = s.obsm["X_umap"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    colscatter(axes[0], U, batch, "cap PsychAD@200k — batch (mixing ALL=0.158 / POST=0.228)")
    colscatter(axes[1], U, logage, "cap PsychAD@200k — log2 age", cat=False)
    fig.suptitle("Controlled one-variable test: dev30 with PsychAD capped at 200k (X_scVI UMAP)",
                 fontweight="bold")
    fig.tight_layout()
    for p in (PLOTS / "umaps_latent_capcheck.png", OUT / "y8c_cap_umap.png"):
        fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {PLOTS/'umaps_latent_capcheck.png'}")
    print(f"saved {OUT/'y8c_cap_umap.png'}")
    print("DONE")


if __name__ == "__main__":
    main()
