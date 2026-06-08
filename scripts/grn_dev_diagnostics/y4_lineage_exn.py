#!/usr/bin/env python3
"""
Y4 — a principled ExN definition using the embedding + lineage TFs, replacing
the arbitrary GAD>=10 threshold.

The hard cases are immature neurons (RBFOX3+/DCX+, low SLC17A7, some GAD<10).
Excitatory vs inhibitory LINEAGE is separable in these cells by lineage TFs
even before SLC17A7 turns on:
  excitatory lineage: NEUROD2/6, TBR1, EOMES, SLC17A7/6, SATB2, BCL11B, FEZF2
  inhibitory lineage: GAD1/2, SLC32A1, DLX1/2/5, LHX6, ADARB2, SP8

Question (user): do the GAD<10 "immature ExN" candidates actually cluster with
the InN cells in the latent space, or with ExN? Method:
  1. Define confident anchors: InN (GAD>=10), ExN (SLC17A7/6 >=1 & GAD<10);
     glia by glia markers; ambiguous = neuron (RBFOX3/DCX) but neither anchor.
  2. kNN graph on X_scVI -> for every cell, fraction of anchor neighbours that
     are ExN vs InN (a data-driven, threshold-free lineage vote).
  3. Assign ambiguous cells ExN/InN by neighbour vote; cross-check vs lineage
     TF scores. Report what happens to the young-PsychAD "gained" cells.
  4. Write the principled ExN id set per source (confident ExN + ExN-voted
     ambiguous, minus glia/InN) for the Stage-3 rebuild.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=02:00:00 --mem=300G --cpus-per-task=16 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y4_lineage_exn.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("scripts/grn_dev_diagnostics/outputs")
STAGE1 = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"

EXC_TF = ["NEUROD2", "NEUROD6", "TBR1", "EOMES", "SLC17A7", "SLC17A6", "SATB2", "BCL11B", "FEZF2"]
INH_TF = ["GAD1", "GAD2", "SLC32A1", "DLX1", "DLX2", "DLX5", "LHX6", "ADARB2", "SP8"]
GLIA = ["AQP4", "GFAP", "MBP", "PLP1", "CX3CR1", "P2RY12", "PDGFRA"]
NEUR = ["RBFOX3", "DCX", "RBFOX1"]
ALL = sorted(set(EXC_TF + INH_TF + GLIA + NEUR))
KNN = 30
AGE_BINS = [(-1, 0), (0, 1), (1, 10), (10, 18), (18, 31)]
AGE_LABS = ["prenatal", "0-1", "1-10", "10-18", "18-30"]


def agebin(a):
    for (lo, hi), lab in zip(AGE_BINS, AGE_LABS):
        if lo <= a < hi:
            return lab
    return "NA"


def main():
    a = sc.read_h5ad(STAGE1, backed="r")
    sym = a.var["gene_symbol"].astype(str)
    pos = {}
    for i, s in enumerate(sym.values):
        pos.setdefault(s, i)
    have = [g for g in ALL if g in pos]
    print("lineage markers present:", {g: (g in pos) for g in ALL})
    Z = np.asarray(a.obsm["X_scVI"][:])
    sub = a[:, [pos[g] for g in have]].to_memory()
    Xs = sp.csr_matrix(sub.X)
    # CP10k log1p for scores; raw for thresholds
    raw = {g: np.asarray(Xs[:, i].todense()).ravel() for i, g in enumerate(have)}
    tot = np.asarray(a.X[:].sum(1)).ravel() if False else None  # avoid full read; use HVG sum proxy
    # library size proxy = sum of all HVG counts (read in chunks)
    print("computing library size (chunked) ...")
    n = a.n_obs; ls = np.zeros(n); step = 100000
    for s0 in range(0, n, step):
        ls[s0:s0+step] = np.asarray(a.X[s0:s0+step].sum(1)).ravel()
    inv = 1.0 / np.where(ls > 0, ls, 1)

    def score(genes):
        gs = [g for g in genes if g in raw]
        M = np.vstack([np.log1p(raw[g] * inv * 1e4) for g in gs]).T
        return M.mean(1)
    exc = score(EXC_TF); inh = score(INH_TF)
    gad_max = np.maximum.reduce([raw.get(g, np.zeros(n)) for g in ["GAD1", "GAD2", "SLC32A1"]])
    slc17 = np.maximum.reduce([raw.get(g, np.zeros(n)) for g in ["SLC17A7", "SLC17A6"]])
    glia_det = np.maximum.reduce([raw.get(g, np.zeros(n)) for g in GLIA]) >= 1
    is_neuron = (raw.get("RBFOX3", np.zeros(n)) >= 1) | (raw.get("DCX", np.zeros(n)) >= 1)

    conf_inn = is_neuron & (gad_max >= 10)
    conf_exn = is_neuron & (gad_max < 10) & (slc17 >= 1)
    glia = (~is_neuron) & glia_det
    ambiguous = is_neuron & (~conf_inn) & (~conf_exn)
    print(f"anchors: conf_ExN={conf_exn.sum():,} conf_InN={conf_inn.sum():,} "
          f"glia={glia.sum():,} ambiguous_neuron={ambiguous.sum():,} "
          f"non-neuron-non-glia={(~is_neuron & ~glia).sum():,}")

    # kNN graph on latent
    print("neighbors on X_scVI ...")
    adata = ad.AnnData(X=sp.csr_matrix((n, 1)))
    adata.obsm["X_scVI"] = Z
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=KNN)
    G = adata.obsp["distances"]  # kNN-1 nonzeros per row
    G = (G > 0).astype(np.float32)
    # neighbour anchor fractions
    ie = G.dot(conf_exn.astype(np.float32))
    ii = G.dot(conf_inn.astype(np.float32))
    denom = ie + ii
    exn_nb_frac = np.divide(ie, denom, out=np.full(n, np.nan), where=denom > 0)
    # data-driven vote for ambiguous cells (and all): ExN if more ExN anchors nearby
    vote_exn = exn_nb_frac >= 0.5

    obs = a.obs[["source", "age_years", "cell_class"]].copy()
    obs["agebin"] = obs["age_years"].map(agebin)
    obs["exc"] = exc; obs["inh"] = inh
    obs["grp"] = np.select([conf_exn, conf_inn, glia, ambiguous],
                           ["conf_ExN", "conf_InN", "glia", "ambiguous"], default="other")
    obs["exn_nb_frac"] = exn_nb_frac
    obs["vote_exn"] = vote_exn

    # principled ExN = confident ExN OR (ambiguous voted ExN); never glia/conf_InN
    principled_exn = conf_exn | (ambiguous & vote_exn)
    obs["principled_exn"] = principled_exn
    print(f"\nprincipled ExN total: {principled_exn.sum():,}")

    # how do ambiguous cells vote? overall and lineage-TF check
    amb = obs[ambiguous]
    print(f"\nambiguous neurons: {len(amb):,}; voted ExN={amb['vote_exn'].mean()*100:.1f}%")
    print("  ambiguous mean exc-lineage={:.3f} vs inh-lineage={:.3f}".format(
        amb['exc'].mean(), amb['inh'].mean()))
    print("  ambiguous voted-ExN: exc={:.3f} inh={:.3f}; voted-InN: exc={:.3f} inh={:.3f}".format(
        amb[amb.vote_exn]['exc'].mean(), amb[amb.vote_exn]['inh'].mean(),
        amb[~amb.vote_exn]['exc'].mean(), amb[~amb.vote_exn]['inh'].mean()))

    # the young-PsychAD "gained" cells (marker-rule view): ambiguous PsychAD <10y
    yp = obs[(obs.source == "PSYCHAD") & obs.agebin.isin(["0-1", "1-10"]) & ambiguous]
    print(f"\nyoung PsychAD (<10y) ambiguous: {len(yp):,}; "
          f"neighbour-voted ExN={yp['vote_exn'].mean()*100:.1f}%")
    print("  their mean exc-lineage={:.3f} inh-lineage={:.3f}".format(yp['exc'].mean(), yp['inh'].mean()))

    # ExN fraction per source x agebin under the principled definition
    tab = obs.groupby(["source", "agebin"]).agg(
        n=("principled_exn", "size"),
        principled=("principled_exn", "mean")).round(3)
    tab.to_csv(OUT / "y4_principled_exn_fraction.csv")
    print("\nprincipled ExN fraction by source x agebin:\n", tab.to_string())

    # write principled ExN id sets (only for kept chemistry V3 cells -> all here)
    for s in ["VELMESHEV", "PSYCHAD"]:
        ids = a.obs_names[(obs.source.values == s) & principled_exn]
        pd.DataFrame(index=pd.Index(ids, name="cell_id")).to_parquet(
            OUT / f"y4_principledexn_cellids_{s}.parquet")
        print(f"  wrote y4_principledexn_cellids_{s}.parquet: {len(ids):,}")
    obs[["source", "age_years", "agebin", "grp", "exc", "inh",
         "exn_nb_frac", "vote_exn", "principled_exn"]].to_parquet(
        OUT / "y4_percell_lineage.parquet")

    # ---- figures ----
    rng = np.random.default_rng(0)
    sidx = rng.choice(n, min(120000, n), replace=False)
    sa = ad.AnnData(X=sp.csr_matrix((len(sidx), 1))); sa.obsm["X_scVI"] = Z[sidx]
    sc.pp.neighbors(sa, use_rep="X_scVI", n_neighbors=15); sc.tl.umap(sa)
    U = sa.obsm["X_umap"]; grp = obs["grp"].values[sidx]
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    cmap = {"conf_ExN": "#C0392B", "conf_InN": "#2471A3", "glia": "#BDC3C7",
            "ambiguous": "#F39C12", "other": "#EAECEE"}
    for grp_name, c in cmap.items():
        m = grp == grp_name
        ax[0].scatter(U[m, 0], U[m, 1], s=2, c=c, alpha=0.5, label=grp_name, linewidths=0)
    ax[0].legend(markerscale=4, fontsize=8); ax[0].set_title("anchors + ambiguous on UMAP")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    # ambiguous colored by neighbour ExN fraction
    ma = grp == "ambiguous"
    scat = ax[1].scatter(U[ma, 0], U[ma, 1], s=4, c=obs["exn_nb_frac"].values[sidx][ma],
                         cmap="RdBu_r", vmin=0, vmax=1)
    ax[1].set_title("ambiguous: neighbour ExN-anchor fraction\n(red=ExN-like, blue=InN-like)")
    ax[1].set_xticks([]); ax[1].set_yticks([]); fig.colorbar(scat, ax=ax[1], fraction=0.046)
    # exc vs inh lineage scatter for ambiguous, colored by vote
    amb_s = obs.iloc[sidx][ma]
    ax[2].scatter(amb_s["inh"], amb_s["exc"], s=4,
                  c=np.where(amb_s["vote_exn"], "#C0392B", "#2471A3"), alpha=0.5)
    lim = max(amb_s["inh"].max(), amb_s["exc"].max())
    ax[2].plot([0, lim], [0, lim], "k--", lw=0.6)
    ax[2].set_xlabel("inhibitory-lineage TF score"); ax[2].set_ylabel("excitatory-lineage TF score")
    ax[2].set_title("ambiguous cells: lineage TFs (color=neighbour vote)")
    fig.suptitle("Y4 — principled ExN: do GAD<10 immature candidates sit with ExN or InN?",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "y4_lineage_exn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved y4_principled_exn_fraction.csv, y4_principledexn_cellids_*.parquet, "
          f"y4_percell_lineage.parquet, y4_lineage_exn.png in {OUT}")


if __name__ == "__main__":
    main()
