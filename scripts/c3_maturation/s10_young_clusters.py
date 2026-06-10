#!/usr/bin/env python3
"""Young-donor (<5y) cluster-based labeling + embedding-quality diagnostic (sbatch).

Addresses:
  1. SLC17A7-SPECIFIC excitatory definition (not pan-neuronal RBFOX3).
  2. CLUSTER-based labels (Leiden), not per-cell marker cutoffs — each cluster
     labeled by its dominant marker SIGNATURE (averages out dropout).
  3. Fixed native broad labels (derived from native_fine; Velmeshev's stored broad
     lumps all neurons into 'Other').
  4. Embedding-quality / fragmentation diagnostic: compare three UMAPs —
     (a) scVI latent recomputed on the <5y subset, (b) the PRECOMPUTED full-data
     scVI UMAP subset to <5y cells, (c) raw-counts PCA — with Leiden cluster counts
     and silhouette(marker-call) in each representation, to tell whether the scVI
     fragmentation is a UMAP-subset artefact or a poor embedding, and which
     representation to cluster on.

Figures per dataset: a MAIN figure (large classification panels on top, marker-gene
panels below) and a FRAGMENTATION figure (3 embeddings side by side).

SUBMIT:
  sbatch --time=02:30:00 --mem=200G scripts/run_script.sh scripts/c3_maturation/s10_young_clusters.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L

AGE_MAX = 5.0
ENS = {
    "SOX2": "ENSG00000181449", "MKI67": "ENSG00000148773", "DCX": "ENSG00000077279",
    "STMN2": "ENSG00000104435", "NEUROD6": "ENSG00000164600", "RBFOX3": "ENSG00000167281",
    "SLC17A7": "ENSG00000104888", "SATB2": "ENSG00000119042",
    "GAD1": "ENSG00000128683", "GAD2": "ENSG00000136750", "SLC32A1": "ENSG00000101438",
    "DLX2": "ENSG00000115844",
    "AQP4": "ENSG00000171885", "PDGFRA": "ENSG00000134853", "PLP1": "ENSG00000123560",
    "CSF1R": "ENSG00000182578", "GFAP": "ENSG00000131095", "MBP": "ENSG00000197971",
}
MARKER_PANEL = ["SOX2", "MKI67", "DCX", "STMN2", "NEUROD6", "RBFOX3",
                "SLC17A7", "SATB2", "GAD1", "GAD2", "SLC32A1", "DLX2",
                "AQP4", "PDGFRA", "PLP1", "CSF1R"]
DATASETS = {
    "PsychAD": dict(path=L.B / "PsychAD_noage_tuning5/scvi_output/integrated.h5ad", chem=None),
    "Velmeshev-V3": dict(path=L.B / "Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad", chem="V3"),
}


def fine_to_broad(fine):
    f = pd.Series(fine, dtype=object).astype(str)
    import re
    def cl(x):
        if re.search(r"prog|cycl|divid|radial|^rg$|ipc|neurobl|glial_prog", x, re.I):
            return "Progenitor"
        if re.search(r"astro|oligo|opc|micro|endo|vlmc|mural|peri|fibrous|protoplasmic|^pc$|vasc|immune|^pvm$", x, re.I):
            return "Glia/vasc"
        if re.search(r"interneuron|^int$|^in_|sst|pvalb|^pv$|vip|lamp5|calb2|adarb2|cck|reln|sncg|sv2c|cge|mge|chc|^id2$", x, re.I):
            return "IN"
        if re.search(r"^l[2-6]|^en_|excit|_it|^it$|^et$|^ct$|^np$|l5-6|newborn|^sp$", x, re.I):
            return "EN"
        return "other"
    return f.map(cl).values


def signatures(expr):
    """expr: dict sym -> log1p CPM array. Return DataFrame of lineage signatures."""
    g = lambda *s: np.sum([expr[x] for x in s if x in expr], axis=0)
    return pd.DataFrame({
        "EN_sig": g("SLC17A7", "SATB2"),
        "IN_sig": g("GAD1", "GAD2", "SLC32A1"),
        "Prog_sig": g("SOX2", "MKI67"),
        "Imm_sig": g("DCX", "STMN2"),
        "Glia_sig": g("AQP4", "PLP1", "PDGFRA", "CSF1R", "GFAP", "MBP"),
    })


def percell_call(sig):
    """SLC17A7-specific per-cell call."""
    lab = np.full(len(sig), "Unknown", dtype=object)
    en, inn, prog, imm, glia = (sig[c].values for c in ["EN_sig", "IN_sig", "Prog_sig", "Imm_sig", "Glia_sig"])
    is_glia = (glia >= 1) & (en < 1) & (inn < 1)
    lab[is_glia] = "Glia"
    lab[(prog >= 1) & (en < 1) & (inn < 1) & ~is_glia] = "Progenitor"
    neur = (~is_glia) & ((en >= 1) | (inn >= 1))
    lab[neur & (en >= inn)] = "ExN"
    lab[neur & (inn > en)] = "InN"
    lab[(lab == "Unknown") & (imm >= 1)] = "Immature_neuron"
    return lab


def cluster_vote(leiden, sig):
    """Label each Leiden cluster by dominant signature (EN/IN/Glia/Progenitor)."""
    df = sig.copy(); df["cl"] = np.asarray(leiden)
    means = df.groupby("cl").mean()
    lab = {}
    for cl, row in means.iterrows():
        order = row[["EN_sig", "IN_sig", "Glia_sig", "Prog_sig"]].sort_values(ascending=False)
        top = order.index[0].replace("_sig", "")
        top = {"EN": "ExN", "IN": "InN", "Glia": "Glia", "Prog": "Progenitor"}[top]
        # immature flag if Imm high and neuronal top
        if top in ("ExN", "InN") and row["Imm_sig"] > row[["EN_sig", "IN_sig"]].max():
            top = "Immature_neuron"
        lab[cl] = top
    return np.array([lab[c] for c in np.asarray(leiden)]), means


def leiden_on(rep_mat, n_neighbors=15, res=1.0):
    tmp = ad.AnnData(np.zeros((rep_mat.shape[0], 1)))
    tmp.obsm["R"] = rep_mat
    sc.pp.neighbors(tmp, use_rep="R", n_neighbors=n_neighbors)
    try:
        sc.tl.leiden(tmp, resolution=res, flavor="igraph", n_iterations=2, directed=False)
    except Exception:
        sc.tl.leiden(tmp, resolution=res)
    return tmp.obs["leiden"].values, tmp


def umap_from_neighbors(tmp):
    sc.tl.umap(tmp)
    return tmp.obsm["X_umap"]


def sil(rep_mat, labels, seed=0, cap=8000):
    m = labels != "Unknown"
    X, y = rep_mat[m], labels[m]
    if len(np.unique(y)) < 2:
        return np.nan
    if len(y) > cap:
        rng = np.random.default_rng(seed); idx = rng.choice(len(y), cap, replace=False)
        X, y = X[idx], y[idx]
    return float(silhouette_score(X, y))


CMAP = {"ExN": "#d62728", "InN": "#1f77b4", "Glia": "#2ca02c", "Progenitor": "#9467bd",
        "Immature_neuron": "#ff7f0e", "Unknown": "#cccccc", "EN": "#d62728", "other": "#cccccc",
        "Glia/vasc": "#2ca02c"}


def scatter_cat(ax, um, lab, title, s=4, legend=True):
    lab = pd.Series(lab).astype(str).values
    for k in pd.unique(lab):
        m = lab == k
        ax.scatter(um[m, 0], um[m, 1], s=s, c=CMAP.get(k, None), label=f"{k} ({m.sum()})")
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(markerscale=3, fontsize=8, loc="best")


def process(name, cfg):
    print(f"\n=== {name}", flush=True)
    a = ad.read_h5ad(cfg["path"], backed="r")
    obs = a.obs
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    mask = (age >= 0) & (age < AGE_MAX)
    if cfg["chem"] and "chemistry" in obs:
        mask &= obs["chemistry"].astype(str).values == cfg["chem"]
    idx = np.where(mask)[0]
    print(f"  <{AGE_MAX}y cells: {len(idx):,}", flush=True)
    sub = a[idx].to_memory()
    vn = list(sub.var_names)

    # raw counts -> log1p CPM for the panel + signature genes
    X = sub.layers["counts"] if "counts" in sub.layers else sub.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    expr = {}
    for s, ens in ENS.items():
        expr[s] = np.log1p(np.asarray(X[:, vn.index(ens)].todense()).ravel() / tot * 1e4) if ens in vn else np.zeros(len(idx))
    sig = signatures(expr)

    # native fine + fixed broad
    nat_fine = next((c for c in ["cell_type_raw", "subclass", "Cell_Type"] if c in sub.obs), None)
    sub.obs["native_fine"] = sub.obs[nat_fine].astype(str) if nat_fine else "?"
    sub.obs["native_broad_fixed"] = fine_to_broad(sub.obs["native_fine"].values)

    # representations
    scvi = sub.obsm["X_scVI"] if "X_scVI" in sub.obsm else sub.obsm["X_scANVI"]
    # raw PCA
    b = sub.copy(); b.X = X.copy()
    sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    sc.pp.highly_variable_genes(b, n_top_genes=2000); b = b[:, b.var.highly_variable].copy()
    sc.pp.scale(b, max_value=10); sc.tl.pca(b, n_comps=30)
    pca = b.obsm["X_pca"]

    # leiden on each representation
    leiden_scvi, tmp_scvi = leiden_on(scvi)
    leiden_pca, tmp_pca = leiden_on(pca)
    # cluster-vote labels (use PCA clustering as primary — model-free)
    cvote_pca, means_pca = cluster_vote(leiden_pca, sig)
    cvote_scvi, _ = cluster_vote(leiden_scvi, sig)
    pcall = percell_call(sig)

    # UMAPs: (a) scVI recomputed, (b) scVI precomputed full subset, (c) PCA
    um_scvi = umap_from_neighbors(tmp_scvi)
    um_pca = umap_from_neighbors(tmp_pca)
    pre_key = next((k for k in ["X_umap_scvi", "X_umap"] if k in sub.obsm), None)
    um_pre = sub.obsm[pre_key] if pre_key else None

    # diagnostics
    print(f"  Leiden clusters: scVI={len(np.unique(leiden_scvi))}, PCA={len(np.unique(leiden_pca))}")
    print(f"  silhouette(percell-call): scVI-latent={sil(scvi,pcall):.3f}  PCA={sil(pca,pcall):.3f}")
    print(f"  cluster-vote(PCA) composition: {dict(pd.Series(cvote_pca).value_counts())}")
    print(f"  per-cell call composition:     {dict(pd.Series(pcall).value_counts())}")
    ct = pd.crosstab(sub.obs['native_broad_fixed'], pd.Series(cvote_pca, name='cluster_vote'),
                     normalize='index').round(3)
    print("  native_broad_fixed x cluster_vote(PCA):"); print(ct.to_string())
    pd.DataFrame({"native_fine": sub.obs["native_fine"].values,
                  "native_broad_fixed": sub.obs["native_broad_fixed"].values,
                  "percell_call": pcall, "cvote_pca": cvote_pca, "cvote_scvi": cvote_scvi,
                  "leiden_pca": leiden_pca, "leiden_scvi": leiden_scvi,
                  "age": age[idx]}, index=sub.obs_names).to_parquet(
        L.OUT_DIR / f"s10_{name.replace('-', '_').lower()}_obs.parquet")

    # ---------- MAIN figure: big top panels (on PCA UMAP) + marker grid ----------
    tag = name.replace("-", "_").lower()
    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(5, 4, height_ratios=[2.4, 1, 1, 1, 1], hspace=0.25, wspace=0.18)
    # top row: 4 large panels
    ax = fig.add_subplot(gs[0, 0]); scatter_cat(ax, um_pca, sub.obs["native_broad_fixed"].values, "native (fixed broad, from fine)")
    ax = fig.add_subplot(gs[0, 1]); scatter_cat(ax, um_pca, pcall, "per-cell SLC17A7-specific call")
    ax = fig.add_subplot(gs[0, 2]); scatter_cat(ax, um_pca, cvote_pca, "CLUSTER-vote label (Leiden/PCA)")
    ax = fig.add_subplot(gs[0, 3]); sca = ax.scatter(um_pca[:, 0], um_pca[:, 1], c=age[idx], s=4, cmap="viridis"); plt.colorbar(sca, ax=ax, fraction=.04); ax.set_title("age (years)", fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    # marker grid (16) in rows 1-4
    for i, g in enumerate(MARKER_PANEL):
        r, c = 1 + i // 4, i % 4
        ax = fig.add_subplot(gs[r, c])
        sca = ax.scatter(um_pca[:, 0], um_pca[:, 1], c=expr[g], s=3, cmap="magma")
        plt.colorbar(sca, ax=ax, fraction=.04); ax.set_title(g, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{name}: <{AGE_MAX:.0f}y (n={len(idx):,}) — raw-counts PCA UMAP. "
                 f"Large: classifications; below: marker genes", y=0.995, fontsize=15)
    fig.savefig(L.OUT_DIR / f"s10_young_main_{tag}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---------- FRAGMENTATION figure: 3 embeddings, coloured by cluster-vote ----------
    panels = [("scVI latent\n(recomputed on subset)", um_scvi, cvote_scvi),
              ("PCA on raw counts", um_pca, cvote_pca)]
    if um_pre is not None:
        panels.insert(1, (f"scVI precomputed full UMAP\n({pre_key}, subset)", um_pre, cvote_scvi))
    fig2, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6.5))
    for ax, (ttl, um, lab) in zip(np.atleast_1d(axes), panels):
        scatter_cat(ax, um, lab, ttl, s=4, legend=True)
    fig2.suptitle(f"{name}: <{AGE_MAX:.0f}y — embedding comparison (colour = cluster-vote). "
                  f"Leiden clusters scVI={len(np.unique(leiden_scvi))} PCA={len(np.unique(leiden_pca))}; "
                  f"silhouette scVI={sil(scvi,pcall):.2f} PCA={sil(pca,pcall):.2f}", y=1.02, fontsize=12)
    fig2.tight_layout(); fig2.savefig(L.OUT_DIR / f"s10_young_embed_{tag}.png", dpi=120, bbox_inches="tight"); plt.close(fig2)
    print(f"  saved s10_young_main_{tag}.png, s10_young_embed_{tag}.png", flush=True)


def main():
    for name, cfg in DATASETS.items():
        process(name, cfg)
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
