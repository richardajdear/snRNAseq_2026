#!/usr/bin/env python3
"""Neuron+progenitor manifold for trajectory feasibility (sbatch).

Motivation (user): we cannot get a clean trajectory UMAP without first removing the
major non-neuronal classes. Idea: subset to NEURONS (EN+IN+immature) and neuronal
PROGENITORS using a PAN-neuronal cluster-vote (RBFOX3/DCX/STMN2 etc.) BEFORE making
any EN-vs-IN call, then recompute the embedding and ask whether neurons form a
*continuous* spread amenable to pseudotime/trajectory analysis.

Also answers: why do the UMAPs show isolated islands rather than a continuous spread,
and which parameters control spread-vs-clustered? -> a parameter sweep + PAGA-init +
ForceAtlas2 + diffusion-map embeddings of the SAME neuron subset.

Label fix (user): in Velmeshev the native label 'Interneurons' actually denotes
IMMATURE EXCITATORY cells (they express SLC17A7/SATB2, not GAD); true inhibitory
neurons are the granular SST/PV/VIP/CALB2/CCK/INT labels. fine_to_broad() now
special-cases this so the native reference panel is not misleading.

Per dataset, all ages, stratified-subsampled to ~300k cells:
  - representation = PCA(30) on RAW COUNTS (one batch per dataset; no scVI);
  - identify neurons+progenitors by pan-neuronal cluster-vote (Leiden, nn=50);
  - recompute the neuron-only embedding on that PCA;
  - DPT pseudotime rooted at the progenitor/immature pole;
  - UMAP parameter sweep + PAGA-init + ForceAtlas2 + diffmap to show islands->continuum.

SUBMIT (himem: ~300k cells):
  sbatch --time=05:00:00 --mem=400G --partition=icelake-himem scripts/run_script.sh scripts/c3_maturation/s11_neuron_manifold.py
"""
from pathlib import Path
import sys, re
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L

AGE_MAX = 1000.0        # all ages (stratified subsample keeps the young/immature end represented)
MAX_CELLS = 300_000     # stratified subsample before clustering
NN = 50                 # n_neighbors for the neuron-manifold graph (user: increase to 50)
SEED = 0

ENS = {
    "SOX2": "ENSG00000181449", "MKI67": "ENSG00000148773", "DCX": "ENSG00000077279",
    "STMN2": "ENSG00000104435", "NEUROD6": "ENSG00000164600", "RBFOX3": "ENSG00000167281",
    "SLC17A7": "ENSG00000104888", "SATB2": "ENSG00000119042",
    "GAD1": "ENSG00000128683", "GAD2": "ENSG00000136750", "SLC32A1": "ENSG00000101438",
    "DLX2": "ENSG00000115844",
    "AQP4": "ENSG00000171885", "PDGFRA": "ENSG00000134853", "PLP1": "ENSG00000123560",
    "CSF1R": "ENSG00000182578", "GFAP": "ENSG00000131095", "MBP": "ENSG00000197971",
}
DATASETS = {
    "PsychAD": dict(path=L.B / "PsychAD_noage_tuning5/scvi_output/integrated.h5ad", chem=None),
    "Velmeshev-V3": dict(path=L.B / "Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad", chem="V3"),
}

CMAP = {"ExN": "#d62728", "InN": "#1f77b4", "Glia": "#2ca02c", "Progenitor": "#9467bd",
        "Immature_neuron": "#ff7f0e", "Unknown": "#cccccc", "EN": "#d62728", "other": "#bbbbbb",
        "Glia/vasc": "#2ca02c", "IN": "#1f77b4", "ImmatureEN(native-misnomer)": "#ff7f0e"}


def fine_to_broad(fine, dataset=None):
    """Native fine -> broad. Velmeshev 'Interneurons'/'INT'... special-cased:
    'Interneurons' (exact) = IMMATURE EXCITATORY (Velmeshev misnomer)."""
    f = pd.Series(fine, dtype=object).astype(str)

    def cl(x):
        xl = x.strip().lower()
        if dataset and "velmeshev" in dataset.lower() and xl == "interneurons":
            return "ImmatureEN(native-misnomer)"   # express SLC17A7/SATB2, not GAD
        if re.search(r"prog|cycl|divid|radial|^rg$|ipc|neurobl|glial_prog", x, re.I):
            return "Progenitor"
        if re.search(r"astro|oligo|opc|micro|endo|vlmc|mural|peri|fibrous|protoplasmic|^pc$|vasc|immune|^pvm$", x, re.I):
            return "Glia/vasc"
        # genuine inhibitory (incl. Velmeshev granular INT/SST/PV/VIP/CALB2/CCK/RELN/SV2C/NOS)
        if re.search(r"^int$|^in_|sst|pvalb|^pv$|^pv_|vip|lamp5|calb2|adarb2|cck|reln|sncg|sv2c|cge|mge|chc|^id2$|^nos$|interneuron", x, re.I):
            return "IN"
        if re.search(r"^l[2-6]|^en_|excit|_it|^it$|^et$|^ct$|^np$|l5-6|newborn|^sp$", x, re.I):
            return "EN"
        return "other"
    return f.map(cl).values


def signatures(expr):
    g = lambda *s: np.sum([expr[x] for x in s if x in expr], axis=0)
    return pd.DataFrame({
        "EN_sig":   g("SLC17A7", "SATB2"),
        "IN_sig":   g("GAD1", "GAD2", "SLC32A1"),
        "Prog_sig": g("SOX2", "MKI67"),
        "Imm_sig":  g("DCX", "STMN2"),
        "Pan_sig":  g("RBFOX3", "DCX", "STMN2", "NEUROD6"),   # pan-neuronal / neurogenic
        "Glia_sig": g("AQP4", "PLP1", "PDGFRA", "CSF1R", "GFAP", "MBP"),
    })


def vote_neuron_glia(leiden, sig):
    """Per-cluster: is it neuronal/progenitor (KEEP) or glia (DROP)?
    Neuronal score = max(EN,IN,Pan); compare to Glia and Prog."""
    df = sig.copy(); df["cl"] = np.asarray(leiden)
    means = df.groupby("cl").mean()
    keep, sub_lab = {}, {}
    for cl, row in means.iterrows():
        neuro = max(row["EN_sig"], row["IN_sig"], row["Pan_sig"])
        glia, prog = row["Glia_sig"], row["Prog_sig"]
        is_neuro = (neuro >= glia)            # neuronal signal at least matches glia
        is_prog = (prog >= glia) and (prog >= 0.5)
        keep[cl] = bool(is_neuro or is_prog)
        # sub-identity for KEPT clusters
        if prog >= max(row["EN_sig"], row["IN_sig"], row["Imm_sig"]) and prog >= 0.5:
            sub_lab[cl] = "Progenitor"
        elif row["Imm_sig"] > max(row["EN_sig"], row["IN_sig"]):
            sub_lab[cl] = "Immature_neuron"
        elif row["EN_sig"] >= row["IN_sig"]:
            sub_lab[cl] = "ExN"
        else:
            sub_lab[cl] = "InN"
    keep_arr = np.array([keep[c] for c in np.asarray(leiden)])
    lab_arr = np.array([sub_lab[c] for c in np.asarray(leiden)])
    return keep_arr, lab_arr, means


def umap_coords(rep, n_neighbors=15, min_dist=0.5, spread=1.0, init="spectral", paga_groups=None):
    """Compute a UMAP on a representation with given params; optional PAGA init."""
    tmp = ad.AnnData(np.zeros((rep.shape[0], 1)))
    tmp.obsm["R"] = rep
    sc.pp.neighbors(tmp, use_rep="R", n_neighbors=n_neighbors)
    init_pos = init
    if init == "paga" and paga_groups is not None:
        tmp.obs["g"] = pd.Categorical(paga_groups.astype(str))
        sc.tl.paga(tmp, groups="g")
        sc.pl.paga(tmp, plot=False)
        init_pos = "paga"
    sc.tl.umap(tmp, min_dist=min_dist, spread=spread, init_pos=init_pos)
    return tmp.obsm["X_umap"]


def scatter_cat(ax, um, lab, title, s=4, legend=True):
    lab = pd.Series(lab).astype(str).values
    order = sorted(pd.unique(lab), key=lambda k: -(lab == k).sum())
    for k in order:
        m = lab == k
        ax.scatter(um[m, 0], um[m, 1], s=s, c=CMAP.get(k, None), label=f"{k} ({m.sum()})")
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(markerscale=3, fontsize=8, loc="best")


def scatter_val(ax, um, val, title, cmap="viridis", s=4):
    sca = ax.scatter(um[:, 0], um[:, 1], c=val, s=s, cmap=cmap)
    plt.colorbar(sca, ax=ax, fraction=.045); ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])


def process(name, cfg):
    print(f"\n=== {name}", flush=True)
    rng = np.random.default_rng(SEED)
    a = ad.read_h5ad(cfg["path"], backed="r")
    obs = a.obs
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    mask = (age >= 0) & (age < AGE_MAX)
    if cfg["chem"] and "chemistry" in obs:
        mask &= obs["chemistry"].astype(str).values == cfg["chem"]
    idx = np.where(mask)[0]
    print(f"  age[0,{AGE_MAX}) cells: {len(idx):,}", flush=True)
    # stratified subsample by age decile so the maturation spectrum is represented
    if len(idx) > MAX_CELLS:
        ag = age[idx]
        bins = pd.qcut(ag, q=min(10, len(np.unique(ag))), duplicates="drop")
        per = MAX_CELLS // len(bins.categories)
        keep = []
        for c in bins.categories:
            sel = idx[bins == c]
            keep.append(rng.choice(sel, min(per, len(sel)), replace=False))
        idx = np.sort(np.concatenate(keep))
    print(f"  subsampled to: {len(idx):,}", flush=True)
    sub = a[idx].to_memory()
    vn = list(sub.var_names)

    X = sub.layers["counts"] if "counts" in sub.layers else sub.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    expr = {}
    for s, ens in ENS.items():
        expr[s] = (np.log1p(np.asarray(X[:, vn.index(ens)].todense()).ravel() / tot * 1e4)
                   if ens in vn else np.zeros(len(idx)))
    sig = signatures(expr)

    nat_fine = next((c for c in ["cell_type_raw", "subclass", "Cell_Type"] if c in sub.obs), None)
    sub.obs["native_fine"] = sub.obs[nat_fine].astype(str) if nat_fine else "?"
    sub.obs["native_broad_fixed"] = fine_to_broad(sub.obs["native_fine"].values, dataset=name)

    # ---- representation: PCA on RAW COUNTS (single batch per dataset; no scVI) ----
    b = ad.AnnData(X.copy(), obs=sub.obs[[]].copy(), var=sub.var.copy())
    sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    sc.pp.highly_variable_genes(b, n_top_genes=2000)
    b = b[:, b.var.highly_variable].copy()
    sc.pp.scale(b, max_value=10); sc.tl.pca(b, n_comps=30)
    rep = b.obsm["X_pca"]; del b
    print(f"  representation = PCA(30) on raw counts (HVG 2000); no scVI", flush=True)

    # ---- 1. cluster ALL (subsampled) cells, vote neuron vs glia ----
    sub.obsm["R"] = rep
    sc.pp.neighbors(sub, use_rep="R", n_neighbors=NN)
    try:
        sc.tl.leiden(sub, resolution=1.5, flavor="igraph", n_iterations=2, directed=False, key_added="leiden_all")
    except Exception:
        sc.tl.leiden(sub, resolution=1.5, key_added="leiden_all")
    leiden_all = sub.obs["leiden_all"].values
    keep, sublab_all, means = vote_neuron_glia(leiden_all, sig)
    print(f"  Leiden(all)={len(np.unique(leiden_all))} clusters; "
          f"neuron+prog kept = {keep.sum():,}/{len(keep):,} ({keep.mean():.1%})", flush=True)
    print(f"  kept sub-identity: {dict(pd.Series(sublab_all[keep]).value_counts())}", flush=True)
    ct = pd.crosstab(sub.obs['native_broad_fixed'].values, np.where(keep, 'NEURON', 'glia'),
                     normalize='index').round(3)
    print("  native_broad_fixed x kept (row-normalized):"); print(ct.to_string())

    # ---- 2. neuron subset: recompute embedding on the PCA representation ----
    nsub = sub[keep].copy()
    nrep = rep[keep]
    nexpr = {k: v[keep] for k, v in expr.items()}
    nsig = sig[keep].reset_index(drop=True)
    nlab = sublab_all[keep]
    nage = age[idx][keep]
    # re-cluster within neurons for sub-identity coloring + PAGA
    nsub.obsm["R"] = nrep
    sc.pp.neighbors(nsub, use_rep="R", n_neighbors=NN)
    try:
        sc.tl.leiden(nsub, resolution=1.0, flavor="igraph", n_iterations=2, directed=False, key_added="leiden_n")
    except Exception:
        sc.tl.leiden(nsub, resolution=1.0, key_added="leiden_n")
    leiden_n = nsub.obs["leiden_n"].values

    # DPT pseudotime rooted at progenitor/immature pole
    sc.tl.diffmap(nsub)
    root_score = (nsig["Prog_sig"].values + nsig["Imm_sig"].values)
    nsub.uns["iroot"] = int(np.argmax(root_score))
    sc.tl.dpt(nsub)
    dpt = nsub.obs["dpt_pseudotime"].values

    # ---- 3. embeddings: default + parameter sweep + PAGA-init + diffmap + FA2 ----
    embeds = {}
    embeds["UMAP local\n(nn=15, min_dist=0.5)"] = umap_coords(nrep, 15, 0.5)
    embeds["UMAP global\n(nn=50, min_dist=0.5)"] = umap_coords(nrep, 50, 0.5)
    embeds["UMAP spread\n(nn=50, min_dist=0.9, spread=2)"] = umap_coords(nrep, 50, 0.9, spread=2.0)
    embeds["UMAP PAGA-init\n(nn=50, graph-aware)"] = umap_coords(nrep, NN, 0.5, init="paga", paga_groups=leiden_n)
    # diffmap (DC1 vs DC2) — intrinsically continuous
    embeds["DiffusionMap\n(DC1 vs DC2)"] = nsub.obsm["X_diffmap"][:, 1:3]
    # ForceAtlas2 (optional; needs fa2)
    try:
        sc.tl.draw_graph(nsub, layout="fa")
        embeds["ForceAtlas2\n(draw_graph fa)"] = nsub.obsm["X_draw_graph_fa"]
    except Exception as e:
        print(f"  ForceAtlas2 unavailable ({e}); trying 'fr'", flush=True)
        try:
            sc.tl.draw_graph(nsub, layout="fr")
            embeds["Fruchterman-Reingold\n(draw_graph fr)"] = nsub.obsm["X_draw_graph_fr"]
        except Exception as e2:
            print(f"  draw_graph unavailable ({e2})", flush=True)

    tag = name.replace("-", "_").lower()
    main_um = embeds["UMAP PAGA-init\n(nn=50, graph-aware)"]   # trajectory-friendly default for main fig

    # ---- Figure A: neuron manifold (PAGA-init UMAP) — identity, age, DPT, markers ----
    panel_markers = ["SOX2", "DCX", "STMN2", "NEUROD6", "RBFOX3", "SLC17A7", "SATB2", "GAD1"]
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(4, 4, height_ratios=[2.2, 2.2, 1, 1], hspace=0.28, wspace=0.2)
    ax = fig.add_subplot(gs[0, 0]); scatter_cat(ax, main_um, nlab, "cluster-vote sub-identity")
    ax = fig.add_subplot(gs[0, 1]); scatter_cat(ax, main_um, nsub.obs["native_broad_fixed"].values, "native (fixed broad)")
    ax = fig.add_subplot(gs[0, 2]); scatter_val(ax, main_um, nage, "age (years)")
    ax = fig.add_subplot(gs[0, 3]); scatter_val(ax, main_um, dpt, "DPT pseudotime", cmap="plasma")
    ax = fig.add_subplot(gs[1, 0]); scatter_cat(ax, main_um, leiden_n, "Leiden (neurons)", legend=False)
    # DPT-vs-age trend on EN lineage
    ax = fig.add_subplot(gs[1, 1])
    en_m = np.isin(nlab, ["ExN", "Immature_neuron"])
    ax.scatter(nage[en_m], dpt[en_m], s=3, alpha=.3, c="#d62728")
    ax.set_xlabel("age (years)"); ax.set_ylabel("DPT pseudotime"); ax.set_title("EN-lineage: pseudotime vs age", fontsize=12)
    ax = fig.add_subplot(gs[1, 2]); scatter_val(ax, main_um, nsig["EN_sig"].values, "EN_sig (SLC17A7+SATB2)", cmap="magma")
    ax = fig.add_subplot(gs[1, 3]); scatter_val(ax, main_um, nsig["IN_sig"].values, "IN_sig (GAD1/2+SLC32A1)", cmap="magma")
    for i, g in enumerate(panel_markers):
        r, c = 2 + i // 4, i % 4
        ax = fig.add_subplot(gs[r, c]); scatter_val(ax, main_um, nexpr[g], g, cmap="magma", s=3)
    fig.suptitle(f"{name}: neuron+progenitor manifold (n={keep.sum():,}, all ages) — "
                 f"PAGA-init UMAP (nn={NN}) on PCA-of-counts. Is it a continuous maturation spread?",
                 y=0.995, fontsize=15)
    fig.savefig(L.OUT_DIR / f"s11_neuron_manifold_{tag}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- Figure B: embedding/parameter sweep (islands -> continuum), coloured by DPT ----
    items = list(embeds.items())
    ncol = 3; nrow = int(np.ceil(len(items) / ncol))
    fig2, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (ttl, um) in zip(axes, items):
        scatter_val(ax, um, dpt, ttl, cmap="plasma", s=3)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig2.suptitle(f"{name}: same neuron subset (PCA-of-counts), different embeddings (colour=DPT). "
                  f"n_neighbors / min_dist / spread / PAGA-init / diffmap control islands-vs-continuum",
                  y=1.0, fontsize=13)
    fig2.tight_layout(); fig2.savefig(L.OUT_DIR / f"s11_embedding_sweep_{tag}.png", dpi=115, bbox_inches="tight"); plt.close(fig2)

    # ---- save obs ----
    pd.DataFrame({
        "native_fine": nsub.obs["native_fine"].values,
        "native_broad_fixed": nsub.obs["native_broad_fixed"].values,
        "cluster_vote": nlab, "leiden_n": leiden_n,
        "dpt": dpt, "age": nage,
        "EN_sig": nsig["EN_sig"].values, "IN_sig": nsig["IN_sig"].values,
        "Imm_sig": nsig["Imm_sig"].values, "Prog_sig": nsig["Prog_sig"].values,
    }, index=nsub.obs_names).to_parquet(L.OUT_DIR / f"s11_{tag}_neurons.parquet")

    # ---- save a COMPACT neuron-manifold .h5ad next to integrated.h5ad (for trajectory pipeline) ----
    # placeholder X (no gene matrix); representation in obsm, neighbor graph in obsp.
    obs_export = pd.DataFrame({
        "native_fine": nsub.obs["native_fine"].values,
        "native_broad_fixed": nsub.obs["native_broad_fixed"].values,
        "cluster_vote": nlab, "leiden_n": np.asarray(leiden_n).astype(str),
        "age": nage, "dpt_seed": dpt,
        "EN_sig": nsig["EN_sig"].values, "IN_sig": nsig["IN_sig"].values,
        "Imm_sig": nsig["Imm_sig"].values, "Prog_sig": nsig["Prog_sig"].values,
        "Pan_sig": nsig["Pan_sig"].values, "Glia_sig": nsig["Glia_sig"].values,
    }, index=nsub.obs_names)
    for g in ENS:  # key marker expression (log1p CPM) for coloring/diagnostics
        obs_export[f"expr_{g}"] = nexpr[g]
    exp = ad.AnnData(X=sp.csr_matrix((len(nlab), 1), dtype="float32"), obs=obs_export)
    exp.obsm["X_pca"] = np.asarray(nrep)
    exp.obsm["X_diffmap"] = np.asarray(nsub.obsm["X_diffmap"])
    exp.obsm["X_umap_pagainit"] = np.asarray(main_um)
    # carry the neighbor graph (computed on X_pca, nn=NN) so PAGA/CellRank can reuse it
    exp.obsp["connectivities"] = nsub.obsp["connectivities"]
    exp.obsp["distances"] = nsub.obsp["distances"]
    exp.uns["neighbors"] = nsub.uns["neighbors"]
    exp.uns["iroot"] = int(nsub.uns["iroot"])
    exp.uns["manifold_meta"] = {"dataset": name, "n_neighbors": NN, "representation": "PCA30_on_counts",
                                "n_cells": int(len(nlab)), "age_max": AGE_MAX}
    tdir = Path(cfg["path"]).parent / "trajectory"
    tdir.mkdir(parents=True, exist_ok=True)
    exp.write(tdir / "neuron_manifold.h5ad")
    print(f"  saved s11_neuron_manifold_{tag}.png, s11_embedding_sweep_{tag}.png, "
          f"s11_{tag}_neurons.parquet, {tdir}/neuron_manifold.h5ad", flush=True)


def main():
    for name, cfg in DATASETS.items():
        process(name, cfg)
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
