#!/usr/bin/env python3
"""Build a neuron+progenitor trajectory MANIFOLD from an `integrated.h5ad`
(the output of `code/pipeline/run_pipeline.py`).

STANDALONE: this module has **no dependency on anything under `scripts/`** — it
reads only the integrated object plus the AHBA weights CSV under `reference/`, so
the core prep can run directly on the main pipeline's output.

Pipeline:
  chemistry/age subset (optionally ALL cells incl. prenatal)
    -> PCA(30) on raw counts (single batch per dataset; no scVI)
    -> Leiden -> pan-neuronal cluster-vote (keep neurons+progenitors, drop glia)
    -> neuron subset -> nn graph + diffmap + DPT-seed (rooted at progenitor/immature pole)
    -> per-cell C3 projection (signed depth-robust + C3+ pole)
    -> write neuron_manifold.h5ad (X_pca, neighbor graph, diffmap, PAGA-init UMAP,
       cluster-vote labels, signatures, C3 scores, marker expr, uns['iroot']).

The output is consumed by `code/trajectory/run_trajectory.py`.

CLI
  python code/trajectory/manifold.py \
      --integrated /…/scvi_output/integrated.h5ad \
      --out        /…/scvi_output/trajectory/neuron_manifold.h5ad \
      --dataset    Velmeshev-V3 --chem V3 [--all-cells] [--no-c3] \
      [--fig-dir /…/figs --fig-prefix s11]
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
NN = 50                 # n_neighbors for the neuron-manifold graph
MAX_CELLS = 300_000     # stratified subsample cap (disabled by all_cells=True)
AGE_MAX = 1000.0        # upper age bound when not using all_cells
FA2_MAX = 120_000       # skip ForceAtlas2 above this many neurons (too slow)
SEED = 0

DEFAULT_REF_WEIGHTS = Path(__file__).resolve().parents[2] / "reference/ahba_dme_hcp_top8kgenes_weights.csv"

ENS = {
    "SOX2": "ENSG00000181449", "MKI67": "ENSG00000148773", "DCX": "ENSG00000077279",
    "STMN2": "ENSG00000104435", "NEUROD6": "ENSG00000164600", "RBFOX3": "ENSG00000167281",
    "SLC17A7": "ENSG00000104888", "SATB2": "ENSG00000119042",
    "GAD1": "ENSG00000128683", "GAD2": "ENSG00000136750", "SLC32A1": "ENSG00000101438",
    "DLX2": "ENSG00000115844",
    "AQP4": "ENSG00000171885", "PDGFRA": "ENSG00000134853", "PLP1": "ENSG00000123560",
    "CSF1R": "ENSG00000182578", "GFAP": "ENSG00000131095", "MBP": "ENSG00000197971",
}
MARKER_PANEL = ["SOX2", "DCX", "STMN2", "NEUROD6", "RBFOX3", "SLC17A7", "SATB2", "GAD1"]

CMAP = {"ExN": "#d62728", "InN": "#1f77b4", "Glia": "#2ca02c", "Progenitor": "#9467bd",
        "Immature_neuron": "#ff7f0e", "Unknown": "#cccccc", "EN": "#d62728", "other": "#bbbbbb",
        "Glia/vasc": "#2ca02c", "IN": "#1f77b4", "ImmatureEN(native-misnomer)": "#ff7f0e"}


# --------------------------------------------------------------------------- #
# labeling / signatures
# --------------------------------------------------------------------------- #
def fine_to_broad(fine, dataset=None):
    """Native fine label -> broad class. Velmeshev 'Interneurons' (exact) is a
    misnomer for IMMATURE EXCITATORY cells (express SLC17A7/SATB2, not GAD)."""
    f = pd.Series(fine, dtype=object).astype(str)

    def cl(x):
        xl = x.strip().lower()
        if dataset and "velmeshev" in dataset.lower() and xl == "interneurons":
            return "ImmatureEN(native-misnomer)"
        if re.search(r"prog|cycl|divid|radial|^rg$|ipc|neurobl|glial_prog", x, re.I):
            return "Progenitor"
        if re.search(r"astro|oligo|opc|micro|endo|vlmc|mural|peri|fibrous|protoplasmic|^pc$|vasc|immune|^pvm$", x, re.I):
            return "Glia/vasc"
        if re.search(r"^int$|^in_|sst|pvalb|^pv$|^pv_|vip|lamp5|calb2|adarb2|cck|reln|sncg|sv2c|cge|mge|chc|^id2$|^nos$|interneuron", x, re.I):
            return "IN"
        if re.search(r"^l[2-6]|^en_|excit|_it|^it$|^et$|^ct$|^np$|l5-6|newborn|^sp$", x, re.I):
            return "EN"
        return "other"
    return f.map(cl).values


def signatures(expr):
    """expr: dict sym -> per-cell log1p-CPM array. Returns lineage signature frame."""
    g = lambda *s: np.sum([expr[x] for x in s if x in expr], axis=0)
    return pd.DataFrame({
        "EN_sig":   g("SLC17A7", "SATB2"),
        "IN_sig":   g("GAD1", "GAD2", "SLC32A1"),
        "Prog_sig": g("SOX2", "MKI67"),
        "Imm_sig":  g("DCX", "STMN2"),
        "Pan_sig":  g("RBFOX3", "DCX", "STMN2", "NEUROD6"),
        "Glia_sig": g("AQP4", "PLP1", "PDGFRA", "CSF1R", "GFAP", "MBP"),
    })


def vote_neuron_glia(leiden, sig):
    """Per-cluster keep/drop + sub-identity. Keep if max(EN,IN,Pan) >= Glia (or
    progenitor-dominated). Sub-identity = dominant signature (averages out dropout)."""
    df = sig.copy(); df["cl"] = np.asarray(leiden)
    means = df.groupby("cl").mean()
    keep, sub_lab = {}, {}
    for cl, row in means.iterrows():
        neuro = max(row["EN_sig"], row["IN_sig"], row["Pan_sig"])
        glia, prog = row["Glia_sig"], row["Prog_sig"]
        keep[cl] = bool((neuro >= glia) or ((prog >= glia) and (prog >= 0.5)))
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


# --------------------------------------------------------------------------- #
# AHBA C3 weights (standalone) + per-cell C3 projection
# --------------------------------------------------------------------------- #
def load_c3_weights(var, component="C3", ref_csv=DEFAULT_REF_WEIGHTS):
    """Signed AHBA `component` weights indexed by Ensembl, mapped from the
    symbol-indexed AHBA CSV via the integrated object's own var symbol column.
    Standalone — needs only the CSV + the input adata's var."""
    ahba = pd.read_csv(ref_csv, index_col=0)[[component]]   # index = gene symbol
    sym_col = next((c for c in ["gene_symbol", "feature_name", "gene_name", "symbol"]
                    if c in var.columns), None)
    if sym_col is None:                                     # assume var_names are symbols
        return ahba[component]
    sym2ens = pd.Series(np.asarray(var.index), index=var[sym_col].astype(str).values)
    sym2ens = sym2ens[~sym2ens.index.duplicated(keep="first")]
    common = ahba.index.intersection(sym2ens.index)
    w = pd.Series(ahba.loc[common, component].values, index=sym2ens.loc[common].values)
    return w[~w.index.duplicated(keep="first")]


def c3_per_cell(Xcounts, var_names, weights):
    """Per-cell C3 scores from raw counts, sparse-friendly (no densify).
    Returns (signed, pos): signed = Σ w_g·log1p(CPM_g) over all C3 genes
    (depth-robust signed_logcpm); pos = same restricted to C3+ (w>0) genes."""
    vn = pd.Index(var_names)
    common = vn.intersection(weights.index)
    cols = np.array([vn.get_loc(g) for g in common])
    wv = weights.loc[common].values.astype(np.float64)
    tot = np.asarray(Xcounts.sum(1)).ravel(); tot[tot == 0] = 1.0
    Xc = Xcounts[:, cols].astype(np.float64)
    cpm = sp.diags(1e6 / tot) @ Xc
    cpm = cpm.tocsr(); cpm.data = np.log1p(cpm.data)
    signed = np.asarray(cpm @ wv).ravel()
    pm = wv > 0
    pos = np.asarray(cpm[:, pm] @ wv[pm]).ravel()
    print(f"  C3 per-cell: {len(common)} genes ({int(pm.sum())}+ / {int((~pm).sum())}-)", flush=True)
    return signed, pos


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def umap_coords(rep, n_neighbors=15, min_dist=0.5, spread=1.0, init="spectral", paga_groups=None):
    """UMAP on a representation with given params; optional PAGA-init."""
    tmp = ad.AnnData(np.zeros((rep.shape[0], 1)))
    tmp.obsm["R"] = rep
    sc.pp.neighbors(tmp, use_rep="R", n_neighbors=n_neighbors)
    init_pos = init
    if init == "paga" and paga_groups is not None:
        tmp.obs["g"] = pd.Categorical(np.asarray(paga_groups).astype(str))
        sc.tl.paga(tmp, groups="g")
        sc.pl.paga(tmp, plot=False)
        init_pos = "paga"
    sc.tl.umap(tmp, min_dist=min_dist, spread=spread, init_pos=init_pos)
    return tmp.obsm["X_umap"]


def _leiden(adata, res, key):
    try:
        sc.tl.leiden(adata, resolution=res, flavor="igraph", n_iterations=2,
                     directed=False, key_added=key)
    except TypeError:
        sc.tl.leiden(adata, resolution=res, key_added=key)


# --------------------------------------------------------------------------- #
# plotting helpers (diagnostics)
# --------------------------------------------------------------------------- #
def _sc_cat(ax, um, lab, title, s=4, legend=True):
    lab = pd.Series(lab).astype(str).values
    for k in sorted(pd.unique(lab), key=lambda k: -(lab == k).sum()):
        m = lab == k
        ax.scatter(um[m, 0], um[m, 1], s=s, c=CMAP.get(k, None), label=f"{k} ({m.sum()})")
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(markerscale=3, fontsize=8, loc="best")


def _sc_val(ax, um, val, title, cmap="viridis", s=4):
    sca = ax.scatter(um[:, 0], um[:, 1], c=val, s=s, cmap=cmap)
    plt.colorbar(sca, ax=ax, fraction=.045); ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------------------- #
# main build
# --------------------------------------------------------------------------- #
def build_manifold(integrated_path, out_path, *, dataset, chem=None, all_cells=False,
                   nn=NN, max_cells=MAX_CELLS, age_max=AGE_MAX, compute_c3=True,
                   ref_csv=DEFAULT_REF_WEIGHTS, fig_dir=None, fig_prefix="manifold", seed=SEED):
    """Build and write a neuron_manifold.h5ad from an integrated.h5ad. Returns the
    exported AnnData. If `fig_dir` is given, also writes diagnostic figures."""
    print(f"\n=== build_manifold {dataset} (all_cells={all_cells})", flush=True)
    rng = np.random.default_rng(seed)
    a = ad.read_h5ad(integrated_path, backed="r")
    obs = a.obs
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    mask = np.ones(len(age), bool) if all_cells else ((age >= 0) & (age < age_max))
    if chem and "chemistry" in obs:
        mask &= obs["chemistry"].astype(str).values == chem
    idx = np.where(mask)[0]
    aidx = age[idx]
    print(f"  matched cells: {len(idx):,} "
          f"(prenatal age<0: {int(np.nansum(aidx < 0)):,}; NaN-age: {int(np.isnan(aidx).sum()):,})", flush=True)
    if not all_cells and len(idx) > max_cells:
        bins = pd.qcut(aidx, q=min(10, len(np.unique(aidx))), duplicates="drop")
        per = max_cells // len(bins.categories)
        idx = np.sort(np.concatenate([rng.choice(idx[bins == c], min(per, int((bins == c).sum())),
                                                  replace=False) for c in bins.categories]))
    print(f"  using: {len(idx):,} cells", flush=True)
    sub = a[idx].to_memory()
    vn = list(sub.var_names)

    X = sub.layers["counts"] if "counts" in sub.layers else sub.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    expr = {s: (np.log1p(np.asarray(X[:, vn.index(e)].todense()).ravel() / tot * 1e4)
                if e in vn else np.zeros(len(idx))) for s, e in ENS.items()}
    sig = signatures(expr)

    nat_fine = next((c for c in ["cell_type_raw", "subclass", "Cell_Type"] if c in sub.obs), None)
    sub.obs["native_fine"] = sub.obs[nat_fine].astype(str) if nat_fine else "?"
    sub.obs["native_broad_fixed"] = fine_to_broad(sub.obs["native_fine"].values, dataset=dataset)

    # representation: PCA(30) on raw counts (single batch; no scVI)
    b = ad.AnnData(X.copy(), obs=sub.obs[[]].copy(), var=sub.var.copy())
    sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    sc.pp.highly_variable_genes(b, n_top_genes=2000)
    b = b[:, b.var.highly_variable].copy()
    sc.pp.scale(b, max_value=10); sc.tl.pca(b, n_comps=30)
    rep = b.obsm["X_pca"]; del b
    print("  representation = PCA(30) on raw counts (HVG 2000); no scVI", flush=True)

    # cluster all -> vote neuron vs glia
    sub.obsm["R"] = rep
    sc.pp.neighbors(sub, use_rep="R", n_neighbors=nn)
    _leiden(sub, 1.5, "leiden_all")
    leiden_all = sub.obs["leiden_all"].values
    keep, sublab_all, _ = vote_neuron_glia(leiden_all, sig)
    print(f"  Leiden(all)={len(np.unique(leiden_all))}; neuron+prog kept = "
          f"{keep.sum():,}/{len(keep):,} ({keep.mean():.1%})", flush=True)
    print(f"  kept sub-identity: {dict(pd.Series(sublab_all[keep]).value_counts())}", flush=True)
    print("  native_broad_fixed x kept:")
    print(pd.crosstab(sub.obs['native_broad_fixed'].values, np.where(keep, 'NEURON', 'glia'),
                      normalize='index').round(3).to_string())

    # neuron subset
    nsub = sub[keep].copy()
    nrep = rep[keep]
    nexpr = {k: v[keep] for k, v in expr.items()}
    nsig = sig[keep].reset_index(drop=True)
    nlab = sublab_all[keep]
    nage = age[idx][keep]
    c3_signed_n = c3_pos_n = None
    if compute_c3:
        weights = load_c3_weights(sub.var, "C3", ref_csv)
        c3_signed_n, c3_pos_n = c3_per_cell(X[keep], vn, weights)

    nsub.obsm["R"] = nrep
    sc.pp.neighbors(nsub, use_rep="R", n_neighbors=nn)
    _leiden(nsub, 1.0, "leiden_n")
    leiden_n = nsub.obs["leiden_n"].values

    # DPT-seed rooted at progenitor/immature pole
    sc.tl.diffmap(nsub)
    nsub.uns["iroot"] = int(np.argmax(nsig["Prog_sig"].values + nsig["Imm_sig"].values))
    sc.tl.dpt(nsub)
    dpt = nsub.obs["dpt_pseudotime"].values

    main_um = umap_coords(nrep, nn, 0.5, init="paga", paga_groups=leiden_n)

    # ---- write compact neuron_manifold.h5ad ----
    obs_export = pd.DataFrame({
        "native_fine": nsub.obs["native_fine"].values,
        "native_broad_fixed": nsub.obs["native_broad_fixed"].values,
        "cluster_vote": nlab, "leiden_n": np.asarray(leiden_n).astype(str),
        "age": nage, "dpt_seed": dpt,
        "EN_sig": nsig["EN_sig"].values, "IN_sig": nsig["IN_sig"].values,
        "Imm_sig": nsig["Imm_sig"].values, "Prog_sig": nsig["Prog_sig"].values,
        "Pan_sig": nsig["Pan_sig"].values, "Glia_sig": nsig["Glia_sig"].values,
    }, index=nsub.obs_names)
    if compute_c3:
        obs_export["C3_signed"] = c3_signed_n
        obs_export["C3_pos"] = c3_pos_n
    for g in ENS:
        obs_export[f"expr_{g}"] = nexpr[g]
    exp = ad.AnnData(X=sp.csr_matrix((len(nlab), 1), dtype="float32"), obs=obs_export)
    exp.obsm["X_pca"] = np.asarray(nrep)
    exp.obsm["X_diffmap"] = np.asarray(nsub.obsm["X_diffmap"])
    exp.obsm["X_umap_pagainit"] = np.asarray(main_um)
    exp.obsp["connectivities"] = nsub.obsp["connectivities"]
    exp.obsp["distances"] = nsub.obsp["distances"]
    exp.uns["neighbors"] = nsub.uns["neighbors"]
    exp.uns["iroot"] = int(nsub.uns["iroot"])
    exp.uns["manifold_meta"] = {"dataset": dataset, "n_neighbors": nn, "all_cells": all_cells,
                                "representation": "PCA30_on_counts", "n_cells": int(len(nlab))}
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    exp.write(out_path)
    print(f"  wrote {out_path}", flush=True)

    if fig_dir is not None:
        _diagnostic_figures(Path(fig_dir), fig_prefix, dataset, nsub, nrep, nlab, leiden_n,
                            nage, dpt, nsig, nexpr, main_um, c3_signed_n, c3_pos_n, nn)
    return exp


def _diagnostic_figures(fig_dir, prefix, dataset, nsub, nrep, nlab, leiden_n, nage, dpt,
                        nsig, nexpr, main_um, c3_signed, c3_pos, nn):
    """Manifold panel (identity/age/DPT/C3/markers) + embedding parameter sweep."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = dataset.replace("-", "_").lower()
    en_m = np.isin(nlab, ["ExN", "Immature_neuron"])
    has_c3 = c3_signed is not None

    # parameter sweep (islands -> continuum)
    embeds = {
        "UMAP local\n(nn=15)": umap_coords(nrep, 15, 0.5),
        "UMAP global\n(nn=50)": umap_coords(nrep, 50, 0.5),
        "UMAP spread\n(nn=50,md=0.9,spr=2)": umap_coords(nrep, 50, 0.9, spread=2.0),
        "UMAP PAGA-init\n(nn=50)": main_um,
        "DiffusionMap\n(DC1 vs DC2)": np.asarray(nsub.obsm["X_diffmap"])[:, 1:3],
    }
    if nsub.n_obs <= FA2_MAX:
        try:
            sc.tl.draw_graph(nsub, layout="fa")
            embeds["ForceAtlas2"] = nsub.obsm["X_draw_graph_fa"]
        except Exception as e:
            print(f"  ForceAtlas2 unavailable ({e})", flush=True)
    else:
        print(f"  skipping ForceAtlas2 (n={nsub.n_obs:,} > {FA2_MAX:,})", flush=True)

    # ---- Figure A: manifold (5 rows: identity/age/DPT; signatures; C3; markers x2) ----
    fig = plt.figure(figsize=(22, 20))
    gs = gridspec.GridSpec(5, 4, height_ratios=[2.0, 2.0, 2.0, 1, 1], hspace=0.3, wspace=0.2)
    ax = fig.add_subplot(gs[0, 0]); _sc_cat(ax, main_um, nlab, "cluster-vote sub-identity")
    ax = fig.add_subplot(gs[0, 1]); _sc_cat(ax, main_um, nsub.obs["native_broad_fixed"].values, "native (fixed broad)")
    ax = fig.add_subplot(gs[0, 2]); _sc_val(ax, main_um, nage, "age (years)")
    ax = fig.add_subplot(gs[0, 3]); _sc_val(ax, main_um, dpt, "DPT-seed pseudotime", cmap="plasma")
    ax = fig.add_subplot(gs[1, 0]); _sc_cat(ax, main_um, leiden_n, "Leiden (neurons)", legend=False)
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(nage[en_m], dpt[en_m], s=3, alpha=.3, c="#d62728")
    ax.set_xlabel("age (years)"); ax.set_ylabel("DPT-seed"); ax.set_title("EN-lineage: DPT vs age", fontsize=12)
    ax = fig.add_subplot(gs[1, 2]); _sc_val(ax, main_um, nsig["EN_sig"].values, "EN_sig (SLC17A7+SATB2)", cmap="magma")
    ax = fig.add_subplot(gs[1, 3]); _sc_val(ax, main_um, nsig["IN_sig"].values, "IN_sig (GAD1/2+SLC32A1)", cmap="magma")
    if has_c3:
        ax = fig.add_subplot(gs[2, 0]); _sc_val(ax, main_um, c3_pos, "C3+ (pos pole)", cmap="viridis")
        ax = fig.add_subplot(gs[2, 1]); _sc_val(ax, main_um, c3_signed, "C3 signed_logcpm", cmap="viridis")
        ax = fig.add_subplot(gs[2, 2])
        ax.scatter(nage[en_m], c3_pos[en_m], s=3, alpha=.3, c="#1f77b4")
        ax.set_xlabel("age"); ax.set_ylabel("C3+"); ax.set_title("EN-lineage: C3+ vs age", fontsize=12)
        ax = fig.add_subplot(gs[2, 3])
        ax.scatter(dpt[en_m], c3_pos[en_m], s=3, alpha=.3, c="#9467bd")
        ax.set_xlabel("DPT-seed"); ax.set_ylabel("C3+"); ax.set_title("EN-lineage: C3+ vs DPT-seed", fontsize=12)
    for i, g in enumerate(MARKER_PANEL):
        ax = fig.add_subplot(gs[3 + i // 4, i % 4]); _sc_val(ax, main_um, nexpr[g], g, cmap="magma", s=3)
    fig.suptitle(f"{dataset}: neuron+progenitor manifold (n={nsub.n_obs:,}) — PAGA-init UMAP (nn={nn}) "
                 f"on PCA-of-counts", y=0.995, fontsize=15)
    fig.savefig(fig_dir / f"{prefix}_neuron_manifold_{tag}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- Figure B: embedding sweep coloured by DPT ----
    items = list(embeds.items())
    ncol = 3; nrow = int(np.ceil(len(items) / ncol))
    fig2, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (ttl, um) in zip(axes, items):
        _sc_val(ax, um, dpt, ttl, cmap="plasma", s=3)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig2.suptitle(f"{dataset}: same neuron subset (PCA-of-counts), embeddings coloured by DPT — "
                  f"n_neighbors/min_dist/spread/PAGA-init/diffmap control islands-vs-continuum", y=1.0, fontsize=13)
    fig2.tight_layout(); fig2.savefig(fig_dir / f"{prefix}_embedding_sweep_{tag}.png", dpi=115, bbox_inches="tight"); plt.close(fig2)
    print(f"  wrote figures -> {fig_dir}/{prefix}_neuron_manifold_{tag}.png (+sweep)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integrated", required=True, help="integrated.h5ad (run_pipeline.py output)")
    ap.add_argument("--out", required=True, help="output neuron_manifold.h5ad path")
    ap.add_argument("--dataset", required=True, help="dataset name (label)")
    ap.add_argument("--chem", default=None, help="chemistry filter (e.g. V3)")
    ap.add_argument("--all-cells", action="store_true", help="keep all cells incl. prenatal/NaN-age, no subsample")
    ap.add_argument("--no-c3", action="store_true", help="skip per-cell C3 projection")
    ap.add_argument("--nn", type=int, default=NN)
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--fig-prefix", default="manifold")
    a = ap.parse_args()
    build_manifold(a.integrated, a.out, dataset=a.dataset, chem=a.chem, all_cells=a.all_cells,
                   nn=a.nn, compute_c3=not a.no_c3, fig_dir=a.fig_dir, fig_prefix=a.fig_prefix)


if __name__ == "__main__":
    main()
