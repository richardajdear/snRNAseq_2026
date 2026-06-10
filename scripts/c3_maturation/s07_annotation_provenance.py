#!/usr/bin/env python3
"""Annotation provenance / cross-check for PsychAD (sbatch).

Three independent labelings exist; this script characterises and compares them:
  (1) native `cell_type_raw` (= PsychAD `subclass`, from the aging/dementia reference)
  (2) `cell_type_aligned`  = scANVI trained on cell_type_raw (=> inherits the native
      reference; this is what Steps 1/2 used for EN subtypes)
  (3) `marker_annotation`  = code/annotation_by_markers.py: hard-threshold marker
      classifier on RAW counts (InN if max(GAD1,GAD2,SLC32A1)>=10; ExN if RBFOX3
      or DCX>=1; glia by AQP4/PLP1/...); independent of reference AND of scANVI.

Outputs (for REPORT_annotation.md):
  - UMAP computed on the scVI/scANVI LATENT space (the embedding used for
    integration + label transfer), coloured by each labeling, by age, and by
    ground-truth marker genes (SLC17A7, GAD1, AQP4).
  - EN-fraction vs age for all three labelings (reconcile the "5% EN in young" claim).
  - native-vs-marker confusion, focused on young (<10y) donors.

SUBMIT (do NOT run on login node — 68 GB object):
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=02:00:00 --mem=200G scripts/run_script.sh scripts/c3_maturation/s07_annotation_provenance.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L

INTEGRATED = L.B / "PsychAD_noage_tuning5/scvi_output/integrated.h5ad"
PARQUET = L.B / "PsychAD_noage_tuning5/pseudobulk_output/manual_annotations.parquet"
N_SUB = 80_000
MARKERS = {"SLC17A7": "ENSG00000104888", "GAD1": "ENSG00000128683",
           "AQP4": "ENSG00000171885", "RBFOX3": "ENSG00000167281"}
EXN_MARKER = {"ExN_mature", "ExN_immature", "ExN_weak"}


def broad_native(s):
    s = s.astype(str)
    return np.where(s.str.contains("EN_|^EN$|Excit", case=False, regex=True), "EN",
           np.where(s.str.contains("IN_|Inhib|^IN$", case=False, regex=True), "IN",
           np.where(s.str.contains("Astro|Oligo|OPC|Micro|Endo|VLMC|PVM|Immune|Mural", case=False, regex=True),
                    "Glia/other", "other")))


def main():
    print(f"Loading {INTEGRATED} (backed) ...", flush=True)
    a = ad.read_h5ad(INTEGRATED, backed="r")
    print(f"  shape {a.shape}", flush=True)
    print(f"  obs cols: {list(a.obs.columns)}", flush=True)
    print(f"  obsm keys: {list(a.obsm.keys())}", flush=True)

    obs = a.obs
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    # subsample (oversample young so they are visible)
    rng = np.random.default_rng(0)
    w = np.where(age < 10, 6.0, 1.0)
    w = np.where(np.isfinite(age), w, 0)
    p = w / w.sum()
    idx = np.sort(rng.choice(a.n_obs, size=min(N_SUB, a.n_obs), replace=False, p=p))
    print(f"  subsample: {len(idx)} cells (young-oversampled)", flush=True)

    sub = a[idx].to_memory()
    # marker annotation join
    ma = pd.read_parquet(PARQUET)
    sub.obs["marker_annotation"] = ma["marker_annotation"].reindex(sub.obs_names).values

    # latent rep for UMAP
    lat = "X_scANVI" if "X_scANVI" in sub.obsm else ("X_scVI" if "X_scVI" in sub.obsm else None)
    print(f"  latent rep for UMAP: {lat}", flush=True)
    if lat is None:
        raise RuntimeError(f"no scVI/scANVI latent in obsm: {list(sub.obsm.keys())}")
    sc.pp.neighbors(sub, use_rep=lat, n_neighbors=15)
    sc.tl.umap(sub)

    # native + aligned broad labels
    nat_col = "cell_type_raw" if "cell_type_raw" in sub.obs else "subclass"
    sub.obs["native_broad"] = broad_native(sub.obs[nat_col])
    sub.obs["aligned_broad"] = broad_native(sub.obs.get("cell_type_aligned", sub.obs[nat_col]))
    sub.obs["age_years_n"] = pd.to_numeric(sub.obs["age_years"], errors="coerce").values

    # marker gene log1p CPM for ground truth
    import scipy.sparse as sp
    X = sub.layers["counts"] if "counts" in sub.layers else sub.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    vn = list(sub.var_names)
    for g, ens in MARKERS.items():
        if ens in vn:
            col = np.asarray(X[:, vn.index(ens)].todense()).ravel()
            sub.obs[f"mk_{g}"] = np.log1p(col / tot * 1e4)

    um = sub.obsm["X_umap"]
    sub.obs["UMAP1"], sub.obs["UMAP2"] = um[:, 0], um[:, 1]
    sub.obs.to_parquet(L.OUT_DIR / "s07_umap_obs.parquet")

    # ---- figure 1: UMAP coloured by labelings / age / markers ----
    panels = [("native_broad", "cat"), ("aligned_broad", "cat"),
              ("marker_annotation", "cat"), ("age_years_n", "num"),
              ("mk_SLC17A7", "num"), ("mk_GAD1", "num"), ("mk_AQP4", "num"),
              ("mk_RBFOX3", "num")]
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for ax, (col, kind) in zip(axes.ravel(), panels):
        if col not in sub.obs:
            ax.set_visible(False); continue
        if kind == "num":
            sca = ax.scatter(um[:, 0], um[:, 1], c=sub.obs[col].values, s=2, cmap="viridis")
            plt.colorbar(sca, ax=ax, fraction=.04)
        else:
            cats = pd.Categorical(sub.obs[col].astype(str))
            for k, code in zip(cats.categories, range(len(cats.categories))):
                m = cats == k
                ax.scatter(um[m, 0], um[m, 1], s=2, label=f"{k} ({m.sum()})")
            ax.legend(markerscale=4, fontsize=7, loc="best")
        ax.set_title(col); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("PsychAD UMAP on scVI/scANVI latent — annotation comparison", y=1.0, fontsize=14)
    fig.tight_layout()
    fig.savefig(L.OUT_DIR / "s07_umap_annotations.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- EN fraction vs age, three labelings (full dataset obs, not subsample) ----
    full = obs.copy()
    full["age"] = age
    full["marker_annotation"] = ma["marker_annotation"].reindex(full.index).values
    full["native_EN"] = broad_native(full[nat_col]) == "EN"
    full["aligned_EN"] = broad_native(full.get("cell_type_aligned", full[nat_col])) == "EN"
    full["marker_EN"] = full["marker_annotation"].astype(str).isin(EXN_MARKER)
    full["agebin"] = pd.cut(full["age"], [-1, 2, 5, 10, 20, 40, 200],
                            labels=["<2", "2-5", "5-10", "10-20", "20-40", "40+"])
    comp = full.groupby("agebin", observed=True).agg(
        n_cells=("age", "size"),
        native_EN_frac=("native_EN", "mean"),
        aligned_EN_frac=("aligned_EN", "mean"),
        marker_EN_frac=("marker_EN", "mean"),
        marker_unknown_frac=("marker_annotation", lambda s: (s.astype(str) == "Unknown").mean()),
    ).round(3)
    print("\n--- EN fraction vs age, by labeling (full PsychAD) ---")
    print(comp.to_string())
    comp.to_csv(L.OUT_DIR / "s07_en_fraction_by_labeling.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(comp))
    for c, lab in [("native_EN_frac", "native (subclass/aging-ref)"),
                   ("aligned_EN_frac", "cell_type_aligned (scANVI)"),
                   ("marker_EN_frac", "marker_annotation (GAD/RBFOX3)")]:
        ax.plot(x, comp[c], marker="o", label=lab)
    ax.set_xticks(list(x)); ax.set_xticklabels(comp.index)
    ax.set_xlabel("age bin"); ax.set_ylabel("EN fraction"); ax.set_ylim(0, 1)
    ax.legend(); ax.set_title("PsychAD EN fraction vs age — three labelings")
    fig.tight_layout(); fig.savefig(L.OUT_DIR / "s07_en_fraction_by_labeling.png", dpi=140, bbox_inches="tight")

    # ---- native vs marker confusion in young (<10y) ----
    young = full[full["age"] < 10]
    ct = pd.crosstab(broad_native(young[nat_col]), young["marker_annotation"].astype(str),
                     normalize="index").round(3)
    print("\n--- young (<10y) native_broad x marker_annotation (row-normalised) ---")
    print(ct.to_string())
    ct.to_csv(L.OUT_DIR / "s07_young_native_vs_marker.csv")
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
