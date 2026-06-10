#!/usr/bin/env python3
"""Young-donor (<5y) UMAPs to validate the ExN definition (sbatch).

For PsychAD and Velmeshev-V3 SEPARATELY, take ALL cells from <5y donors, recompute
a UMAP on that subset's **scVI latent** (X_scVI — unsupervised, batch-corrected
within the dataset; chosen over scANVI to avoid label circularity, since scANVI is
trained on the native labels we are scrutinising), and show:
  - native broad + native fine labels (the aging/dev reference)
  - our marker-based labels (computed here from raw counts, same logic as
    code/annotation_by_markers.py: InN if max(GAD1,GAD2,SLC32A1)>=10; ExN if
    RBFOX3 or DCX>=1; glia by AQP4/PLP1/...)
  - age
  - per-marker expression for neuronal-differentiation and ExN/InN-identity genes
Goal: do the unsupervised clusters of young cells agree with native labels or with
the marker labels, and does the EN-% discrepancy make visual sense in each dataset?

SUBMIT:
  sbatch --time=02:00:00 --mem=200G scripts/run_script.sh scripts/c3_maturation/s09_young_umaps.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L

AGE_MAX = 5.0

# marker Ensembl IDs
GENES = {
    # neuronal differentiation axis (progenitor -> immature -> mature)
    "SOX2": "ENSG00000181449", "MKI67": "ENSG00000148773", "DCX": "ENSG00000077279",
    "STMN2": "ENSG00000104435", "NEUROD6": "ENSG00000164600", "RBFOX3": "ENSG00000167281",
    # ExN identity
    "SLC17A7": "ENSG00000104888", "SATB2": "ENSG00000119042",
    # InN identity
    "GAD1": "ENSG00000128683", "GAD2": "ENSG00000136750", "SLC32A1": "ENSG00000101438",
    "DLX2": "ENSG00000115844",
    # glia context
    "AQP4": "ENSG00000171885", "PDGFRA": "ENSG00000134853", "PLP1": "ENSG00000123560",
    "CSF1R": "ENSG00000182578",
}
PANEL_ORDER = ["SOX2", "MKI67", "DCX", "STMN2", "NEUROD6", "RBFOX3",
               "SLC17A7", "SATB2", "GAD1", "GAD2", "SLC32A1", "DLX2",
               "AQP4", "PDGFRA", "PLP1", "CSF1R"]

DATASETS = {
    "PsychAD": dict(path=L.B / "PsychAD_noage_tuning5/scvi_output/integrated.h5ad",
                    parquet=L.B / "PsychAD_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
                    chem=None),
    "Velmeshev-V3": dict(path=L.B / "Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad",
                         parquet=L.B / "Vel_prepost_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
                         chem="V3"),
}


def marker_to_broad(s):
    """ExN*->EN, InN->IN, glia types->Glia, Unknown/skipped/NaN->Unknown."""
    s = pd.Series(s, dtype=object).fillna("Unknown").astype(str)
    return np.where(s.str.startswith("ExN"), "EN",
           np.where(s == "InN", "IN",
           np.where(s.isin(["Astro", "Oligo", "OPC", "Micro"]), "Glia", "Unknown")))


def marker_classify(counts_csr, vn):
    """Vectorised version of annotation_by_markers.classify_cell on a cell x gene
    CSR (raw counts). Returns array of labels."""
    pos = {g: i for i, g in enumerate(vn)}
    def col(sym):
        ens = GENES.get(sym)
        if ens is None or ens not in pos:
            # some (GFAP/MBP/etc) not in GENES dict; map by known IDs
            extra = {"GFAP": "ENSG00000131095", "MBP": "ENSG00000197971",
                     "CX3CR1": "ENSG00000168329", "P2RY12": "ENSG00000169313",
                     "RBFOX1": "ENSG00000078328"}
            ens = extra.get(sym)
        if ens is None or ens not in pos:
            return np.zeros(counts_csr.shape[0])
        return np.asarray(counts_csr[:, pos[ens]].todense()).ravel()
    GAD1, GAD2, SLC32A1 = col("GAD1"), col("GAD2"), col("SLC32A1")
    RBFOX3, DCX, RBFOX1 = col("RBFOX3"), col("DCX"), col("RBFOX1")
    AQP4, GFAP = col("AQP4"), col("GFAP")
    MBP, PLP1 = col("MBP"), col("PLP1")
    CX3CR1, P2RY12 = col("CX3CR1"), col("P2RY12")
    PDGFRA = col("PDGFRA")
    n = counts_csr.shape[0]
    lab = np.full(n, "Unknown", dtype=object)
    is_inn = np.maximum.reduce([GAD1, GAD2, SLC32A1]) >= 10
    rbf3, dcx = RBFOX3 >= 1, DCX >= 1
    is_astro = (AQP4 >= 1) | (GFAP >= 1)
    is_oligo = (MBP >= 1) | (PLP1 >= 1)
    is_micro = (CX3CR1 >= 1) | (P2RY12 >= 1)
    is_opc = PDGFRA >= 1
    # order matters (later assignments only on still-Unknown via masks)
    lab[(~is_inn) & is_opc] = "OPC"
    lab[(~is_inn) & is_micro] = "Micro"
    lab[(~is_inn) & is_oligo] = "Oligo"
    lab[(~is_inn) & is_astro] = "Astro"
    lab[(~is_inn) & (RBFOX1 >= 1)] = "ExN_weak"
    lab[(~is_inn) & dcx & (~rbf3)] = "ExN_immature"
    lab[(~is_inn) & rbf3] = "ExN_mature"
    lab[is_inn] = "InN"
    return lab


def broad_of(marker_lab):
    s = pd.Series(marker_lab)
    return np.where(s.str.startswith("ExN"), "EN",
           np.where(s == "InN", "IN",
           np.where(s == "Unknown", "Unknown", "Glia")))


def process(name, cfg):
    print(f"\n=== {name}: {cfg['path']}", flush=True)
    a = ad.read_h5ad(cfg["path"], backed="r")
    obs = a.obs
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    mask = (age >= 0) & (age < AGE_MAX)
    if cfg["chem"] is not None and "chemistry" in obs:
        mask &= (obs["chemistry"].astype(str).values == cfg["chem"])
    idx = np.where(mask)[0]
    print(f"  <{AGE_MAX}y cells: {len(idx):,}", flush=True)
    sub = a[idx].to_memory()
    vn = list(sub.var_names)

    X = sub.layers["counts"] if "counts" in sub.layers else sub.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    # marker labels: use the VALIDATED cached parquet (annotation_by_markers.py on
    # RAW per-dataset h5ads), NOT a recompute from the integrated counts layer
    # (which is not the same raw counts and over-calls EN).
    ma = pd.read_parquet(cfg["parquet"])
    sub.obs["marker"] = ma["marker_annotation"].reindex(sub.obs_names).astype(str).values
    sub.obs["marker_broad"] = marker_to_broad(sub.obs["marker"].values)
    n_match = (sub.obs["marker"] != "nan").sum()
    print(f"  marker_annotation joined from parquet: {n_match}/{len(sub)} matched", flush=True)

    # native labels
    nat_fine = next((c for c in ["cell_type_raw", "subclass", "Cell_Type"] if c in sub.obs), None)
    nat_broad = next((c for c in ["cell_class", "cell_class_original"] if c in sub.obs), None)
    sub.obs["native_fine"] = sub.obs[nat_fine].astype(str) if nat_fine else "?"
    sub.obs["native_broad"] = sub.obs[nat_broad].astype(str) if nat_broad else "?"
    sub.obs["age_y"] = pd.to_numeric(sub.obs["age_years"], errors="coerce").values

    # marker expression (log1p CPM) — same for both embeddings
    tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
    expr = {}
    for g in PANEL_ORDER:
        ens = GENES[g]
        expr[g] = np.log1p(np.asarray(X[:, vn.index(ens)].todense()).ravel() / tot * 1e4) if ens in vn else np.zeros(len(idx))

    # ---- two embeddings ----
    # (1) scVI latent (unsupervised, batch-corrected within dataset)
    rep = "X_scVI" if "X_scVI" in sub.obsm else "X_scANVI"
    print(f"  scVI UMAP rep: {rep}", flush=True)
    sc.pp.neighbors(sub, use_rep=rep, n_neighbors=15)
    sc.tl.umap(sub)
    um_scvi = sub.obsm["X_umap"].copy()
    # (2) PCA on the young cells' OWN raw counts (no model): normalize_total ->
    #     log1p -> HVG(2000) -> scale -> PCA(30) -> UMAP
    print("  PCA-on-raw-counts UMAP ...", flush=True)
    b = sub.copy()
    b.X = (b.layers["counts"].copy() if "counts" in b.layers else b.X.copy())
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    sc.pp.highly_variable_genes(b, n_top_genes=2000)
    b = b[:, b.var.highly_variable].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=30)
    sc.pp.neighbors(b, use_rep="X_pca", n_neighbors=15)
    sc.tl.umap(b)
    um_pca = b.obsm["X_umap"].copy()

    def plot_panels(um, tag, rep_desc):
        label_panels = ["native_broad", "native_fine", "marker", "age_y"]
        n_panels = len(label_panels) + len(PANEL_ORDER)
        ncol = 5
        nrow = int(np.ceil(n_panels / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
        axes = axes.ravel()
        pi = 0
        for col in label_panels:
            ax = axes[pi]; pi += 1
            if col == "age_y":
                sca = ax.scatter(um[:, 0], um[:, 1], c=sub.obs[col].values, s=3, cmap="viridis")
                plt.colorbar(sca, ax=ax, fraction=.04)
            else:
                top = sub.obs[col].astype(str).value_counts().head(12).index
                for k in pd.Categorical(sub.obs[col].astype(str)).categories:
                    m = (sub.obs[col].astype(str) == k).values
                    ax.scatter(um[m, 0], um[m, 1], s=3,
                               label=(f"{k} ({m.sum()})" if k in top else None))
                ax.legend(markerscale=3, fontsize=5, loc="best", ncol=1)
            ax.set_title(col, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        for g in PANEL_ORDER:
            ax = axes[pi]; pi += 1
            sca = ax.scatter(um[:, 0], um[:, 1], c=expr[g], s=3, cmap="magma")
            plt.colorbar(sca, ax=ax, fraction=.04)
            ax.set_title(g, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
        for j in range(pi, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"{name}: <{AGE_MAX:.0f}y donors — {rep_desc}. n={len(idx):,} cells. "
                     f"Native (aging/dev ref) vs marker labels + differentiation markers",
                     y=1.005, fontsize=13)
        fig.tight_layout()
        out = L.OUT_DIR / f"s09_young_{tag}_{name.replace('-', '_').lower()}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}", flush=True)

    plot_panels(um_scvi, "scvi", f"UMAP on {rep} (scVI latent, unsupervised, batch-corrected)")
    plot_panels(um_pca, "pca", "UMAP on PCA of raw counts (normalize_total+log1p+HVG2000+scale+PCA30; no model)")

    # composition + native-vs-marker crosstab
    print(f"  marker_broad composition (<{AGE_MAX}y): "
          f"{dict(pd.Series(sub.obs['marker_broad']).value_counts())}")
    ct = pd.crosstab(sub.obs["native_broad"], sub.obs["marker_broad"], normalize="index").round(3)
    print("  native_broad x marker_broad (row-normalised):")
    print(ct.to_string())
    ct.to_csv(L.OUT_DIR / f"s09_{name.replace('-', '_').lower()}_native_vs_marker.csv")
    sub.obs[["age_y", "native_broad", "native_fine", "marker", "marker_broad"]].to_parquet(
        L.OUT_DIR / f"s09_{name.replace('-', '_').lower()}_obs.parquet")


def main():
    for name, cfg in DATASETS.items():
        process(name, cfg)
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
