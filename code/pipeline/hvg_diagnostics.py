"""
HVG method comparison: seurat vs pearson_residuals integration diagnostics.

Loads integrated.h5ad from two scVI runs (differing only in HVG selection method)
and computes quantitative metrics + figures to determine which produces better
batch integration and cell type resolution.

Metrics
-------
1. HVG overlap
   - Jaccard similarity between selected gene sets
   - Recovery of known PFC cell type marker genes per method

2. Batch integration quality  (higher = better)
   - iLISI: mean local inverse Simpson index over batch labels in latent space.
     Range [1, n_batches]; normalized to [0, 1]. Measures how well cells from
     different batches are interleaved in the neighbourhood.
   - ASW_batch_norm: 1 - |silhouette(batch)|. Values near 1 = good mixing.

3. Cell type separation quality  (higher = better)
   - cLISI separation: 1 - normalised cLISI. Measures how cleanly cell types
     form distinct clusters in the neighbourhood.
   - ASW_celltype: silhouette width for broad cell class labels. Higher = tighter
     within-type clusters.

4. scANVI label transfer quality  (Wang cells only, ground truth available)
   - Accuracy and weighted F1 of cell_type_aligned vs cell_type_for_scanvi.
   - Confusion matrix (seurat run).

5. Prediction confidence
   - Distribution of cell_type_aligned_confidence across batches and methods.
   - % of cells above 0.5 and 0.8 thresholds.

Both iLISI and cLISI are computed on a subsample of the scANVI latent space
(falling back to X_scVI if X_scANVI is absent). All LISI computations use
k=60 nearest neighbours; ASW uses sklearn on the same subsample.

Usage
-----
    PYTHONPATH=code python -m pipeline.hvg_diagnostics \\
        --seurat   .../VelWangPsychAD_100k/scvi_output/integrated.h5ad \\
        --pearson  .../VelWangPsychAD_100k_pearson/scvi_output/integrated.h5ad \\
        --output_dir .../hvg_method_comparison

    # Faster (smaller subsample, skip UMAP re-plot):
    PYTHONPATH=code python -m pipeline.hvg_diagnostics ... --n_sample 10000 --no_umap
"""

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Known PFC cell type markers for recovery analysis
# ---------------------------------------------------------------------------
MARKERS = {
    "Excitatory":       ["SLC17A7", "SATB2", "CUX2", "RORB", "TBR1", "FEZF2", "NRGN"],
    "Inhibitory":       ["GAD1", "GAD2", "SLC32A1", "PVALB", "SST", "VIP", "LAMP5"],
    "Astrocytes":       ["GFAP", "AQP4", "SLC1A3", "ALDH1L1", "GJA1"],
    "Oligodendrocytes": ["MBP", "MOBP", "MOG", "PLP1", "OPALIN"],
    "OPC":              ["PDGFRA", "VCAN", "OLIG1", "OLIG2"],
    "Microglia":        ["CX3CR1", "P2RY12", "TMEM119", "CSF1R", "IBA1"],
    "Endothelial":      ["CLDN5", "FLT1", "VWF", "PECAM1"],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("hvg_diagnostics")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(output_dir / "hvg_diagnostics.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _load(path: Path, name: str, logger: logging.Logger) -> ad.AnnData:
    """Load h5ad in backed mode — we only need .obs, .var, .obsm."""
    logger.info(f"Loading {name}: {path}")
    adata = ad.read_h5ad(str(path), backed=True)
    logger.info(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    logger.info(f"  obsm: {list(adata.obsm.keys())}")
    logger.info(f"  obs:  {list(adata.obs.columns)}")
    return adata


def _pick_embedding(adata: ad.AnnData, preferred: str) -> str:
    """Return preferred embedding key, falling back to X_scANVI then X_scVI."""
    for key in [preferred, "X_scANVI", "X_scVI"]:
        if key in adata.obsm:
            return key
    raise KeyError(f"No recognised embedding found in obsm keys: {list(adata.obsm.keys())}")


def _encode(series: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    encoded = le.fit_transform(series.fillna("__NA__").astype(str))
    return encoded, le


# ---------------------------------------------------------------------------
# 1. HVG overlap
# ---------------------------------------------------------------------------

def hvg_overlap(
    adata_s: ad.AnnData,
    adata_p: ad.AnnData,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """Gene set overlap and known-marker recovery."""
    logger.info("\n── HVG Overlap ──────────────────────────────────────────")

    metrics = {}

    if "highly_variable" not in adata_s.var.columns or "highly_variable" not in adata_p.var.columns:
        logger.warning("'highly_variable' missing from .var in one or both files — overlap skipped")
        return metrics

    seurat_hvgs  = set(adata_s.var_names[adata_s.var["highly_variable"]])
    pearson_hvgs = set(adata_p.var_names[adata_p.var["highly_variable"]])
    shared       = seurat_hvgs & pearson_hvgs
    union        = seurat_hvgs | pearson_hvgs
    jaccard      = len(shared) / len(union)

    logger.info(f"  seurat: {len(seurat_hvgs):,}  pearson: {len(pearson_hvgs):,}")
    logger.info(f"  shared: {len(shared):,}  seurat-only: {len(seurat_hvgs - pearson_hvgs):,}"
                f"  pearson-only: {len(pearson_hvgs - seurat_hvgs):,}")
    logger.info(f"  Jaccard similarity: {jaccard:.3f}")

    metrics.update(
        n_seurat_hvgs=len(seurat_hvgs),
        n_pearson_hvgs=len(pearson_hvgs),
        n_shared_hvgs=len(shared),
        hvg_jaccard=jaccard,
    )

    # Marker recovery
    all_genes = set(adata_s.var_names)
    rows = []
    for ct, markers in MARKERS.items():
        present  = [m for m in markers if m in all_genes]
        in_s     = [m for m in present if m in seurat_hvgs]
        in_p     = [m for m in present if m in pearson_hvgs]
        rows.append(dict(
            cell_type=ct,
            n_markers=len(markers),
            n_in_data=len(present),
            n_seurat=len(in_s),
            n_pearson=len(in_p),
            pct_seurat=100 * len(in_s) / max(len(present), 1),
            pct_pearson=100 * len(in_p) / max(len(present), 1),
            seurat_genes=", ".join(in_s),
            pearson_genes=", ".join(in_p),
        ))
        logger.info(f"  {ct:20s}: seurat {len(in_s)}/{len(present)}  pearson {len(in_p)}/{len(present)}")

    marker_df = pd.DataFrame(rows)
    marker_df.to_csv(output_dir / "marker_recovery.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stacked bar of gene set composition
    ax = axes[0]
    n_seurat_only  = len(seurat_hvgs) - len(shared)
    n_pearson_only = len(pearson_hvgs) - len(shared)
    vals  = [n_seurat_only, len(shared), n_pearson_only]
    lbls  = ["seurat only", "shared", "pearson only"]
    cols  = ["#4878CF", "#6ACC65", "#D65F5F"]
    bars  = ax.bar(lbls, vals, color=cols, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%d", fontsize=10)
    ax.set_title(f"HVG overlap  (Jaccard = {jaccard:.3f})", fontsize=12)
    ax.set_ylabel("Number of genes")
    ax.set_ylim(0, max(vals) * 1.18)

    # Right: marker recovery per cell type
    ax = axes[1]
    x = np.arange(len(MARKERS))
    w = 0.36
    ax.bar(x - w / 2, marker_df["pct_seurat"],  w, label="seurat",            color="#4878CF")
    ax.bar(x + w / 2, marker_df["pct_pearson"], w, label="pearson_residuals", color="#D65F5F")
    ax.set_xticks(x)
    ax.set_xticklabels(marker_df["cell_type"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% known markers in HVG set")
    ax.set_title("Known marker recovery by cell type")
    ax.set_ylim(0, 120)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "hvg_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved hvg_overlap.png, marker_recovery.csv")

    return metrics


# ---------------------------------------------------------------------------
# 2. LISI + ASW integration metrics
# ---------------------------------------------------------------------------

def _lisi(X: np.ndarray, labels: np.ndarray, k: int = 60) -> np.ndarray:
    """
    Per-cell Local Inverse Simpson's Index.

    For each cell, the k nearest neighbours are found and the inverse Simpson
    index of their label distribution is computed:

        LISI_i = 1 / sum_l( (count_l / k)^2 )

    Range: [1, n_unique_labels]. High iLISI = well-mixed batches.
    Low cLISI = cell types are locally pure.

    Reference: Korsunsky et al. 2019, Nature Methods (Harmony).
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean",
                            n_jobs=-1, algorithm="auto")
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)
    neighbor_labels = labels[indices[:, 1:]]   # exclude self

    lisi_scores = np.empty(len(X), dtype=float)
    for i in range(len(X)):
        freqs = np.bincount(neighbor_labels[i], minlength=int(labels.max()) + 1)
        freqs = freqs[freqs > 0].astype(float) / k
        lisi_scores[i] = 1.0 / np.sum(freqs ** 2)
    return lisi_scores


def integration_metrics(
    adata: ad.AnnData,
    name: str,
    emb_key: str,
    batch_key: str,
    celltype_key: str,
    n_sample: int,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return (metrics_dict, ilisi_scores, clisi_scores) on a subsample."""
    logger.info(f"  {name}  embedding={emb_key}  n_sample={n_sample:,}")

    X       = np.asarray(adata.obsm[emb_key])
    batch,  le_b  = _encode(adata.obs[batch_key])
    ct,     le_ct = _encode(adata.obs[celltype_key])
    n_batch = len(le_b.classes_)
    n_ct    = len(le_ct.classes_)

    idx    = rng.choice(len(X), min(n_sample, len(X)), replace=False)
    X_sub  = X[idx]
    b_sub  = batch[idx]
    ct_sub = ct[idx]

    # iLISI — batch mixing (want HIGH)
    ilisi = _lisi(X_sub, b_sub, k=60)
    ilisi_norm = (ilisi - 1) / max(n_batch - 1, 1)          # → [0, 1]

    # cLISI — cell type purity (want LOW raw; report as separation score)
    clisi = _lisi(X_sub, ct_sub, k=60)
    clisi_norm = (clisi - 1) / max(n_ct - 1, 1)             # → [0, 1]
    ct_separation = 1.0 - float(clisi_norm.mean())           # → higher is better

    # ASW
    asw_batch  = silhouette_score(X_sub, b_sub,  metric="euclidean")
    asw_ct     = silhouette_score(X_sub, ct_sub, metric="euclidean")
    asw_batch_norm = 1.0 - abs(asw_batch)                    # → [0, 1] higher = better mixing

    m = {
        f"{name}_ilisi_norm":       float(ilisi_norm.mean()),
        f"{name}_ct_separation":    ct_separation,
        f"{name}_asw_batch_norm":   asw_batch_norm,
        f"{name}_asw_celltype":     float(asw_ct),
    }
    logger.info(
        f"    iLISI(norm)={m[f'{name}_ilisi_norm']:.3f}  "
        f"cLISI-sep={m[f'{name}_ct_separation']:.3f}  "
        f"ASW_batch(norm)={m[f'{name}_asw_batch_norm']:.3f}  "
        f"ASW_ct={m[f'{name}_asw_celltype']:.3f}"
    )
    return m, ilisi, clisi


# ---------------------------------------------------------------------------
# 3. scANVI label transfer quality (Wang cells)
# ---------------------------------------------------------------------------

def label_transfer_quality(
    adata_s: ad.AnnData,
    adata_p: ad.AnnData,
    gt_col: str,
    pred_col: str,
    conf_col: str,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    """
    Accuracy / F1 on Wang cells (gt_col != 'Unknown').

    Wang cells were used to train scANVI — this is an in-sample quality check,
    but divergence between methods still reveals which learned better
    representations (a method with poor batch integration will misclassify
    Wang cells whose neighbours are contaminated by other batches).
    """
    logger.info("\n── scANVI Label Transfer (Wang cells) ──────────────────")
    metrics = {}

    for name, adata in [("seurat", adata_s), ("pearson", adata_p)]:
        if pred_col not in adata.obs.columns:
            logger.warning(f"  {name}: '{pred_col}' missing — scANVI not run")
            continue

        obs = adata.obs.copy()
        wang = obs[(obs[gt_col] != "Unknown") & obs[gt_col].notna()]
        logger.info(f"  {name}: {len(wang):,} Wang cells with ground-truth labels")

        acc = accuracy_score(wang[gt_col], wang[pred_col])
        f1  = f1_score(wang[gt_col], wang[pred_col], average="weighted", zero_division=0)
        metrics[f"{name}_wang_accuracy"] = acc
        metrics[f"{name}_wang_f1"]       = f1
        logger.info(f"    accuracy={acc:.3f}  weighted-F1={f1:.3f}")

        if conf_col in obs.columns:
            conf = obs[conf_col].values
            metrics[f"{name}_median_conf"]  = float(np.median(conf))
            metrics[f"{name}_pct_conf_gt80"] = float((conf > 0.8).mean() * 100)
            metrics[f"{name}_pct_conf_gt50"] = float((conf > 0.5).mean() * 100)
            logger.info(
                f"    median confidence={metrics[f'{name}_median_conf']:.3f}  "
                f">0.5: {metrics[f'{name}_pct_conf_gt50']:.1f}%  "
                f">0.8: {metrics[f'{name}_pct_conf_gt80']:.1f}%"
            )

    # Confusion matrix for each method
    for name, adata in [("seurat", adata_s), ("pearson", adata_p)]:
        if pred_col not in adata.obs.columns:
            continue
        obs  = adata.obs.copy()
        wang = obs[(obs[gt_col] != "Unknown") & obs[gt_col].notna()]
        labels = sorted(wang[gt_col].unique())
        if len(labels) > 30:
            logger.info(f"  {name}: {len(labels)} label classes — skipping confusion matrix (>30)")
            continue
        cm = confusion_matrix(wang[gt_col], wang[pred_col], labels=labels, normalize="true")
        sz = max(7, len(labels) * 0.55)
        fig, ax = plt.subplots(figsize=(sz, sz * 0.9))
        sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
                    cmap="Blues", vmin=0, vmax=1, ax=ax, square=True,
                    linewidths=0.3, annot=len(labels) <= 15, fmt=".2f",
                    cbar_kws={"label": "Recall", "shrink": 0.6})
        ax.set_title(f"{name} HVG — Wang self-prediction (n={len(wang):,})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True label")
        plt.tight_layout()
        plt.savefig(output_dir / f"wang_confusion_{name}.png", dpi=130, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved wang_confusion_{name}.png")

    return metrics


# ---------------------------------------------------------------------------
# 4. Confidence histograms
# ---------------------------------------------------------------------------

def plot_confidence(
    adata_s: ad.AnnData,
    adata_p: ad.AnnData,
    conf_col: str,
    output_dir: Path,
    logger: logging.Logger,
):
    logger.info("\n── Confidence Distribution ──────────────────────────────")
    if conf_col not in adata_s.obs.columns or conf_col not in adata_p.obs.columns:
        logger.warning("Confidence column missing — skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    bins = np.linspace(0, 1, 41)
    palette = sns.color_palette("tab10", 4)

    for ax, (name, adata) in zip(axes, [("seurat", adata_s), ("pearson", adata_p)]):
        obs = adata.obs
        for i, src in enumerate(sorted(obs["source"].unique())):
            conf = obs.loc[obs["source"] == src, conf_col].values
            ax.hist(conf, bins=bins, alpha=0.55, label=src, density=True,
                    histtype="stepfilled", color=palette[i])
        ax.axvline(0.5, color="black", ls="--", lw=1.2, label="0.5 threshold")
        ax.axvline(0.8, color="grey",  ls=":",  lw=1.0, label="0.8 threshold")
        ax.set_title(f"{name} HVG — prediction confidence")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved confidence_comparison.png")


# ---------------------------------------------------------------------------
# 5. UMAP comparison
# ---------------------------------------------------------------------------

def plot_umap(
    adata_s: ad.AnnData,
    adata_p: ad.AnnData,
    output_dir: Path,
    logger: logging.Logger,
    n_plot: int = 50_000,
):
    logger.info("\n── UMAP Comparison ──────────────────────────────────────")

    # Prefer scANVI UMAP if available
    umap_key = "X_umap_scANVI" if "X_umap_scANVI" in adata_s.obsm else "X_umap"
    if umap_key not in adata_s.obsm or umap_key not in adata_p.obsm:
        logger.warning(f"UMAP key '{umap_key}' not in both files — skipping")
        return

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    batch_pal = sns.color_palette("tab10", 4)

    for row, (name, adata) in enumerate([("seurat", adata_s), ("pearson", adata_p)]):
        xy  = np.asarray(adata.obsm[umap_key])
        obs = adata.obs
        idx = rng.choice(len(xy), min(n_plot, len(xy)), replace=False)
        xy_s = xy[idx]
        src  = obs["source"].values[idx]
        ct   = (obs["cell_type_aligned"].values[idx]
                if "cell_type_aligned" in obs.columns
                else obs["cell_class"].values[idx])

        # Batch panel
        ax = axes[row, 0]
        for i, s in enumerate(sorted(set(src))):
            m = src == s
            ax.scatter(xy_s[m, 0], xy_s[m, 1], s=0.4, alpha=0.25,
                       color=batch_pal[i], label=s, rasterized=True)
        ax.set_title(f"{name} HVG — coloured by batch", fontsize=11)
        ax.axis("off")
        ax.legend(markerscale=8, fontsize=9, loc="lower right",
                  framealpha=0.8, title="source")

        # Cell type panel
        ax = axes[row, 1]
        ct_uniq  = sorted(set(ct))
        ct_pal   = sns.color_palette("tab20", min(len(ct_uniq), 20))
        for i, c in enumerate(ct_uniq[:20]):
            m = ct == c
            ax.scatter(xy_s[m, 0], xy_s[m, 1], s=0.4, alpha=0.25,
                       color=ct_pal[i % 20], label=c, rasterized=True)
        ax.set_title(f"{name} HVG — coloured by cell type", fontsize=11)
        ax.axis("off")
        ax.legend(markerscale=8, fontsize=8, loc="lower right", ncol=2,
                  framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "umap_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved umap_comparison.png  (n={min(n_plot, len(xy)):,} cells per panel)")


# ---------------------------------------------------------------------------
# 6. Summary bar chart
# ---------------------------------------------------------------------------

def plot_summary(metrics: dict, output_dir: Path, logger: logging.Logger):
    logger.info("\n── Summary Metrics ──────────────────────────────────────")

    groups = {
        "Batch mixing\n(higher = better)": [
            ("iLISI (norm)",   "seurat_ilisi_norm",     "pearson_ilisi_norm"),
            ("ASW batch",      "seurat_asw_batch_norm", "pearson_asw_batch_norm"),
        ],
        "Cell type separation\n(higher = better)": [
            ("cLISI sep.",     "seurat_ct_separation",  "pearson_ct_separation"),
            ("ASW cell type",  "seurat_asw_celltype",   "pearson_asw_celltype"),
        ],
        "Label transfer\n(higher = better)": [
            ("Wang accuracy",  "seurat_wang_accuracy",  "pearson_wang_accuracy"),
            ("Wang F1",        "seurat_wang_f1",        "pearson_wang_f1"),
            ("Median conf.",   "seurat_median_conf",    "pearson_median_conf"),
        ],
    }

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5.5 * n_groups, 5.5))

    for ax, (title, items) in zip(np.atleast_1d(axes), groups.items()):
        valid = [(lbl, sk, pk) for lbl, sk, pk in items
                 if sk in metrics and pk in metrics]
        if not valid:
            ax.set_visible(False)
            continue

        lbls, s_vals, p_vals = zip(*[
            (lbl, metrics[sk], metrics[pk]) for lbl, sk, pk in valid
        ])
        x = np.arange(len(lbls))
        w = 0.36
        b1 = ax.bar(x - w / 2, s_vals, w, label="seurat",
                    color="#4878CF", edgecolor="white")
        b2 = ax.bar(x + w / 2, p_vals, w, label="pearson_residuals",
                    color="#D65F5F", edgecolor="white")
        ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
        ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.22)
        ax.legend(fontsize=8)

    plt.suptitle("scVI integration quality: seurat vs pearson_residuals HVG selection",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved metrics_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seurat",   required=True, help="integrated.h5ad from seurat HVG run")
    parser.add_argument("--pearson",  required=True, help="integrated.h5ad from pearson_residuals HVG run")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embedding", default="X_scANVI",
                        help="Preferred obsm key for LISI/ASW (default: X_scANVI, falls back to X_scVI)")
    parser.add_argument("--n_sample", type=int, default=30_000,
                        help="Cells to subsample for LISI/ASW (default: 30000)")
    parser.add_argument("--batch_key",     default="source")
    parser.add_argument("--celltype_key",  default="cell_class",
                        help="Broad cell type column for cLISI/ASW (default: cell_class)")
    parser.add_argument("--gt_col",   default="cell_type_for_scanvi",
                        help="Column with Wang ground-truth labels (others = 'Unknown')")
    parser.add_argument("--pred_col", default="cell_type_aligned")
    parser.add_argument("--conf_col", default="cell_type_aligned_confidence")
    parser.add_argument("--no_umap", action="store_true",
                        help="Skip UMAP comparison plot (faster)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(out)

    # Validate inputs
    for p in [args.seurat, args.pearson]:
        if not Path(p).exists():
            logger.error(f"File not found: {p}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("HVG Method Comparison: seurat vs pearson_residuals")
    logger.info("=" * 60)

    adata_s = _load(Path(args.seurat),  "seurat",  logger)
    adata_p = _load(Path(args.pearson), "pearson", logger)

    metrics = {}
    rng = np.random.default_rng(42)

    # 1. HVG overlap
    metrics.update(hvg_overlap(adata_s, adata_p, out, logger))

    # 2. Integration metrics
    logger.info("\n── Integration Metrics (LISI + ASW) ────────────────────")
    emb_s = _pick_embedding(adata_s, args.embedding)
    emb_p = _pick_embedding(adata_p, args.embedding)
    m_s, _, _ = integration_metrics(adata_s, "seurat",  emb_s, args.batch_key,
                                    args.celltype_key, args.n_sample, rng, logger)
    m_p, _, _ = integration_metrics(adata_p, "pearson", emb_p, args.batch_key,
                                    args.celltype_key, args.n_sample, rng, logger)
    metrics.update(m_s)
    metrics.update(m_p)

    # 3. Label transfer quality
    metrics.update(label_transfer_quality(
        adata_s, adata_p, args.gt_col, args.pred_col, args.conf_col, out, logger
    ))

    # 4. Confidence histograms
    plot_confidence(adata_s, adata_p, args.conf_col, out, logger)

    # 5. UMAP comparison
    if not args.no_umap:
        plot_umap(adata_s, adata_p, out, logger)

    # 6. Summary bar chart
    plot_summary(metrics, out, logger)

    # Save all scalar metrics
    df = pd.DataFrame(
        [(k, v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))],
        columns=["metric", "value"],
    )
    df.to_csv(out / "metrics.csv", index=False)
    logger.info(f"\nAll metrics written to {out / 'metrics.csv'}")
    logger.info("\n" + df.to_string(index=False))
    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
