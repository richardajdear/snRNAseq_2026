"""UMAP computation and comparison plots for batch correction QC."""

import logging
from pathlib import Path
from typing import List, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

from .utils import Timer


# ── Style utilities ────────────────────────────────────────────────────────────

def _set1_palette(n: int) -> list:
    """Return n colours from RColorBrewer's Set1 palette (matching R's default)."""
    # Set1 has 9 distinct colors; cycle if needed
    set1_hex = [
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
        "#A65628", "#F781BF", "#999999", "#66C2A5",
    ]
    colors = []
    for i in range(n):
        hex_color = set1_hex[i % len(set1_hex)]
        # Convert hex to RGB tuple
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        colors.append((r, g, b))
    return colors


def _apply_ggplot_theme(ax):
    """Apply ggplot2 theme_bw() styling to a matplotlib Axes."""
    ax.set_facecolor("white")
    ax.grid(True, color="#E0E0E0", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#AAAAAA")
        spine.set_linewidth(0.8)
    ax.tick_params(left=False, bottom=False, labelsize=8, colors="#555555")
    for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        item.set_fontfamily("sans-serif")
        item.set_color("#333333")


# ── UMAP computation ───────────────────────────────────────────────────────────

def compute_umap(
    adata: ad.AnnData,
    obsm_key: str,
    neighbors_key: str,
    umap_key: str,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    logger: Optional[logging.Logger] = None,
):
    """Compute neighbors + UMAP from a latent representation in obsm."""
    if logger:
        logger.info(f"Computing UMAP from {obsm_key} (n_neighbors={n_neighbors})")
    sc.pp.neighbors(adata, use_rep=obsm_key, n_neighbors=n_neighbors, key_added=neighbors_key)
    sc.tl.umap(adata, min_dist=min_dist, neighbors_key=neighbors_key)
    adata.obsm[umap_key] = adata.obsm["X_umap"].copy()
    if logger:
        logger.info(f"Stored UMAP in obsm['{umap_key}']")


def compute_raw_pca_umap(
    adata: ad.AnnData,
    config,
    logger: logging.Logger,
):
    """
    Compute PCA on log-normalized HVG counts (no batch correction), then UMAP.

    Stored in obsm['X_pca_raw'] and obsm['X_umap_raw'].
    This serves as the uncorrected baseline for comparison plots.
    """
    logger.info("Computing raw (uncorrected) PCA + UMAP on HVG counts")

    # Work on a temporary copy of the HVG subset to avoid mutating adata.X
    hvg_mask = adata.var.get("highly_variable", None)
    if hvg_mask is not None and hvg_mask.any():
        tmp = adata[:, adata.var["highly_variable"]].copy()
    else:
        tmp = adata.copy()

    # Normalize and log-transform for PCA (standard scanpy preprocessing)
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.pca(tmp, n_comps=50)

    adata.obsm["X_pca_raw"] = tmp.obsm["X_pca"]
    compute_umap(
        adata,
        obsm_key="X_pca_raw",
        neighbors_key="neighbors_raw",
        umap_key="X_umap_raw",
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        logger=logger,
    )


def compute_umaps(
    adata: ad.AnnData,
    config,
    logger: logging.Logger,
):
    """Compute UMAPs: raw (uncorrected), scVI, and scANVI (if available)."""
    with Timer("Computing raw (uncorrected) PCA UMAP", logger):
        compute_raw_pca_umap(adata, config, logger)

    if "X_scVI" in adata.obsm:
        with Timer("Computing scVI UMAP", logger):
            compute_umap(
                adata,
                obsm_key="X_scVI",
                neighbors_key="neighbors_scvi",
                umap_key="X_umap_scvi",
                n_neighbors=config.umap_n_neighbors,
                min_dist=config.umap_min_dist,
                logger=logger,
            )

    if "X_scANVI" in adata.obsm:
        with Timer("Computing scANVI UMAP", logger):
            compute_umap(
                adata,
                obsm_key="X_scANVI",
                neighbors_key="neighbors_scanvi",
                umap_key="X_umap_scanvi",
                n_neighbors=config.umap_n_neighbors,
                min_dist=config.umap_min_dist,
                logger=logger,
            )


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_umap_comparison(
    adata: ad.AnnData,
    color_by: str,
    umap_keys: List[str],
    umap_labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    panel_size: float = 4.0,
    point_size: float = 1.0,
    alpha: float = 0.5,
    log2_vars: Optional[List[str]] = None,
    log2_ticks: Optional[List[float]] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Plot side-by-side square UMAPs colored by a single variable.

    Uses ggplot2-inspired styling with a shared legend/colorbar on the right.
    Variables in log2_vars are log2(x+1) transformed for coloring; the colorbar
    ticks are shown at original (unlogged) values specified by log2_ticks.
    """
    if umap_labels is None:
        umap_labels = umap_keys
    if log2_vars is None:
        log2_vars = []
    if log2_ticks is None:
        log2_ticks = [0, 1, 3, 9, 25, 40]

    n_panels = len(umap_keys)

    if color_by not in adata.obs.columns:
        if logger:
            logger.warning(f"'{color_by}' not in .obs, skipping plot")
        return

    values = adata.obs[color_by]
    is_categorical = values.dtype.name == "category" or values.dtype == object
    is_log2 = (not is_categorical) and (color_by in log2_vars)

    # Shuffle point order for fair overlap
    rng = np.random.RandomState(42)
    order = rng.permutation(adata.n_obs)

    # Pre-compute colors / transformed values
    if is_categorical:
        cats = list(values.cat.categories) if hasattr(values, "cat") else sorted(values.unique())
        palette = _set1_palette(len(cats))
        color_map = dict(zip(cats, palette))
        legend_handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=color_map[c], markersize=6, label=c,
            )
            for c in cats
        ]
    else:
        raw_vals = np.array(values, dtype=float)
        plot_vals = np.log2(raw_vals + 1) if is_log2 else raw_vals
        vmin = float(np.nanmin(plot_vals))
        vmax = float(np.nanmax(plot_vals))

    # Figure layout: panels + fixed annotation strip on the right.
    # Using explicit subplots_adjust + add_axes for the colorbar keeps panel
    # sizes identical between categorical and continuous variables.
    ANNO_W = 1.5  # inches reserved for legend / colorbar
    fig_w = panel_size * n_panels + ANNO_W
    fig_h = panel_size + (0.8 if is_categorical else 0.0)
    plots_right = (panel_size * n_panels) / fig_w  # figure-fraction where panels end

    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.flatten()
    fig.subplots_adjust(
        left=0.02, right=plots_right - 0.01,
        top=0.92, bottom=0.08, wspace=0.12,
    )

    cbar_mappable = None

    for ax, umap_key, label in zip(axes, umap_keys, umap_labels):
        _apply_ggplot_theme(ax)
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP1", fontsize=9)
        ax.set_ylabel("UMAP2", fontsize=9)
        ax.set_title(label, fontsize=10)

        if umap_key not in adata.obsm:
            ax.text(
                0.5, 0.5, "not computed",
                ha="center", va="center",
                transform=ax.transAxes, color="#888888", fontsize=9,
            )
            continue

        coords = adata.obsm[umap_key][order]

        if is_categorical:
            point_colors = [color_map[v] for v in values.iloc[order]]
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c=point_colors, s=point_size, alpha=alpha,
                rasterized=True, linewidths=0,
            )
        else:
            scat = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=plot_vals[order], s=point_size, alpha=alpha,
                cmap="viridis", rasterized=True,
                vmin=vmin, vmax=vmax, linewidths=0,
            )
            if cbar_mappable is None:
                cbar_mappable = scat

    # Shared legend / colorbar in the annotation strip
    anno_left = plots_right + 0.01  # start of annotation strip in figure fractions
    if is_categorical:
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(anno_left, 0.5),
            fontsize=8,
            frameon=False,
            title=color_by,
            title_fontsize=9,
            markerscale=1.5,
        )
    elif cbar_mappable is not None:
        # Dedicated colorbar axes — does not steal space from the main panels
        cb_left = anno_left + 0.08 / fig_w
        cb_width = 0.18 / fig_w
        cax = fig.add_axes([cb_left, 0.25, cb_width, 0.50])
        cbar = fig.colorbar(cbar_mappable, cax=cax)
        cbar.set_label(color_by, fontsize=9)
        if is_log2:
            cbar.set_ticks([np.log2(t + 1) for t in log2_ticks])
            cbar.set_ticklabels([str(t) for t in log2_ticks])
        cbar.ax.tick_params(labelsize=8)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        if logger:
            logger.info(f"Saved plot: {output_path}")

    plt.close(fig)


def plot_batch_comparison(
    adata: ad.AnnData,
    color_vars: List[str],
    config,
    logger: logging.Logger,
):
    """
    Plot comparison UMAPs for each color variable across available embeddings.

    Generates one figure per color variable, with panels for each UMAP embedding.
    """
    plots_dir = config._resolved_output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Collect available UMAP keys — raw first, then corrected
    umap_keys = []
    umap_labels = []
    if "X_umap_raw" in adata.obsm:
        umap_keys.append("X_umap_raw")
        umap_labels.append("Uncorrected (PCA)")
    if "X_umap_scvi" in adata.obsm:
        umap_keys.append("X_umap_scvi")
        umap_labels.append("scVI")
    if "X_umap_scanvi" in adata.obsm:
        umap_keys.append("X_umap_scanvi")
        umap_labels.append("scANVI")

    if not umap_keys:
        logger.warning("No UMAP embeddings found, skipping plots")
        return

    log2_vars = getattr(config, "umap_log2_vars", ["age_years"])
    log2_ticks = getattr(config, "umap_log2_ticks", [0, 1, 3, 9, 25, 40])

    for var in color_vars:
        if var not in adata.obs.columns:
            logger.warning(f"'{var}' not in .obs, skipping")
            continue

        plot_umap_comparison(
            adata,
            color_by=var,
            umap_keys=umap_keys,
            umap_labels=umap_labels,
            output_path=str(plots_dir / f"umap_{var}.png"),
            panel_size=4.0,
            point_size=config.umap_point_size,
            log2_vars=log2_vars,
            log2_ticks=log2_ticks,
            logger=logger,
        )
