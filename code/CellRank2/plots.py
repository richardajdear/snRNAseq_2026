"""Diagnostic plots for CellRank 2 lineage tracing outputs."""

import logging
from pathlib import Path
from typing import List, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from .config import CellRankConfig
from .utils import Timer


def _apply_ggplot_theme(ax):
    """Apply ggplot2 theme_bw()-inspired styling."""
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


def plot_umap_colored(
    adata: ad.AnnData,
    color_by: str,
    umap_key: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    point_size: float = 1.0,
    alpha: float = 0.5,
    logger: Optional[logging.Logger] = None,
    figsize: tuple = (5, 5),
) -> None:
    """Plot a single UMAP coloured by a continuous or categorical variable."""
    if umap_key not in adata.obsm:
        if logger:
            logger.warning(f"'{umap_key}' not in adata.obsm; skipping plot.")
        return
    if color_by not in adata.obs.columns:
        if logger:
            logger.warning(f"'{color_by}' not in adata.obs; skipping plot.")
        return

    coords = adata.obsm[umap_key]
    values = adata.obs[color_by]
    is_cat = values.dtype.name == "category" or values.dtype == object

    fig, ax = plt.subplots(figsize=figsize)
    _apply_ggplot_theme(ax)
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=9)
    ax.set_ylabel("UMAP2", fontsize=9)
    ax.set_title(title or color_by, fontsize=10)

    rng = np.random.RandomState(42)
    order = rng.permutation(adata.n_obs)
    xy = coords[order]

    if is_cat:
        cats = (
            list(values.cat.categories)
            if hasattr(values, "cat")
            else sorted(values.dropna().unique())
        )
        cmap = plt.get_cmap("tab20", max(len(cats), 1))
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        # NaN / unassigned cells are plotted in light grey
        nan_color = (0.8, 0.8, 0.8, 0.3)
        c_vals = [
            cmap(cat_to_idx[v]) if (v == v and v in cat_to_idx) else nan_color
            for v in values.iloc[order]
        ]
        ax.scatter(xy[:, 0], xy[:, 1], c=c_vals, s=point_size, alpha=alpha,
                   rasterized=True, linewidths=0)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cmap(i), markersize=5, label=c)
            for i, c in enumerate(cats)
        ]
        ax.legend(handles=handles, fontsize=7, frameon=False,
                  bbox_to_anchor=(1.01, 1), loc="upper left", title=color_by,
                  title_fontsize=8)
    else:
        vals = np.array(values, dtype=float)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals[order], s=point_size,
                        alpha=alpha, cmap="viridis", rasterized=True,
                        linewidths=0)
        plt.colorbar(sc, ax=ax, shrink=0.7, label=color_by)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        if logger:
            logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_macrostates(
    adata: ad.AnnData,
    g,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Plot macrostate memberships on the UMAP."""
    plots_dir = config.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    if g.macrostates is None:
        logger.warning("No macrostates computed; skipping macrostate plot.")
        return

    adata.obs["macrostates"] = g.macrostates

    with Timer("Plotting macrostates", logger):
        plot_umap_colored(
            adata,
            color_by="macrostates",
            umap_key=config.umap_key,
            output_path=str(plots_dir / "macrostates.png"),
            title="Macrostates",
            point_size=config.point_size,
            logger=logger,
        )


def plot_fate_probabilities(
    adata: ad.AnnData,
    g,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Plot per-lineage fate probabilities on the UMAP (one panel per lineage)."""
    if g.fate_probabilities is None:
        logger.warning("No fate probabilities computed; skipping fate prob plots.")
        return

    plots_dir = config.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    lineages = list(g.fate_probabilities.names)
    coords = adata.obsm.get(config.umap_key)
    if coords is None:
        logger.warning(
            f"'{config.umap_key}' not in adata.obsm; skipping fate prob plots."
        )
        return

    n = len(lineages)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle("Fate Probabilities", fontsize=12)

    rng = np.random.RandomState(42)
    order = rng.permutation(adata.n_obs)
    xy = coords[order]

    for idx, lin in enumerate(lineages):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        _apply_ggplot_theme(ax)
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(lin, fontsize=9)

        probs = g.fate_probabilities[:, idx].X.ravel()
        sc_ = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=probs[order], s=config.point_size, alpha=0.6,
            cmap="viridis", rasterized=True,
            vmin=0, vmax=1, linewidths=0,
        )
        plt.colorbar(sc_, ax=ax, shrink=0.7)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    out_path = plots_dir / "fate_probabilities.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close(fig)


def plot_coarse_transition_matrix(
    g,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Plot the coarse-grained transition matrix as a heatmap."""
    if g.coarse_T is None:
        logger.warning("No coarse-T; skipping coarse transition matrix plot.")
        return

    plots_dir = config.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    with Timer("Plotting coarse transition matrix", logger):
        try:
            import matplotlib.pyplot as plt
            ax = g.plot_coarse_T(show=False, figsize=(6, 5))
            if ax is not None:
                fig = ax.get_figure()
                out_path = plots_dir / "coarse_T.png"
                fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
                logger.info(f"Saved: {out_path}")
                plt.close(fig)
        except TypeError:
            # Some CellRank versions don't support show=False; use figure capture
            try:
                import matplotlib
                matplotlib.use("Agg")
                g.plot_coarse_T()
                fig = plt.gcf()
                out_path = plots_dir / "coarse_T.png"
                fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
                logger.info(f"Saved: {out_path}")
                plt.close("all")
            except Exception as exc2:
                logger.warning(f"  Could not plot coarse_T: {exc2}")
        except Exception as exc:
            logger.warning(f"  Could not plot coarse_T: {exc}")


def plot_obs_vars(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Plot obs variables (cell type, age, batch) on the UMAP."""
    plots_dir = config.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    for var in config.plot_color_vars:
        if var not in adata.obs.columns:
            logger.warning(f"  '{var}' not in adata.obs; skipping plot.")
            continue
        plot_umap_colored(
            adata,
            color_by=var,
            umap_key=config.umap_key,
            output_path=str(plots_dir / f"umap_{var}.png"),
            point_size=config.point_size,
            logger=logger,
        )
