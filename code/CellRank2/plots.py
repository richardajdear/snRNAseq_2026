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
    out_path = plots_dir / "coarse_T.png"

    with Timer("Plotting coarse transition matrix", logger):
        try:
            # CellRank 2 supports a native `save` parameter on plot_coarse_T
            g.plot_coarse_T(save=str(out_path), figsize=(6, 5))
            logger.info(f"Saved: {out_path}")
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


def plot_excitatory_l23_fate_umap(
    adata: ad.AnnData,
    g,
    config: CellRankConfig,
    logger: logging.Logger,
    excitatory_pattern: str = "excit",
    l23_pattern: str = "l2",
) -> None:
    """UMAP of all excitatory neurons with layer 2-3 fate cells highlighted.

    Plots every excitatory neuron in the dataset on the UMAP embedding.
    Cells whose most likely cell fate (argmax of fate probabilities) matches
    a layer 2-3 terminal state are highlighted in red; all other excitatory
    neurons are shown in grey.

    Parameters
    ----------
    excitatory_pattern
        Case-insensitive substring used to identify excitatory neurons in
        ``config.cell_type_key`` (default ``"excit"``).
    l23_pattern
        Case-insensitive substring used to match layer 2-3 terminal state
        names (default ``"l2"``).
    """
    if g.fate_probabilities is None:
        logger.warning(
            "No fate probabilities; skipping excitatory L2-3 fate UMAP plot."
        )
        return

    if config.umap_key not in adata.obsm:
        logger.warning(
            f"'{config.umap_key}' not in adata.obsm; "
            "skipping excitatory L2-3 fate UMAP plot."
        )
        return

    if config.cell_type_key not in adata.obs.columns:
        logger.warning(
            f"'{config.cell_type_key}' not in adata.obs; "
            "skipping excitatory L2-3 fate UMAP plot."
        )
        return

    # Identify excitatory neurons
    cell_types = adata.obs[config.cell_type_key].astype(str)
    excit_mask = cell_types.str.lower().str.contains(
        excitatory_pattern.lower(), na=False
    )

    if excit_mask.sum() == 0:
        logger.warning(
            f"No cells matching '{excitatory_pattern}' in "
            f"'{config.cell_type_key}'; skipping excitatory L2-3 fate UMAP plot."
        )
        return

    n_excit = int(excit_mask.sum())
    logger.info(f"  Excitatory neurons found: {n_excit} cells")

    # Fate probabilities matrix (n_cells × n_lineages)
    probs = g.fate_probabilities.X
    lineage_names = [str(n) for n in g.fate_probabilities.names]

    # Find lineage indices that correspond to layer 2-3
    l23_indices = [
        i for i, name in enumerate(lineage_names)
        if l23_pattern.lower() in name.lower()
    ]

    if l23_indices:
        logger.info(
            f"  L2-3 lineages matched: "
            + ", ".join(lineage_names[i] for i in l23_indices)
        )
    else:
        logger.error(
            f"  No lineages matching '{l23_pattern}' found in available lineages: "
            + ", ".join(lineage_names) + ". "
            "Excitatory UMAP will be plotted without L2-3 highlighting. "
            "Increase n_macrostates so that L2-3 neurons form a distinct macrostate, "
            "or set terminal_states explicitly in the config."
        )

    # Per-cell argmax fate index
    argmax_fate = np.argmax(probs, axis=1)

    # Cells that are excitatory AND whose most likely fate is an L2-3 lineage
    l23_fate_mask = np.isin(argmax_fate, l23_indices) if l23_indices else np.zeros(
        adata.n_obs, dtype=bool
    )
    highlight_mask = excit_mask.values & l23_fate_mask

    n_highlighted = int(highlight_mask.sum())
    logger.info(
        f"  Cells highlighted (most likely fate = L2-3): {n_highlighted}/{n_excit}"
    )

    # UMAP coordinates for excitatory neurons only
    coords = adata.obsm[config.umap_key]
    excit_coords = coords[excit_mask.values]
    excit_highlight = highlight_mask[excit_mask.values]  # boolean within excit subset

    # Build title
    l23_names_str = (
        ", ".join(lineage_names[i] for i in l23_indices)
        if l23_indices
        else f"(no match for '{l23_pattern}')"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_ggplot_theme(ax)
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=9)
    ax.set_ylabel("UMAP2", fontsize=9)
    ax.set_title(
        f"Excitatory neurons — most-likely fate\nL2-3 lineage: {l23_names_str}",
        fontsize=8,
    )

    # Grey background: all other excitatory neurons
    background_coords = excit_coords[~excit_highlight]
    ax.scatter(
        background_coords[:, 0], background_coords[:, 1],
        c="#CCCCCC", s=config.point_size, alpha=0.4,
        rasterized=True, linewidths=0, label="Other fate",
    )

    # Red foreground: excitatory neurons with L2-3 most-likely fate
    if excit_highlight.any():
        fg_coords = excit_coords[excit_highlight]
        ax.scatter(
            fg_coords[:, 0], fg_coords[:, 1],
            c="#E41A1C", s=config.point_size * 2.5, alpha=0.85,
            rasterized=True, linewidths=0,
            label=f"L2-3 most-likely fate (n={n_highlighted})",
        )

    ax.legend(
        fontsize=7, frameon=True, loc="upper right",
        title=f"Excitatory neurons (n={n_excit})", title_fontsize=7,
    )

    plots_dir = config.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "umap_excitatory_l23_fate.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close(fig)
