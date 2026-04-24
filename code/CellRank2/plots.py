"""Diagnostic plots for CellRank 2 lineage tracing outputs."""

import logging
import re
from pathlib import Path
from typing import Optional

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CellRankConfig
from .utils import Timer


# ── Style utilities ────────────────────────────────────────────────────────────

def _set1_palette(n: int) -> list:
    """Return n colours from RColorBrewer's Set1 palette (matching R's default)."""
    set1_hex = [
        "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
        "#A65628", "#F781BF", "#999999", "#66C2A5",
    ]
    colors = []
    for i in range(n):
        hex_color = set1_hex[i % len(set1_hex)]
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


# ── cell_type_aligned palette (shared with scVI/visualize.py) ─────────────────

def _is_immature(label: str) -> bool:
    return any(x in label for x in ("Immature", "Newborn")) or label.startswith("IPC-")


def _cell_type_group(label: str) -> str:
    if label.startswith("EN-") or label == "IPC-EN":
        return "EN"
    if label.startswith("IN-"):
        return "IN"
    if "Oligodendrocyte" in label or label == "OPC":
        return "Oligo"
    if label.startswith("Astrocyte"):
        return "Astro"
    if label == "Microglia":
        return "Microglia"
    if label.startswith("RG-") or label in ("Tri-IPC", "Cajal-Retzius cell"):
        return "Progenitor"
    return "Other"


_GROUP_CMAP = {
    "EN":         ("Reds",    (0.45, 0.90), (0.25, 0.45)),
    "IN":         ("Blues",   (0.40, 0.88), (0.22, 0.42)),
    "Oligo":      ("Greens",  (0.45, 0.85), (0.22, 0.38)),
    "Astro":      ("GnBu",    (0.45, 0.80), (0.22, 0.38)),
    "Microglia":  ("YlGn",    (0.52, 0.68), (0.30, 0.45)),
    "Progenitor": ("Purples", (0.45, 0.85), (0.25, 0.40)),
    "Other":      ("Greys",   (0.35, 0.65), (0.20, 0.35)),
}


def build_cell_type_aligned_palette(categories: list) -> dict:
    """Build a hue+shade colour dict for cell_type_aligned categories.

    Groups cell types by broad class (EN, IN, Oligo, etc.) and assigns darker
    shades to mature subtypes and lighter shades to immature/newborn subtypes,
    making the maturation axis readable at a glance.
    """
    from matplotlib import colormaps

    groups: dict = {}
    for cat in categories:
        g = _cell_type_group(cat)
        if g not in groups:
            groups[g] = {"mature": [], "immature": []}
        (groups[g]["immature"] if _is_immature(cat) else groups[g]["mature"]).append(cat)

    for g in groups:
        groups[g]["mature"].sort()
        groups[g]["immature"].sort()

    palette = {}
    for g_name, buckets in groups.items():
        cmap_name, mature_range, immature_range = _GROUP_CMAP.get(g_name, _GROUP_CMAP["Other"])
        cmap = colormaps[cmap_name]

        def _sample(labels, lo, hi):
            n = len(labels)
            vals = [lo] if n == 1 else [lo + (hi - lo) * i / (n - 1) for i in range(n)]
            for lbl, v in zip(labels, vals):
                palette[lbl] = cmap(v)[:3]

        _sample(buckets["mature"],   *mature_range)
        _sample(buckets["immature"], *immature_range)

    return palette


def _macrostate_base_name(name: str) -> str:
    """Strip trailing numeric suffix: 'EN-L2_3-IT_1' → 'EN-L2_3-IT'."""
    return re.sub(r"_\d+$", "", name)


# ── Single-panel UMAP helper ───────────────────────────────────────────────────

def plot_umap_colored(
    adata: ad.AnnData,
    color_by: str,
    umap_key: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    point_size: float = 2.0,
    alpha: float = 0.5,
    cmap: str = "viridis",
    palette: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
    figsize: tuple = (6, 6),
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
        if palette is not None:
            fallback = _set1_palette(len(cats))
            color_map = {c: palette.get(c, fallback[i]) for i, c in enumerate(cats)}
        else:
            color_map = dict(zip(cats, _set1_palette(len(cats))))

        nan_color = (0.85, 0.85, 0.85, 0.5)
        c_vals = [
            color_map[v] if (v == v and v in color_map) else nan_color
            for v in values.iloc[order]
        ]
        ax.scatter(xy[:, 0], xy[:, 1], c=c_vals, s=point_size, alpha=alpha,
                   rasterized=True, linewidths=0)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map.get(c, (0.5, 0.5, 0.5)),
                       markersize=6, label=c)
            for c in cats if c in color_map
        ]
        ax.legend(handles=handles, fontsize=7, frameon=False,
                  bbox_to_anchor=(1.01, 1), loc="upper left", title=color_by,
                  title_fontsize=8)
    else:
        vals = np.array(values, dtype=float)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals[order], s=point_size,
                        alpha=alpha, cmap=cmap, rasterized=True, linewidths=0)
        plt.colorbar(sc, ax=ax, shrink=0.7, label=color_by)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        if logger:
            logger.info(f"Saved: {output_path}")
    plt.close(fig)


# ── CellRank-specific plots ────────────────────────────────────────────────────

def plot_macrostates(
    adata: ad.AnnData,
    g,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Plot macrostate assignments on the UMAP using a tab20 qualitative palette.

    Uses soft memberships (dominant state per cell) so all cells are coloured.
    tab20 is used so that even EN-only subsets (where all macrostates are
    EN subtypes) have clearly distinguishable colours.
    """
    from matplotlib import colormaps

    plots_dir = config.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    if g.macrostates is None:
        logger.warning("No macrostates computed; skipping macrostate plot.")
        return

    # Assign every cell to its dominant macrostate using soft memberships
    color_var = "macrostates_dominant"
    if g.macrostates_memberships is not None:
        memberships = g.macrostates_memberships.X
        state_names = list(g.macrostates_memberships.names)
        dominant_idx = np.argmax(memberships, axis=1)
        adata.obs[color_var] = pd.Categorical(
            [state_names[i] for i in dominant_idx],
            categories=state_names,
        )
    else:
        adata.obs["macrostates"] = g.macrostates
        color_var = "macrostates"

    # Use tab20 so EN-only subsets (all macrostates the same broad type) have
    # clearly distinguishable colours rather than similar shades of one hue.
    ms_cats = list(adata.obs[color_var].cat.categories)
    cmap_t20 = colormaps["tab20"]
    ms_palette = {
        c: cmap_t20(i / max(len(ms_cats) - 1, 1))[:3]
        for i, c in enumerate(ms_cats)
    }

    with Timer("Plotting macrostates", logger):
        plot_umap_colored(
            adata,
            color_by=color_var,
            umap_key=config.umap_key,
            output_path=str(plots_dir / "macrostates.png"),
            title="Macrostates",
            point_size=config.point_size,
            palette=ms_palette,
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
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

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    out_path = plots_dir / "fate_probabilities.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
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

        var_palette = None
        if var == config.cell_type_key:
            cats = list(adata.obs[var].dropna().unique())
            var_palette = build_cell_type_aligned_palette(cats)

        plot_umap_colored(
            adata,
            color_by=var,
            umap_key=config.umap_key,
            output_path=str(plots_dir / f"umap_{var}.png"),
            point_size=config.point_size,
            palette=var_palette,
            logger=logger,
        )


def plot_combined_umap_panel(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Six-panel 2×3 UMAP summary for the excitatory neuron lineage analysis.

    All panels share the same UMAP embedding (config.umap_key) and EN-only cells.

    Row 0:  source (batch)  |  age_log_pc  |  cell_type_aligned
    Row 1:  DPT pseudotime  |  L2-3 fate probability  |  L2-3 lineage membership

    Saves a single combined PNG: umap_panel_excitatory.png
    """
    from matplotlib import colormaps
    from matplotlib.patches import Patch

    plots_dir = config.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    umap_key = config.umap_key
    if umap_key not in adata.obsm:
        logger.warning(f"'{umap_key}' not in adata.obsm; skipping combined UMAP panel.")
        return

    n = adata.n_obs
    coords = adata.obsm[umap_key]
    rng = np.random.RandomState(42)
    order = rng.permutation(n)
    xy = coords[order]

    cmap_t20 = colormaps["tab20"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Excitatory neuron lineage  (n={n:,})", fontsize=12, y=0.99)

    def _setup(ax, title):
        _apply_ggplot_theme(ax)
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP1", fontsize=8)
        ax.set_ylabel("UMAP2", fontsize=8)
        ax.set_title(title, fontsize=9)

    def _inset_cbar(ax, sc_, label):
        """Colorbar as an inset — doesn't steal layout space from ax."""
        cax = ax.inset_axes([1.02, 0.05, 0.04, 0.90])
        fig.colorbar(sc_, cax=cax, label=label)
        cax.tick_params(labelsize=6)
        cax.yaxis.label.set_size(7)

    def _cat_scatter(ax, col, title, palette=None):
        _setup(ax, title)
        if col not in adata.obs.columns:
            ax.text(0.5, 0.5, f"'{col}' not found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="red")
            return
        vals = adata.obs[col].astype(str)
        cats = sorted(vals.dropna().unique())
        if palette is None:
            palette = {c: cmap_t20(i / max(len(cats) - 1, 1))[:3]
                       for i, c in enumerate(cats)}
        nan_c = (0.85, 0.85, 0.85, 0.5)
        c_arr = [palette.get(v, nan_c) for v in vals.iloc[order]]
        ax.scatter(xy[:, 0], xy[:, 1], c=c_arr, s=config.point_size, alpha=0.5,
                   rasterized=True, linewidths=0)
        handles = [plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=palette.get(c, (0.5,) * 3),
                               markersize=4, label=c)
                   for c in cats if c in palette]
        ncols = 2 if len(handles) > 6 else 1
        ax.legend(handles=handles, fontsize=5, frameon=True, framealpha=0.7,
                  loc="lower right", title=col, title_fontsize=6,
                  ncol=ncols, handlelength=0.8, handletextpad=0.3, borderpad=0.4)

    def _cont_scatter(ax, col, title, cmap="viridis", vmin=None, vmax=None):
        _setup(ax, title)
        if col not in adata.obs.columns:
            ax.text(0.5, 0.5, f"'{col}' not found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="red")
            return
        vals = np.array(adata.obs[col], dtype=float)
        if vmin is None:
            vmin = float(np.nanpercentile(vals, 1))
        if vmax is None:
            vmax = float(np.nanpercentile(vals, 99))
        sc_ = ax.scatter(xy[:, 0], xy[:, 1], c=vals[order], s=config.point_size,
                         alpha=0.6, cmap=cmap, rasterized=True, linewidths=0,
                         vmin=vmin, vmax=vmax)
        _inset_cbar(ax, sc_, col)

    # ── Row 0: batch/age/cell-type ────────────────────────────────────────────
    _cat_scatter(axes[0, 0], config.batch_key, title="Source (batch)")

    _cont_scatter(axes[0, 1], "age_log_pc", title="Age (log1p years)", cmap="viridis")

    ct_key = config.cell_type_key
    if ct_key in adata.obs.columns:
        ct_cats = sorted(adata.obs[ct_key].dropna().astype(str).unique())
        ct_pal = {c: cmap_t20(i / max(len(ct_cats) - 1, 1))[:3]
                  for i, c in enumerate(ct_cats)}
    else:
        ct_pal = None
    _cat_scatter(axes[0, 2], ct_key, title="Cell type", palette=ct_pal)

    # ── Row 1: pseudotime/fate/lineage ────────────────────────────────────────
    pt_key = config.absorption_pseudotime_key or ""
    if pt_key and pt_key in adata.obs.columns:
        pt = np.array(adata.obs[pt_key], dtype=float)
        logger.info(
            f"  DPT pseudotime: min={np.nanmin(pt):.3f}, "
            f"median={np.nanmedian(pt):.3f}, max={np.nanmax(pt):.3f}"
        )
    _cont_scatter(axes[1, 0], pt_key,
                  title="DPT pseudotime\n(0=progenitor → 1=mature)",
                  cmap="magma", vmin=0.0, vmax=1.0)

    fate_key = "fate_prob_l23"
    if fate_key in adata.obs.columns:
        fp = np.array(adata.obs[fate_key], dtype=float)
        vmin_fp = float(np.nanpercentile(fp, 1))
        vmax_fp = float(np.nanpercentile(fp, 99))
        logger.info(
            f"  L2-3 fate prob: p1={vmin_fp:.3f}, "
            f"median={np.nanmedian(fp):.3f}, p99={vmax_fp:.3f}"
        )
        _cont_scatter(axes[1, 1], fate_key, title="L2-3 fate probability",
                      cmap="plasma", vmin=vmin_fp, vmax=vmax_fp)
    else:
        _setup(axes[1, 1], "L2-3 fate probability")
        axes[1, 1].text(0.5, 0.5, f"'{fate_key}' not found",
                        transform=axes[1, 1].transAxes,
                        ha="center", va="center", fontsize=8, color="red")

    # ── Panel (1,2): L2-3 lineage membership ─────────────────────────────────
    ax6 = axes[1, 2]
    _setup(ax6, f"L2-3 lineage  (fate_prob ≥ {config.fate_prob_threshold})")
    if fate_key in adata.obs.columns:
        fp = np.array(adata.obs[fate_key], dtype=float)
        in_lineage = fp >= config.fate_prob_threshold
        n_on = int(in_lineage.sum())
        logger.info(
            f"  L2-3 lineage: {n_on:,} / {n:,} cells "
            f"({100 * n_on / n:.1f}%)"
        )
        col_arr = np.where(in_lineage, 0, 1).astype(float)
        lin_cmap = matplotlib.colors.ListedColormap(
            [(0.85, 0.85, 0.85), (0.13, 0.47, 0.71)]
        )
        ax6.scatter(xy[:, 0], xy[:, 1], c=col_arr[order], s=config.point_size,
                    alpha=0.5, cmap=lin_cmap, rasterized=True, linewidths=0,
                    vmin=0, vmax=1)
        ax6.legend(
            handles=[
                Patch(facecolor=(0.85, 0.85, 0.85), label=f"Other  ({n - n_on:,})"),
                Patch(facecolor=(0.13, 0.47, 0.71), label=f"L2-3 lineage  ({n_on:,})"),
            ],
            fontsize=6, frameon=True, framealpha=0.7, loc="lower right",
        )
    else:
        ax6.text(0.5, 0.5, f"'{fate_key}' not found", transform=ax6.transAxes,
                 ha="center", va="center", fontsize=8, color="red")

    # tight_layout works correctly now: all legends are inside axes and all
    # colorbars are inset_axes, so nothing external steals layout space.
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = plots_dir / "umap_panel_excitatory.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    logger.info(f"Saved: {out}")
    plt.close(fig)
