"""UMAP computation and comparison plots for batch correction QC."""

import gc
import logging
from pathlib import Path
from typing import List, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy.sparse as sp

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


# ── cell_type_aligned palette ─────────────────────────────────────────────────

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


# Colormap and shade range (mature dark end, immature light end) per group
_GROUP_CMAP = {
    "EN":         ("Reds",    (0.45, 0.90), (0.25, 0.45)),   # (cmap, mature_range, immature_range)
    "IN":         ("Blues",   (0.40, 0.88), (0.22, 0.42)),
    "Oligo":      ("Greens",  (0.45, 0.85), (0.22, 0.38)),
    "Astro":      ("GnBu",    (0.45, 0.80), (0.22, 0.38)),
    "Microglia":  ("YlGn",    (0.52, 0.68), (0.30, 0.45)),
    "Progenitor": ("Purples", (0.45, 0.85), (0.25, 0.40)),
    "Other":      ("Greys",   (0.35, 0.65), (0.20, 0.35)),
}


def build_cell_type_aligned_palette(categories: list) -> dict:
    """Build a hue+shade colour dict for cell_type_aligned.

    Cell types are grouped by broad class (EN, IN, Oligo, Astro, Microglia,
    Progenitor, Other).  Within each group, mature subtypes are assigned darker
    shades and immature/newborn/IPC subtypes lighter shades of the same hue,
    making the maturation axis readable at a glance.
    """
    from matplotlib import colormaps

    # Bucket each category into (group, mature/immature)
    groups: dict = {}
    for cat in categories:
        g = _cell_type_group(cat)
        if g not in groups:
            groups[g] = {"mature": [], "immature": []}
        (groups[g]["immature"] if _is_immature(cat) else groups[g]["mature"]).append(cat)

    for g in groups:
        groups[g]["mature"].sort()
        groups[g]["immature"].sort()

    # Single cell class: use full colour spectrum so subtypes are clearly
    # distinct rather than bunched in similar shades of one hue.
    if len(groups) == 1:
        cmap = colormaps["tab20"]
        buckets = next(iter(groups.values()))
        all_types = buckets["mature"] + buckets["immature"]
        n = len(all_types)
        return {lbl: cmap(i / max(n - 1, 1))[:3] for i, lbl in enumerate(all_types)}

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


# ── UMAP computation ───────────────────────────────────────────────────────────

def compute_umap(
    adata: ad.AnnData,
    obsm_key: str,
    neighbors_key: str,
    umap_key: str,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    backend: str = "exact",
    subsample_n: int = 200_000,
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
):
    """Compute UMAP from a latent representation in obsm.

    Two backends:
      - "exact": sc.pp.neighbors + sc.tl.umap on every cell (default; matches
        the previous behaviour).
      - "subsample": fit umap-learn on a random subset of size subsample_n and
        transform the rest. Scales to millions of cells in pure Python without
        building the full kNN graph in scanpy. Falls back to "exact" when the
        dataset already fits inside subsample_n.
    """
    if backend == "subsample" and adata.n_obs > subsample_n:
        import umap as umap_learn  # local import — only needed on this path

        if obsm_key not in adata.obsm:
            if logger:
                logger.warning(f"  {obsm_key} not in obsm — cannot compute {umap_key}")
            return

        if logger:
            logger.info(
                f"Computing UMAP from {obsm_key} via subsample backend "
                f"(fit on {subsample_n:,} / {adata.n_obs:,} cells, "
                f"n_neighbors={n_neighbors})"
            )
        rng = np.random.default_rng(random_state)
        sub = np.sort(rng.choice(adata.n_obs, size=subsample_n, replace=False))
        rep = np.asarray(adata.obsm[obsm_key])

        reducer = umap_learn.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        ).fit(rep[sub])

        embedding = np.empty((adata.n_obs, 2), dtype=np.float32)
        batch = 50_000
        for i in range(0, adata.n_obs, batch):
            embedding[i:i + batch] = reducer.transform(rep[i:i + batch])
        adata.obsm[umap_key] = embedding
        if logger:
            logger.info(f"Stored UMAP in obsm['{umap_key}'] (subsample backend)")
        return

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

    # Avoid adata[:, hvg_mask].copy() — for a HVG-only h5ad that duplicates the
    # full payload (including any dense layers). Pull only the sparse X slice
    # we need into a fresh minimal AnnData and run sparse-aware PCA.
    hvg_mask = adata.var.get("highly_variable", None)
    if hvg_mask is not None and hvg_mask.any() and not hvg_mask.all():
        hvg_idx = np.where(np.asarray(hvg_mask))[0]
        X_hvg = adata.X[:, hvg_idx]
    else:
        X_hvg = adata.X  # already HVG-only

    if sp.issparse(X_hvg):
        X_work = X_hvg.copy()
    else:
        X_work = np.asarray(X_hvg, dtype=np.float32, order="C")

    tmp = ad.AnnData(X=X_work)
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    # zero_center=False keeps PCA on the sparse matrix (TruncatedSVD under the
    # hood) — avoids densifying ~43 GiB for a (767k, 15k) HVG matrix.
    sc.pp.pca(tmp, n_comps=50, zero_center=False)

    adata.obsm["X_pca_raw"] = tmp.obsm["X_pca"]
    del tmp, X_work
    gc.collect()

    compute_umap(
        adata,
        obsm_key="X_pca_raw",
        neighbors_key="neighbors_raw",
        umap_key="X_umap_raw",
        n_neighbors=config.umap_n_neighbors,
        min_dist=config.umap_min_dist,
        backend=getattr(config, "umap_backend", "exact"),
        subsample_n=getattr(config, "umap_subsample_n", 200_000),
        random_state=getattr(config, "random_seed", 42),
        logger=logger,
    )


def compute_inferred_pca_umaps(
    adata: ad.AnnData,
    config,
    logger: logging.Logger,
    chunk_size: int = 20_000,
):
    """Compute PCA+UMAP on scvi_normalized and scanvi_normalized layers.

    Unlike the latent-space UMAPs, these preserve age gradients because the
    normalized expression encodes biological variation that the latent space
    removes when age is an explicit covariate.

    Memory model: each dense layer is (n_obs × n_hvg) float32. We stream it
    through IncrementalPCA in chunks of `chunk_size` rows, applying log1p on
    the fly. Peak extra RAM ≈ chunk_size × n_hvg × 4 B (~1.2 GB at 20k × 15k).

    Results stored in:
        obsm['X_umap_scvi_inferred']    — from scvi_normalized PCA
        obsm['X_umap_scanvi_inferred']  — from scanvi_normalized PCA
    """
    try:
        from sklearn.decomposition import IncrementalPCA
    except ImportError:
        logger.warning("sklearn not available — skipping inferred PCA UMAPs")
        return

    tasks = [
        (config.output_layer_scvi,   "X_pca_scvi_inferred",
         "X_umap_scvi_inferred",   "neighbors_scvi_inferred"),
        (config.output_layer_scanvi, "X_pca_scanvi_inferred",
         "X_umap_scanvi_inferred", "neighbors_scanvi_inferred"),
    ]

    hvg_mask = adata.var.get("highly_variable", None)
    if hvg_mask is not None and hvg_mask.any() and not hvg_mask.all():
        hvg_idx = np.where(np.asarray(hvg_mask))[0]
        logger.info(f"Inferred PCA UMAPs: using {len(hvg_idx):,} HVGs")
        use_all_cols = False
    else:
        hvg_idx = None
        logger.info(f"Inferred PCA UMAPs: data already HVG-only ({adata.n_vars:,} genes)")
        use_all_cols = True

    def _load_chunk(layer, start, stop):
        # Slice rows first, then optionally HVG cols. Keeps reads contiguous
        # for backed h5ad layers and avoids fancy-indexing the full matrix.
        if use_all_cols:
            block = layer[start:stop]
        else:
            block = layer[start:stop][:, hvg_idx]
        if sp.issparse(block):
            block = block.toarray()
        block = np.asarray(block, dtype=np.float32)
        np.log1p(block, out=block)
        return block

    for layer_name, pca_key, umap_key, nbrs_key in tasks:
        if layer_name not in adata.layers:
            logger.info(f"  {layer_name} layer not found — skipping {umap_key}")
            continue

        with Timer(f"Inferred PCA+UMAP from {layer_name}", logger):
            layer = adata.layers[layer_name]
            n_obs = layer.shape[0]
            n_cols = len(hvg_idx) if not use_all_cols else layer.shape[1]
            n_pcs = min(50, n_cols - 1)
            logger.info(
                f"  IncrementalPCA n_components={n_pcs} on ({n_obs:,}, {n_cols:,}) "
                f"in chunks of {chunk_size:,}"
            )

            ipca = IncrementalPCA(n_components=n_pcs, batch_size=chunk_size)
            for start in range(0, n_obs, chunk_size):
                stop = min(start + chunk_size, n_obs)
                chunk = _load_chunk(layer, start, stop)
                if chunk.shape[0] < n_pcs:
                    # IncrementalPCA requires batch_size ≥ n_components; merge
                    # the final stub into the previous partial_fit by skipping.
                    logger.info(
                        f"  Skipping partial_fit on final stub of {chunk.shape[0]} rows "
                        "(< n_components); transform pass still covers it."
                    )
                    continue
                ipca.partial_fit(chunk)

            pcs = np.empty((n_obs, n_pcs), dtype=np.float32)
            for start in range(0, n_obs, chunk_size):
                stop = min(start + chunk_size, n_obs)
                chunk = _load_chunk(layer, start, stop)
                pcs[start:stop] = ipca.transform(chunk)

            adata.obsm[pca_key] = pcs

            # Persist gene loadings (transpose: components_ is (n_pcs, n_features)).
            # Stored in full gene space with NaN for non-HVGs, matching the
            # ldvae_loadings pattern, so downstream code can use adata.var_names
            # directly to look up per-gene weights.
            comps = ipca.components_.astype(np.float32)
            pc_columns = [f"PC{i + 1}" for i in range(n_pcs)]
            loadings_key = pca_key.lstrip("X_") + "_loadings"
            if use_all_cols:
                adata.varm[loadings_key] = comps.T
            else:
                full = np.full((adata.n_vars, n_pcs), np.nan, dtype=np.float32)
                full[hvg_idx, :] = comps.T
                adata.varm[loadings_key] = full
            adata.uns[loadings_key + "_columns"] = pc_columns
            adata.uns[loadings_key + "_explained_variance_ratio"] = (
                ipca.explained_variance_ratio_.astype(np.float32)
            )
            logger.info(
                f"  Stored loadings in varm['{loadings_key}'] "
                f"shape={adata.varm[loadings_key].shape}; "
                f"top-3 explained variance ratios: "
                f"{ipca.explained_variance_ratio_[:3].round(4).tolist()}"
            )

            del ipca
            gc.collect()

            compute_umap(
                adata,
                obsm_key=pca_key,
                neighbors_key=nbrs_key,
                umap_key=umap_key,
                n_neighbors=config.umap_n_neighbors,
                min_dist=config.umap_min_dist,
                backend=getattr(config, "umap_backend", "exact"),
                subsample_n=getattr(config, "umap_subsample_n", 200_000),
                random_state=getattr(config, "random_seed", 42),
                logger=logger,
            )


def compute_latent_pcas(
    adata: ad.AnnData,
    logger: Optional[logging.Logger] = None,
):
    """Compute PCA of the scVI/scANVI latent representations.

    The 50-dim latent space has no guaranteed ordering by explained variance, so
    plotting dim0 vs dim1 is uninformative. This runs sklearn PCA to find the top
    components and stores them for PCA comparison plots.

    Results stored in:
        obsm['X_pca_scvi_latent']   — top PCs of X_scVI
        obsm['X_pca_scanvi_latent'] — top PCs of X_scANVI
    """
    try:
        from sklearn.decomposition import PCA as _PCA
    except ImportError:
        if logger:
            logger.warning("sklearn not available — skipping latent PCA")
        return

    for obsm_key, pca_key in [
        ("X_scVI",   "X_pca_scvi_latent"),
        ("X_scANVI", "X_pca_scanvi_latent"),
    ]:
        if obsm_key not in adata.obsm:
            if logger:
                logger.info(f"  {obsm_key} not in obsm — skipping {pca_key}")
            continue
        X = np.asarray(adata.obsm[obsm_key], dtype=np.float32)
        n_pcs = min(50, X.shape[1] - 1)
        if logger:
            logger.info(f"  PCA of {obsm_key} ({X.shape}) → {n_pcs} components → {pca_key}")
        adata.obsm[pca_key] = _PCA(n_components=n_pcs).fit_transform(X)


def compute_umaps(
    adata: ad.AnnData,
    config,
    logger: logging.Logger,
):
    """Compute UMAPs: raw (uncorrected), scVI/scANVI latent, scVI/scANVI inferred."""
    backend = getattr(config, "umap_backend", "exact")
    subsample_n = getattr(config, "umap_subsample_n", 200_000)
    random_state = getattr(config, "random_seed", 42)

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
                backend=backend,
                subsample_n=subsample_n,
                random_state=random_state,
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
                backend=backend,
                subsample_n=subsample_n,
                random_state=random_state,
                logger=logger,
            )

    if getattr(config, "compute_inferred_umaps", False):
        with Timer("Computing inferred PCA UMAPs (normalized expression)", logger):
            compute_inferred_pca_umaps(adata, config, logger)
    else:
        logger.info(
            "Skipping inferred PCA UMAPs (compute_inferred_umaps=False). "
            "Enable in config to compute UMAPs of scvi_normalized / scanvi_normalized."
        )

    with Timer("Computing PCA of latent representations", logger):
        compute_latent_pcas(adata, logger)


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
    palette: Optional[dict] = None,
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
        if palette is not None:
            fallback = _set1_palette(len(cats))
            color_map = {c: palette.get(c, fallback[i]) for i, c in enumerate(cats)}
        else:
            color_map = dict(zip(cats, _set1_palette(len(cats))))
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


def _plot_grid(
    adata: ad.AnnData,
    row_specs: list,
    col_specs: list,
    output_path: str,
    config,
    logger: Optional[logging.Logger] = None,
    panel_size: float = 3.5,
    point_size: Optional[float] = None,
    alpha: float = 0.45,
    x_axis_label: str = "UMAP1",
    y_axis_label: str = "UMAP2",
):
    """Render a n_rows × n_cols embedding grid with per-row legends.

    row_specs     : list of (obs_col, row_label) — one row per variable
    col_specs     : list of (obsm_key, col_label) — one column per embedding
    x_axis_label  : label for the x axis of each panel (default "UMAP1")
    y_axis_label  : label for the y axis of non-leftmost panels (default "UMAP2")
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    n_rows = len(row_specs)
    n_cols = len(col_specs)
    pt_size = point_size if point_size is not None else config.umap_point_size

    LEGEND_W = 2.4   # inches for the per-row legend/colorbar column
    fig_w = n_cols * panel_size + LEGEND_W
    fig_h = n_rows * panel_size + 0.5   # 0.5" for column-header row

    fig = plt.figure(figsize=(fig_w, fig_h))
    # GridSpec: header row (thin) + n_rows data rows × (n_cols data cols + 1 legend col)
    col_widths = [panel_size] * n_cols + [LEGEND_W]
    row_heights = [0.4] + [panel_size] * n_rows   # first row for column headers

    gs = gridspec.GridSpec(
        n_rows + 1, n_cols + 1,
        width_ratios=col_widths,
        height_ratios=row_heights,
        hspace=0.08,
        wspace=0.06,
        left=0.06, right=0.99,
        top=0.97, bottom=0.03,
    )

    rng = np.random.RandomState(42)
    order = rng.permutation(adata.n_obs)

    # Column headers (row 0)
    for col_i, (obsm_key, col_label) in enumerate(col_specs):
        ax_hdr = fig.add_subplot(gs[0, col_i])
        ax_hdr.axis("off")
        ax_hdr.text(0.5, 0.5, col_label, ha="center", va="center",
                    fontsize=11, fontweight="bold", transform=ax_hdr.transAxes)

    log2_ticks = getattr(config, "umap_log2_ticks", [0, 1, 3, 9, 25, 40])

    # Data rows
    for row_i, (obs_col, row_label) in enumerate(row_specs):
        gs_row = row_i + 1   # gs row 0 is the header

        if obs_col not in adata.obs.columns:
            if logger:
                logger.warning(f"'{obs_col}' not in .obs — row skipped")
            for col_i in range(n_cols):
                ax = fig.add_subplot(gs[gs_row, col_i])
                ax.axis("off")
                ax.text(0.5, 0.5, f"{obs_col}\nnot found",
                        ha="center", va="center", fontsize=9, color="#888888",
                        transform=ax.transAxes)
            continue

        values = adata.obs[obs_col]
        is_log = obs_col == "age_years"
        is_categorical = (values.dtype.name == "category" or
                          values.dtype == object or
                          not is_log and values.dtype.kind not in "fiu")

        # Build color mapping for this row
        if is_categorical:
            cats = (list(values.cat.categories)
                    if hasattr(values, "cat") else sorted(values.unique()))
            if obs_col == "cell_type_aligned":
                pal = build_cell_type_aligned_palette(cats)
                fallback = _set1_palette(len(cats))
                color_map = {c: pal.get(c, fallback[i]) for i, c in enumerate(cats)}
            else:
                color_map = dict(zip(cats, _set1_palette(len(cats))))
        else:
            raw_vals = np.array(values, dtype=float)
            plot_vals = np.log2(raw_vals + 1) if is_log else raw_vals
            vmin, vmax = float(np.nanmin(plot_vals)), float(np.nanmax(plot_vals))
            cmap_obj = cm.get_cmap("viridis")
            norm = Normalize(vmin=vmin, vmax=vmax)

        for col_i, (obsm_key, _) in enumerate(col_specs):
            ax = fig.add_subplot(gs[gs_row, col_i])
            _apply_ggplot_theme(ax)
            ax.set_box_aspect(1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(x_axis_label, fontsize=7, color="#777777")
            ax.set_ylabel(y_axis_label, fontsize=7, color="#777777")

            # Row label on leftmost column
            if col_i == 0:
                ax.set_ylabel(row_label, fontsize=10, labelpad=6)

            if obsm_key not in adata.obsm:
                ax.text(0.5, 0.5, "not computed", ha="center", va="center",
                        fontsize=9, color="#888888", transform=ax.transAxes)
                continue

            coords = adata.obsm[obsm_key][order]

            if is_categorical:
                point_colors = [color_map[v] for v in values.iloc[order]]
                ax.scatter(coords[:, 0], coords[:, 1],
                           c=point_colors, s=pt_size, alpha=alpha,
                           rasterized=True, linewidths=0)
            else:
                ax.scatter(coords[:, 0], coords[:, 1],
                           c=plot_vals[order], s=pt_size, alpha=alpha,
                           cmap=cmap_obj, norm=norm,
                           rasterized=True, linewidths=0)

        # Legend / colorbar in last column
        leg_ax = fig.add_subplot(gs[gs_row, -1])
        leg_ax.axis("off")

        if is_categorical:
            handles = [
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_map[c], markersize=5, label=c)
                for c in cats
            ]
            leg_ax.legend(
                handles=handles, loc="center left",
                bbox_to_anchor=(0.0, 0.5),
                fontsize=6.5, frameon=False,
                title=row_label, title_fontsize=8,
                markerscale=1.4,
                ncol=max(1, len(handles) // 20),
            )
        else:
            # Inset colorbar
            cax = leg_ax.inset_axes([0.05, 0.15, 0.25, 0.70])
            sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(row_label, fontsize=8)
            if is_log:
                tick_vals = [t for t in log2_ticks if vmin <= np.log2(t + 1) <= vmax]
                if tick_vals:
                    cbar.set_ticks([np.log2(t + 1) for t in tick_vals])
                    cbar.set_ticklabels([str(t) for t in tick_vals])
            cbar.ax.tick_params(labelsize=7)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if logger:
        logger.info(f"Saved: {output_path}")
    plt.close(fig)


def plot_umap_grids(
    adata: ad.AnnData,
    config,
    logger: logging.Logger,
):
    """Output umaps_latent.png and umaps_inferred.png in scvi_output/plots/.

    Each is a 4-row × 3-column grid:
      Rows    : batch (batch_key), age (log years), cell_class, cell_type_aligned
      Latent cols  : Uncorrected PCA | scVI latent | scANVI latent
      Inferred cols: Uncorrected PCA | scVI normalized | scANVI normalized
    """
    plots_dir = config._resolved_output_dir.parent / "plots" / "all"

    batch_col = getattr(config, "batch_key", "source")

    row_specs = [
        (batch_col,          f"Batch ({batch_col})"),
        ("age_years",        "Age (log years)"),
        ("cell_class",       "Cell class"),
        ("cell_type_aligned","Cell type"),
    ]

    latent_cols = [
        ("X_umap_raw",     "Uncorrected (PCA)"),
        ("X_umap_scvi",    "scVI latent"),
        ("X_umap_scanvi",  "scANVI latent"),
    ]
    inferred_cols = [
        ("X_umap_raw",              "Uncorrected (PCA)"),
        ("X_umap_scvi_inferred",    "scVI normalized"),
        ("X_umap_scanvi_inferred",  "scANVI normalized"),
    ]

    _plot_grid(
        adata, row_specs, latent_cols,
        output_path=str(plots_dir / "umaps_latent.png"),
        config=config, logger=logger,
        point_size=config.umap_point_size,
    )
    _plot_grid(
        adata, row_specs, inferred_cols,
        output_path=str(plots_dir / "umaps_inferred.png"),
        config=config, logger=logger,
        point_size=config.umap_point_size,
    )


def plot_pca_grids(
    adata: ad.AnnData,
    config,
    logger: logging.Logger,
):
    """Output pca_latent.png and pca_inferred.png in scvi_output/plots/.

    Mirrors the layout of plot_umap_grids but shows PC1 vs PC2 scatter instead
    of UMAP coordinates. Useful for inspecting whether age or batch gradients
    are captured in the top PCs of each representation.

    Latent cols  : Uncorrected PCA | PCA of scVI latent | PCA of scANVI latent
    Inferred cols: Uncorrected PCA | PCA of scVI normalised | PCA of scANVI normalised

    Calls compute_latent_pcas() as a fallback so this function works when
    re-plotting an existing integrated.h5ad that predates the latent PCA step.
    """
    compute_latent_pcas(adata, logger)

    plots_dir = config._resolved_output_dir.parent / "plots" / "all"
    batch_col = getattr(config, "batch_key", "source")

    row_specs = [
        (batch_col,           f"Batch ({batch_col})"),
        ("age_years",         "Age (log years)"),
        ("cell_class",        "Cell class"),
        ("cell_type_aligned", "Cell type"),
    ]
    latent_cols = [
        ("X_pca_raw",           "Uncorrected (PCA)"),
        ("X_pca_scvi_latent",   "scVI latent"),
        ("X_pca_scanvi_latent", "scANVI latent"),
    ]
    inferred_cols = [
        ("X_pca_raw",             "Uncorrected (PCA)"),
        ("X_pca_scvi_inferred",   "scVI normalised"),
        ("X_pca_scanvi_inferred", "scANVI normalised"),
    ]

    _plot_grid(
        adata, row_specs, latent_cols,
        output_path=str(plots_dir / "pca_latent.png"),
        config=config, logger=logger,
        point_size=config.umap_point_size,
        x_axis_label="PC1", y_axis_label="PC2",
    )
    _plot_grid(
        adata, row_specs, inferred_cols,
        output_path=str(plots_dir / "pca_inferred.png"),
        config=config, logger=logger,
        point_size=config.umap_point_size,
        x_axis_label="PC1", y_axis_label="PC2",
    )


def plot_excitatory_grids(
    adata: ad.AnnData,
    config,
    logger: logging.Logger,
):
    """Recompute UMAPs/PCAs on excitatory neurons and write 4 plot files.

    UMAPs and PCAs are computed fresh on the excitatory subset so the embedding
    reflects only excitatory cell variation. Plots show 3 rows coloured by
    batch, age, and cell type (cell_type_aligned if present from scANVI label
    transfer, else cell_type_raw). Written to <run>/plots/excitatory/.
    """
    if "cell_class" not in adata.obs.columns:
        logger.warning("cell_class column not found — skipping excitatory plots")
        return

    mask = adata.obs["cell_class"] == "Excitatory"
    excitatory = adata[mask].copy()
    if excitatory.n_obs == 0:
        logger.warning("No excitatory cells found — skipping excitatory plots")
        return
    logger.info(f"Excitatory subset: {excitatory.n_obs:,} cells")

    if "cell_type_aligned" in excitatory.obs.columns:
        ct_col, ct_label = "cell_type_aligned", "Cell type (aligned)"
    else:
        ct_col, ct_label = "cell_type_raw", "Cell type (raw)"

    with Timer("UMAP computation (excitatory subset)", logger):
        compute_umaps(excitatory, config, logger)

    plots_dir = config._resolved_output_dir.parent / "plots" / "excitatory"
    batch_col = getattr(config, "batch_key", "source")
    row_specs = [
        (batch_col,   f"Batch ({batch_col})"),
        ("age_years", "Age (log years)"),
        (ct_col,      ct_label),
    ]

    latent_cols = [
        ("X_umap_raw",    "Uncorrected (PCA)"),
        ("X_umap_scvi",   "scVI latent"),
        ("X_umap_scanvi", "scANVI latent"),
    ]
    inferred_cols = [
        ("X_umap_raw",             "Uncorrected (PCA)"),
        ("X_umap_scvi_inferred",   "scVI normalized"),
        ("X_umap_scanvi_inferred", "scANVI normalized"),
    ]
    _plot_grid(excitatory, row_specs, latent_cols,
               output_path=str(plots_dir / "umaps_latent.png"),
               config=config, logger=logger, point_size=config.umap_point_size)
    _plot_grid(excitatory, row_specs, inferred_cols,
               output_path=str(plots_dir / "umaps_inferred.png"),
               config=config, logger=logger, point_size=config.umap_point_size)

    compute_latent_pcas(excitatory, logger)
    pca_latent_cols = [
        ("X_pca_raw",           "Uncorrected (PCA)"),
        ("X_pca_scvi_latent",   "scVI latent"),
        ("X_pca_scanvi_latent", "scANVI latent"),
    ]
    pca_inferred_cols = [
        ("X_pca_raw",             "Uncorrected (PCA)"),
        ("X_pca_scvi_inferred",   "scVI normalised"),
        ("X_pca_scanvi_inferred", "scANVI normalised"),
    ]
    _plot_grid(excitatory, row_specs, pca_latent_cols,
               output_path=str(plots_dir / "pca_latent.png"),
               config=config, logger=logger, point_size=config.umap_point_size,
               x_axis_label="PC1", y_axis_label="PC2")
    _plot_grid(excitatory, row_specs, pca_inferred_cols,
               output_path=str(plots_dir / "pca_inferred.png"),
               config=config, logger=logger, point_size=config.umap_point_size,
               x_axis_label="PC1", y_axis_label="PC2")


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

        var_palette = None
        if var == "cell_type_aligned":
            cats = list(adata.obs[var].unique())
            var_palette = build_cell_type_aligned_palette(cats)

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
            palette=var_palette,
            logger=logger,
        )
