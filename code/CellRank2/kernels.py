"""Kernel construction and combination for CellRank 2 lineage tracing.

Two kernels are built and combined:

  ConnectivityKernel (CK)
    Captures transcriptomic similarity via a kNN graph computed on the
    scANVI-corrected latent representation (X_scANVI).  Symmetric, so it
    handles self-transitions within the same time point.

  RealTimeKernel (RTK)
    Captures directional cell fate using donor chronological age as the
    experimental time axis.  Optimal transport (moscot) maps cells between
    consecutive age bins, producing asymmetric couplings that drive the
    transition matrix forward in developmental / maturational time.

Combined kernel = rtk_weight * RTK + (1 - rtk_weight) * CK
"""

import logging
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .config import CellRankConfig
from .utils import Timer, log_memory


# ── Neighbour graph ────────────────────────────────────────────────────────────

def validate_scanvi_outputs(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Raise an informative error if the scANVI pipeline was not run.

    Checks that the scANVI latent representation (``config.latent_key``,
    typically ``X_scANVI``) is present in ``adata.obsm``.  If it is missing,
    a :class:`RuntimeError` is raised immediately so the user knows they must
    run the scVI/scANVI step before CellRank.

    This is called once at the start of the pipeline so the error is surfaced
    early rather than buried in a later step.
    """
    if config.latent_key not in adata.obsm:
        available = list(adata.obsm.keys())
        raise RuntimeError(
            f"scANVI latent key '{config.latent_key}' not found in adata.obsm "
            f"(available: {available}). "
            "The scANVI step must be run before the CellRank pipeline. "
            "Re-run the scVI/scANVI pipeline (step 'scvi' or 'scanvi') and "
            "ensure the integrated h5ad is saved with the scANVI latent space."
        )
    logger.info(
        f"  scANVI validation passed: '{config.latent_key}' found in adata.obsm."
    )


def ensure_umap(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Ensure the UMAP embedding for plotting exists in ``adata.obsm``.

    If ``config.umap_key`` (e.g. ``X_umap_scanvi``) is already present,
    nothing is done.  Otherwise, a UMAP is computed from ``config.latent_key``
    (``X_scANVI``) using the neighbour graph stored under
    ``config.neighbors_key`` and stored as ``config.umap_key``.

    Raises
    ------
    RuntimeError
        If ``config.latent_key`` is not in ``adata.obsm`` (scANVI not run).
    RuntimeError
        If the neighbour graph (``config.neighbors_key``) has not been
        computed yet (run the ``neighbors`` step first).
    """
    umap_key = config.umap_key
    if umap_key in adata.obsm and not config.overwrite:
        logger.info(
            f"  '{umap_key}' already present in adata.obsm, skipping UMAP "
            "computation (set overwrite=True to recompute)."
        )
        return

    # Guard: scANVI must have been run
    if config.latent_key not in adata.obsm:
        raise RuntimeError(
            f"Cannot compute UMAP: scANVI latent key '{config.latent_key}' "
            "not found in adata.obsm. Run the scANVI step first."
        )

    # Guard: neighbour graph must exist
    if config.neighbors_key not in adata.uns:
        raise RuntimeError(
            f"Cannot compute UMAP: neighbour graph '{config.neighbors_key}' "
            "not found in adata.uns. Run the 'neighbors' step first."
        )

    logger.info(
        f"  '{umap_key}' not found in adata.obsm — computing UMAP from "
        f"'{config.latent_key}' using neighbour graph '{config.neighbors_key}'."
    )
    with Timer(f"UMAP on '{config.latent_key}'", logger):
        sc.tl.umap(
            adata,
            neighbors_key=config.neighbors_key,
            random_state=config.random_seed,
        )
    adata.obsm[umap_key] = adata.obsm["X_umap"].copy()
    logger.info(f"  UMAP computed and stored as '{umap_key}'.")


def ensure_neighbors(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Compute kNN graph on the scANVI latent space (if not already stored).

    The graph is stored under ``neighbors_key`` in ``adata.uns`` /
    ``adata.obsp``, matching the key expected by the ConnectivityKernel.
    """
    if config.latent_key not in adata.obsm:
        raise KeyError(
            f"Latent key '{config.latent_key}' not found in adata.obsm. "
            "Run the scVI/scANVI pipeline first to generate the latent "
            "representation."
        )

    uns_key = config.neighbors_key
    if uns_key in adata.uns and not config.overwrite:
        logger.info(
            f"  Neighbour graph '{uns_key}' already present, skipping "
            "(set overwrite=True to recompute)."
        )
        return

    with Timer(
        f"Computing kNN graph (n_neighbors={config.n_neighbors}) "
        f"on '{config.latent_key}'",
        logger,
    ):
        sc.pp.neighbors(
            adata,
            use_rep=config.latent_key,
            n_neighbors=config.n_neighbors,
            key_added=uns_key,
            random_state=config.random_seed,
        )
    logger.info(f"  Stored kNN graph as '{uns_key}'.")


# ── Age binning ────────────────────────────────────────────────────────────────

def bin_ages(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Map continuous ``time_key`` values into discrete age bins.

    Bins are defined by ``config.age_bin_edges``.  The resulting categorical
    column is stored in ``adata.obs[config.age_bin_key]``.  If no bin edges
    are provided, the unique values of ``time_key`` are used directly (assumed
    already discrete / categorical).
    """
    if config.time_key not in adata.obs.columns:
        raise KeyError(
            f"Time key '{config.time_key}' not found in adata.obs. "
            "Ensure donor age is stored before running CellRank."
        )

    age_bin_key = config.age_bin_key
    if age_bin_key in adata.obs.columns and not config.overwrite:
        logger.info(
            f"  Age bins '{age_bin_key}' already present, skipping."
        )
        return

    edges = config.age_bin_edges
    if edges:
        # Use numeric bin midpoints as category values — required by moscot.
        midpoints = [
            round((edges[i] + edges[i + 1]) / 2, 4) for i in range(len(edges) - 1)
        ]
        bins = pd.cut(
            adata.obs[config.time_key].astype(float),
            bins=edges,
            labels=midpoints,
            right=False,
            include_lowest=True,
        )
        # Preserve numeric dtype in the category (moscot requires numeric cats)
        adata.obs[age_bin_key] = bins.astype(float).astype("category")
        n_bins = adata.obs[age_bin_key].nunique()
        bin_counts = adata.obs[age_bin_key].value_counts().to_dict()
        logger.info(
            f"  Age binning: {n_bins} bins (midpoints: {midpoints})"
        )
        for lbl, cnt in sorted(bin_counts.items()):
            logger.info(f"    bin {lbl}: {cnt} cells")
    else:
        # Use raw values as-is — must already be numeric and categorical
        raw = adata.obs[config.time_key].astype(float)
        adata.obs[age_bin_key] = pd.Categorical(raw)
        logger.info(
            f"  No bin edges provided; using raw '{config.time_key}' as "
            f"'{age_bin_key}'."
        )


# ── Optimal transport (moscot RealTimeKernel) ──────────────────────────────────

def run_moscot_ot(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
):
    """Solve moscot TemporalProblem for all consecutive age-bin pairs.

    Returns the solved moscot ``TemporalProblem`` object, which is passed
    directly to ``RealTimeKernel.from_moscot``.
    """
    import moscot.problems as mp

    time_col = config.age_bin_key
    sorted_times = sorted(adata.obs[time_col].cat.categories)
    logger.info(
        f"  Running moscot OT across {len(sorted_times)} time points: "
        + ", ".join(str(t) for t in sorted_times)
    )

    with Timer("moscot TemporalProblem (prepare + solve)", logger):
        problem = mp.TemporalProblem(adata)
        problem = problem.prepare(
            time_key=time_col,
        )
        problem = problem.solve(
            epsilon=config.ot_epsilon,
            max_iterations=config.ot_max_iterations,
        )

    logger.info("  moscot OT solved successfully.")
    return problem


# ── Kernel construction ────────────────────────────────────────────────────────

def build_connectivity_kernel(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
):
    """Build and compute the ConnectivityKernel on the scANVI latent graph."""
    import cellrank as cr

    with Timer("ConnectivityKernel", logger):
        ck = cr.kernels.ConnectivityKernel(
            adata, conn_key=f"{config.neighbors_key}_connectivities"
        )
        ck.compute_transition_matrix()

    logger.info(
        f"  ConnectivityKernel: transition matrix shape "
        f"{ck.transition_matrix.shape}"
    )
    return ck


def build_realtime_kernel(
    adata: ad.AnnData,
    moscot_problem,
    config: CellRankConfig,
    logger: logging.Logger,
):
    """Build RealTimeKernel from a solved moscot TemporalProblem."""
    import cellrank as cr

    with Timer("RealTimeKernel (from moscot)", logger):
        rtk = cr.kernels.RealTimeKernel.from_moscot(moscot_problem)
        rtk.compute_transition_matrix(
            self_transitions="connectivities",
            conn_kwargs={
                "key_added": config.neighbors_key,
            },
        )

    logger.info(
        f"  RealTimeKernel: transition matrix shape "
        f"{rtk.transition_matrix.shape}"
    )
    return rtk


def combine_kernels(
    rtk,
    ck,
    config: CellRankConfig,
    logger: logging.Logger,
):
    """Combine RealTimeKernel and ConnectivityKernel into a weighted sum.

    Combined = rtk_weight * RTK + (1 - rtk_weight) * CK
    """
    w_rtk = config.rtk_weight
    w_ck = 1.0 - w_rtk
    combined = w_rtk * rtk + w_ck * ck
    logger.info(
        f"  Combined kernel: "
        f"RTK weight={w_rtk:.2f}, CK weight={w_ck:.2f}"
    )
    return combined


# ── Public entry point ─────────────────────────────────────────────────────────

def build_kernels(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
    moscot_problem=None,
):
    """Build the ConnectivityKernel, optionally combine with RealTimeKernel.

    If ``moscot_problem`` is provided (from a prior ``run_moscot_ot`` call),
    the function builds and combines both kernels.  Otherwise, only the
    ConnectivityKernel is built (useful for testing without age data).

    Returns
    -------
    combined_kernel
        The final kernel expression passed to the GPCCA estimator.
    ck
        The raw ConnectivityKernel (stored separately for diagnostics).
    rtk
        The raw RealTimeKernel, or None if not built.
    """
    ck = build_connectivity_kernel(adata, config, logger)

    if moscot_problem is not None:
        rtk = build_realtime_kernel(adata, moscot_problem, config, logger)
        combined = combine_kernels(rtk, ck, config, logger)
    else:
        logger.warning(
            "No moscot problem provided; using ConnectivityKernel only. "
            "Results will lack temporal directionality."
        )
        combined = ck
        rtk = None

    log_memory("After kernel construction", logger)
    return combined, ck, rtk
