"""Kernel construction and combination for CellRank 2 lineage tracing.

Three kernels are built and combined:

  ConnectivityKernel (CK)
    Captures transcriptomic similarity via a kNN graph computed on the
    scANVI-corrected latent representation (X_scANVI).  Symmetric, so it
    handles self-transitions within the same time point.

  RealTimeKernel (RTK)
    Captures directional cell fate using donor chronological age as the
    experimental time axis.  Optimal transport (moscot) maps cells between
    consecutive age bins, producing asymmetric couplings that drive the
    transition matrix forward in developmental / maturational time.

  CytoTRACEKernel (CTK)  [optional, enabled when cytotrace_weight > 0]
    Captures cell-intrinsic developmental directionality from transcriptomic
    complexity: cells expressing more genes are less differentiated.  Drives
    the Markov chain from high-complexity progenitors to low-complexity mature
    neurons without relying on age-bin OT couplings.

Combined kernel = cytotrace_weight * CTK + rtk_weight * RTK + (1-both) * CK
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
        neigh_meta = adata.uns.get(uns_key, {})
        conn_key = neigh_meta.get("connectivities_key", f"{uns_key}_connectivities")
        dist_key = neigh_meta.get("distances_key", f"{uns_key}_distances")

        stale_reason = None
        if conn_key not in adata.obsp:
            stale_reason = f"missing connectivities '{conn_key}'"
        else:
            conn = adata.obsp[conn_key]
            if conn.shape != (adata.n_obs, adata.n_obs):
                stale_reason = (
                    f"connectivity matrix shape {conn.shape} does not match "
                    f"current cells ({adata.n_obs}, {adata.n_obs})"
                )
            else:
                row_sums = np.asarray(conn.sum(axis=1)).ravel()
                n_zero = int(np.sum(row_sums <= 0))
                if n_zero > 0:
                    stale_reason = (
                        f"{n_zero} cells have zero connectivity degree"
                    )

        if stale_reason is None:
            logger.info(
                f"  Neighbour graph '{uns_key}' already present, skipping "
                "(set overwrite=True to recompute)."
            )
            return

        logger.warning(
            f"  Existing neighbour graph '{uns_key}' is stale/invalid "
            f"({stale_reason}); recomputing."
        )
        adata.uns.pop(uns_key, None)
        adata.obsp.pop(conn_key, None)
        adata.obsp.pop(dist_key, None)

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

    # Resolve device: "auto" → "cuda" if available, else "cpu"
    device = config.ot_device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    logger.info(f"  moscot OT device: {device}")

    # Use the batch-corrected latent embedding (X_scANVI, 30D) as the OT cost
    # rather than adata.X (15,540D → PCA fallback). This makes couplings
    # transcriptomically specific: prenatal cells couple to the adult cells they
    # most resemble in the corrected latent space.
    joint_attr = {"attr": "obsm", "key": config.latent_key}
    logger.info(f"  OT joint_attr: obsm['{config.latent_key}']")

    with Timer("moscot TemporalProblem (prepare + solve)", logger):
        problem = mp.TemporalProblem(adata)
        problem = problem.prepare(
            time_key=time_col,
            joint_attr=joint_attr,
        )
        problem = problem.solve(
            epsilon=config.ot_epsilon,
            max_iterations=config.ot_max_iterations,
            device=device,
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


def build_cytotrace_kernel(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
):
    """Build CytoTRACEKernel using transcriptomic complexity (genes per cell).

    Drives the Markov chain from high-complexity progenitors toward
    low-complexity mature neurons, providing cell-intrinsic directionality
    that the RealTimeKernel alone cannot supply.

    Returns None if CytoTRACEKernel is not available in the installed
    CellRank version.
    """
    try:
        from cellrank.kernels import CytoTRACEKernel
    except ImportError:
        logger.warning(
            "CytoTRACEKernel not available in this CellRank version; "
            "set cytotrace_weight=0 to silence this warning."
        )
        return None

    try:
        # CytoTRACEKernel reads adata.uns['neighbors'] by default regardless of
        # what conn_key is passed. Temporarily expose our custom graph under the
        # standard key so the kernel can find it.
        _prior = adata.uns.get("neighbors")
        adata.uns["neighbors"] = adata.uns[config.neighbors_key]
        try:
            with Timer("CytoTRACEKernel", logger):
                ctk = CytoTRACEKernel(adata)
                ctk.compute_transition_matrix(threshold_scheme="soft")
        finally:
            if _prior is not None:
                adata.uns["neighbors"] = _prior
            else:
                adata.uns.pop("neighbors", None)

        logger.info(
            f"  CytoTRACEKernel: transition matrix shape "
            f"{ctk.transition_matrix.shape}"
        )
        return ctk
    except Exception as exc:
        logger.warning(
            f"  CytoTRACEKernel.compute_transition_matrix failed ({exc}); "
            "continuing without CytoTRACEKernel."
        )
        return None


def combine_kernels(
    rtk,
    ck,
    config: CellRankConfig,
    logger: logging.Logger,
    ctk=None,
):
    """Combine kernels into a weighted sum.

    With CytoTRACEKernel:
        combined = cytotrace_weight * CTK + rtk_weight * RTK + (1-both) * CK
    Without:
        combined = rtk_weight * RTK + (1 - rtk_weight) * CK
    """
    w_rtk = config.rtk_weight
    w_ctk = config.cytotrace_weight if ctk is not None else 0.0
    w_ck = 1.0 - w_rtk - w_ctk

    if ctk is not None:
        combined = w_ctk * ctk + w_rtk * rtk + w_ck * ck
        logger.info(
            f"  Combined kernel: "
            f"CTK weight={w_ctk:.2f}, RTK weight={w_rtk:.2f}, CK weight={w_ck:.2f}"
        )
    else:
        combined = w_rtk * rtk + w_ck * ck
        logger.info(
            f"  Combined kernel: "
            f"RTK weight={w_rtk:.2f}, CK weight={w_ck:.2f}"
        )
    return combined


def compute_lineage_umap(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Recompute UMAP on the EN-only subset from the existing kNN graph.

    The neighbours graph (config.neighbors_key) was computed on X_scANVI of
    the filtered EN subset, so this UMAP focuses all dimensionality-reduction
    budget on within-EN developmental variance rather than cross-cell-type
    separation.  Result stored in adata.obsm[config.lineage_umap_key].
    """
    if not config.recompute_umap:
        return

    if config.lineage_umap_key in adata.obsm and not config.overwrite:
        logger.info(
            f"  '{config.lineage_umap_key}' already present; "
            "skipping UMAP recomputation (set overwrite=True to redo)."
        )
        return

    if config.neighbors_key not in adata.uns:
        logger.warning(
            f"  '{config.neighbors_key}' not in adata.uns; "
            "cannot recompute UMAP — run neighbors step first."
        )
        return

    with Timer("UMAP on EN-only subset", logger):
        sc.tl.umap(
            adata,
            neighbors_key=config.neighbors_key,
            random_state=config.random_seed,
        )

    adata.obsm[config.lineage_umap_key] = adata.obsm["X_umap"].copy()
    logger.info(
        f"  EN-only UMAP stored as '{config.lineage_umap_key}' "
        f"({adata.n_obs} cells × 2 dims)"
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def build_kernels(
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
    moscot_problem=None,
):
    """Build and combine kernels for GPCCA.

    Always builds ConnectivityKernel.  If ``moscot_problem`` is provided,
    also builds RealTimeKernel.  If ``config.cytotrace_weight > 0``, also
    builds CytoTRACEKernel for cell-intrinsic developmental directionality.

    Returns
    -------
    combined_kernel
        The final weighted kernel expression for the GPCCA estimator.
    ck
        The raw ConnectivityKernel.
    rtk
        The raw RealTimeKernel, or None if not built.
    ctk
        The raw CytoTRACEKernel, or None if not built.
    """
    ck = build_connectivity_kernel(adata, config, logger)

    # CytoTRACEKernel (optional — provides cell-intrinsic directionality)
    ctk = None
    if config.cytotrace_weight > 0:
        ctk = build_cytotrace_kernel(adata, config, logger)
        if ctk is None:
            logger.warning(
                "  CytoTRACEKernel build failed; proceeding without it. "
                "Kernel weights will be renormalised automatically."
            )

    if moscot_problem is not None:
        rtk = build_realtime_kernel(adata, moscot_problem, config, logger)
        combined = combine_kernels(rtk, ck, config, logger, ctk=ctk)
    else:
        logger.warning(
            "No moscot problem provided; using ConnectivityKernel only. "
            "Results will lack temporal directionality."
        )
        combined = ck
        rtk = None

    log_memory("After kernel construction", logger)
    return combined, ck, rtk, ctk
