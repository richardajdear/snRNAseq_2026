"""GPCCA estimator: macrostates, terminal/initial states, fate probabilities.

Wraps the CellRank 2 GPCCA estimator with sensible defaults for snRNA-seq
developmental / maturational pseudotime analysis.
"""

import logging
from typing import List, Optional, Union

import anndata as ad

from .config import CellRankConfig
from .utils import Timer, log_memory


def build_gpcca(
    combined_kernel,
    config: CellRankConfig,
    logger: logging.Logger,
):
    """Instantiate the GPCCA estimator from a combined kernel.

    Returns
    -------
    g : cellrank.estimators.GPCCA
    """
    import cellrank as cr

    logger.info("Building GPCCA estimator...")
    g = cr.estimators.GPCCA(combined_kernel)
    return g


def compute_macrostates(
    g,
    config: CellRankConfig,
    logger: logging.Logger,
):
    """Compute macrostates using GPCCA (Schur decomposition).

    If ``config.use_minchi`` is True, a range [n_min, n_max] is passed so
    that GPCCA selects the optimal number automatically via the minChi
    criterion.  Otherwise the fixed ``config.n_macrostates`` value is used.
    """
    if config.use_minchi:
        n_states = list(
            range(config.n_macrostates_min, config.n_macrostates_max + 1)
        )
        logger.info(
            f"Computing macrostates with minChi: "
            f"testing n_states={n_states[0]}–{n_states[-1]}"
        )
    else:
        n_states = config.n_macrostates
        logger.info(f"Computing macrostates: n_states={n_states}")

    cluster_key = (
        config.cluster_key
        if config.cluster_key
        else None
    )

    with Timer("GPCCA compute_macrostates", logger):
        g.compute_macrostates(
            n_states=n_states,
            cluster_key=cluster_key,
            n_cells=30,
        )

    if g.macrostates is not None:
        n_actual = g.macrostates.nunique()
        logger.info(f"  Macrostates identified: {n_actual}")
        for state, count in g.macrostates.value_counts().items():
            logger.info(f"    {state}: {count} cells")

    log_memory("After compute_macrostates", logger)
    return g


def _predict_terminal_states_robust(g, logger: logging.Logger) -> None:
    """Try several methods to auto-select terminal states, falling back gracefully.

    CellRank's ``predict_terminal_states(method='stability', threshold=0.96)``
    can fail on small or noisy datasets where no macrostate meets the threshold.
    We try progressively more permissive strategies:
      1. stability (default, threshold=0.96)
      2. stability with lower threshold (0.5)
      3. top_n=1 (always succeeds)
    ``allow_overlap=True`` is used throughout so that the same states can appear
    as both initial and terminal (common on small / noisy datasets).
    """
    # Strategy 1: default stability threshold
    try:
        g.predict_terminal_states(method="stability", allow_overlap=True)
        return
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            f"  predict_terminal_states(stability, 0.96) failed: {exc}. "
            "Trying lower threshold..."
        )

    # Strategy 2: relaxed stability threshold
    try:
        g.predict_terminal_states(
            method="stability", stability_threshold=0.5, allow_overlap=True
        )
        return
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            f"  predict_terminal_states(stability, 0.5) failed: {exc}. "
            "Falling back to top_n=1..."
        )

    # Strategy 3: always works — select top-1 most stable state
    try:
        g.predict_terminal_states(method="top_n", n_states=1, allow_overlap=True)
    except Exception as exc:
        logger.warning(
            f"  All terminal state prediction strategies failed: {exc}. "
            "No terminal states assigned."
        )



def set_terminal_and_initial_states(
    g,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Set terminal and initial states on the GPCCA estimator.

    If ``config.terminal_states`` is non-empty, those states are set
    explicitly.  Otherwise, ``predict_terminal_states()`` is called to
    select them automatically based on the coarse-grained transition matrix.

    Same logic applies to ``config.initial_states``.
    """
    # Terminal states
    if config.terminal_states:
        logger.info(
            f"Setting terminal states explicitly: {config.terminal_states}"
        )
        g.set_terminal_states(states=config.terminal_states)
    else:
        logger.info("Auto-predicting terminal states...")
        with Timer("predict_terminal_states", logger):
            _predict_terminal_states_robust(g, logger)
        if g.terminal_states is not None:
            logger.info(
                f"  Terminal states: "
                f"{sorted(g.terminal_states.cat.categories.tolist())}"
            )

    # Initial states
    if config.initial_states:
        logger.info(
            f"Setting initial states explicitly: {config.initial_states}"
        )
        g.set_initial_states(states=config.initial_states)
    else:
        logger.info("Auto-predicting initial states...")
        with Timer("predict_initial_states", logger):
            try:
                g.predict_initial_states(allow_overlap=True)
                if g.initial_states is not None:
                    logger.info(
                        f"  Initial states: "
                        f"{sorted(g.initial_states.cat.categories.tolist())}"
                    )
            except Exception as exc:
                logger.warning(
                    f"  predict_initial_states failed ({exc}); "
                    "skipping initial state assignment."
                )


def set_terminal_states_from_cell_types(
    g,
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Set terminal states directly from cell_type_key, bypassing GPCCA.

    Each unique value in ``config.cell_type_key`` becomes an absorbing
    terminal state with a hard 0/1 membership array.  This guarantees that
    every cell type in your annotation is represented as a distinct lineage
    (e.g. 'ExcitatoryNeuron_L2-3' will always appear), at the cost of not
    using the spectral structure of the transition matrix to infer which
    states are truly absorbing.

    Use when you trust your cell-type annotation more than GPCCA's automatic
    macrostate detection, or when ``n_macrostates`` is too low to resolve
    fine-grained subtypes.
    """
    import numpy as np

    col = adata.obs[config.cell_type_key].astype(str)
    all_types = sorted(col.unique().tolist())
    n_cells = config.n_terminal_cells
    patterns = config.immature_cell_type_patterns

    def _is_immature(ct: str) -> bool:
        return any(pat in ct for pat in patterns)

    mature_types = [ct for ct in all_types if not _is_immature(ct)]
    immature_types = [ct for ct in all_types if _is_immature(ct)]

    logger.info(
        f"  Setting terminal states from '{config.cell_type_key}': "
        f"{len(mature_types)} mature (terminal) + {len(immature_types)} immature (transient) — "
        f"skipping GPCCA Schur decomposition."
    )
    if immature_types:
        logger.info(f"  Immature/transient (not absorbing): {immature_types}")

    # Sample n_cells representative cells per MATURE type only.
    # Immature cells (RG, IPC, Immature*) are left fully transient so the
    # solver can assign them fate probabilities towards the mature endpoints.
    rng = np.random.default_rng(0)
    states_dict = {}
    for ct in mature_types:
        all_names = adata.obs_names[(col == ct).values].tolist()
        n_sample = min(n_cells, len(all_names))
        sampled = rng.choice(all_names, size=n_sample, replace=False).tolist()
        states_dict[ct] = sampled
        logger.info(f"    {ct}: {n_sample} / {len(all_names)} cells sampled as terminal")

    with Timer("set_terminal_states (cell-type assignment)", logger):
        g.set_terminal_states(states=states_dict)

    assigned = sorted(g.terminal_states.cat.categories.tolist())
    logger.info(f"  Terminal states set ({len(assigned)}): {assigned}")

    log_memory("After cell-type state assignment", logger)


def compute_fate_probabilities(
    g,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Compute per-cell fate probabilities towards terminal states."""
    with Timer("compute_fate_probabilities", logger):
        g.compute_fate_probabilities()

    if g.fate_probabilities is not None:
        lineages = list(g.fate_probabilities.names)
        logger.info(f"  Fate probabilities computed for lineages: {lineages}")

    log_memory("After compute_fate_probabilities", logger)


def compute_lineage_drivers(
    g,
    adata: ad.AnnData,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Compute gene-level lineage driver scores (slow; optional)."""
    with Timer("compute_lineage_drivers", logger):
        g.compute_lineage_drivers(
            cluster_key=config.driver_cluster_key,
        )
    logger.info("  Lineage drivers stored in g.lineage_drivers")


def subset_to_lineage(
    adata: ad.AnnData,
    g,
    config: CellRankConfig,
    logger: logging.Logger,
) -> Optional[ad.AnnData]:
    """Return a subset of ``adata`` enriched for the target lineage(s).

    Cells with fate probability >= ``config.fate_prob_threshold`` towards
    *any* of ``config.lineage_targets`` are retained.  Fate probability
    columns are stored in ``adata.obs`` (added by CellRank automatically
    after ``compute_fate_probabilities``).

    Returns
    -------
    adata_subset : AnnData or None
        None if ``config.lineage_targets`` is empty.
    """
    if not config.lineage_targets:
        logger.info(
            "No lineage_targets specified; skipping lineage subset."
        )
        return None

    if g.fate_probabilities is None:
        raise RuntimeError(
            "Fate probabilities not computed. Run compute_fate_probabilities first."
        )

    lineages = list(g.fate_probabilities.names)
    target_mask = None

    for target in config.lineage_targets:
        # Find matching lineage name (case-insensitive, partial match)
        matched = [l for l in lineages if target.lower() in l.lower()]
        if not matched:
            logger.warning(
                f"  Lineage target '{target}' not found in computed lineages "
                f"{lineages}; skipping."
            )
            continue
        for m in matched:
            obs_col = f"lineage_{m}_probs"
            if obs_col not in adata.obs.columns:
                # CellRank stores fate probs as obsm["lineage_..."]
                # Extract to obs for subsetting
                col_idx = list(lineages).index(m)
                adata.obs[obs_col] = g.fate_probabilities[:, col_idx].X.ravel()
            mask = adata.obs[obs_col] >= config.fate_prob_threshold
            target_mask = mask if target_mask is None else (target_mask | mask)
            n_sel = mask.sum()
            logger.info(
                f"  Lineage '{m}': {n_sel} cells "
                f"(fate_prob >= {config.fate_prob_threshold})"
            )

    if target_mask is None:
        logger.warning("  No valid lineage targets matched; returning None.")
        return None

    adata_sub = adata[target_mask].copy()
    logger.info(
        f"  Subset: {adata_sub.n_obs} / {adata.n_obs} cells retained "
        f"({100 * adata_sub.n_obs / adata.n_obs:.1f}%)"
    )
    return adata_sub
