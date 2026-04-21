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


def _assign_states_by_pattern(g, pattern: str, logger: logging.Logger) -> None:
    """Classify macrostates into initial/terminal using a regex pattern.

    Macrostates whose names match *pattern* (case-insensitive) become initial
    states (progenitors); all others become terminal states (mature cell types).
    This bypasses GPCCA's stability-based auto-prediction, which fails when the
    transition matrix lacks strong absorbing states.
    """
    import re

    all_states = list(g.macrostates.cat.categories)
    rx = re.compile(pattern, re.IGNORECASE)
    initial = [s for s in all_states if rx.search(s)]
    terminal = [s for s in all_states if not rx.search(s)]

    logger.info(
        f"  Pattern '{pattern}' classified macrostates:"
    )
    logger.info(f"    Initial  (progenitor): {initial}")
    logger.info(f"    Terminal (mature):     {terminal}")

    if not terminal:
        logger.warning(
            "  Pattern matched ALL macrostates as initial — no terminal states. "
            "Falling back to auto-prediction."
        )
        _predict_terminal_states_robust(g, logger)
        return

    if not initial:
        logger.warning(
            "  Pattern matched NO macrostates as initial — "
            "all states will be terminal. Falling back to auto-prediction."
        )
        _predict_terminal_states_robust(g, logger)
        return

    g.set_terminal_states(states=terminal)
    g.set_initial_states(states=initial)


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
    # Priority 1: explicit lists override everything
    if config.terminal_states or config.initial_states:
        if config.terminal_states:
            logger.info(
                f"Setting terminal states explicitly: {config.terminal_states}"
            )
            g.set_terminal_states(states=config.terminal_states)
        if config.initial_states:
            logger.info(
                f"Setting initial states explicitly: {config.initial_states}"
            )
            g.set_initial_states(states=config.initial_states)

    # Priority 2: pattern-based classification (recommended default)
    elif config.immature_state_pattern:
        logger.info(
            f"Assigning states by immature_state_pattern: "
            f"'{config.immature_state_pattern}'"
        )
        with Timer("assign_states_by_pattern", logger):
            _assign_states_by_pattern(g, config.immature_state_pattern, logger)

    # Priority 3: fully automatic (fallback — unreliable for this dataset)
    else:
        logger.info("Auto-predicting terminal states...")
        with Timer("predict_terminal_states", logger):
            _predict_terminal_states_robust(g, logger)
        if g.terminal_states is not None:
            logger.info(
                f"  Terminal states: "
                f"{sorted(g.terminal_states.cat.categories.tolist())}"
            )
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


def compute_fate_probabilities(
    g,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Compute per-cell fate probabilities towards terminal states."""
    import numpy as np

    with Timer("compute_fate_probabilities", logger):
        g.compute_fate_probabilities()

    if g.fate_probabilities is None:
        log_memory("After compute_fate_probabilities", logger)
        return

    lineages = list(g.fate_probabilities.names)
    logger.info(f"  Fate probabilities computed for lineages: {lineages}")

    # Write L2-3 pseudotime: sum of fate probs for all L2-3-matching terminal
    # states, normalised to [0, 1].  Higher value = more committed to L2-3 fate.
    if config.pseudotime_key:
        l23_indices = [
            i for i, name in enumerate(lineages)
            if config.l23_lineage_pattern.lower() in name.lower()
        ]
        if l23_indices:
            probs = g.fate_probabilities.X
            raw = probs[:, l23_indices].sum(axis=1)
            mn, mx = raw.min(), raw.max()
            g.adata.obs[config.pseudotime_key] = (raw - mn) / (mx - mn + 1e-9)
            g.adata.obs["fate_prob_l23"] = raw
            logger.info(
                f"  Pseudotime '{config.pseudotime_key}' written from L2-3 "
                f"terminal states: {[lineages[i] for i in l23_indices]}"
            )
        else:
            logger.warning(
                f"  No terminal states matched l23_lineage_pattern "
                f"'{config.l23_lineage_pattern}' in {lineages}; "
                f"'{config.pseudotime_key}' not written."
            )

    log_memory("After compute_fate_probabilities", logger)


def compute_absorption_pseudotime(
    g,
    config: CellRankConfig,
    logger: logging.Logger,
) -> None:
    """Compute absorption-time pseudotime via GPCCA mean first-passage times.

    This is the Monocle3-equivalent trajectory ordering: each cell is assigned
    the expected number of Markov chain steps until it reaches any terminal
    state.  Values are high near progenitors (many steps to maturity) and low
    near mature neurons (already at or near a terminal state).  We invert and
    normalise to [0, 1] so that 0 = progenitor, 1 = fully mature.
    """
    import numpy as np

    if not config.absorption_pseudotime_key:
        return

    if g.terminal_states is None:
        logger.warning(
            "  No terminal states set; skipping absorption-time pseudotime."
        )
        return

    try:
        with Timer("compute_absorption_times", logger):
            g.compute_absorption_times()

        at = g.absorption_times
        if at is None:
            logger.warning("  absorption_times is None after computation; skipping.")
            return

        # absorption_times is a DataFrame (cells × terminal states); take the
        # row-wise mean across all terminal states.
        import pandas as pd
        if isinstance(at, pd.DataFrame):
            mean_at = at.mean(axis=1).values
        else:
            mean_at = np.asarray(at).mean(axis=1)

        # Invert so high = mature (far from progenitor → short absorption time
        # in raw terms, but conceptually "far along the trajectory").
        # Raw absorption time is *high* for progenitors, *low* for mature cells.
        # We invert: pseudotime = (max - raw) / (max - min) so 0 = progenitor,
        # 1 = mature.
        mn, mx = float(np.nanmin(mean_at)), float(np.nanmax(mean_at))
        if mx - mn < 1e-9:
            logger.warning("  Absorption times have zero range; pseudotime not written.")
            return

        pseudotime = (mx - mean_at) / (mx - mn)
        g.adata.obs[config.absorption_pseudotime_key] = pseudotime
        logger.info(
            f"  Absorption-time pseudotime written to "
            f"'{config.absorption_pseudotime_key}' "
            f"(range [{pseudotime.min():.3f}, {pseudotime.max():.3f}])."
        )

    except (AttributeError, NotImplementedError) as exc:
        logger.warning(
            f"  compute_absorption_times not available in this CellRank version "
            f"({exc}); skipping absorption-time pseudotime."
        )
    except Exception as exc:
        logger.warning(f"  compute_absorption_times failed unexpectedly: {exc}")


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
