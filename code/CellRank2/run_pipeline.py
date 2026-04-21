"""
Main orchestration script for the CellRank 2 lineage tracing pipeline.

Usage (from project root):
    # With YAML config:
    PYTHONPATH=code python -m CellRank2.run_pipeline --config code/CellRank2/config.yaml

    # With CLI overrides:
    PYTHONPATH=code python -m CellRank2.run_pipeline \\
        --config code/CellRank2/config.yaml \\
        --n_macrostates 10

    # Run specific steps only (e.g. re-plot without re-computing):
    PYTHONPATH=code python -m CellRank2.run_pipeline \\
        --config code/CellRank2/config.yaml \\
        --steps save

Pipeline steps:
    neighbors   — compute kNN in scANVI latent space (skipped if already stored)
    ot          — run moscot optimal transport between age bins
    kernels     — build ConnectivityKernel + RealTimeKernel, combine them
    gpcca       — GPCCA macrostates → terminal/initial state prediction
    fate_probs  — per-cell fate probabilities towards terminal states
    save        — checkpoint adata to h5ad
"""

import logging
from pathlib import Path

import anndata as ad
import numpy as np

from .config import CellRankConfig
from .estimator import (
    build_gpcca,
    compute_absorption_pseudotime,
    compute_fate_probabilities,
    compute_lineage_drivers,
    compute_macrostates,
    set_terminal_and_initial_states,
    subset_to_lineage,
)
from .kernels import bin_ages, build_kernels, ensure_neighbors, run_moscot_ot
from .plots import (
    plot_coarse_transition_matrix,
    plot_excitatory_l23_plots,
    plot_fate_probabilities,
    plot_macrostates,
    plot_obs_vars,
)
from .utils import Timer, get_device_info, log_memory, setup_logger


def run(config: CellRankConfig) -> ad.AnnData:
    """Execute the CellRank 2 pipeline according to config.steps."""

    out = config._resolved_output_dir
    out.mkdir(parents=True, exist_ok=True)
    log_path = str(out / "pipeline.log")
    logger = setup_logger(log_file=log_path)

    logger.info("=" * 60)
    logger.info("CellRank 2 Lineage Tracing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Output dir: {out}")

    # Save config for reproducibility
    config.to_yaml(str(out / "cellrank_config.yaml"))
    logger.info(f"Config saved to {out / 'cellrank_config.yaml'}")

    device_info = get_device_info(logger)
    np.random.seed(config.random_seed)

    steps = config.steps
    logger.info(f"Steps: {steps}")

    # Load data
    if not config.input_h5ad:
        raise ValueError("input_h5ad must be set in config.")
    logger.info(f"Loading: {config.input_h5ad}")
    with Timer("Loading h5ad", logger):
        adata = ad.read_h5ad(config.input_h5ad)
    if adata.obs_names.duplicated().any():
        if config.batch_key not in adata.obs.columns:
            raise KeyError(
                f"obs_names are not unique but batch_key '{config.batch_key}' "
                "is not in adata.obs. Cannot construct unique cell identifiers."
            )
        logger.warning(
            f"obs_names are not unique — prepending '{config.batch_key}' "
            "column to make them unique."
        )
        adata.obs_names = (
            adata.obs[config.batch_key].astype(str) + "_" + adata.obs_names
        )
    logger.info(f"  {adata.n_obs} cells × {adata.n_vars} genes")
    log_memory("After loading", logger)

    # ── CELL TYPE PRE-FILTER ───────────────────────────────────────────────────
    if config.cell_type_filter_pattern:
        if config.cell_type_key not in adata.obs.columns:
            raise KeyError(
                f"cell_type_key '{config.cell_type_key}' not found in adata.obs. "
                "Cannot apply cell_type_filter_pattern."
            )
        mask = adata.obs[config.cell_type_key].astype(str).str.contains(
            config.cell_type_filter_pattern, case=False, na=False, regex=True
        )
        n_before = adata.n_obs
        adata = adata[mask].copy()
        logger.info(
            f"Cell-type filter '{config.cell_type_filter_pattern}': "
            f"{n_before} → {adata.n_obs} cells retained "
            f"({100 * adata.n_obs / n_before:.1f}%)"
        )
        if adata.n_obs == 0:
            raise ValueError(
                f"Cell-type filter '{config.cell_type_filter_pattern}' removed all cells."
            )
        log_memory("After cell-type filter", logger)

    # ── FILTER TO CELLS WITH VALID AGE BINS ───────────────────────────────────
    # Bin ages now (before neighbors) so the kNN graph is computed only on
    # temporally-valid cells.  Cells without a valid bin (NaN age or age outside
    # the configured edges) cannot participate in the RealTimeKernel and their
    # inclusion causes _restich_couplings to fail.
    bin_ages(adata, config, logger)
    valid_mask = adata.obs[config.age_bin_key].notna()
    cells_removed = False
    if not valid_mask.all():
        n_excl = int((~valid_mask).sum())
        logger.warning(
            f"  Excluding {n_excl}/{adata.n_obs} cells with no valid age bin "
            "(NaN age or age outside bin edges). These cells are retained in "
            "the input h5ad but excluded from CellRank analysis."
        )
        adata = adata[valid_mask].copy()
        cells_removed = True
        logger.info(f"  Retaining {adata.n_obs} cells for CellRank analysis.")

    # ── NEIGHBORS ──────────────────────────────────────────────────────────────
    # Always (re)compute the kNN graph after filtering.  If cells were removed
    # the precomputed graph is a subset of the original and some rows may be
    # all-zero (all neighbours fell in the excluded set), making the
    # ConnectivityKernel transition matrix non-row-stochastic.
    nk = config.neighbors_key
    if cells_removed and nk in adata.uns:
        logger.info(
            f"  Cells were removed after binning — dropping stale neighbour "
            f"graph '{nk}' and recomputing on the filtered subset."
        )
        adata.uns.pop(nk, None)
        adata.obsp.pop(f"{nk}_connectivities", None)
        adata.obsp.pop(f"{nk}_distances", None)

    if "neighbors" in steps or cells_removed:
        logger.info("─" * 40)
        logger.info("STEP: neighbors")
        ensure_neighbors(adata, config, logger)
        log_memory("After neighbors", logger)

    # ── OT ─────────────────────────────────────────────────────────────────────
    moscot_problem = None
    if "ot" in steps:
        logger.info("─" * 40)
        logger.info("STEP: ot (moscot optimal transport)")
        # Age binning and filtering already done above, before neighbors.
        moscot_problem = run_moscot_ot(adata, config, logger)
        log_memory("After OT", logger)

    # ── KERNELS ────────────────────────────────────────────────────────────────
    combined_kernel = None
    if "kernels" in steps:
        logger.info("─" * 40)
        logger.info("STEP: kernels")
        combined_kernel, ck, rtk = build_kernels(
            adata, config, logger, moscot_problem=moscot_problem
        )
        log_memory("After kernels", logger)

    # ── GPCCA ──────────────────────────────────────────────────────────────────
    g = None
    if "gpcca" in steps:
        if combined_kernel is None:
            raise RuntimeError(
                "Cannot run GPCCA without a combined kernel. "
                "Ensure 'kernels' step runs before 'gpcca'."
            )
        logger.info("─" * 40)
        logger.info("STEP: gpcca")
        g = build_gpcca(combined_kernel, config, logger)
        g = compute_macrostates(g, config, logger)
        set_terminal_and_initial_states(g, config, logger)

        if config.save_plots:
            plot_macrostates(adata, g, config, logger)
            plot_coarse_transition_matrix(g, config, logger)
            plot_obs_vars(adata, config, logger)

        log_memory("After GPCCA", logger)

    # ── FATE PROBABILITIES ─────────────────────────────────────────────────────
    if "fate_probs" in steps:
        if g is None:
            raise RuntimeError(
                "Cannot compute fate probabilities without GPCCA. "
                "Ensure 'gpcca' step runs before 'fate_probs'."
            )
        logger.info("─" * 40)
        logger.info("STEP: fate_probs")
        compute_fate_probabilities(g, config, logger)
        compute_absorption_pseudotime(g, config, logger)

        if config.compute_drivers:
            logger.info("Computing lineage drivers (this may take a while)...")
            compute_lineage_drivers(g, adata, config, logger)

        if config.save_plots:
            plot_fate_probabilities(adata, g, config, logger)
            plot_excitatory_l23_plots(adata, config, logger)

        # Lineage subsetting
        if config.lineage_targets:
            adata_sub = subset_to_lineage(adata, g, config, logger)
            if adata_sub is not None:
                sub_path = out / "lineage_subset.h5ad"
                adata_sub.write_h5ad(str(sub_path))
                logger.info(f"  Lineage subset saved: {sub_path}")

        log_memory("After fate_probs", logger)

    # ── SAVE ───────────────────────────────────────────────────────────────────
    if "save" in steps:
        logger.info("─" * 40)
        logger.info("STEP: save")
        out_path = config.output_h5ad_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(str(out_path))
        logger.info(f"  Saved: {out_path}")

    logger.info("=" * 60)
    logger.info("CellRank 2 pipeline complete.")
    logger.info("=" * 60)

    return adata


def main():
    config = CellRankConfig.from_cli()
    run(config)


if __name__ == "__main__":
    main()
