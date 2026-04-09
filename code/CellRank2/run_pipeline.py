"""
CellRank 2 pseudotime pipeline for snRNA-seq data.

Computes cell-state transitions along developmental time (age_years) using:
  - RealTimeKernel  : moscot optimal transport between age timepoints
  - ConnectivityKernel : kNN graph on X_scANVI latent space
  - GPCCA estimator : macrostates and fate probabilities

Usage (from project root):
    PYTHONPATH=code python -m CellRank2.run_pipeline --config code/CellRank2/default_config.yaml

    # Run specific steps only:
    PYTHONPATH=code python -m CellRank2.run_pipeline --config code/CellRank2/default_config.yaml \\
        --steps kernel estimator
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Logging / timing helpers
# ---------------------------------------------------------------------------

def _setup_logger(name: str = "cellrank_pipeline",
                  log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)
    for h in logger.handlers:
        logging.getLogger("py.warnings").addHandler(h)
    return logger


class _Timer:
    def __init__(self, label: str, logger: logging.Logger):
        self.label = label
        self.logger = logger

    def __enter__(self):
        self.start = time.time()
        self.logger.info(f"Starting: {self.label}")
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        if elapsed < 1.0:
            dur = f"{elapsed * 1000:.0f}ms"
        else:
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            dur = f"{int(h)}h {int(m)}m {s:.1f}s"
        self.logger.info(f"Completed: {self.label} [{dur}]")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CellRankConfig:
    """Configuration for the CellRank 2 pseudotime pipeline."""

    # -- Paths --
    input_h5ad: str = ""
    output_dir: str = ""

    # -- Data keys --
    time_key: str = "age_years"
    basis: str = "X_scANVI"          # obsm key for ConnectivityKernel neighbours
    umap_basis: str = "X_umap"       # obsm key used for plotting

    # -- Cell filtering (optional) --
    cell_class_filter: Optional[str] = None   # e.g. "Excitatory" — None = all cells
    min_cells_per_timepoint: int = 10         # drop timepoints with fewer cells

    # -- Kernel weights --
    rtk_weight: float = 0.8     # weight of RealTimeKernel in combined kernel
    ck_weight: float = 0.2      # weight of ConnectivityKernel

    # -- RealTimeKernel / moscot OT --
    epsilon: float = 0.5          # entropy regularisation
    threshold: Optional[float] = None   # sparsify transition matrix (None = auto)
    self_transitions: str = "connectivities"  # "uniform" | "connectivities"

    # -- ConnectivityKernel --
    n_neighbors: int = 30

    # -- GPCCA estimator --
    n_components: int = 20        # Schur decomposition components
    n_macrostates: int = 6        # initial number of macrostates to compute

    # -- Pipeline control --
    steps: List[str] = field(
        default_factory=lambda: ["kernel", "estimator", "fate", "plot"]
    )
    overwrite: bool = False
    random_seed: int = 42

    # -- Plot settings --
    plot_basis: str = "umap"      # scanpy/squidpy basis name for sc.pl.embedding
    plot_color_vars: List[str] = field(
        default_factory=lambda: ["cell_class", "cell_type_aligned", "age_years"]
    )
    dpi: int = 150

    @classmethod
    def from_yaml(cls, path: str) -> "CellRankConfig":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_yaml(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @property
    def output_path(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        p = Path(self.input_h5ad)
        return p.parent / p.stem / "cellrank_output"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _load_and_filter(config: CellRankConfig, logger: logging.Logger):
    """Load integrated h5ad and optionally subset by cell_class."""
    import anndata as ad
    import numpy as np
    import scanpy as sc

    logger.info(f"Loading: {config.input_h5ad}")
    adata = ad.read_h5ad(config.input_h5ad)
    logger.info(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # Cell-class filter
    if config.cell_class_filter:
        mask = adata.obs["cell_class"] == config.cell_class_filter
        logger.info(
            f"Filtering to cell_class={config.cell_class_filter!r}: "
            f"{mask.sum()} / {adata.n_obs} cells retained"
        )
        adata = adata[mask].copy()

    # Drop cells with missing time information
    time_col = config.time_key
    if time_col not in adata.obs.columns:
        raise KeyError(
            f"time_key={time_col!r} not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    missing = adata.obs[time_col].isna()
    if missing.any():
        logger.warning(
            f"Dropping {missing.sum()} cells with NaN in {time_col!r}"
        )
        adata = adata[~missing].copy()

    # Check required obsm keys
    if config.basis not in adata.obsm:
        raise KeyError(
            f"basis={config.basis!r} not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    # Check timepoint coverage
    counts = adata.obs[time_col].value_counts()
    too_few = counts[counts < config.min_cells_per_timepoint]
    if not too_few.empty:
        logger.warning(
            f"Dropping {too_few.index.tolist()} timepoints with < "
            f"{config.min_cells_per_timepoint} cells"
        )
        adata = adata[
            ~adata.obs[time_col].isin(too_few.index)
        ].copy()

    logger.info(
        f"After filtering: {adata.n_obs} cells, "
        f"{adata.obs[time_col].nunique()} unique timepoints"
    )

    # Ensure connectivity graph exists on the basis
    existing_params = adata.uns.get("neighbors", {}).get("params", {})
    neighbors_exist = (
        "neighbors" in adata.uns
        and existing_params.get("use_rep") == config.basis
    )
    if not neighbors_exist:
        logger.info(
            f"Computing {config.n_neighbors}-NN graph on {config.basis!r}"
        )
        sc.pp.neighbors(
            adata,
            use_rep=config.basis,
            n_neighbors=config.n_neighbors,
            key_added="neighbors",
        )
    else:
        logger.info("Using existing neighbours graph from adata.uns['neighbors']")

    return adata


def _build_kernel(adata, config: CellRankConfig, logger: logging.Logger):
    """Build and combine RealTimeKernel + ConnectivityKernel."""
    import cellrank as cr
    from cellrank.kernels import ConnectivityKernel

    logger.info(f"CellRank version: {cr.__version__}")

    # ConnectivityKernel on X_scANVI neighbours
    # Use the connectivity key stored by sc.pp.neighbors (default: 'connectivities')
    conn_key = adata.uns.get("neighbors", {}).get("connectivities_key", "connectivities")
    logger.info(f"Building ConnectivityKernel (conn_key={conn_key!r}) …")
    ck = ConnectivityKernel(adata, conn_key=conn_key)
    ck.compute_transition_matrix()
    logger.info("ConnectivityKernel: done")

    # RealTimeKernel (moscot OT)
    logger.info(
        f"Building RealTimeKernel on {config.time_key!r} "
        f"(epsilon={config.epsilon}, self_transitions={config.self_transitions!r}) …"
    )
    try:
        from cellrank.kernels import RealTimeKernel
        rtk = RealTimeKernel.from_adata(
            adata,
            time_key=config.time_key,
            n_neighbors=config.n_neighbors,
        )
        rtk.compute_transition_matrix(
            cost_fn="Sq_Euclidean",
            epsilon=config.epsilon,
            threshold=config.threshold,
            self_transitions=config.self_transitions,
        )
    except Exception as e:
        logger.warning(
            f"RealTimeKernel.from_adata failed ({e!r}); "
            "falling back to ConnectivityKernel only with weight=1.0"
        )
        config.rtk_weight = 0.0
        config.ck_weight = 1.0
        return ck

    logger.info("RealTimeKernel: done")

    # Combine
    w_rtk = config.rtk_weight
    w_ck = config.ck_weight
    # Normalise weights
    total = w_rtk + w_ck
    w_rtk /= total
    w_ck /= total
    logger.info(
        f"Combining kernels: {w_rtk:.2f} × RTK + {w_ck:.2f} × CK"
    )
    combined = w_rtk * rtk + w_ck * ck
    combined.compute_transition_matrix()
    return combined


def _fit_estimator(adata, kernel, config: CellRankConfig, logger: logging.Logger):
    """Fit GPCCA estimator, compute macrostates and terminal states."""
    from cellrank.estimators import GPCCA

    logger.info("Fitting GPCCA estimator …")
    g = GPCCA(kernel)

    with _Timer("Schur decomposition", logger):
        g.compute_schur(n_components=config.n_components)

    with _Timer(f"Macrostates (n={config.n_macrostates})", logger):
        cluster_key = "cell_class" if "cell_class" in adata.obs.columns else None
        g.compute_macrostates(n_states=config.n_macrostates, cluster_key=cluster_key)

    logger.info(f"Macrostates: {g.macrostates.cat.categories.tolist()}")

    # Auto-assign terminal states as macrostates whose name suggests maturity
    # (heuristic: largest coarse-grained stationary distribution component)
    with _Timer("Terminal states", logger):
        try:
            g.set_terminal_states_from_macrostates()
        except AttributeError:
            # Older API: predict_terminal_states
            try:
                g.predict_terminal_states()
            except Exception as e2:
                logger.warning(
                    f"Auto terminal-state assignment failed ({e2!r}). "
                    "Setting all macrostates as terminal states."
                )
                g.set_terminal_states(states=g.macrostates.cat.categories.tolist())

    logger.info(f"Terminal states: {g.terminal_states.cat.categories.tolist()}")
    return g


def _compute_fate(g, logger: logging.Logger):
    """Compute fate probabilities."""
    with _Timer("Fate probabilities", logger):
        g.compute_fate_probabilities()
    logger.info("Fate probabilities computed.")
    return g


def _save_plots(adata, g, config: CellRankConfig, out: Path, logger: logging.Logger):
    """Save diagnostic plots to output_dir/plots/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scanpy as sc

    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plots to {plot_dir}")

    basis = config.plot_basis

    # UMAP coloured by time / cell_class / cell_type
    for col in config.plot_color_vars:
        if col not in adata.obs.columns:
            continue
        try:
            sc.pl.embedding(
                adata, basis=basis, color=col,
                show=False, save=False
            )
            plt.savefig(plot_dir / f"umap_{col}.png", dpi=config.dpi, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot {col!r}: {e}")

    # Macrostates
    try:
        g.plot_macrostates(which="all", basis=basis, show=False)
        plt.savefig(plot_dir / "macrostates.png", dpi=config.dpi, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not plot macrostates: {e}")

    # Terminal states
    try:
        g.plot_terminal_states(basis=basis, show=False)
        plt.savefig(plot_dir / "terminal_states.png", dpi=config.dpi, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not plot terminal states: {e}")

    # Fate probabilities
    try:
        g.plot_fate_probabilities(basis=basis, show=False)
        plt.savefig(plot_dir / "fate_probabilities.png", dpi=config.dpi, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not plot fate probabilities: {e}")

    logger.info("Plots saved.")


def _save_results(adata, g, config: CellRankConfig, out: Path, logger: logging.Logger):
    """Save annotated AnnData and fate probability CSV."""
    import pandas as pd

    # Add fate probabilities and macrostates to adata
    if g.fate_probabilities is not None:
        import re
        for col in g.fate_probabilities.names:
            # Sanitize: replace any non-alphanumeric/underscore characters
            safe_col = re.sub(r"[^A-Za-z0-9_]", "_", col)
            key = f"fate_prob_{safe_col}"
            adata.obs[key] = g.fate_probabilities[:, col].X.squeeze()

    if g.macrostates is not None:
        adata.obs["macrostate"] = g.macrostates

    if g.terminal_states is not None:
        adata.obs["terminal_state"] = g.terminal_states

    out_h5ad = out / "cellrank_output.h5ad"
    logger.info(f"Saving annotated h5ad → {out_h5ad}")
    adata.write_h5ad(str(out_h5ad))

    # Fate probability CSV
    if g.fate_probabilities is not None:
        fate_df = g.fate_probabilities.to_df()
        csv_path = out / "fate_probabilities.csv"
        fate_df.to_csv(csv_path)
        logger.info(f"Saved fate probabilities CSV → {csv_path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(config: CellRankConfig):
    """Execute the CellRank2 pipeline according to config.steps."""
    out = config.output_path
    out.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(log_file=str(out / "cellrank_pipeline.log"))

    logger.info("=" * 60)
    logger.info("CellRank 2 Pseudotime Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input:      {config.input_h5ad}")
    logger.info(f"Output dir: {out}")
    logger.info(f"Steps:      {config.steps}")

    config.to_yaml(str(out / "cellrank_config.yaml"))

    steps = config.steps
    adata = None
    kernel = None
    g = None

    # ── Load & filter ─────────────────────────────────────────────────────
    if any(s in steps for s in ["kernel", "estimator", "fate", "plot"]):
        with _Timer("Data loading and filtering", logger):
            adata = _load_and_filter(config, logger)

    # ── Kernel ────────────────────────────────────────────────────────────
    if "kernel" in steps and adata is not None:
        with _Timer("Kernel computation", logger):
            kernel = _build_kernel(adata, config, logger)

    # ── Estimator ─────────────────────────────────────────────────────────
    if "estimator" in steps and kernel is not None:
        with _Timer("GPCCA estimator", logger):
            g = _fit_estimator(adata, kernel, config, logger)

    # ── Fate probabilities ────────────────────────────────────────────────
    if "fate" in steps and g is not None:
        with _Timer("Fate probabilities", logger):
            g = _compute_fate(g, logger)

    # ── Plots ─────────────────────────────────────────────────────────────
    if "plot" in steps and adata is not None and g is not None:
        with _Timer("Plotting", logger):
            _save_plots(adata, g, config, out, logger)

    # ── Save ──────────────────────────────────────────────────────────────
    if "save" in steps and adata is not None and g is not None:
        with _Timer("Saving results", logger):
            _save_results(adata, g, config, out, logger)
    elif g is not None and adata is not None:
        # Default: always save even if 'save' not explicitly listed
        with _Timer("Saving results", logger):
            _save_results(adata, g, config, out, logger)

    logger.info("=" * 60)
    logger.info("CellRank 2 pipeline complete.")
    logger.info("=" * 60)

    return adata, g


def main():
    parser = argparse.ArgumentParser(
        description="CellRank 2 pseudotime pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--steps", nargs="+",
        choices=["kernel", "estimator", "fate", "plot", "save"],
        default=None,
        help="Pipeline steps to run (default: from config)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = CellRankConfig.from_yaml(args.config)
    if args.steps:
        config.steps = args.steps
    if args.overwrite:
        config.overwrite = True

    run(config)


if __name__ == "__main__":
    main()
