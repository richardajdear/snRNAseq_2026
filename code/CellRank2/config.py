"""Pipeline configuration for CellRank 2 lineage tracing."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class CellRankConfig:
    """Single source of truth for all CellRank 2 pipeline configuration."""

    # -- Paths --
    input_h5ad: str = ""
    output_dir: str = "cellrank_output/"

    # -- Data keys --
    latent_key: str = "X_scANVI"       # obsm key for the corrected latent space
    time_key: str = "age_years"         # obs key for donor chronological age
    cell_type_key: str = "cell_type_aligned"  # obs key for aligned cell type labels
    batch_key: str = "source"

    # -- Neighbour graph (used by ConnectivityKernel + for latent-space kNN) --
    n_neighbors: int = 30
    neighbors_key: str = "neighbors_scanvi"  # key written by scanpy.pp.neighbors

    # -- Optimal transport (moscot RealTimeKernel) --
    # Time points are binned from the continuous age_years column.
    # If age_bins is empty, unique values of time_key are used directly.
    age_bin_edges: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 2.0, 10.0, 20.0, 40.0, 100.0]
    )
    age_bin_key: str = "age_bin"        # new obs column written by the pipeline
    ot_epsilon: float = 0.05            # regularisation strength for OT
    ot_max_iterations: int = 1000

    # -- Kernel combination --
    # Combined kernel = rtk_weight * RealTimeKernel + (1-rtk_weight) * ConnectivityKernel
    rtk_weight: float = 0.8

    # -- GPCCA estimator --
    n_macrostates: int = 8              # passed to compute_macrostates; can be a list for minChi
    n_macrostates_min: int = 2          # lower bound when using minChi (list range)
    n_macrostates_max: int = 12         # upper bound when using minChi
    use_minchi: bool = False            # if True, use [min, max] range instead of fixed n
    cluster_key: str = "cell_type_aligned"   # for naming macrostates

    # -- Terminal / initial state selection --
    # Provide explicit terminal state names (matching cluster labels) when known.
    # Leave empty to rely on automatic selection after compute_macrostates.
    terminal_states: List[str] = field(default_factory=list)
    initial_states: List[str] = field(default_factory=list)

    # -- Fate probabilities --
    compute_drivers: bool = False       # gene-level lineage drivers (slow)
    driver_cluster_key: str = "cell_type_aligned"

    # -- Lineage subsetting --
    # After computing fate probabilities, cells with fate_prob >= threshold
    # towards any of the target lineages are retained.
    lineage_targets: List[str] = field(default_factory=list)
    fate_prob_threshold: float = 0.1

    # -- Output --
    save_plots: bool = True
    plot_color_vars: List[str] = field(
        default_factory=lambda: ["cell_type_aligned", "age_years", "source"]
    )
    umap_key: str = "X_umap_scanvi"    # obsm key for UMAP (used in plots)
    point_size: float = 1.0
    # Patterns used by plot_excitatory_l23_fate_umap.
    # excitatory_cell_type_pattern: case-insensitive regex matched against
    #   cell_type_key to identify excitatory neurons (e.g. "EN-" or "excit|EN-").
    # l23_lineage_pattern: case-insensitive substring matched against
    #   terminal-state names to pick the L2-3 lineage.
    excitatory_cell_type_pattern: str = "excit|EN-"
    l23_lineage_pattern: str = "l2"

    # -- Pipeline control --
    steps: List[str] = field(
        default_factory=lambda: [
            "neighbors",     # compute kNN in latent space (if missing)
            "ot",            # moscot OT between age bins
            "kernels",       # build + combine kernels
            "gpcca",         # GPCCA macrostates + terminal/initial states
            "fate_probs",    # fate probabilities
            "save",          # checkpoint
        ]
    )
    random_seed: int = 42
    overwrite: bool = False

    # -- Derived paths --
    @property
    def _resolved_output_dir(self) -> Path:
        if self.output_dir:
            return Path(self.output_dir)
        p = Path(self.input_h5ad)
        return p.parent / p.stem / "cellrank_output"

    @property
    def output_h5ad_path(self) -> Path:
        return self._resolved_output_dir / "cellrank_output.h5ad"

    @property
    def kernel_dir(self) -> Path:
        return self._resolved_output_dir / "kernels"

    @property
    def plots_dir(self) -> Path:
        return self._resolved_output_dir / "plots"

    # -- Serialization --
    @classmethod
    def from_yaml(cls, path: str) -> "CellRankConfig":
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f) or {}
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_yaml(self, path: str):
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_cli(cls, args=None) -> "CellRankConfig":
        import argparse

        parser = argparse.ArgumentParser(
            description="CellRank 2 lineage tracing pipeline"
        )
        parser.add_argument("--config", type=str, help="Path to YAML config file")

        def _str2bool(v):
            if v.lower() in ("true", "1", "yes"):
                return True
            elif v.lower() in ("false", "0", "no"):
                return False
            raise argparse.ArgumentTypeError(f"Boolean expected, got: {v!r}")

        for name, fld in cls.__dataclass_fields__.items():
            t = fld.type
            if t in (bool, "bool"):
                parser.add_argument(
                    f"--{name}", type=_str2bool, nargs="?", const=True, default=None
                )
            elif "List" in str(t) and "float" in str(t):
                parser.add_argument(f"--{name}", nargs="+", type=float, default=None)
            elif "List" in str(t):
                parser.add_argument(f"--{name}", nargs="+", default=None)
            elif "Optional[str]" in str(t):
                parser.add_argument(f"--{name}", type=str, default=None)
            elif "Optional[int]" in str(t):
                parser.add_argument(f"--{name}", type=int, default=None)
            elif t in (int, "int"):
                parser.add_argument(f"--{name}", type=int, default=None)
            elif t in (float, "float"):
                parser.add_argument(f"--{name}", type=float, default=None)
            elif t in (str, "str"):
                parser.add_argument(f"--{name}", type=str, default=None)

        parsed = parser.parse_args(args)

        cfg = cls.from_yaml(parsed.config) if parsed.config else cls()

        for name in cls.__dataclass_fields__:
            val = getattr(parsed, name, None)
            if val is not None:
                setattr(cfg, name, val)

        return cfg
