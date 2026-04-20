"""Pipeline configuration as a single dataclass."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class PipelineConfig:
    """Single source of truth for all scVI pipeline configuration."""

    # -- Paths --
    input_h5ad: str = ""
    output_dir: str = "output/"
    scvi_model_dir: str = "scvi_model"
    scanvi_model_dir: str = "scanvi_model"
    output_h5ad: str = "integrated.h5ad"

    # -- Data keys --
    batch_key: str = "source"
    cell_type_key: str = "cell_class"
    counts_layer: str = "counts"
    continuous_covariate_keys: Optional[List[str]] = None
    categorical_covariate_keys: Optional[List[str]] = None

    # -- Preprocessing --
    n_top_genes: int = 10000
    hvg_flavor: str = "seurat_v3"
    hvg_batch_key: Optional[str] = None

    # -- Model architecture --
    n_latent: int = 30
    n_hidden: int = 128
    n_layers: int = 2
    gene_likelihood: str = "zinb"  # "nb" or "zinb"; tunable via scvi_tuning/best_hyperparameters.yaml

    # -- Training --
    max_epochs_scvi: int = 400
    max_epochs_scanvi: int = 20
    train_size: float = 0.9
    early_stopping: bool = True
    early_stopping_patience: int = 30
    batch_size: int = 128
    num_workers: int = 4
    random_seed: int = 42

    # -- Inference --
    run_scvi_inference: bool = True
    run_scanvi: bool = False
    run_scanvi_inference: bool = False
    predict_cell_types: bool = False  # run model.predict() after scANVI for label transfer
    n_mc_samples: int = 10
    transform_batch: Optional[str] = None
    chunk_size: Optional[int] = None
    target_vram_fraction: float = 0.25
    max_chunk_size: int = 50000
    output_layer_scvi: str = "scvi_normalized"
    output_layer_scanvi: str = "scanvi_normalized"
    save_npy_backup: bool = False

    # -- UMAP --
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.3
    umap_color_vars: List[str] = field(
        default_factory=lambda: ["source", "chemistry", "cell_class", "cell_type_aligned", "age_years"]
    )
    umap_point_size: float = 1.0
    umap_log2_vars: List[str] = field(
        default_factory=lambda: ["age_years"]
    )
    umap_log2_ticks: List[float] = field(
        default_factory=lambda: [0.0, 1.0, 3.0, 9.0, 25.0, 40.0]
    )

    # -- Pipeline control --
    steps: List[str] = field(
        default_factory=lambda: ["prep", "train_scvi", "infer", "umap", "plot", "save"]
    )
    overwrite_scvi: bool = False
    overwrite_scanvi: bool = False

    # -- Derived paths --
    @property
    def _resolved_output_dir(self) -> Path:
        """
        If output_dir is empty or not set, derive it from input_h5ad:
            <input_dir>/<input_stem>/scvi_output/
        This keeps large output files next to the data (not in the code dir).
        """
        if self.output_dir:
            return Path(self.output_dir)
        p = Path(self.input_h5ad)
        return p.parent / p.stem / "scvi_output"

    @property
    def scvi_model_path(self) -> Path:
        return self._resolved_output_dir / self.scvi_model_dir

    @property
    def scanvi_model_path(self) -> Path:
        return self._resolved_output_dir / self.scanvi_model_dir

    @property
    def output_h5ad_path(self) -> Path:
        return self._resolved_output_dir / self.output_h5ad

    # -- Serialization --
    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
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
    def from_cli(cls, args=None) -> "PipelineConfig":
        import argparse

        parser = argparse.ArgumentParser(description="scVI batch correction pipeline")
        parser.add_argument("--config", type=str, help="Path to YAML config file")

        def _str2bool(v):
            if v.lower() in ("true", "1", "yes"):
                return True
            elif v.lower() in ("false", "0", "no"):
                return False
            raise argparse.ArgumentTypeError(f"Boolean expected, got: {v!r}")

        for name, fld in cls.__dataclass_fields__.items():
            if fld.type in (bool, "bool"):
                # Support both --flag (store_true) and --flag true/false
                parser.add_argument(f"--{name}", type=_str2bool,
                                    nargs="?", const=True, default=None)
            elif fld.type == List[float] or (
                "List" in str(fld.type) and "float" in str(fld.type)
            ):
                parser.add_argument(f"--{name}", nargs="+", type=float, default=None)
            elif fld.type == List[str] or "List" in str(fld.type):
                parser.add_argument(f"--{name}", nargs="+", default=None)
            elif fld.type == Optional[str] or "Optional[str]" in str(fld.type):
                parser.add_argument(f"--{name}", type=str, default=None)
            elif fld.type == Optional[int] or "Optional[int]" in str(fld.type):
                parser.add_argument(f"--{name}", type=int, default=None)
            elif fld.type in (int, "int"):
                parser.add_argument(f"--{name}", type=int, default=None)
            elif fld.type in (float, "float"):
                parser.add_argument(f"--{name}", type=float, default=None)
            elif fld.type in (str, "str"):
                parser.add_argument(f"--{name}", type=str, default=None)

        parsed = parser.parse_args(args)

        # Start from YAML if provided
        if parsed.config:
            cfg = cls.from_yaml(parsed.config)
        else:
            cfg = cls()

        # Override with explicitly set CLI args
        for name in cls.__dataclass_fields__:
            val = getattr(parsed, name, None)
            if val is not None:
                setattr(cfg, name, val)

        return cfg
