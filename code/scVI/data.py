"""Data loading, validation, and preprocessing for scVI."""

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

from .config import PipelineConfig
from .utils import Timer, log_memory


def load_adata(config: PipelineConfig, logger: logging.Logger) -> ad.AnnData:
    """Load h5ad file and validate expected structure."""
    input_path = Path(config.input_h5ad)
    if not input_path.exists():
        fallback = config._resolved_output_dir / config.output_h5ad
        if fallback.exists():
            logger.info(
                f"input_h5ad not found at {input_path}; "
                f"falling back to {fallback}"
            )
            input_path = fallback
        else:
            raise FileNotFoundError(
                f"input_h5ad not found: {input_path}\n"
                f"Fallback also missing: {fallback}"
            )
    with Timer(f"Loading {input_path}", logger):
        adata = sc.read_h5ad(input_path)

    logger.info(f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes")
    logger.info(f"Layers: {list(adata.layers.keys())}")
    logger.info(f"obs columns: {list(adata.obs.columns)}")

    # Validate batch key
    if config.batch_key not in adata.obs.columns:
        raise ValueError(
            f"Batch key '{config.batch_key}' not in .obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    batches = adata.obs[config.batch_key].value_counts()
    logger.info(f"Batch key '{config.batch_key}' — {len(batches)} batches:")
    for name, count in batches.items():
        logger.info(f"  {name}: {count} cells")

    # Validate cell type key if scANVI will be used
    if config.run_scanvi and config.cell_type_key not in adata.obs.columns:
        raise ValueError(
            f"Cell type key '{config.cell_type_key}' not in .obs (needed for scANVI). "
            f"Available: {list(adata.obs.columns)}"
        )

    log_memory("After load", logger)
    return adata


def prepare_for_scvi(
    adata: ad.AnnData,
    config: PipelineConfig,
    logger: logging.Logger,
) -> ad.AnnData:
    """
    Ensure counts layer exists, select HVGs, return subsetted copy.

    Does NOT mutate the original adata (except adding counts layer if missing).
    """
    # Ensure counts layer
    if config.counts_layer not in adata.layers:
        logger.info(f"No '{config.counts_layer}' layer. Copying .X as counts.")
        adata.layers[config.counts_layer] = adata.X.copy()
    else:
        logger.info(f"Using existing '{config.counts_layer}' layer.")

    # Check if counts look like integers (scVI expects raw counts)
    sample = adata.layers[config.counts_layer]
    if hasattr(sample, "toarray"):
        sample = sample[:100].toarray()
    else:
        sample = sample[:100]
    if not np.allclose(sample, sample.astype(int)):
        logger.warning(
            "Counts layer does not appear to contain integers. "
            "scVI expects raw (unnormalized) counts."
        )

    # HVG selection
    # Force re-selection if overwrite_scvi is true, or if "highly_variable" not yet marked
    force_reselect = config.overwrite_scvi
    if "highly_variable" not in adata.var.columns or force_reselect:
        with Timer(
            f"Selecting {config.n_top_genes} HVGs ({config.hvg_flavor})", logger
        ):
            if config.hvg_flavor == "pearson_residuals":
                # sc.experimental.pp.highly_variable_genes handles pearson_residuals
                kwargs = {"n_top_genes": config.n_top_genes, "layer": config.counts_layer}
                if config.hvg_batch_key:
                    logger.warning(
                        "pearson_residuals does not support batch_key in this scanpy version "
                        f"— ignoring hvg_batch_key='{config.hvg_batch_key}'"
                    )
                sc.experimental.pp.highly_variable_genes(adata, **kwargs)
            else:
                kwargs = {"n_top_genes": config.n_top_genes, "flavor": config.hvg_flavor}
                if config.hvg_flavor == "seurat_v3":
                    kwargs["layer"] = config.counts_layer
                elif config.hvg_flavor == "seurat":
                    logger.info("Log-normalizing .X for seurat HVG selection")
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                if config.hvg_batch_key:
                    kwargs["batch_key"] = config.hvg_batch_key
                sc.pp.highly_variable_genes(adata, **kwargs)
    elif "highly_variable" in adata.var.columns:
        n_hvg_existing = int(adata.var["highly_variable"].sum())
        if n_hvg_existing != config.n_top_genes:
            logger.warning(
                f"Existing HVG selection has {n_hvg_existing} genes, but "
                f"config.n_top_genes={config.n_top_genes}. To re-select, use --overwrite_scvi true"
            )
    n_hvg = int(adata.var["highly_variable"].sum())
    logger.info(f"HVGs: {n_hvg} genes selected")

    # Subset to HVGs — copy so original adata is unchanged
    adata_scvi = adata[:, adata.var["highly_variable"]].copy()
    logger.info(f"scVI AnnData: {adata_scvi.shape}")
    log_memory("After prep", logger)

    return adata_scvi


def save_checkpoint(adata: ad.AnnData, path: str, logger: logging.Logger):
    """Save adata to h5ad."""
    with Timer(f"Saving to {path}", logger):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(path)
    logger.info(f"Saved: {adata.shape}")
