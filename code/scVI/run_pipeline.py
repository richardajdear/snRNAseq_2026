"""
Main orchestration script for the scVI batch correction pipeline.

Usage (from project root):
    # With YAML config:
    PYTHONPATH=code python -m scVI.run_pipeline --config code/scVI/default_config.yaml

    # With CLI overrides:
    PYTHONPATH=code python -m scVI.run_pipeline --config code/scVI/default_config.yaml --max_epochs_scvi 100

    # Run specific steps only:
    PYTHONPATH=code python -m scVI.run_pipeline --config code/scVI/default_config.yaml --steps infer save
"""

from pathlib import Path

import anndata as ad
import numpy as np
import torch

from .config import PipelineConfig
from .data import load_adata, prepare_for_scvi, save_checkpoint
from .inference import get_normalized_expression
from .train import train_scanvi, train_scvi
from .utils import Timer, get_device_info, log_memory, setup_logger
from .visualize import compute_umaps, plot_batch_comparison


def run(config: PipelineConfig):
    """Execute the pipeline according to config.steps."""

    # Setup output dir and logging
    out = config._resolved_output_dir
    out.mkdir(parents=True, exist_ok=True)
    log_path = str(out / "pipeline.log")
    logger = setup_logger(log_file=log_path)

    logger.info("=" * 60)
    logger.info("scVI Batch Correction Pipeline")
    logger.info("=" * 60)
    logger.info(f"Output dir: {out}")

    # Save config for reproducibility
    config_path = str(out / "config.yaml")
    config.to_yaml(config_path)
    logger.info(f"Config saved to {config_path}")

    # Device
    device_info = get_device_info(logger)

    # Seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    steps = config.steps
    logger.info(f"Steps: {steps}")

    adata = None
    adata_scvi = None

    # --- PREP ---
    needs_data = any(s in steps for s in ["prep", "train_scvi", "train_scanvi", "infer"])
    if needs_data:
        with Timer("Data loading and preparation", logger):
            adata = load_adata(config, logger)
            adata_scvi = prepare_for_scvi(adata, config, logger)
        log_memory("After prep", logger)

    # Load saved checkpoint when running downstream steps (umap/plot/save) without training
    if adata is None and any(s in steps for s in ["umap", "plot", "save"]):
        checkpoint = config.output_h5ad_path
        if checkpoint.exists():
            logger.info(f"Loading checkpoint: {checkpoint}")
            adata = ad.read_h5ad(str(checkpoint))
        else:
            logger.warning(f"No checkpoint found at {checkpoint}; umap/plot/save steps will be skipped")

    # --- TRAIN SCVI ---
    scvi_model = None
    if "train_scvi" in steps:
        scvi_model = train_scvi(adata_scvi, config, device_info, logger)

        # Extract latent representation
        with Timer("scVI latent representation", logger):
            adata.obsm["X_scVI"] = scvi_model.get_latent_representation(
                adata=adata_scvi
            )
        logger.info(f"X_scVI latent: shape={adata.obsm['X_scVI'].shape}")

        save_checkpoint(adata, str(config.output_h5ad_path), logger)

    # --- TRAIN SCANVI (optional) ---
    scanvi_model = None
    if "train_scanvi" in steps and config.run_scanvi:
        # Pass scvi_model so scANVI inherits learned weights (from_scvi_model)
        scanvi_model = train_scanvi(
            adata_scvi, config, device_info, logger, scvi_model=scvi_model
        )

        with Timer("scANVI latent representation", logger):
            adata.obsm["X_scANVI"] = scanvi_model.get_latent_representation(
                adata=adata_scvi
            )
        logger.info(f"X_scANVI latent: shape={adata.obsm['X_scANVI'].shape}")

        save_checkpoint(adata, str(config.output_h5ad_path), logger)

    # --- INFERENCE ---
    if "infer" in steps:
        # Load scVI model if not in memory
        if scvi_model is None and config.run_scvi_inference:
            from scvi.model import SCVI

            logger.info("Loading scVI model for inference...")
            SCVI.setup_anndata(
                adata_scvi, layer=config.counts_layer, batch_key=config.batch_key
            )
            scvi_model = SCVI.load(str(config.scvi_model_path), adata=adata_scvi)

            # Recompute latent representation (skipped when not training)
            logger.info("Computing X_scVI latent representation from loaded model...")
            adata.obsm["X_scVI"] = scvi_model.get_latent_representation(adata=adata_scvi)
            logger.info(f"X_scVI latent: shape={adata.obsm['X_scVI'].shape}")

        # scVI inference
        if config.run_scvi_inference and scvi_model is not None:
            with Timer("scVI inference", logger):
                get_normalized_expression(
                    model=scvi_model,
                    adata_scvi=adata_scvi,
                    adata_full=adata,
                    config=config,
                    device_info=device_info,
                    logger=logger,
                    layer_name=config.output_layer_scvi,
                )
            log_memory("After scVI inference", logger)

        # scANVI inference (optional)
        if config.run_scanvi_inference and config.run_scanvi:
            if scanvi_model is None:
                from scvi.model import SCANVI

                logger.info("Loading scANVI model for inference...")
                SCANVI.setup_anndata(
                    adata_scvi,
                    layer=config.counts_layer,
                    batch_key=config.batch_key,
                    labels_key=config.cell_type_key,
                    unlabeled_category="Unknown",
                )
                scanvi_model = SCANVI.load(
                    str(config.scanvi_model_path), adata=adata_scvi
                )

                # Recompute latent representation (skipped when not training)
                logger.info("Computing X_scANVI latent representation from loaded model...")
                adata.obsm["X_scANVI"] = scanvi_model.get_latent_representation(adata=adata_scvi)
                logger.info(f"X_scANVI latent: shape={adata.obsm['X_scANVI'].shape}")

            with Timer("scANVI inference", logger):
                get_normalized_expression(
                    model=scanvi_model,
                    adata_scvi=adata_scvi,
                    adata_full=adata,
                    config=config,
                    device_info=device_info,
                    logger=logger,
                    layer_name=config.output_layer_scanvi,
                )
            log_memory("After scANVI inference", logger)

    # --- UMAP ---
    if "umap" in steps and adata is not None:
        with Timer("UMAP computation", logger):
            compute_umaps(adata, config, logger)

    # --- PLOT ---
    if "plot" in steps and adata is not None:
        with Timer("UMAP plots", logger):
            plot_batch_comparison(adata, config.umap_color_vars, config, logger)

    # --- SAVE ---
    if "save" in steps and adata is not None:
        save_checkpoint(adata, str(config.output_h5ad_path), logger)

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)

    return adata


def main():
    config = PipelineConfig.from_cli()
    run(config)


if __name__ == "__main__":
    main()
