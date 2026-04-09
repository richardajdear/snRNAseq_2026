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
import pandas as pd
import torch

from .config import PipelineConfig
from .data import load_adata, prepare_for_scvi, save_checkpoint
from .inference import get_normalized_expression
from .train import train_scanvi, train_scvi
from .utils import Timer, get_device_info, log_memory, setup_logger
from .visualize import compute_umaps, plot_batch_comparison
from pipeline.label_transfer.transfer import aligned_to_class


def _update_cell_class_from_aligned(adata, logger):
    """Recompute cell_class from cell_type_aligned using aligned_to_class mapping.

    Called after scANVI label transfer assigns new cell_type_aligned values.
    Some cells may be remapped across classes (e.g. Inhibitory → Excitatory),
    so cell_class must be updated to stay consistent with cell_type_aligned.
    """
    new_class = adata.obs['cell_type_aligned'].map(aligned_to_class)
    if 'cell_class' in adata.obs.columns:
        old_class = adata.obs['cell_class'].astype(str)
        n_changed = (old_class != new_class.astype(str)).sum()
        logger.info(
            f"cell_class: updated {n_changed} cells whose class changed after "
            "cell_type_aligned label transfer"
        )
    else:
        logger.info("cell_class: column not present; creating from cell_type_aligned")
    adata.obs['cell_class'] = pd.Categorical(new_class)


def _predict_scanvi_with_confidence(scanvi_model, adata_scvi, logger):
    """Return (predicted_labels, confidence_vector) across scvi-tools versions."""
    predictions = scanvi_model.predict(adata_scvi)

    # Newer versions may expose predict_proba, while others support predict(soft=True).
    if hasattr(scanvi_model, "predict_proba"):
        proba = scanvi_model.predict_proba(adata_scvi)
        if hasattr(proba, "values"):
            confidence = proba.values.max(axis=1)
        else:
            confidence = np.asarray(proba).max(axis=1)
        return predictions, confidence

    try:
        soft = scanvi_model.predict(adata_scvi, soft=True)
        if hasattr(soft, "values"):
            confidence = soft.values.max(axis=1)
        else:
            confidence = np.asarray(soft).max(axis=1)
    except TypeError:
        logger.warning(
            "Could not compute probabilistic confidence for scANVI predictions; "
            "setting cell_type_aligned_confidence to 1.0 for all cells."
        )
        confidence = np.ones(len(predictions), dtype=float)

    return predictions, confidence


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
        # scanvi-only reruns may skip train_scvi. In that case, initialize from
        # an existing saved scVI model when available.
        if scvi_model is None and config.scvi_model_path.exists():
            from scvi.model import SCVI

            logger.info(
                "train_scvi step skipped; loading existing scVI model "
                f"from {config.scvi_model_path} for scANVI initialization"
            )
            SCVI.setup_anndata(
                adata_scvi,
                layer=config.counts_layer,
                batch_key=config.batch_key,
            )
            with Timer("Loading scVI model for scANVI init", logger):
                scvi_model = SCVI.load(str(config.scvi_model_path), adata=adata_scvi)

        # Pass scvi_model so scANVI inherits learned weights (from_scvi_model)
        scanvi_model = train_scanvi(
            adata_scvi, config, device_info, logger, scvi_model=scvi_model
        )

        with Timer("scANVI latent representation", logger):
            adata.obsm["X_scANVI"] = scanvi_model.get_latent_representation(
                adata=adata_scvi
            )
        logger.info(f"X_scANVI latent: shape={adata.obsm['X_scANVI'].shape}")

        # Label transfer via scANVI predictions (used when cell_type_for_scanvi labels
        # are only set for WANG cells; model predicts labels for all other cells)
        if config.predict_cell_types:
            with Timer("scANVI label transfer (model.predict)", logger):
                predictions, confidence = _predict_scanvi_with_confidence(
                    scanvi_model, adata_scvi, logger
                )
            adata.obs['cell_type_aligned'] = predictions
            adata.obs['cell_type_aligned_confidence'] = confidence
            n_types = adata.obs['cell_type_aligned'].nunique()
            n_low = (adata.obs['cell_type_aligned_confidence'] < 0.5).sum()
            logger.info(
                f"cell_type_aligned: {n_types} types assigned to {len(predictions)} cells "
                f"({n_low} low-confidence <0.5)"
            )
            _update_cell_class_from_aligned(adata, logger)

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

                # Re-run label transfer predictions if not already stored
                if config.predict_cell_types and 'cell_type_aligned' not in adata.obs.columns:
                    logger.info("Re-running scANVI label transfer from loaded model...")
                    with Timer("scANVI label transfer (model.predict)", logger):
                        predictions, confidence = _predict_scanvi_with_confidence(
                            scanvi_model, adata_scvi, logger
                        )
                    adata.obs['cell_type_aligned'] = predictions
                    adata.obs['cell_type_aligned_confidence'] = confidence
                    logger.info(
                        f"cell_type_aligned: {adata.obs['cell_type_aligned'].nunique()} types"
                    )
                    _update_cell_class_from_aligned(adata, logger)
                    save_checkpoint(adata, str(config.output_h5ad_path), logger)

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
