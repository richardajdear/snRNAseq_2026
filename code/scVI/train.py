"""scVI and scANVI model training (or loading from checkpoint)."""

import logging
from typing import Optional

import anndata as ad
import scvi
from scvi.model import SCANVI, SCVI

from .config import PipelineConfig
from .utils import Timer


def _configure_scvi_settings(config: PipelineConfig, logger: logging.Logger):
    """Set global scvi-tools settings."""
    scvi.settings.seed = config.random_seed
    if config.num_workers > 0:
        scvi.settings.num_threads = config.num_workers
    logger.info(f"scvi settings: seed={config.random_seed}")


def _get_accelerator(device_info: dict) -> dict:
    """Return Lightning Trainer accelerator kwargs."""
    device = device_info["device"]
    if device == "cuda":
        return {"accelerator": "gpu", "devices": 1}
    elif device == "mps":
        return {"accelerator": "mps", "devices": 1}
    else:
        return {"accelerator": "cpu", "devices": 1}


def train_scvi(
    adata_scvi: ad.AnnData,
    config: PipelineConfig,
    device_info: dict,
    logger: logging.Logger,
) -> SCVI:
    """Train or load an scVI model."""
    _configure_scvi_settings(config, logger)
    model_path = config.scvi_model_path

    # Setup anndata (required before both training and loading)
    SCVI.setup_anndata(
        adata_scvi, layer=config.counts_layer, batch_key=config.batch_key
    )

    # Load existing model if available
    if model_path.exists() and not config.overwrite_scvi:
        logger.info(f"Loading existing scVI model from {model_path}")
        with Timer("Loading scVI model", logger):
            model = SCVI.load(str(model_path), adata=adata_scvi)
        return model

    # Build and train new model
    model = SCVI(
        adata_scvi,
        n_hidden=config.n_hidden,
        n_latent=config.n_latent,
        n_layers=config.n_layers,
    )
    logger.info(
        f"scVI model: n_latent={config.n_latent}, "
        f"n_hidden={config.n_hidden}, n_layers={config.n_layers}"
    )

    accel = _get_accelerator(device_info)
    extra_kwargs = {}
    if config.early_stopping:
        extra_kwargs["early_stopping_patience"] = config.early_stopping_patience

    with Timer(f"Training scVI ({config.max_epochs_scvi} max epochs)", logger):
        model.train(
            max_epochs=config.max_epochs_scvi,
            early_stopping=config.early_stopping,
            train_size=config.train_size,
            validation_size=1.0 - config.train_size,
            batch_size=config.batch_size,
            enable_progress_bar=True,
            **extra_kwargs,
            **accel,
        )

    # Save
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), overwrite=True)
    logger.info(f"scVI model saved to {model_path}")

    return model


def train_scanvi(
    adata_scvi: ad.AnnData,
    config: PipelineConfig,
    device_info: dict,
    logger: logging.Logger,
    scvi_model: Optional[SCVI] = None,
) -> SCANVI:
    """
    Train or load a scANVI model (semi-supervised with cell type labels).

    If scvi_model is provided, initializes scANVI from it via from_scvi_model(),
    which transfers learned weights and typically improves convergence.
    """
    _configure_scvi_settings(config, logger)
    model_path = config.scanvi_model_path

    # Validate cell type key
    if config.cell_type_key not in adata_scvi.obs.columns:
        raise ValueError(
            f"Cell type key '{config.cell_type_key}' not in .obs for scANVI. "
            f"Available: {list(adata_scvi.obs.columns)}"
        )

    n_labels = adata_scvi.obs[config.cell_type_key].nunique()
    logger.info(
        f"scANVI labels: '{config.cell_type_key}' with {n_labels} categories"
    )

    accel = _get_accelerator(device_info)
    extra_kwargs = {}
    if config.early_stopping:
        extra_kwargs["early_stopping_patience"] = config.early_stopping_patience

    # Load existing model if available
    if model_path.exists() and not config.overwrite_scanvi:
        logger.info(f"Loading existing scANVI model from {model_path}")
        # Loading requires setup_anndata first
        SCANVI.setup_anndata(
            adata_scvi,
            layer=config.counts_layer,
            batch_key=config.batch_key,
            labels_key=config.cell_type_key,
            unlabeled_category="Unknown",
        )
        with Timer("Loading scANVI model", logger):
            model = SCANVI.load(str(model_path), adata=adata_scvi)
        return model

    # Build from scVI weights (preferred) or from scratch
    if scvi_model is not None:
        logger.info("Initializing scANVI from pre-trained scVI model (from_scvi_model)")
        model = SCANVI.from_scvi_model(
            scvi_model,
            labels_key=config.cell_type_key,
            unlabeled_category="Unknown",
        )
    else:
        logger.info("Initializing scANVI from scratch (no scVI model provided)")
        SCANVI.setup_anndata(
            adata_scvi,
            layer=config.counts_layer,
            batch_key=config.batch_key,
            labels_key=config.cell_type_key,
            unlabeled_category="Unknown",
        )
        model = SCANVI(
            adata_scvi,
            n_hidden=config.n_hidden,
            n_latent=config.n_latent,
            n_layers=config.n_layers,
        )

    with Timer(f"Training scANVI ({config.max_epochs_scanvi} max epochs)", logger):
        model.train(
            max_epochs=config.max_epochs_scanvi,
            early_stopping=config.early_stopping,
            train_size=config.train_size,
            validation_size=1.0 - config.train_size,
            batch_size=config.batch_size,
            enable_progress_bar=True,
            **extra_kwargs,
            **accel,
        )

    model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), overwrite=True)
    logger.info(f"scANVI model saved to {model_path}")

    return model
