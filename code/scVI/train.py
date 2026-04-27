"""scVI and scANVI model training (or loading from checkpoint)."""

import logging
import psutil
from typing import Optional

import anndata as ad
import scvi
from scvi.model import SCANVI, SCVI

from .config import PipelineConfig
from .utils import Timer, log_memory


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
    covariate_kwargs = {}
    if config.continuous_covariate_keys:
        covariate_kwargs['continuous_covariate_keys'] = config.continuous_covariate_keys
    if config.categorical_covariate_keys:
        covariate_kwargs['categorical_covariate_keys'] = config.categorical_covariate_keys
    if covariate_kwargs:
        logger.info(f"scVI covariates: {covariate_kwargs}")
    SCVI.setup_anndata(
        adata_scvi, layer=config.counts_layer, batch_key=config.batch_key,
        **covariate_kwargs,
    )

    # Determine whether to load or train
    logger.info(f"scVI model path: {model_path}")
    logger.info(f"Model exists: {model_path.exists()}, overwrite_scvi: {config.overwrite_scvi}")

    # Load existing model if available
    if model_path.exists() and not config.overwrite_scvi:
        logger.info(f"Loading existing scVI model from {model_path}")
        with Timer("Loading scVI model", logger):
            model = SCVI.load(str(model_path), adata=adata_scvi)
        logger.info(f"Loaded model: n_genes={model.adata.n_vars}")
        return model

    if model_path.exists() and config.overwrite_scvi:
        logger.info(f"Overwrite flag set: removing old model at {model_path}")

    # Build and train new model
    model = SCVI(
        adata_scvi,
        n_hidden=config.n_hidden,
        n_latent=config.n_latent,
        n_layers=config.n_layers,
        gene_likelihood=config.gene_likelihood,
    )
    logger.info(
        f"scVI model: n_latent={config.n_latent}, n_hidden={config.n_hidden}, "
        f"n_layers={config.n_layers}, gene_likelihood={config.gene_likelihood}"
    )

    accel = _get_accelerator(device_info)
    extra_kwargs = {}
    if config.early_stopping:
        extra_kwargs["early_stopping_patience"] = config.early_stopping_patience

    # Pre-training memory check — smart thresholds based on dataset size
    log_memory("Before scVI training", logger)
    mem = psutil.virtual_memory()
    available_gb = mem.available / 1e9
    # Estimate dataset size: ~8 bytes per float32 element
    dataset_size_gb = (adata_scvi.shape[0] * adata_scvi.shape[1] * 8 / 1e9)
    is_large_dataset = adata_scvi.shape[0] > 100000 or adata_scvi.shape[1] > 15000

    logger.info(f"System memory: {mem.percent:.1f}% used ({available_gb:.1f}GB available), "
                f"Dataset: {adata_scvi.shape[0]} cells × {adata_scvi.shape[1]} genes (~{dataset_size_gb:.2f}GB)")

    # Only reject if truly insufficient: <2GB available AND dataset is large
    if available_gb < 2.0 and is_large_dataset:
        logger.error(
            f"INSUFFICIENT MEMORY: Only {available_gb:.1f}GB available. "
            f"Dataset ({adata_scvi.shape[0]} cells × {adata_scvi.shape[1]} genes) too large. "
            f"For large datasets: submit to HPC with 20+ GB GPU memory."
        )
        raise MemoryError(f"Only {available_gb:.1f}GB available; dataset too large for this machine.")
    elif available_gb < 3.0 and is_large_dataset:
        logger.warning(f"Low memory ({available_gb:.1f}GB available) for large dataset. Training may be slow.")

    try:
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
    except (RuntimeError, MemoryError) as e:
        if "out of memory" in str(e).lower():
            logger.error(
                f"Out of memory during scVI training. This dataset ({adata_scvi.shape[0]} cells, "
                f"{adata_scvi.shape[1]} genes) is too large for your system. "
                f"Try: (1) reducing n_top_genes, (2) reducing batch_size, "
                f"(3) running on an HPC with more GPU memory."
            )
        raise

    # Save
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path), overwrite=True)
    logger.info(f"scVI model saved to {model_path}")
    log_memory("After scVI training", logger)

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

    labels = adata_scvi.obs[config.cell_type_key].astype(str)
    n_labels = labels.nunique()
    n_labeled = int((labels != "Unknown").sum())
    n_labeled_classes = int(labels[labels != "Unknown"].nunique())
    logger.info(
        f"scANVI labels: '{config.cell_type_key}' with {n_labels} categories "
        f"({n_labeled} labeled cells across {n_labeled_classes} non-Unknown classes)"
    )
    if n_labeled_classes == 0:
        raise ValueError(
            "No labeled reference classes found for scANVI (all cells are 'Unknown'). "
            f"Check '{config.cell_type_key}' population in the combined input and rerun "
            "downsample/combine before scanvi."
        )

    accel = _get_accelerator(device_info)
    extra_kwargs = {}
    if config.early_stopping:
        extra_kwargs["early_stopping_patience"] = config.early_stopping_patience

    # Load existing model if available (check gene count mismatch first)
    if model_path.exists() and not config.overwrite_scanvi:
        # Try to load saved model config to check gene count compatibility
        try:
            import json
            model_config_path = model_path / "model.pt"
            # If model exists, attempt load but catch dimension mismatch
            logger.info(f"Loading existing scANVI model from {model_path}")
            scanvi_covariate_kwargs = {}
            if config.continuous_covariate_keys:
                scanvi_covariate_kwargs['continuous_covariate_keys'] = config.continuous_covariate_keys
            if config.categorical_covariate_keys:
                scanvi_covariate_kwargs['categorical_covariate_keys'] = config.categorical_covariate_keys
            SCANVI.setup_anndata(
                adata_scvi,
                layer=config.counts_layer,
                batch_key=config.batch_key,
                labels_key=config.cell_type_key,
                unlabeled_category="Unknown",
                **scanvi_covariate_kwargs,
            )
            with Timer("Loading scANVI model", logger):
                model = SCANVI.load(str(model_path), adata=adata_scvi)
            return model
        except ValueError as e:
            if "Number of vars" in str(e) and "not the same" in str(e):
                logger.warning(
                    f"scANVI model gene count mismatch: {e}. "
                    f"Re-training from scVI model instead. "
                    f"To suppress this, use --overwrite_scanvi true"
                )
                # Fall through to train new model
            else:
                raise

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
        scanvi_covariate_kwargs = {}
        if config.continuous_covariate_keys:
            scanvi_covariate_kwargs['continuous_covariate_keys'] = config.continuous_covariate_keys
        if config.categorical_covariate_keys:
            scanvi_covariate_kwargs['categorical_covariate_keys'] = config.categorical_covariate_keys
        SCANVI.setup_anndata(
            adata_scvi,
            layer=config.counts_layer,
            batch_key=config.batch_key,
            labels_key=config.cell_type_key,
            unlabeled_category="Unknown",
            **scanvi_covariate_kwargs,
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
