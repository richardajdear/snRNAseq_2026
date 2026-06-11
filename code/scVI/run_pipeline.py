"""
See README.md at the repo root for full documentation, environment setup,
and usage guidelines.

Main orchestration script for the scVI batch correction pipeline.

Usage (from project root):
    # With YAML config:
    PYTHONPATH=code python -m scVI.run_pipeline --config code/scVI/default_config.yaml

    # With CLI overrides:
    PYTHONPATH=code python -m scVI.run_pipeline --config code/scVI/default_config.yaml --max_epochs_scvi 100

    # Run specific steps only:
    PYTHONPATH=code python -m scVI.run_pipeline --config code/scVI/default_config.yaml --steps infer save
"""

import gc
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from .config import PipelineConfig
from .data import load_adata, prepare_for_scvi, save_checkpoint
from .inference import get_normalized_expression
from .train import train_scanvi, train_scvi
from scvi.model import LinearSCVI
from .utils import Timer, get_device_info, log_memory, setup_logger
from .visualize import compute_umaps, plot_umap_grids, plot_pca_grids, plot_excitatory_grids
from pipeline.label_transfer.transfer import aligned_to_class


def _write_adata_chunked(adata, path, chunk_rows=10_000, logger=None):
    """Write adata to h5ad, writing np.memmap layers in row-chunks via h5py.

    adata.write_h5ad() calls numpy.asarray(layer) which loads an entire memmap
    into RAM. This function detects memmap layers, writes everything else via
    anndata, then appends the large layers incrementally using h5py directly.
    Peak RAM per layer is chunk_rows × n_genes × 4 bytes (~600 MB at 10k rows).
    """
    import h5py

    memmap_layers = {k: v for k, v in adata.layers.items()
                     if isinstance(v, np.memmap)}

    if not memmap_layers:
        adata.write_h5ad(str(path))
        return

    for k in memmap_layers:
        del adata.layers[k]
    try:
        adata.write_h5ad(str(path))
    finally:
        for k, v in memmap_layers.items():
            adata.layers[k] = v

    with h5py.File(str(path), 'a') as h5f:
        layers_grp = h5f.require_group('layers')
        for lname, arr in memmap_layers.items():
            n_rows, n_cols = arr.shape
            ds = layers_grp.create_dataset(
                lname, shape=(n_rows, n_cols), dtype='float32',
                chunks=(min(chunk_rows, n_rows), n_cols),
            )
            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                ds[start:end] = np.asarray(arr[start:end])
            if logger:
                logger.info(f"  Wrote layer '{lname}' ({n_rows}×{n_cols}) to {path}")


def _update_cell_class_from_aligned(adata, logger):
    """Recompute cell_class from cell_type_aligned using aligned_to_class mapping.

    Called after scANVI label transfer assigns new cell_type_aligned values.
    Some cells may be remapped across classes (e.g. Inhibitory → Excitatory),
    so cell_class must be updated to stay consistent with cell_type_aligned.

    The original cell_class is preserved as cell_class_original so that
    downstream diagnostics can detect which cells changed class.
    """
    new_class = adata.obs['cell_type_aligned'].map(aligned_to_class)
    if 'cell_class' in adata.obs.columns:
        old_class = adata.obs['cell_class'].astype(str)
        n_changed = (old_class != new_class.astype(str)).sum()
        logger.info(
            f"cell_class: updated {n_changed} cells whose class changed after "
            "cell_type_aligned label transfer"
        )
        # Preserve pre-transfer cell_class for diagnostics (only set once)
        if 'cell_class_original' not in adata.obs.columns:
            adata.obs['cell_class_original'] = adata.obs['cell_class'].copy()
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


def _export_pca_loadings_csv(adata, config, logger):
    """Write per-gene PCA loadings as CSV next to integrated.h5ad.

    One CSV per inferred-PCA varm key (pca_scvi_inferred_loadings,
    pca_scanvi_inferred_loadings). Rows are gene IDs (adata.var_names);
    columns are PC1..PCk. Genes not used in the PCA fit have NaN values
    and are dropped from the CSV. These files are small (~MB) and meant
    to be downloaded from HPC so notebooks can treat PC1 as a GRN.
    """
    out_dir = config._resolved_output_dir
    for key in list(adata.varm.keys()):
        if not key.endswith("_loadings"):
            continue
        if not key.startswith("pca_") or "inferred" not in key:
            continue
        loadings = np.asarray(adata.varm[key])
        col_key = key + "_columns"
        if col_key in adata.uns:
            columns = list(adata.uns[col_key])
        else:
            columns = [f"PC{i + 1}" for i in range(loadings.shape[1])]
        df = pd.DataFrame(loadings, index=adata.var_names, columns=columns)
        df = df.dropna(how="all")
        # Short filename: drop the "_inferred" suffix for readability.
        # e.g. pca_scanvi_inferred_loadings → pca_scanvi_loadings.csv
        short = key.replace("_inferred", "")
        out_path = out_dir / f"{short}.csv"
        df.to_csv(out_path, index_label="gene_id")
        logger.info(
            f"Wrote {out_path} (shape={df.shape}; "
            f"genes={len(df):,} × components={len(columns)})"
        )


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

        # adata_scvi is a copy of the HVG subset and inherits all layers from the
        # input h5ad (including any existing scvi_normalized / scanvi_normalized).
        # Only the counts layer is needed for scVI/scANVI; drop the rest to free
        # ~2×29 GB when re-running inference on an already-normalised h5ad.
        _unneeded_layers = [l for l in list(adata_scvi.layers.keys()) if l != config.counts_layer]
        if _unneeded_layers:
            for _l in _unneeded_layers:
                del adata_scvi.layers[_l]
            gc.collect()
            logger.info(f"Freed adata_scvi layers not needed for inference: {_unneeded_layers}")
            log_memory("After freeing adata_scvi layers", logger)

        # Strip stale normalized layers from adata early — before any training or
        # checkpoint save. Without this, a copied integrated.h5ad (e.g. from a
        # previous run) carries ~65 GB dense layers that (a) waste RAM throughout
        # training and (b) get written to the checkpoint, so inference-only reruns
        # load a full-size file unnecessarily.
        _stale_norm = [
            l for l in [config.output_layer_scvi, config.output_layer_scanvi]
            if l in adata.layers
        ]
        if _stale_norm:
            for _l in _stale_norm:
                del adata.layers[_l]
            gc.collect()
            logger.info(f"Freed stale normalized layers from adata before training: {_stale_norm}")
            log_memory("After freeing stale normalized layers", logger)

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

        # --- batch-integration quality metric (auto-warn if poorly mixed) ---
        try:
            from scVI.batch_metric import batch_mixing_score
            _bk = getattr(config, "batch_key", "source")
            if _bk in adata.obs.columns and adata.obs[_bk].nunique() > 1:
                adata.uns["batch_mixing_score"] = batch_mixing_score(
                    adata.obsm["X_scVI"], adata.obs[_bk].astype(str).values,
                    logger=logger)
        except Exception as _e:
            logger.warning(f"batch_mixing_score failed (non-fatal): {_e}")

        # Extract LDVAE factor loadings (gene weights per latent dimension).
        # Done here, before any scANVI step modifies the latent space, so the
        # loadings always reflect the LDVAE decoder regardless of downstream steps.
        if config.linear_decoder:
            with Timer("LDVAE factor loadings", logger):
                loadings_df = scvi_model.get_loadings()  # DataFrame: n_hvgs × n_latent
                adata_scvi.varm["ldvae_loadings"] = loadings_df.values
                # Expand to full gene space; NaN for non-HVGs.
                loadings_full = pd.DataFrame(
                    np.nan,
                    index=adata.var_names,
                    columns=loadings_df.columns,
                )
                loadings_full.loc[adata_scvi.var_names] = loadings_df.values
                adata.varm["ldvae_loadings"] = loadings_full.values
                adata.uns["ldvae_loading_columns"] = list(loadings_df.columns)
            logger.info(
                f"ldvae_loadings: shape={adata.varm['ldvae_loadings'].shape} "
                f"(NaN for {adata.n_vars - adata_scvi.n_vars} non-HVGs)"
            )

        save_checkpoint(adata, str(config.output_h5ad_path), logger)

    # --- TRAIN SCANVI (optional) ---
    scanvi_model = None
    if "train_scanvi" in steps and config.run_scanvi:
        # scanvi-only reruns may skip train_scvi. In that case, initialize from
        # an existing saved model when available.
        if scvi_model is None and config.scvi_model_path.exists():
            from scvi.model import SCVI

            _scvi_cls = LinearDecoderSCVI if config.linear_decoder else SCVI
            _model_label = "LinearDecoderSCVI (LDVAE)" if config.linear_decoder else "scVI"
            logger.info(
                f"train_scvi step skipped; loading existing {_model_label} model "
                f"from {config.scvi_model_path} for scANVI initialization"
            )
            _scvi_cls.setup_anndata(
                adata_scvi,
                layer=config.counts_layer,
                batch_key=config.batch_key,
            )
            with Timer(f"Loading {_model_label} model for scANVI init", logger):
                scvi_model = _scvi_cls.load(str(config.scvi_model_path), adata=adata_scvi)

        # Pass scvi_model so scANVI inherits learned weights (from_scvi_model).
        # For LinearDecoderSCVI, from_scvi_model should work (same encoder architecture)
        # but scvi-tools may reject the non-SCVI class; fall back to scratch if so.
        if config.linear_decoder and scvi_model is not None:
            try:
                scanvi_model = train_scanvi(
                    adata_scvi, config, device_info, logger, scvi_model=scvi_model
                )
                logger.info("scANVI initialized from LinearDecoderSCVI encoder via from_scvi_model")
            except (TypeError, ValueError, AssertionError, RuntimeError) as e:
                logger.warning(
                    f"SCANVI.from_scvi_model() rejected LinearDecoderSCVI ({e}). "
                    "Falling back to scANVI from scratch — cell type annotation will still "
                    "work but scANVI latent will not share the LDVAE encoder weights."
                )
                scanvi_model = train_scanvi(
                    adata_scvi, config, device_info, logger, scvi_model=None
                )
        else:
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
        # Free any existing normalized layers that we're about to recompute.
        # In retransform runs the input h5ad already has these layers (~32 GB
        # each, dense float32) which waste ~64 GB of the 250 GB job allocation
        # throughout inference. Free them now — they're overwritten in this step.
        for _layer in [config.output_layer_scvi, config.output_layer_scanvi]:
            if _layer in adata.layers:
                logger.info(f"Freeing existing layer '{_layer}' before inference (will be recomputed).")
                del adata.layers[_layer]
        gc.collect()
        log_memory("After freeing old normalized layers", logger)

        # File-backed temp dir for memmap output arrays — avoids holding 2×56 GB in RAM simultaneously
        _temp_dir = config.temp_dir or str(config._resolved_output_dir / "tmp_inference")
        logger.info(f"Inference temp dir (memmap files): {_temp_dir}")

        # Load scVI/LDVAE model if not in memory
        if scvi_model is None and config.run_scvi_inference:
            from scvi.model import SCVI

            _scvi_cls = LinearDecoderSCVI if config.linear_decoder else SCVI
            _model_label = "LinearDecoderSCVI (LDVAE)" if config.linear_decoder else "scVI"
            logger.info(f"Loading {_model_label} model for inference...")
            _scvi_cls.setup_anndata(
                adata_scvi, layer=config.counts_layer, batch_key=config.batch_key
            )
            scvi_model = _scvi_cls.load(str(config.scvi_model_path), adata=adata_scvi)

            # Recompute latent representation (skipped when not training)
            logger.info(f"Computing X_scVI latent representation from loaded {_model_label} model...")
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
                    temp_dir=_temp_dir,
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
                    temp_dir=_temp_dir,
                )
            log_memory("After scANVI inference", logger)

    # --- UMAP ---
    if "umap" in steps and adata is not None:
        with Timer("UMAP computation", logger):
            compute_umaps(adata, config, logger)

    # --- PLOT ---
    if "plot" in steps and adata is not None:
        with Timer("UMAP + PCA plots (all cells)", logger):
            plot_umap_grids(adata, config, logger)
            plot_pca_grids(adata, config, logger)
        with Timer("UMAP + PCA plots (excitatory)", logger):
            plot_excitatory_grids(adata, config, logger)

    # --- SAVE ---
    if "save" in steps and adata is not None:
        with Timer(f"Saving to {config.output_h5ad_path}", logger):
            _write_adata_chunked(adata, config.output_h5ad_path, chunk_rows=10_000, logger=logger)
        logger.info(f"Saved: {adata.shape}")
        _export_pca_loadings_csv(adata, config, logger)

        # Remove temp memmap files now that layers are safely written to h5ad
        import shutil
        _temp_dir_path = Path(config.temp_dir or str(config._resolved_output_dir / "tmp_inference"))
        if _temp_dir_path.exists():
            shutil.rmtree(str(_temp_dir_path), ignore_errors=True)
            logger.info(f"Cleaned up temp inference files: {_temp_dir_path}")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)

    return adata


def main():
    config = PipelineConfig.from_cli()
    run(config)


if __name__ == "__main__":
    main()
