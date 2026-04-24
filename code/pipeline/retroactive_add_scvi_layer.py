"""Retroactively add scvi_normalized and fix latent-space UMAPs for existing models.

Run this after a pipeline run that:
  (a) skipped scVI inference (run_scvi_inference=False), so scvi_normalized is missing, OR
  (b) produced a degenerate scANVI UMAP because the saved scANVI model had more
      latent dimensions than the current config (e.g. old n_latent=60 model loaded
      by a n_latent=40 config), causing many near-zero-variance dims in X_scANVI.

What this script does (without retraining any model):
  1. Loads the saved scVI model and runs normalized expression → adds scvi_normalized.
  2. Re-computes latent-space UMAPs (X_umap_scvi, X_umap_scanvi) with PCA
     dimensionality reduction before the neighbor graph, which collapses
     near-zero-variance latent dims and fixes degenerate UMAPs.
  3. Re-computes inferred PCA UMAPs (X_umap_scvi_inferred, X_umap_scanvi_inferred).
  4. Regenerates umaps_latent.png and umaps_inferred.png.
  5. Saves the updated integrated.h5ad.

Usage (from the repo root):
    PYTHONPATH=code python3 code/pipeline/retroactive_add_scvi_layer.py \\
        --output_dir /path/to/scvi_output \\
        [--skip_scvi_inference]     # skip step 1 if scvi_normalized already exists
        [--skip_umap]               # skip UMAP recomputation (just add layer + plots)
        [--skip_plots]              # skip plot regeneration
        [--overwrite_h5ad]          # actually save the updated h5ad (default: dry-run)
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import numpy as np
import scanpy as sc


def _setup_logger(name="retroactive"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def _stat_obsm(adata, key, logger):
    if key not in adata.obsm:
        logger.info(f"  {key}: not present")
        return
    emb = np.array(adata.obsm[key])
    var = np.var(emb, axis=0)
    n_dead = int((var < 0.01).sum())
    logger.info(
        f"  {key}: shape={emb.shape} "
        f"var_range=[{var.min():.4f},{var.max():.4f}] "
        f"n_near-zero-var={n_dead}/{emb.shape[1]}"
    )


def main():
    p = argparse.ArgumentParser(description="Retroactive scvi_normalized + UMAP fix")
    p.add_argument("--output_dir", required=True,
                   help="scvi_output directory (contains config.yaml, integrated.h5ad, scvi_model/)")
    p.add_argument("--skip_scvi_inference", action="store_true",
                   help="Skip scVI inference even if scvi_normalized is missing")
    p.add_argument("--skip_umap", action="store_true",
                   help="Skip UMAP recomputation (only add layer + regenerate plots)")
    p.add_argument("--skip_plots", action="store_true",
                   help="Skip plot regeneration")
    p.add_argument("--overwrite_h5ad", action="store_true",
                   help="Save updated integrated.h5ad (default: dry-run, no save)")
    args = p.parse_args()

    logger = _setup_logger()
    output_dir = Path(args.output_dir)
    h5ad_path  = output_dir / "integrated.h5ad"
    config_path = output_dir / "config.yaml"
    scvi_model_dir = output_dir / "scvi_model"

    if not h5ad_path.exists():
        logger.error(f"integrated.h5ad not found: {h5ad_path}")
        sys.exit(1)
    if not config_path.exists():
        logger.error(f"config.yaml not found: {config_path}")
        sys.exit(1)

    # ── Load config ───────────────────────────────────────────────────────────
    from scVI.config import PipelineConfig
    from scVI.data import prepare_for_scvi, save_checkpoint
    from scVI.inference import get_normalized_expression
    from scVI.utils import Timer, get_device_info
    from scVI.visualize import compute_umap, compute_inferred_pca_umaps, plot_umap_grids

    config = PipelineConfig.from_yaml(str(config_path))

    logger.info(f"Config: n_latent={config.n_latent}, batch_key={config.batch_key}, "
                f"transform_batch={config.transform_batch}")

    # ── Load h5ad (not backed — we'll modify in place) ────────────────────────
    logger.info(f"Loading {h5ad_path} …")
    adata = sc.read_h5ad(str(h5ad_path))
    logger.info(f"  {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    logger.info(f"  layers: {list(adata.layers.keys())}")
    logger.info(f"  obsm:   {list(adata.obsm.keys())}")

    logger.info("Latent space statistics:")
    _stat_obsm(adata, "X_scVI",   logger)
    _stat_obsm(adata, "X_scANVI", logger)

    device_info = get_device_info(logger)

    # ── Step 1: scVI normalized expression ───────────────────────────────────
    if not args.skip_scvi_inference and "scvi_normalized" not in adata.layers:
        if not scvi_model_dir.exists():
            logger.error(f"scVI model not found: {scvi_model_dir}")
            sys.exit(1)

        logger.info("Building adata_scvi (HVG subset) …")
        adata_scvi = prepare_for_scvi(adata, config, logger)

        logger.info("Loading scVI model …")
        from scvi.model import SCVI
        SCVI.setup_anndata(
            adata_scvi,
            layer=config.counts_layer,
            batch_key=config.batch_key,
            continuous_covariate_keys=config.continuous_covariate_keys or [],
            categorical_covariate_keys=config.categorical_covariate_keys or [],
        )
        scvi_model = SCVI.load(str(scvi_model_dir), adata=adata_scvi)
        logger.info(f"  Loaded: n_vars={scvi_model.adata.n_vars}")

        logger.info(f"Running scVI inference (transform_batch={config.transform_batch}) …")
        with Timer("scVI inference", logger):
            get_normalized_expression(
                model=scvi_model,
                adata_scvi=adata_scvi,
                adata_full=adata,
                config=config,
                device_info=device_info,
                logger=logger,
                layer_name="scvi_normalized",
            )
        logger.info("  scvi_normalized layer added.")
    elif "scvi_normalized" in adata.layers:
        logger.info("scvi_normalized already present — skipping inference.")
    else:
        logger.info("--skip_scvi_inference set — skipping scVI inference.")

    # ── Step 2: Re-compute latent-space UMAPs with PCA ───────────────────────
    if not args.skip_umap:
        for obsm_key, nbrs_key, umap_key in [
            ("X_scVI",   "neighbors_scvi",   "X_umap_scvi"),
            ("X_scANVI", "neighbors_scanvi", "X_umap_scanvi"),
        ]:
            if obsm_key not in adata.obsm:
                logger.info(f"  {obsm_key} not in obsm — skipping UMAP recompute")
                continue
            with Timer(f"Latent UMAP from {obsm_key}", logger):
                compute_umap(
                    adata,
                    obsm_key=obsm_key,
                    neighbors_key=nbrs_key,
                    umap_key=umap_key,
                    n_neighbors=config.umap_n_neighbors,
                    min_dist=config.umap_min_dist,
                    logger=logger,
                )
            logger.info(f"  {umap_key} recomputed.")

        # ── Step 3: Inferred PCA UMAPs ────────────────────────────────────────
        with Timer("Inferred PCA UMAPs", logger):
            compute_inferred_pca_umaps(adata, config, logger)
    else:
        logger.info("--skip_umap set — skipping UMAP recomputation.")

    # ── Step 4: Regenerate plots ───────────────────────────────────────────────
    if not args.skip_plots:
        with Timer("Regenerating UMAP grid plots", logger):
            plot_umap_grids(adata, config, logger)
        logger.info("  plots/umaps_latent.png and plots/umaps_inferred.png regenerated.")
    else:
        logger.info("--skip_plots set — skipping plot regeneration.")

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    if args.overwrite_h5ad:
        logger.info(f"Saving updated h5ad to {h5ad_path} …")
        save_checkpoint(adata, str(h5ad_path), logger)
        logger.info("Done.")
    else:
        logger.info(
            "DRY RUN — h5ad not saved. Pass --overwrite_h5ad to write the file."
        )
        logger.info(f"  layers now: {list(adata.layers.keys())}")
        logger.info(f"  obsm now:   {list(adata.obsm.keys())}")


if __name__ == "__main__":
    main()
