"""Retroactively add scvi_normalized layer and regenerate inferred UMAP plots.

Loads an existing integrated.h5ad + saved scVI model, runs scVI
get_normalized_expression (which was skipped when scANVI inference was the
primary output), then recomputes:
    X_pca_scvi_inferred   → added to obsm
    X_umap_scvi_inferred  → added to obsm
And regenerates:
    scvi_output/plots/umaps_latent.png
    scvi_output/plots/umaps_inferred.png

Existing layers / obsm keys are preserved; only the additions are new.

Usage
-----
    PYTHONPATH=code python3 -m pipeline.retroactive_add_scvi_inferred \\
        --config  path/to/scvi_output/scvi_config.yaml

    # or pass paths directly:
    PYTHONPATH=code python3 -m pipeline.retroactive_add_scvi_inferred \\
        --h5ad    path/to/scvi_output/integrated.h5ad \\
        --model   path/to/scvi_output/scvi_model \\
        --output  path/to/scvi_output
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import scanpy as sc


def _setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("retroactive")


def main():
    p = argparse.ArgumentParser(description="Add scvi_normalized + inferred UMAP")
    p.add_argument("--config", help="Path to scvi_config.yaml (overrides other flags)")
    p.add_argument("--h5ad",   help="Path to integrated.h5ad")
    p.add_argument("--model",  help="Path to scvi_model directory")
    p.add_argument("--output", help="Output directory (scvi_output/); plots go in plots/")
    p.add_argument("--transform_batch", default=None,
                   help="transform_batch passed to get_normalized_expression "
                        "(default: read from config, or None)")
    p.add_argument("--n_mc_samples", type=int, default=10)
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite scvi_normalized if it already exists")
    p.add_argument("--plots_only", action="store_true",
                   help="Skip inference; only regenerate plots from existing obsm")
    args = p.parse_args()

    logger = _setup_logger()

    # ── resolve paths ─────────────────────────────────────────────────────────
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        h5ad_path    = Path(cfg.get("output_dir", ".")) / cfg.get("output_h5ad", "integrated.h5ad")
        model_path   = Path(cfg.get("output_dir", ".")) / cfg.get("scvi_model_dir", "scvi_model")
        output_dir   = Path(cfg.get("output_dir", "."))
        transform_batch = args.transform_batch or cfg.get("transform_batch")
        n_mc_samples    = cfg.get("n_mc_samples", args.n_mc_samples)
        batch_key       = cfg.get("batch_key", "source")
        counts_layer    = cfg.get("counts_layer", "counts")
    else:
        if not (args.h5ad and args.model and args.output):
            p.error("Provide --config OR all three of --h5ad, --model, --output")
        h5ad_path    = Path(args.h5ad)
        model_path   = Path(args.model)
        output_dir   = Path(args.output)
        transform_batch = args.transform_batch
        n_mc_samples    = args.n_mc_samples
        batch_key    = "source"
        counts_layer = "counts"

    logger.info(f"h5ad:          {h5ad_path}")
    logger.info(f"scVI model:    {model_path}")
    logger.info(f"output dir:    {output_dir}")
    logger.info(f"transform_batch: {transform_batch}")

    # ── load h5ad ─────────────────────────────────────────────────────────────
    logger.info(f"Loading {h5ad_path} …")
    adata = sc.read_h5ad(str(h5ad_path))
    logger.info(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    logger.info(f"  layers: {list(adata.layers.keys())}")
    logger.info(f"  obsm:   {list(adata.obsm.keys())}")

    if not args.plots_only:
        if "scvi_normalized" in adata.layers and not args.overwrite:
            logger.info("scvi_normalized already present — skipping inference "
                        "(use --overwrite to force)")
        else:
            # ── build adata_scvi (HVG subset) ─────────────────────────────────
            if "highly_variable" not in adata.var.columns:
                logger.error("'highly_variable' column not in adata.var — "
                             "cannot reconstruct HVG subset without it.")
                sys.exit(1)
            adata_scvi = adata[:, adata.var["highly_variable"]].copy()
            logger.info(f"HVG subset: {adata_scvi.shape}")

            # ── load scVI model ────────────────────────────────────────────────
            try:
                from scvi.model import SCVI
            except ImportError:
                logger.error("scvi-tools not available. Activate the scVI conda env first.")
                sys.exit(1)

            logger.info(f"Loading scVI model from {model_path} …")
            SCVI.setup_anndata(adata_scvi, layer=counts_layer, batch_key=batch_key)
            scvi_model = SCVI.load(str(model_path), adata=adata_scvi)
            logger.info("Model loaded.")

            # ── run scVI normalized expression ────────────────────────────────
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from scVI.config import PipelineConfig
            from scVI.inference import get_normalized_expression
            from scVI.utils import get_device_info, log_memory

            pseudo_config = PipelineConfig(
                transform_batch=transform_batch,
                n_mc_samples=n_mc_samples,
                chunk_size=None,
                target_vram_fraction=0.25,
                max_chunk_size=50000,
                save_npy_backup=False,
            )
            device_info = get_device_info(logger)
            log_memory("Before scVI inference", logger)

            get_normalized_expression(
                model=scvi_model,
                adata_scvi=adata_scvi,
                adata_full=adata,
                config=pseudo_config,
                device_info=device_info,
                logger=logger,
                layer_name="scvi_normalized",
            )
            log_memory("After scVI inference", logger)
            logger.info("scvi_normalized layer written to adata.")

    # ── compute/recompute inferred PCA + UMAP ─────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scVI.config import PipelineConfig
    from scVI.visualize import compute_inferred_pca_umaps, plot_umap_grids

    pseudo_config = PipelineConfig(
        output_dir=str(output_dir),
        umap_n_neighbors=30,
        umap_min_dist=0.3,
        umap_point_size=1.0,
        umap_log2_ticks=[0.0, 1.0, 3.0, 9.0, 25.0, 40.0],
        batch_key=batch_key,
    )

    if not args.plots_only:
        logger.info("Computing inferred PCA + UMAP from scvi_normalized …")
        compute_inferred_pca_umaps(adata, pseudo_config, logger)
        logger.info(f"obsm now: {list(adata.obsm.keys())}")

    # ── regenerate UMAP grid plots ─────────────────────────────────────────────
    logger.info("Regenerating UMAP grid plots …")
    plot_umap_grids(adata, pseudo_config, logger)

    # ── save updated h5ad ────────────────────────────────────────────────────
    if not args.plots_only:
        logger.info(f"Saving updated h5ad to {h5ad_path} …")
        adata.write_h5ad(str(h5ad_path))
        logger.info("Done.")


if __name__ == "__main__":
    main()
