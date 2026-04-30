"""Chunked inference for batch-corrected normalized expression with OOM retry."""

import logging
import psutil
from typing import Optional

import anndata as ad
import numpy as np
import torch
from tqdm import tqdm

from .config import PipelineConfig
from .utils import Timer, estimate_chunk_size, log_memory


def get_normalized_expression(
    model,
    adata_scvi: ad.AnnData,
    adata_full: ad.AnnData,
    config: PipelineConfig,
    device_info: dict,
    logger: logging.Logger,
    layer_name: str = "scvi_normalized",
) -> np.ndarray:
    """
    Extract batch-corrected normalized expression via chunked inference.

    Stores results in adata_full.layers[layer_name], mapped back to the
    full gene set (zeros for non-HVG genes).

    Returns the (n_cells, n_all_genes) array.
    """
    n_cells, n_hvg_genes = adata_scvi.shape
    n_all_genes = adata_full.shape[1]

    # Pre-inference memory check
    log_memory("Before inference", logger)
    mem = psutil.virtual_memory()
    if mem.percent > 90:
        logger.warning(
            f"Very high memory usage before inference: {mem.percent:.1f}%. "
            f"OOM may occur during chunked processing."
        )

    # Determine chunk size
    chunk_size = config.chunk_size
    if chunk_size is None:
        chunk_size = estimate_chunk_size(
            n_genes=n_hvg_genes,
            n_mc_samples=config.n_mc_samples,
            vram_bytes=device_info["vram_bytes"],
            target_fraction=config.target_vram_fraction,
            max_chunk=config.max_chunk_size,
        )
    logger.info(f"Inference chunk size: {chunk_size} (total cells: {n_cells})")

    # Run chunked inference
    hvg_results = _chunked_inference(
        model=model,
        adata=adata_scvi,
        n_mc_samples=config.n_mc_samples,
        transform_batch=config.transform_batch,
        chunk_size=chunk_size,
        logger=logger,
    )

    # Map HVG results to full gene space
    hvg_genes = adata_scvi.var_names
    gene_indices = adata_full.var_names.get_indexer(hvg_genes)
    valid = gene_indices >= 0
    n_mapped = valid.sum()

    if n_mapped < len(hvg_genes):
        logger.warning(
            f"{len(hvg_genes) - n_mapped} HVG genes not found in full AnnData. "
            f"These will be zero in the output."
        )

    logger.info(f"Mapping {n_mapped} HVG genes into full gene space ({n_all_genes})")

    full_result = np.zeros((n_cells, n_all_genes), dtype=np.float32)
    full_result[:, gene_indices[valid]] = hvg_results[:, np.where(valid)[0]]

    # Store
    adata_full.layers[layer_name] = full_result
    logger.info(f"Stored in adata.layers['{layer_name}'] shape={full_result.shape}")

    # Optional .npy backup
    if config.save_npy_backup:
        npy_path = config._resolved_output_dir / f"{layer_name}.npy"
        np.save(str(npy_path), full_result)
        logger.info(f"Backup saved: {npy_path}")

    return full_result


def _chunked_inference(
    model,
    adata: ad.AnnData,
    n_mc_samples: int,
    transform_batch: Optional[str],
    chunk_size: int,
    logger: logging.Logger,
    min_chunk_size: int = 100,
) -> np.ndarray:
    """
    Process inference in chunks with OOM retry logic.

    On OOM: clears cache, halves chunk size, retries the failed chunk.
    Raises RuntimeError if chunk size drops below min_chunk_size.
    """
    n_cells = adata.shape[0]

    # Single pass if small enough
    if chunk_size >= n_cells:
        logger.info(f"Processing all {n_cells} cells in one pass")
        kwargs = {"n_samples": n_mc_samples, "library_size": 1e4}
        if transform_batch is not None:
            kwargs["transform_batch"] = transform_batch
        return model.get_normalized_expression(adata=adata, **kwargs).values

    n_chunks = (n_cells + chunk_size - 1) // chunk_size
    logger.info(f"Processing {n_cells} cells in ~{n_chunks} chunks of {chunk_size}")

    chunks = []
    current_chunk_size = chunk_size
    start_idx = 0
    chunk_num = 0

    pbar = tqdm(total=n_cells, desc="Inference", unit=" cells")

    while start_idx < n_cells:
        end_idx = min(start_idx + current_chunk_size, n_cells)
        chunk_adata = adata[start_idx:end_idx].copy()

        try:
            kwargs = {"n_samples": n_mc_samples, "library_size": 1e4}
            if transform_batch is not None:
                kwargs["transform_batch"] = transform_batch
            result = model.get_normalized_expression(adata=chunk_adata, **kwargs)
            chunks.append(result.values)
            chunk_num += 1
            cells_processed = end_idx - start_idx
            logger.info(
                f"  Chunk {chunk_num}: [{start_idx}:{end_idx}) "
                f"({cells_processed} cells) OK"
            )
            pbar.update(cells_processed)
            start_idx = end_idx
            current_chunk_size = chunk_size  # restore after success

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() or isinstance(
                e, torch.cuda.OutOfMemoryError
            ):
                torch.cuda.empty_cache()
                new_size = current_chunk_size // 2
                if new_size < min_chunk_size:
                    raise RuntimeError(
                        f"OOM with chunk_size={current_chunk_size}. "
                        f"Minimum ({min_chunk_size}) reached."
                    ) from e
                logger.warning(
                    f"  OOM at [{start_idx}:{end_idx}). "
                    f"Halving chunk: {current_chunk_size} -> {new_size}"
                )
                current_chunk_size = new_size
                # Don't advance start_idx — retry this chunk
            else:
                raise

    pbar.close()
    return np.vstack(chunks)
