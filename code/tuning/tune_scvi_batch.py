"""Hyperparameter tuning for scVI batch correction with expression-based integration metrics.

This script is intentionally separate from `code/pipeline/` so tuning can run as an
overnight optimization job and then feed best parameters back into the pipeline config.

Search space covers model architecture and gene likelihood only (n_latent, n_hidden,
n_layers, gene_likelihood, batch_size).  Training-schedule knobs (lr, weight_decay,
dropout_rate) are left at scVI defaults: they are secondary to architecture for batch
integration quality and work well with early stopping at the production epoch horizon.
The output best_hyperparameters.yaml maps directly to fields in scVI/config.py.

Optimization objective (50:50 cross-trial normalized combination):
  1. Age-binned batch mixing computed on PCA of batch-corrected normalized expression
     (get_normalized_expression with transform_batch), not the latent space.  This
     directly measures integration quality in the expression space that is used in
     production (scanvi_normalized), rather than in the latent representation.
  2. Decoder reconstruction accuracy (get_reconstruction_error without transform_batch).
     Including this term prevents architectures that improve batch mixing by collapsing
     the latent space at the expense of accurate generative modeling.

Both components are min-max normalized across all successful trials so they contribute
equally to the final objective regardless of their absolute scales.  The raw component
values are also recorded so the user can inspect trade-offs.

After all scVI trials, the top-k models are evaluated with scANVI (using
SCANVI.from_scvi_model) to report whether scANVI fine-tuning changes integration
quality — since production inference uses scANVI, not scVI.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
import torch
import yaml
from scvi.model import SCANVI, SCVI
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

EXPRESSION_PCA_SEED_OFFSET = 2000


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("tune_scvi_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, mode="w")  # 'w' so reruns don't interleave
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Config / search space utilities
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _as_list(space: dict[str, Any], key: str, default: list[Any]) -> list[Any]:
    vals = space.get(key, default)
    if not isinstance(vals, list) or len(vals) == 0:
        raise ValueError(f"search_space.{key} must be a non-empty list")
    return vals


def _sample_trial_params(search_space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Sample one trial's hyperparameters from the discrete search space.

    Only model-architecture and gene-likelihood parameters are tuned.
    lr / weight_decay / dropout_rate use scVI defaults; they have secondary
    impact on batch integration quality and do not cleanly map back to
    production config fields.
    """
    return {
        "n_latent":        int(rng.choice(_as_list(search_space, "n_latent",        [20, 30, 40]))),
        "n_hidden":        int(rng.choice(_as_list(search_space, "n_hidden",        [64, 128, 256]))),
        "n_layers":        int(rng.choice(_as_list(search_space, "n_layers",        [1, 2]))),
        "gene_likelihood": str(rng.choice(_as_list(search_space, "gene_likelihood", ["nb", "zinb"]))),
        "batch_size":      int(rng.choice(_as_list(search_space, "batch_size",      [256, 512]))),
    }


def _search_space_size(search_space: dict[str, Any]) -> int:
    size = 1
    for key, default in [
        ("n_latent",        [20, 30, 40]),
        ("n_hidden",        [64, 128, 256]),
        ("n_layers",        [1, 2]),
        ("gene_likelihood", ["nb", "zinb"]),
        ("batch_size",      [256, 512]),
    ]:
        size *= len(_as_list(search_space, key, default))
    return size


# ---------------------------------------------------------------------------
# Input data recovery
# ---------------------------------------------------------------------------

def _ensure_input_h5ad(
    input_h5ad: Path,
    counts_layer: str,
    logger: logging.Logger,
) -> Path:
    """If input_h5ad (combined.h5ad) is absent, rebuild it from integrated.h5ad.

    The pipeline deletes combined.h5ad after the scVI step unless
    keep_intermediates=true.  integrated.h5ad is always present and retains
    layers['counts'] with the original raw integer counts.  Reading in backed
    mode avoids loading embeddings/UMAPs into RAM during the copy; only
    layers['counts'] + obs + var are written to the new file.

    The rebuilt file is saved to the configured input_h5ad path so subsequent
    tuning runs can reuse it without re-running this step.
    """
    if input_h5ad.exists():
        return input_h5ad

    fallback = input_h5ad.parent / "scvi_output" / "integrated.h5ad"
    if not fallback.exists():
        raise FileNotFoundError(
            f"Input file not found:  {input_h5ad}\n"
            f"Fallback also missing: {fallback}\n"
            "Either set keep_intermediates: true in the pipeline config and re-run\n"
            "downsample+combine, or point input_h5ad to an existing file."
        )

    logger.info(f"{input_h5ad.name} absent — rebuilding from integrated.h5ad via backed reads")
    logger.info(f"  Source: {fallback}")
    logger.info(
        "  IMPORTANT: using layers['counts'] (raw integer counts). "
        "adata.X in integrated.h5ad is log-normalised and is NOT used for tuning."
    )

    adata_backed = sc.read_h5ad(str(fallback), backed="r")
    logger.info(f"  Backed handle: {adata_backed.n_obs:,} cells × {adata_backed.n_vars:,} genes")

    if counts_layer not in adata_backed.layers:
        adata_backed.file.close()
        raise KeyError(
            f"Layer '{counts_layer}' not found in {fallback}. "
            f"Available layers: {list(adata_backed.layers.keys())}"
        )

    logger.info(f"  Reading {counts_layer!r} layer into memory ...")
    counts = adata_backed.layers[counts_layer][:]
    if sp.issparse(counts):
        counts = counts.tocsr()

    obs = adata_backed.obs.copy()
    var = adata_backed.var.copy()
    adata_backed.file.close()

    adata_slim = ad.AnnData(obs=obs, var=var)
    adata_slim.layers[counts_layer] = counts

    logger.info(f"  Saving rebuilt combined.h5ad → {input_h5ad}")
    input_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata_slim.write_h5ad(str(input_h5ad))
    logger.info(f"  Done: {adata_slim.n_obs:,} cells × {adata_slim.n_vars:,} genes")
    del adata_slim

    return input_h5ad


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def _make_age_bins(obs: pd.DataFrame, age_key: str, edges: list[float]) -> pd.Series:
    return pd.cut(obs[age_key], bins=edges, right=False, include_lowest=True)


def _subset_for_tuning(
    adata,
    max_cells: int,
    stratify_cols: list[str],
    seed: int,
    logger: logging.Logger,
):
    if adata.n_obs <= max_cells:
        logger.info(f"No subsampling needed: {adata.n_obs:,} cells")
        return adata

    rng = np.random.default_rng(seed)
    obs = adata.obs.copy()
    group_key = obs[stratify_cols].astype(str).agg("|".join, axis=1)
    group_counts = group_key.value_counts()

    frac = max_cells / adata.n_obs
    keep_idx: list[int] = []
    for group, n in group_counts.items():
        idx = np.where(group_key.values == group)[0]
        n_take = max(1, int(round(n * frac)))
        n_take = min(n_take, idx.size)
        keep_idx.extend(rng.choice(idx, size=n_take, replace=False).tolist())

    keep_idx = np.array(sorted(set(keep_idx)), dtype=int)
    if keep_idx.size > max_cells:
        keep_idx = np.sort(rng.choice(keep_idx, size=max_cells, replace=False))

    logger.info(f"Subsampled for tuning: {adata.n_obs:,} → {keep_idx.size:,} cells (stratified on {stratify_cols})")
    return adata[keep_idx].copy()


# ---------------------------------------------------------------------------
# Batch-mixing metric
# ---------------------------------------------------------------------------

def _normalized_neighbor_batch_entropy(
    batch_codes_neighbors: np.ndarray,
    n_batches_global: int,
) -> np.ndarray:
    """Per-cell neighbourhood entropy, normalised by log(n_batches_global).

    Using the global batch count ensures scores are comparable across age bins:
    a prenatal bin with only 2 of 8 batches present is NOT artificially inflated
    to 1.0 — it is scored against log(8), reflecting the globally expected mixing.
    """
    if n_batches_global < 2:
        return np.zeros(batch_codes_neighbors.shape[0], dtype=np.float32)

    out = np.zeros(batch_codes_neighbors.shape[0], dtype=np.float32)
    denom = math.log(n_batches_global)
    for i in range(batch_codes_neighbors.shape[0]):
        counts = np.bincount(batch_codes_neighbors[i], minlength=n_batches_global)
        probs = counts[counts > 0] / counts.sum()
        ent = -np.sum(probs * np.log(probs))
        out[i] = float(ent / denom)
    return out


def _mixing_score_for_subset(
    X: np.ndarray,
    batch_codes: np.ndarray,
    k_neighbors: int,
    n_batches_global: int,
) -> float:
    n = X.shape[0]
    if n < 10:
        return float("nan")

    # Cap neighbours at 1/3 of the group so the neighbourhood is genuinely local.
    # Without this guard, a group of n=25 with k=20 uses 80% of cells as neighbours
    # and entropy just reflects bulk batch frequencies rather than local mixing.
    k_actual = min(k_neighbors, max(2, n // 3))
    if k_actual < 2:
        return float("nan")

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean")
    nn.fit(X)
    idx = nn.kneighbors(return_distance=False)[:, 1:]  # drop self
    neigh_batch_codes = batch_codes[idx]
    ent = _normalized_neighbor_batch_entropy(neigh_batch_codes, n_batches_global)
    return float(np.mean(ent))


def _per_batch_mixing_scores(
    X: np.ndarray,
    batch_labels: np.ndarray,
    k_neighbors: int,
    n_batches_global: int,
    max_total_cells: int = 30000,
    seed: int = 99,
) -> dict[str, float]:
    """Global k-NN batch mixing: for each batch, mean normalised entropy of its cells' neighbourhoods.

    Unlike the age-bin metric (per-bin k-NN), this uses a single global k-NN so a batch
    that is globally isolated in latent space gets a low score regardless of age bin.
    Helps identify which specific batches (e.g. Wang) are poorly integrated.
    """
    rng = np.random.default_rng(seed)
    n_all = X.shape[0]

    if n_all > max_total_cells:
        keep = rng.choice(n_all, max_total_cells, replace=False)
        X_sub = X[keep]
        batch_sub = batch_labels[keep]
    else:
        X_sub = X
        batch_sub = batch_labels

    k_actual = min(k_neighbors + 1, X_sub.shape[0] - 1)
    nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm="ball_tree", n_jobs=1).fit(X_sub)
    _, inds = nbrs.kneighbors(X_sub)

    all_batches = np.unique(batch_labels)
    batch_to_code = {b: i for i, b in enumerate(all_batches)}
    batch_codes_sub = np.array([batch_to_code[b] for b in batch_sub], dtype=int)

    # Exclude self (inds[:, 0]) → shape (n_sub, k_neighbors)
    neighbor_codes = batch_codes_sub[inds[:, 1:]]
    per_cell_ent = _normalized_neighbor_batch_entropy(neighbor_codes, n_batches_global)

    scores: dict[str, float] = {}
    for batch in all_batches:
        mask = (batch_sub == batch)
        if mask.sum() == 0:
            continue
        scores[str(batch)] = float(np.mean(per_cell_ent[mask]))
    return scores


def _evaluate_age_aware_batch_score(
    adata,
    latent_key: str,
    batch_key: str,
    age_key: str,
    age_bin_edges: list[float],
    prenatal_weight: float,
    cell_type_key: str | None,
    k_neighbors: int,
    min_cells_per_bin: int,
    min_cells_per_group: int,
    n_batches_global: int,
) -> tuple[float, dict[str, Any]]:
    obs = adata.obs.copy()
    X = np.asarray(adata.obsm[latent_key])

    age_bins = _make_age_bins(obs, age_key=age_key, edges=age_bin_edges)
    obs = obs.assign(_age_bin=age_bins)

    weighted_scores: list[tuple[float, float]] = []
    details: dict[str, Any] = {}

    for age_bin, idx in obs.groupby("_age_bin", observed=True).indices.items():
        if pd.isna(age_bin):
            continue
        idx = np.asarray(idx, dtype=int)
        if idx.size < min_cells_per_bin:
            continue

        age_left_edge = float(age_bin.left)
        bin_weight = float(prenatal_weight if age_left_edge < 0 else 1.0)
        obs_bin = obs.iloc[idx]

        group_scores: list[tuple[float, int]] = []
        if cell_type_key and cell_type_key in obs_bin.columns:
            for _, idx_rel in obs_bin.groupby(cell_type_key, observed=True).indices.items():
                idx_rel = np.asarray(idx_rel, dtype=int)
                idx_ct = idx[idx_rel]
                if idx_ct.size < min_cells_per_group:
                    continue
                batch_codes = pd.factorize(obs.iloc[idx_ct][batch_key], sort=True)[0].astype(int)
                score_ct = _mixing_score_for_subset(X[idx_ct], batch_codes, k_neighbors, n_batches_global)
                if np.isfinite(score_ct):
                    group_scores.append((score_ct, idx_ct.size))

        n_batches_in_bin = int(obs_bin[batch_key].nunique())
        bin_label = f"[{age_bin.left},{age_bin.right})"
        if group_scores:
            numer = sum(s * w for s, w in group_scores)
            denom = sum(w for _, w in group_scores)
            score_bin = float(numer / max(denom, 1))
            details[f"{bin_label}_mode"] = "celltype-weighted"
        else:
            batch_codes = pd.factorize(obs_bin[batch_key], sort=True)[0].astype(int)
            score_bin = _mixing_score_for_subset(X[idx], batch_codes, k_neighbors, n_batches_global)
            details[f"{bin_label}_mode"] = "bin"

        if np.isfinite(score_bin):
            details[bin_label] = score_bin
            weighted_scores.append((score_bin, bin_weight))
            # Structural ceiling: max normalised entropy for this bin =
            # log(n_batches_in_bin) / log(n_batches_global).  Prenatal bins
            # with fewer present batches have a lower ceiling — reporting
            # score as % of max distinguishes poor mixing from structural cap.
            details[f"{bin_label}_n_batches"] = n_batches_in_bin
            if n_batches_in_bin >= 2 and n_batches_global >= 2:
                max_achievable = math.log(n_batches_in_bin) / math.log(n_batches_global)
                details[f"{bin_label}_max_score"] = round(max_achievable, 4)
                details[f"{bin_label}_pct_of_max"] = round(score_bin / max_achievable * 100, 1)
            else:
                details[f"{bin_label}_max_score"] = float("nan")
                details[f"{bin_label}_pct_of_max"] = float("nan")

    if not weighted_scores:
        return 0.0, details

    numer = sum(score * w for score, w in weighted_scores)
    denom = sum(w for _, w in weighted_scores)
    return float(numer / max(denom, 1.0)), details


# ---------------------------------------------------------------------------
# Trial evaluation: expression-based metric
# ---------------------------------------------------------------------------

def _get_normalized_expr_chunked(
    model,
    adata,
    transform_batch: str | None,
    n_samples: int = 1,
    chunk_size: int = 2000,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """Get normalized expression for all cells in `adata`, processing in chunks.

    Returns a (n_cells, n_genes) float32 array.  Chunking avoids OOM when
    the full expression matrix is larger than available GPU/CPU memory.
    """
    n_cells = adata.n_obs
    n_chunks = math.ceil(n_cells / chunk_size)
    chunks: list[np.ndarray] = []

    for i, start in enumerate(range(0, n_cells, chunk_size)):
        end = min(start + chunk_size, n_cells)
        chunk_adata = adata[start:end].copy()
        kwargs: dict[str, Any] = {"n_samples": n_samples}
        if transform_batch is not None:
            kwargs["transform_batch"] = transform_batch
        result = model.get_normalized_expression(adata=chunk_adata, **kwargs)
        chunks.append(result.values.astype(np.float32))
        if logger and (i + 1) % max(1, n_chunks // 5) == 0:
            logger.info(f"    expr chunk {i + 1}/{n_chunks}: [{start}:{end}) done")

    return np.vstack(chunks)


def _evaluate_trial_components(
    model,
    adata,
    batch_key: str,
    age_key: str,
    cell_type_key: str | None,
    metric_cfg: dict[str, Any],
    n_batches_global: int,
    logger: logging.Logger,
    npy_prefix: "Path | None" = None,
) -> dict[str, Any]:
    """Evaluate a trial with the expression-based metric.

    Returns raw metric components; the combined objective is computed post-hoc
    by _normalize_trial_objectives() so both components are on the same scale.

    Components:
    - batch_mixing_pca_score : age-binned batch mixing on PCA of batch-corrected
                               normalized expression (transform_batch applied).
                               Higher is better; already in [0, 1].
    - recon_error            : mean reconstruction error on the full tuning set
                               without transform_batch.  Lower is better.
    - age_bin_scores         : per-bin batch mixing details dict.
    - batch_scores           : per-batch global mixing scores dict.
    - pca_explained_variance : cumulative variance explained by the PCA (info only).
    """
    transform_batch = metric_cfg.get("transform_batch")
    if not transform_batch:
        raise ValueError(
            "metric.transform_batch is required for the expression-based metric"
        )

    n_pca        = int(metric_cfg.get("n_pca_components",     50))
    n_samples    = int(metric_cfg.get("n_normalized_samples",  1))
    max_cells    = int(metric_cfg.get("max_cells_expr",    20000))
    chunk_size   = int(metric_cfg.get("expr_chunk_size",    2000))
    subsamp_seed = int(metric_cfg.get("expr_subsample_seed",  42))

    # -----------------------------------------------------------------------
    # Component 1: batch mixing on PCA of normalized expression
    # -----------------------------------------------------------------------
    n_cells = adata.n_obs
    if n_cells > max_cells:
        # Stratified subsample so all batches and age bins are represented.
        rng = np.random.default_rng(subsamp_seed)
        obs = adata.obs.copy()
        age_bins = _make_age_bins(obs, age_key, metric_cfg["age_bin_edges"])
        obs["_expr_age_bin"] = age_bins.astype(str)
        group_key = obs[[batch_key, "_expr_age_bin"]].astype(str).agg("|".join, axis=1)
        group_counts = group_key.value_counts()
        frac = max_cells / n_cells
        keep_idx: list[int] = []
        for group, n in group_counts.items():
            idx = np.where(group_key.values == group)[0]
            n_take = max(1, int(round(n * frac)))
            n_take = min(n_take, idx.size)
            keep_idx.extend(rng.choice(idx, size=n_take, replace=False).tolist())
        keep_idx_arr = np.array(sorted(set(keep_idx)), dtype=int)
        if keep_idx_arr.size > max_cells:
            keep_idx_arr = np.sort(rng.choice(keep_idx_arr, size=max_cells, replace=False))
        adata_expr = adata[keep_idx_arr].copy()
        logger.info(
            f"  Expression PCA subsample: {n_cells:,} → {adata_expr.n_obs:,} cells"
        )
    else:
        adata_expr = adata

    logger.info(
        f"  Computing normalized expression "
        f"(transform_batch={transform_batch!r}, n_samples={n_samples}, "
        f"chunk_size={chunk_size}) ..."
    )
    expr = _get_normalized_expr_chunked(
        model=model,
        adata=adata_expr,
        transform_batch=transform_batch,
        n_samples=n_samples,
        chunk_size=chunk_size,
        logger=logger,
    )
    logger.info(f"  Normalized expression shape: {expr.shape}")

    # Fit PCA on the normalized expression.
    n_components = min(n_pca, expr.shape[1], expr.shape[0] - 1)
    logger.info(f"  Fitting PCA (n_components={n_components}) ...")
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(expr).astype(np.float32)
    explained_var = float(np.sum(pca.explained_variance_ratio_))
    logger.info(f"  PCA explained variance (first {n_components} PCs): {explained_var:.3f}")
    del expr  # free memory before batch mixing

    # Save expression PCA + cell indices for UMAP diagnostics.
    if npy_prefix is not None:
        cell_idx = keep_idx_arr if n_cells > max_cells else np.arange(n_cells, dtype=np.int32)
        np.save(str(npy_prefix) + "_expr_pca.npy",     X_pca)
        np.save(str(npy_prefix) + "_expr_pca_idx.npy", cell_idx.astype(np.int32))

    # Age-binned batch mixing on the PCA embedding.
    latent_key_pca = "_X_expr_pca_tmp"
    adata_expr = adata_expr.copy()
    adata_expr.obsm[latent_key_pca] = X_pca

    age_score, age_details = _evaluate_age_aware_batch_score(
        adata=adata_expr,
        latent_key=latent_key_pca,
        batch_key=batch_key,
        age_key=age_key,
        age_bin_edges=metric_cfg["age_bin_edges"],
        prenatal_weight=float(metric_cfg.get("prenatal_weight", 2.0)),
        cell_type_key=cell_type_key,
        k_neighbors=int(metric_cfg.get("k_neighbors", 20)),
        min_cells_per_bin=int(metric_cfg.get("min_cells_per_bin", 200)),
        min_cells_per_group=int(metric_cfg.get("min_cells_per_group", 75)),
        n_batches_global=n_batches_global,
    )

    # Per-batch global mixing scores (single global k-NN on expression PCA).
    batch_labels_expr = np.asarray(adata_expr.obs[batch_key].astype(str))
    batch_scores = _per_batch_mixing_scores(
        X_pca, batch_labels_expr,
        k_neighbors=int(metric_cfg.get("k_neighbors", 20)),
        n_batches_global=n_batches_global,
    )

    # -----------------------------------------------------------------------
    # Component 2: decoder reconstruction error (no transform_batch)
    # -----------------------------------------------------------------------
    logger.info("  Computing reconstruction error (no transform_batch) ...")
    recon_raw = model.get_reconstruction_error()
    if isinstance(recon_raw, dict):
        # scvi-tools ≥1.x may return a dict; prefer "reconstruction_loss" key
        recon_error = float(
            recon_raw.get("reconstruction_loss", next(iter(recon_raw.values())))
        )
    else:
        recon_error = float(recon_raw)
    logger.info(f"  Reconstruction error: {recon_error:.4f}")

    return {
        "batch_mixing_pca_score": float(age_score),
        "recon_error":            float(recon_error),
        "age_bin_scores":         age_details,
        "batch_scores":           batch_scores,
        "pca_explained_variance": float(explained_var),
    }


def _normalize_trial_objectives(
    results: list[dict[str, Any]],
) -> None:
    """Compute 50:50 cross-trial normalized objectives in place.

    Both components are min-max normalized across all successful trials so
    they contribute equally to the final objective regardless of absolute scale:

        bm_pca_norm   = (bm - bm_min) / (bm_max - bm_min)
        decoder_norm  = 1 - (recon - recon_min) / (recon_max - recon_min)
        objective     = 0.5 * bm_pca_norm + 0.5 * decoder_norm

    When all trials share the same value for a component (range ≈ 0), that
    component contributes a neutral 0.5 to every trial so ranking is driven
    entirely by the other component.
    """
    valid = [
        r for r in results
        if r["status"] == "ok"
        and np.isfinite(r.get("batch_mixing_pca_score", float("nan")))
        and np.isfinite(r.get("recon_error", float("nan")))
    ]

    for r in results:
        r.setdefault("batch_mixing_pca_norm", float("nan"))
        r.setdefault("decoder_score_norm",    float("nan"))
        r.setdefault("objective",             float("-inf"))

    if not valid:
        return

    bm_vals = [r["batch_mixing_pca_score"] for r in valid]
    re_vals = [r["recon_error"]             for r in valid]

    bm_min, bm_max = min(bm_vals), max(bm_vals)
    re_min, re_max = min(re_vals), max(re_vals)
    bm_range = bm_max - bm_min
    re_range = re_max - re_min

    for r in valid:
        bm_norm = (
            (r["batch_mixing_pca_score"] - bm_min) / bm_range
            if bm_range > 1e-8 else 0.5
        )
        dec_norm = (
            1.0 - (r["recon_error"] - re_min) / re_range
            if re_range > 1e-8 else 0.5
        )
        r["batch_mixing_pca_norm"] = float(bm_norm)
        r["decoder_score_norm"]    = float(dec_norm)
        r["objective"]             = float(0.5 * bm_norm + 0.5 * dec_norm)


def _compute_objective_with_norms(
    batch_mixing_pca_score: float,
    recon_error: float,
    bm_min: float,
    bm_max: float,
    re_min: float,
    re_max: float,
) -> tuple[float, float, float]:
    """Compute objective for an out-of-sample model using scVI normalization constants.

    Used to put scANVI objectives on the same scale as the scVI trial objectives so
    the scVI→scANVI delta is directly interpretable.  The scANVI values are clamped
    to [0, 1] before combining in case they fall outside the scVI range.
    """
    bm_range = bm_max - bm_min
    re_range = re_max - re_min

    bm_norm = (
        (batch_mixing_pca_score - bm_min) / bm_range if bm_range > 1e-8 else 0.5
    )
    dec_norm = (
        1.0 - (recon_error - re_min) / re_range if re_range > 1e-8 else 0.5
    )
    bm_norm  = max(0.0, min(1.0, bm_norm))
    dec_norm = max(0.0, min(1.0, dec_norm))
    return float(0.5 * bm_norm + 0.5 * dec_norm), float(bm_norm), float(dec_norm)


# ---------------------------------------------------------------------------
# scANVI top-trial evaluation
# ---------------------------------------------------------------------------

def _train_and_evaluate_scanvi(
    adata,
    scvi_model_dir: Path,
    cell_type_key: str,
    latent_key_scanvi: str,
    batch_key: str,
    age_key: str,
    metric_cfg: dict[str, Any],
    n_batches_global: int,
    max_epochs_scanvi: int,
    accel: dict[str, Any],
    logger: logging.Logger,
    norm_constants: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Load a saved scVI model, initialise scANVI from it, train, and evaluate.

    SCANVI.from_scvi_model transfers all learned scVI weights and then fine-tunes
    with cell-type supervision, which is exactly what production does.  Evaluating
    the top scVI trials with scANVI reveals whether scANVI changes the relative
    ranking and quantifies the delta in batch-mixing quality.

    The expression-based metric (PCA of normalized expression + reconstruction error)
    is used so the scANVI objective is on the same scale as the scVI trial objectives.
    If norm_constants is provided (bm_min, bm_max, re_min, re_max from the scVI
    cross-trial normalization), the scANVI objective is computed on the same scale.
    """
    scvi_model = SCVI.load(str(scvi_model_dir), adata=adata)

    # Ensure labels column has no NaN (scANVI requires every cell to have a label
    # or the explicit unlabeled_category string).
    if cell_type_key not in adata.obs.columns:
        logger.warning(f"  '{cell_type_key}' not in obs; all cells treated as unlabeled for scANVI")
        adata.obs[cell_type_key] = "Unknown"
    else:
        adata.obs[cell_type_key] = adata.obs[cell_type_key].astype(object).fillna("Unknown").astype(str)

    scanvi_model = SCANVI.from_scvi_model(
        scvi_model,
        labels_key=cell_type_key,
        unlabeled_category="Unknown",
    )
    scanvi_model.train(
        max_epochs=max_epochs_scanvi,
        early_stopping=True,
        early_stopping_patience=5,
        enable_progress_bar=False,
        **accel,
    )

    # Keep latent representation for potential diagnostics.
    adata.obsm[latent_key_scanvi] = scanvi_model.get_latent_representation()

    components = _evaluate_trial_components(
        model=scanvi_model,
        adata=adata,
        batch_key=batch_key,
        age_key=age_key,
        cell_type_key=cell_type_key,
        metric_cfg=metric_cfg,
        n_batches_global=n_batches_global,
        logger=logger,
    )

    if norm_constants is not None:
        obj, bm_norm, dec_norm = _compute_objective_with_norms(
            batch_mixing_pca_score=components["batch_mixing_pca_score"],
            recon_error=components["recon_error"],
            **norm_constants,
        )
        components["objective"]             = obj
        components["batch_mixing_pca_norm"] = bm_norm
        components["decoder_score_norm"]    = dec_norm
    else:
        components["objective"]             = float("nan")
        components["batch_mixing_pca_norm"] = float("nan")
        components["decoder_score_norm"]    = float("nan")

    return components


# ---------------------------------------------------------------------------
# Budget advisory
# ---------------------------------------------------------------------------

def _budget_advisory(
    results: list[dict[str, Any]],
    n_trials_requested: int,
    search_space_size: int,
    elapsed_hours: float,
    logger: logging.Logger,
) -> None:
    """Log coverage statistics and recommendations for whether a longer run is warranted."""
    n_done = len(results)
    n_ok = sum(r["status"] == "ok" for r in results)
    durations = [r["duration_min"] for r in results if r["status"] == "ok" and r["duration_min"] > 0]

    if not durations:
        logger.warning("No successful trials — budget advisory unavailable.")
        return

    avg_min = float(np.mean(durations))
    std_min = float(np.std(durations)) if len(durations) > 1 else 0.0
    coverage_pct = n_ok / search_space_size * 100
    remaining = max(0, search_space_size - n_ok)
    hours_for_full = remaining * avg_min / 60.0

    logger.info("=" * 80)
    logger.info("BUDGET ADVISORY")
    logger.info(f"  Trials completed / requested : {n_done} / {n_trials_requested}  ({n_ok} succeeded)")
    logger.info(f"  Elapsed                      : {elapsed_hours:.2f} h")
    logger.info(f"  Mean trial duration          : {avg_min:.0f} ± {std_min:.0f} min")
    logger.info(f"  Search space combinations    : {search_space_size}")
    logger.info(f"  Coverage                     : {n_ok} / {search_space_size} = {coverage_pct:.1f}%")
    if coverage_pct < 5:
        logger.info(
            f"  RECOMMENDATION: Coverage is very low ({coverage_pct:.1f}%). "
            f"Exhaustive search would need ~{hours_for_full:.0f} more GPU-hours. "
            "Consider switching to scvi.autotune.run_autotune (Ray Tune + ASHA) for "
            "much better sample efficiency — see the scvi-tools autotune tutorial."
        )
        days = hours_for_full / 12.0
        logger.info(f"  Exhaustive search ≈ {days:.1f} overnight GPU runs at current trial speed.")
    elif coverage_pct < 20:
        logger.info(
            f"  RECOMMENDATION: Coverage is {coverage_pct:.1f}%. "
            f"A follow-up run (~{hours_for_full:.0f} GPU-hours for exhaustive coverage) "
            "would substantially improve confidence. "
            "Consider focusing on the sub-space around the top trials."
        )
    else:
        logger.info(f"  Coverage is {coverage_pct:.1f}% — reasonable for a random search.")
    logger.info("=" * 80)


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

def run_tuning(config_path: Path):
    cfg = _load_yaml(config_path)

    input_h5ad = Path(cfg["input_h5ad"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(output_dir / "tuning.log")
    logger.info(f"Config:     {config_path}")
    logger.info(f"Output dir: {output_dir}")

    batch_key      = cfg.get("batch_key",      "source-chemistry")
    age_key        = cfg.get("age_key",         "age_years")
    cell_type_key  = cfg.get("cell_type_key",   "cell_class")
    counts_layer   = cfg.get("counts_layer",    "counts")

    tuning_cfg            = cfg.get("tuning", {})
    random_seed           = int(tuning_cfg.get("random_seed",   42))
    n_trials              = int(tuning_cfg.get("n_trials",      14))
    max_hours             = float(tuning_cfg.get("max_hours",   12.0))
    max_cells             = int(tuning_cfg.get("max_cells",     120000))
    scanvi_top_k          = int(tuning_cfg.get("scanvi_top_k",  3))

    training_cfg          = cfg.get("training", {})
    max_epochs            = int(training_cfg.get("max_epochs_scvi",           400))
    early_stopping        = bool(training_cfg.get("early_stopping",           True))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 20))
    train_size            = float(training_cfg.get("train_size",              0.9))
    max_epochs_scanvi     = int(training_cfg.get("max_epochs_scanvi",         50))

    hvg_cfg         = cfg.get("hvg", {})
    n_top_genes     = int(hvg_cfg.get("n_top_genes",  10000))
    hvg_flavor      = hvg_cfg.get("hvg_flavor",       "seurat_v3")
    hvg_batch_key   = hvg_cfg.get("hvg_batch_key",    batch_key)

    metric_cfg = cfg.get("metric", {})
    if "age_bin_edges" not in metric_cfg:
        raise ValueError("metric.age_bin_edges is required in the config")
    if "transform_batch" not in metric_cfg:
        raise ValueError(
            "metric.transform_batch is required (the reference batch for normalized expression, "
            "e.g. 'WANG-multiome').  Set it in the metric: section of your tuning YAML config."
        )
    metric_cfg.setdefault("expr_subsample_seed", random_seed + EXPRESSION_PCA_SEED_OFFSET)

    search_space = cfg.get("search_space", {})

    scvi.settings.seed = random_seed
    scvi.settings.num_threads = int(tuning_cfg.get("num_workers", 4))

    # -----------------------------------------------------------------------
    # Input: rebuild combined.h5ad from integrated.h5ad if needed
    # -----------------------------------------------------------------------
    input_h5ad = _ensure_input_h5ad(input_h5ad, counts_layer, logger)
    logger.info(f"Input: {input_h5ad}")

    logger.info("Loading AnnData ...")
    adata = sc.read_h5ad(str(input_h5ad))

    # Log prenatal age distribution by batch — helps calibrate bin edges.
    # Prenatal cells from Wang tend to cluster at specific gestational weeks;
    # this output shows whether bin edges need to be moved to co-locate Wang
    # and Velmeshev cells at similar ages.
    prenatal_mask = adata.obs[age_key] < 0 if age_key in adata.obs.columns else pd.Series(False, index=adata.obs.index)
    if prenatal_mask.sum() > 0 and batch_key in adata.obs.columns:
        logger.info(f"Prenatal cell age distribution by batch ({prenatal_mask.sum():,} cells total):")
        prenatal_obs = adata.obs[prenatal_mask]
        for batch_name, grp in prenatal_obs.groupby(batch_key, observed=True):
            ages = grp[age_key].dropna()
            if len(ages) == 0:
                continue
            logger.info(
                f"  {str(batch_name):40s}  n={len(ages):5d}  "
                f"age [{ages.min():.3f}, {ages.max():.3f}]  "
                f"median={ages.median():.3f}  "
                f"quartiles=({ages.quantile(0.25):.3f}, {ages.quantile(0.75):.3f})"
            )

    # -----------------------------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------------------------
    missing = []
    for key in [batch_key, age_key]:
        if key not in adata.obs.columns:
            missing.append(key)
    for key in list(cfg.get("continuous_covariate_keys") or []) + list(cfg.get("categorical_covariate_keys") or []):
        if key not in adata.obs.columns:
            missing.append(key)
    if missing:
        raise ValueError(f"Required obs columns missing: {missing}. Available: {list(adata.obs.columns)}")

    # Validate that transform_batch is a known batch value (warn, not error, to allow
    # cases where a batch only appears in some age ranges or is spelled differently).
    transform_batch_ref = metric_cfg["transform_batch"]
    if batch_key in adata.obs.columns:
        known_batches = sorted(adata.obs[batch_key].astype(str).unique().tolist())
        if transform_batch_ref not in known_batches:
            logger.warning(
                f"metric.transform_batch='{transform_batch_ref}' not found in "
                f"obs['{batch_key}'].  Available values: {known_batches}.  "
                "get_normalized_expression will raise at evaluation time if this is wrong."
            )

    if cell_type_key not in adata.obs.columns:
        logger.warning(
            f"cell_type_key='{cell_type_key}' not in obs; "
            "within-cell-type batch mixing will fall back to per-bin mixing and "
            "scANVI evaluation will be skipped."
        )
        cell_type_key = None

    if counts_layer not in adata.layers:
        logger.info(f"'{counts_layer}' layer missing; copying .X as counts")
        adata.layers[counts_layer] = adata.X.copy()

    # -----------------------------------------------------------------------
    # HVG selection (once, before all trials)
    # -----------------------------------------------------------------------
    logger.info("Selecting HVGs (once before trials) ...")
    if hvg_flavor == "pearson_residuals":
        if hvg_batch_key:
            logger.info(
                f"  hvg_batch_key='{hvg_batch_key}' is ignored for pearson_residuals "
                "(not supported by sc.experimental.pp.highly_variable_genes)"
            )
        sc.experimental.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, layer=counts_layer)
    else:
        kwargs: dict[str, Any] = {"n_top_genes": n_top_genes, "flavor": hvg_flavor}
        if hvg_flavor == "seurat_v3":
            kwargs["layer"] = counts_layer
        if hvg_batch_key:
            kwargs["batch_key"] = hvg_batch_key
        sc.pp.highly_variable_genes(adata, **kwargs)

    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info(f"HVG AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # -----------------------------------------------------------------------
    # Stratified subsampling
    # -----------------------------------------------------------------------
    age_bins_full = _make_age_bins(adata.obs, age_key, metric_cfg["age_bin_edges"])
    adata.obs["_tune_age_bin"] = age_bins_full.astype(str)
    adata = _subset_for_tuning(
        adata,
        max_cells=max_cells,
        stratify_cols=[batch_key, "_tune_age_bin"],
        seed=random_seed,
        logger=logger,
    )

    n_batches_global = int(adata.obs[batch_key].nunique())
    logger.info(f"Global batch count (for entropy normalisation): {n_batches_global}")

    # Save obs metadata and key names once for diagnostics UMAP.
    # Per-trial latent arrays are saved inside the trial loop below.
    adata.obs[[batch_key, age_key]].to_csv(output_dir / "obs_tuning.csv")
    with open(output_dir / "tuning_metadata.json", "w") as _mf:
        json.dump({"batch_key": batch_key, "age_key": age_key,
                   "cell_type_key": cell_type_key or ""}, _mf)
    logger.info(f"Saved obs metadata ({adata.n_obs:,} cells) → obs_tuning.csv + tuning_metadata.json")

    # -----------------------------------------------------------------------
    # scVI setup (once; reused across all trials and scANVI evaluation)
    # -----------------------------------------------------------------------
    covariate_kwargs: dict[str, Any] = {}
    if cfg.get("continuous_covariate_keys"):
        covariate_kwargs["continuous_covariate_keys"] = list(cfg["continuous_covariate_keys"])
    if cfg.get("categorical_covariate_keys"):
        covariate_kwargs["categorical_covariate_keys"] = list(cfg["categorical_covariate_keys"])
    logger.info(f"Setting up AnnData for scVI (covariates: {covariate_kwargs}) ...")
    SCVI.setup_anndata(adata, layer=counts_layer, batch_key=batch_key, **covariate_kwargs)

    accel: dict[str, Any] = (
        {"accelerator": "gpu", "devices": 1}
        if torch.cuda.is_available()
        else {"accelerator": "cpu", "devices": 1}
    )
    logger.info(f"Trainer accelerator: {accel}")

    space_size = _search_space_size(search_space)
    logger.info(
        f"Search space: {space_size} combinations "
        f"(n_latent × n_hidden × n_layers × gene_likelihood × batch_size). "
        f"Running up to {n_trials} random trials."
    )
    logger.info(
        f"Using production epoch horizon: max_epochs={max_epochs}, "
        f"early_stopping_patience={early_stopping_patience}. "
        f"(Optimal architecture at shorter horizons can differ from production.)"
    )

    # -----------------------------------------------------------------------
    # Trial loop
    # -----------------------------------------------------------------------
    results: list[dict[str, Any]] = []

    # During the loop we track top models by batch_mixing_pca_score (a preliminary
    # rank that is already in [0, 1]).  We save a double-size buffer so that
    # post-loop cross-trial normalization can reshuffle the final top-k without
    # losing any candidate model.
    max_saved_k = max(scanvi_top_k * 2, scanvi_top_k + 3) if scanvi_top_k > 0 else 0
    # top_model_records: sorted ascending by prelim_score so index-0 is the worst kept.
    top_model_records: list[dict[str, Any]] = []

    start = time.time()
    rng = np.random.default_rng(random_seed)

    for trial in range(1, n_trials + 1):
        elapsed_h = (time.time() - start) / 3600.0
        if elapsed_h >= max_hours:
            logger.info(f"Reached max_hours={max_hours}; stopping before trial {trial}")
            break

        params = _sample_trial_params(search_space, rng)
        scvi.settings.seed = random_seed + trial

        logger.info("-" * 80)
        logger.info(f"Trial {trial}/{n_trials} | elapsed={elapsed_h:.2f}h | params={params}")

        trial_t0 = time.time()
        model = None
        try:
            model = SCVI(
                adata,
                n_latent=params["n_latent"],
                n_hidden=params["n_hidden"],
                n_layers=params["n_layers"],
                gene_likelihood=params["gene_likelihood"],
            )
            train_kwargs: dict[str, Any] = {
                "max_epochs":       max_epochs,
                "early_stopping":   early_stopping,
                "train_size":       train_size,
                "validation_size":  max(0.0, 1.0 - train_size),
                "batch_size":       params["batch_size"],
                "enable_progress_bar": True,
                **accel,
            }
            if early_stopping:
                train_kwargs["early_stopping_patience"] = early_stopping_patience
            model.train(**train_kwargs)

            # Save latent representation for the UMAP diagnostic step.
            adata.obsm["X_scVI"] = model.get_latent_representation()
            np.save(str(output_dir / f"trial_{trial:02d}_latent.npy"),
                    adata.obsm["X_scVI"].astype(np.float32))

            # Expression-based metric evaluation (the objective components are
            # stored as raw values; the combined objective is computed post-hoc
            # once all trials are done via cross-trial normalization).
            components = _evaluate_trial_components(
                model=model,
                adata=adata,
                batch_key=batch_key,
                age_key=age_key,
                cell_type_key=cell_type_key,
                metric_cfg=metric_cfg,
                n_batches_global=n_batches_global,
                logger=logger,
                npy_prefix=output_dir / f"trial_{trial:02d}",
            )
            status = "ok"
            error = ""
        except KeyboardInterrupt:
            raise
        except (RuntimeError, FloatingPointError, torch.cuda.OutOfMemoryError) as e:
            status = "failed"
            error = repr(e)
            components = {
                "batch_mixing_pca_score": float("nan"),
                "recon_error":            float("nan"),
                "age_bin_scores":         {},
                "batch_scores":           {},
                "pca_explained_variance": float("nan"),
            }
            logger.exception(f"Trial {trial} failed")

        duration_min = (time.time() - trial_t0) / 60.0
        row: dict[str, Any] = {
            "trial":                    trial,
            "status":                   status,
            "error":                    error,
            "duration_min":             round(duration_min, 2),
            **params,
            # Raw metric components (objective computed post-hoc)
            "batch_mixing_pca_score":   components["batch_mixing_pca_score"],
            "recon_error":              components["recon_error"],
            "pca_explained_variance":   components.get("pca_explained_variance", float("nan")),
            "age_bin_scores_json":      json.dumps(components["age_bin_scores"], sort_keys=True),
            "batch_scores_json":        json.dumps(components.get("batch_scores", {}), sort_keys=True),
            # Placeholders filled by _normalize_trial_objectives after the loop
            "objective":                float("nan"),
            "batch_mixing_pca_norm":    float("nan"),
            "decoder_score_norm":       float("nan"),
        }
        results.append(row)

        logger.info(
            f"Trial {trial} done: "
            f"bm_pca={row['batch_mixing_pca_score']:.4f}  "
            f"recon={row['recon_error']:.4f}  "
            f"({duration_min:.1f} min)"
        )

        # Save model if it enters the preliminary top-k (by batch_mixing_pca_score).
        # The double-buffer (max_saved_k = 2 × scanvi_top_k) ensures that post-loop
        # normalization can reshuffle the final ranking without losing candidate models.
        if status == "ok" and model is not None and max_saved_k > 0:
            prelim_score = row["batch_mixing_pca_score"]
            enters_top_k = (
                np.isfinite(prelim_score)
                and (
                    len(top_model_records) < max_saved_k
                    or prelim_score > top_model_records[0].get("prelim_score", float("-inf"))
                )
            )
            if enters_top_k:
                model_dir = output_dir / f"trial_{trial:02d}_scvi_model"
                model.save(str(model_dir), overwrite=True)
                top_model_records.append({
                    "trial":       trial,
                    "prelim_score": prelim_score,
                    "objective":   float("nan"),  # updated after normalization
                    "model_dir":   model_dir,
                    "params":      params,
                })
                # Keep ascending by prelim_score: index 0 is the worst kept model.
                top_model_records.sort(key=lambda x: x.get("prelim_score", float("-inf")))
                if len(top_model_records) > max_saved_k:
                    evicted = top_model_records.pop(0)
                    shutil.rmtree(evicted["model_dir"], ignore_errors=True)
                    logger.info(
                        f"  Evicted trial {evicted['trial']} model from "
                        f"top-{max_saved_k} buffer (prelim_score={evicted['prelim_score']:.4f})"
                    )

        # Checkpoint after every trial so a walltime kill doesn't lose results.
        pd.DataFrame(results).to_csv(output_dir / "trial_results.csv", index=False)

    elapsed_total_h = (time.time() - start) / 3600.0

    if not results:
        raise RuntimeError("No trials were executed.")

    # -----------------------------------------------------------------------
    # Post-loop: cross-trial normalization → final combined objectives
    # -----------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Computing cross-trial normalized objectives (50:50 bm_pca + decoder) ...")
    _normalize_trial_objectives(results)

    best = max(
        (r for r in results if r["status"] == "ok" and np.isfinite(r.get("objective", float("nan")))),
        key=lambda r: r["objective"],
        default=None,
    )
    if best is None:
        raise RuntimeError("All trials failed; no best hyperparameters available.")

    logger.info(
        f"Best trial: {best['trial']}  objective={best['objective']:.4f}  "
        f"bm_pca={best['batch_mixing_pca_score']:.4f} (norm={best['batch_mixing_pca_norm']:.4f})  "
        f"recon={best['recon_error']:.4f} (dec_norm={best['decoder_score_norm']:.4f})"
    )

    # Update final objectives in top_model_records and trim to the true top-k.
    trial_obj = {r["trial"]: r.get("objective", float("-inf")) for r in results}
    for rec in top_model_records:
        rec["objective"] = trial_obj.get(rec["trial"], float("-inf"))
    top_model_records.sort(key=lambda x: x.get("objective", float("-inf")))  # ascending

    while len(top_model_records) > scanvi_top_k:
        evicted = top_model_records.pop(0)  # remove the worst
        shutil.rmtree(evicted["model_dir"], ignore_errors=True)
        logger.info(
            f"  Post-normalization: evicted trial {evicted['trial']} "
            f"(objective={evicted['objective']:.4f}, not in final top-{scanvi_top_k})"
        )

    # Save final results (sorted best → worst by combined objective).
    df = pd.DataFrame(results).sort_values("objective", ascending=False)
    df.to_csv(output_dir / "trial_results.csv", index=False)

    # Build best_hyperparameters.yaml with keys that map directly to scVI/config.py fields.
    best_params: dict[str, Any] = {
        "scvi": {
            "n_latent":                int(best["n_latent"]),
            "n_hidden":                int(best["n_hidden"]),
            "n_layers":                int(best["n_layers"]),
            "gene_likelihood":         str(best["gene_likelihood"]),
            "batch_size":              int(best["batch_size"]),
            "max_epochs_scvi":         max_epochs,
            "early_stopping":          early_stopping,
            "early_stopping_patience": early_stopping_patience,
        },
        "best_metrics": {
            "objective":               float(best["objective"]),
            "batch_mixing_pca_score":  float(best["batch_mixing_pca_score"]),
            "batch_mixing_pca_norm":   float(best["batch_mixing_pca_norm"]),
            "recon_error":             float(best["recon_error"]),
            "decoder_score_norm":      float(best["decoder_score_norm"]),
            "age_bin_scores":          json.loads(best["age_bin_scores_json"]),
        },
    }

    with open(output_dir / "best_hyperparameters.yaml", "w") as f:
        yaml.safe_dump(best_params, f, sort_keys=False)

    # -----------------------------------------------------------------------
    # scANVI evaluation for top-k trials
    # -----------------------------------------------------------------------
    ctype_for_scanvi = cfg.get("cell_type_key", "cell_class")
    run_scanvi_eval = (
        top_model_records
        and max_epochs_scanvi > 0
        and ctype_for_scanvi in adata.obs.columns
    )
    if top_model_records and not run_scanvi_eval:
        logger.warning(
            f"Skipping scANVI evaluation: cell_type_key='{ctype_for_scanvi}' not in obs."
        )

    if run_scanvi_eval:
        # Build normalization constants from scVI trials so the scANVI objectives
        # are on the same scale and the scVI→scANVI delta is directly interpretable.
        valid_results = [
            r for r in results
            if r["status"] == "ok"
            and np.isfinite(r.get("batch_mixing_pca_score", float("nan")))
            and np.isfinite(r.get("recon_error", float("nan")))
        ]
        bm_vals = [r["batch_mixing_pca_score"] for r in valid_results]
        re_vals = [r["recon_error"]             for r in valid_results]
        norm_constants: dict[str, float] = {
            "bm_min": min(bm_vals) if bm_vals else 0.0,
            "bm_max": max(bm_vals) if bm_vals else 1.0,
            "re_min": min(re_vals) if re_vals else 0.0,
            "re_max": max(re_vals) if re_vals else 1.0,
        }

        logger.info("=" * 80)
        logger.info(
            f"scANVI EVALUATION — top {len(top_model_records)} scVI trials "
            f"(max_epochs_scanvi={max_epochs_scanvi})"
        )
        logger.info(
            "Production inference uses scANVI (not scVI). These metrics quantify "
            "how scANVI fine-tuning changes batch-integration quality for each trial. "
            "scANVI objectives are computed on the same scale as scVI using the "
            "cross-trial normalization constants from the scVI runs."
        )
        # Evaluate best → worst so the most important result appears first in the log.
        scanvi_records_to_eval = sorted(
            top_model_records, key=lambda x: x.get("objective", float("-inf")), reverse=True
        )
        scanvi_results: list[dict[str, Any]] = []

        for rec in scanvi_records_to_eval:
            t = rec["trial"]
            scvi_obj = rec["objective"]
            logger.info(f"  Trial {t}  scVI obj={scvi_obj:.4f} ...")
            try:
                scanvi_metrics = _train_and_evaluate_scanvi(
                    adata=adata,
                    scvi_model_dir=rec["model_dir"],
                    cell_type_key=ctype_for_scanvi,
                    latent_key_scanvi=f"X_scANVI_t{t}",
                    batch_key=batch_key,
                    age_key=age_key,
                    metric_cfg=metric_cfg,
                    n_batches_global=n_batches_global,
                    max_epochs_scanvi=max_epochs_scanvi,
                    accel=accel,
                    logger=logger,
                    norm_constants=norm_constants,
                )
                delta = scanvi_metrics["objective"] - scvi_obj
                # Retrieve actual metric values from the results list (more accurate
                # than prelim_score which was only used for buffer management).
                scvi_row = next((r for r in results if r["trial"] == t), {})
                scanvi_results.append({
                    "trial":                           t,
                    "scvi_objective":                  scvi_obj,
                    "scvi_batch_mixing_pca_score":     scvi_row.get("batch_mixing_pca_score", float("nan")),
                    "scvi_recon_error":                scvi_row.get("recon_error", float("nan")),
                    "scanvi_objective":                scanvi_metrics["objective"],
                    "scanvi_batch_mixing_pca_score":   scanvi_metrics["batch_mixing_pca_score"],
                    "scanvi_recon_error":              scanvi_metrics["recon_error"],
                    "delta_objective":                 delta,
                })
                logger.info(
                    f"    scVI={scvi_obj:.4f}  →  scANVI={scanvi_metrics['objective']:.4f}  "
                    f"(Δ={delta:+.4f})"
                )
            except Exception:
                logger.exception(f"  scANVI evaluation failed for trial {t}")
                scanvi_results.append({
                    "trial":                           t,
                    "scvi_objective":                  scvi_obj,
                    "scvi_batch_mixing_pca_score":     float("nan"),
                    "scvi_recon_error":                float("nan"),
                    "scanvi_objective":                float("nan"),
                    "scanvi_batch_mixing_pca_score":   float("nan"),
                    "scanvi_recon_error":              float("nan"),
                    "delta_objective":                 float("nan"),
                })

        if scanvi_results:
            pd.DataFrame(scanvi_results).to_csv(output_dir / "scanvi_comparison.csv", index=False)
            logger.info(f"Wrote: {output_dir / 'scanvi_comparison.csv'}")

            # Annotate best_hyperparameters.yaml with the best scANVI objective.
            best_scanvi = max(
                (r for r in scanvi_results if np.isfinite(r["scanvi_objective"])),
                key=lambda r: r["scanvi_objective"],
                default=None,
            )
            if best_scanvi is not None:
                best_params["best_metrics"]["scanvi_objective"]     = float(best_scanvi["scanvi_objective"])
                best_params["best_metrics"]["scvi_vs_scanvi_delta"] = float(best_scanvi["delta_objective"])
                with open(output_dir / "best_hyperparameters.yaml", "w") as f:
                    yaml.safe_dump(best_params, f, sort_keys=False)

    # -----------------------------------------------------------------------
    # Cleanup top-k model dirs (no longer needed after scANVI evaluation)
    # -----------------------------------------------------------------------
    for rec in top_model_records:
        shutil.rmtree(rec["model_dir"], ignore_errors=True)

    # -----------------------------------------------------------------------
    # Budget advisory
    # -----------------------------------------------------------------------
    _budget_advisory(results, n_trials, space_size, elapsed_total_h, logger)

    logger.info("=" * 80)
    logger.info(f"Tuning complete.  Trials executed: {len(results)}")
    logger.info(f"Best objective: {best['objective']:.4f}  (trial {best['trial']})")
    logger.info(f"Wrote: {output_dir / 'trial_results.csv'}")
    logger.info(f"Wrote: {output_dir / 'best_hyperparameters.yaml'}")

    # -----------------------------------------------------------------------
    # Auto-generate diagnostic plots
    # -----------------------------------------------------------------------
    try:
        from tuning.tuning_diagnostics import run_diagnostics
        plot_dir = output_dir / "plots"
        logger.info(f"Generating diagnostic plots → {plot_dir}")
        run_diagnostics(output_dir, plot_dir, logger=logger)
    except Exception:
        logger.warning(
            "Diagnostic plotting failed (non-fatal). "
            "Re-run manually: python -m tuning.tuning_diagnostics --input_dir <output_dir>",
            exc_info=True,
        )

    # Remove per-trial latent and expression PCA .npy files — they are only needed
    # during the diagnostic UMAP step above and consume significant disk space.
    npy_files = (
        sorted(output_dir.glob("trial_*_latent.npy"))
        + sorted(output_dir.glob("trial_*_expr_pca.npy"))
        + sorted(output_dir.glob("trial_*_expr_pca_idx.npy"))
    )
    if npy_files:
        for npy in npy_files:
            npy.unlink()
        logger.info(f"Deleted {len(npy_files)} trial latent/expr-PCA .npy file(s)")


def main():
    parser = argparse.ArgumentParser(description="Tune scVI hyperparameters for batch correction")
    parser.add_argument("--config", required=True, help="Path to tuning YAML config")
    args = parser.parse_args()
    run_tuning(Path(args.config))


if __name__ == "__main__":
    main()
