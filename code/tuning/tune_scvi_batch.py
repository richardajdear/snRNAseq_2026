"""Hyperparameter tuning for scVI batch correction with age-aware integration metrics.

This script is intentionally separate from `code/pipeline/` so tuning can run as an
overnight optimization job and then feed best parameters back into the pipeline config.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
import yaml
from scvi.model import SCVI
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("tune_scvi_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _as_list(space: dict[str, Any], key: str, default: list[Any]) -> list[Any]:
    vals = space.get(key, default)
    if not isinstance(vals, list) or len(vals) == 0:
        raise ValueError(f"search_space.{key} must be a non-empty list")
    return vals


def _sample_trial_params(search_space: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    return {
        "n_latent": int(rng.choice(_as_list(search_space, "n_latent", [20, 30, 40]))),
        "n_hidden": int(rng.choice(_as_list(search_space, "n_hidden", [64, 128, 256]))),
        "n_layers": int(rng.choice(_as_list(search_space, "n_layers", [1, 2]))),
        "dropout_rate": float(rng.choice(_as_list(search_space, "dropout_rate", [0.0, 0.1, 0.2]))),
        "gene_likelihood": str(rng.choice(_as_list(search_space, "gene_likelihood", ["nb"]))),
        "batch_size": int(rng.choice(_as_list(search_space, "batch_size", [256, 512]))),
        "lr": float(rng.choice(_as_list(search_space, "lr", [1e-3, 5e-4, 2e-4]))),
        "weight_decay": float(rng.choice(_as_list(search_space, "weight_decay", [1e-6, 1e-5, 1e-4]))),
    }


def _make_age_bins(obs: pd.DataFrame, age_key: str, edges: list[float]) -> pd.Series:
    return pd.cut(obs[age_key], bins=edges, right=False, include_lowest=True)


def _normalized_neighbor_batch_entropy(batch_codes_neighbors: np.ndarray, n_batches: int) -> np.ndarray:
    if n_batches < 2:
        return np.zeros(batch_codes_neighbors.shape[0], dtype=np.float32)

    out = np.zeros(batch_codes_neighbors.shape[0], dtype=np.float32)
    denom = math.log(n_batches)
    for i in range(batch_codes_neighbors.shape[0]):
        counts = np.bincount(batch_codes_neighbors[i], minlength=n_batches)
        probs = counts[counts > 0] / counts.sum()
        ent = -np.sum(probs * np.log(probs))
        out[i] = float(ent / denom)
    return out


def _mixing_score_for_subset(
    X: np.ndarray,
    batch_codes: np.ndarray,
    k_neighbors: int,
) -> float:
    n = X.shape[0]
    if n < 10:
        return float("nan")

    k = min(k_neighbors + 1, n)
    if k <= 2:
        return float("nan")

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    neigh_batch_codes = batch_codes[idx]
    n_batches = int(np.unique(batch_codes).shape[0])
    ent = _normalized_neighbor_batch_entropy(neigh_batch_codes, n_batches)
    return float(np.mean(ent))


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
) -> tuple[float, dict[str, float]]:
    obs = adata.obs.copy()
    X = np.asarray(adata.obsm[latent_key])

    age_bins = _make_age_bins(obs, age_key=age_key, edges=age_bin_edges)
    obs = obs.assign(_age_bin=age_bins)

    weighted_scores = []
    details: dict[str, float] = {}

    for age_bin, idx in obs.groupby("_age_bin", observed=True).indices.items():
        if pd.isna(age_bin):
            continue
        idx = np.asarray(idx, dtype=int)
        if idx.size < min_cells_per_bin:
            continue

        age_left_edge = float(age_bin.left)
        bin_weight = float(prenatal_weight if age_left_edge < 0 else 1.0)

        obs_bin = obs.iloc[idx]

        group_scores = []
        if cell_type_key and cell_type_key in obs_bin.columns:
            for _, idx_rel in obs_bin.groupby(cell_type_key, observed=True).indices.items():
                idx_rel = np.asarray(idx_rel, dtype=int)
                idx_ct = idx[idx_rel]
                if idx_ct.size < min_cells_per_group:
                    continue
                batch_codes = pd.factorize(obs.iloc[idx_ct][batch_key], sort=True)[0].astype(int)
                score_ct = _mixing_score_for_subset(X[idx_ct], batch_codes, k_neighbors)
                if np.isfinite(score_ct):
                    group_scores.append((score_ct, idx_ct.size))

        if group_scores:
            numer = sum(s * w for s, w in group_scores)
            denom = sum(w for _, w in group_scores)
            score_bin = float(numer / max(denom, 1))
        else:
            batch_codes = pd.factorize(obs_bin[batch_key], sort=True)[0].astype(int)
            score_bin = _mixing_score_for_subset(X[idx], batch_codes, k_neighbors)

        if np.isfinite(score_bin):
            details[f"[{age_bin.left},{age_bin.right})"] = score_bin
            weighted_scores.append((score_bin, bin_weight))

    if not weighted_scores:
        return 0.0, details

    numer = sum(score * w for score, w in weighted_scores)
    denom = sum(w for _, w in weighted_scores)
    return float(numer / max(denom, 1e-8)), details


def _evaluate_trial(
    adata,
    batch_key: str,
    age_key: str,
    cell_type_key: str | None,
    metric_cfg: dict[str, Any],
) -> dict[str, Any]:
    age_score, age_details = _evaluate_age_aware_batch_score(
        adata=adata,
        latent_key="X_scVI",
        batch_key=batch_key,
        age_key=age_key,
        age_bin_edges=metric_cfg["age_bin_edges"],
        prenatal_weight=float(metric_cfg.get("prenatal_weight", 2.0)),
        cell_type_key=cell_type_key,
        k_neighbors=int(metric_cfg.get("k_neighbors", 20)),
        min_cells_per_bin=int(metric_cfg.get("min_cells_per_bin", 200)),
        min_cells_per_group=int(metric_cfg.get("min_cells_per_group", 75)),
    )

    sil_weight = float(metric_cfg.get("silhouette_weight", 0.2))
    sil_score_norm = 0.5
    if cell_type_key and cell_type_key in adata.obs.columns:
        labels = adata.obs[cell_type_key].astype(str)
        if labels.nunique() > 1:
            max_n = int(metric_cfg.get("max_silhouette_cells", 20000))
            n = adata.n_obs
            if n > max_n:
                rng = np.random.default_rng(0)
                idx = rng.choice(n, size=max_n, replace=False)
            else:
                idx = np.arange(n)
            sil = silhouette_score(np.asarray(adata.obsm["X_scVI"])[idx], labels.iloc[idx])
            sil_score_norm = float((sil + 1.0) / 2.0)

    total = (1.0 - sil_weight) * age_score + sil_weight * sil_score_norm
    return {
        "objective": float(total),
        "age_batch_score": float(age_score),
        "celltype_silhouette_score_norm": float(sil_score_norm),
        "age_bin_scores": age_details,
    }


def _subset_for_tuning(adata, max_cells: int, stratify_cols: list[str], seed: int, logger: logging.Logger):
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
    elif keep_idx.size < max_cells:
        missing = max_cells - keep_idx.size
        pool = np.setdiff1d(np.arange(adata.n_obs), keep_idx, assume_unique=False)
        if pool.size > 0:
            fill = rng.choice(pool, size=min(missing, pool.size), replace=False)
            keep_idx = np.sort(np.concatenate([keep_idx, fill]))

    logger.info(f"Subsampled for tuning: {adata.n_obs:,} -> {keep_idx.size:,} cells")
    return adata[keep_idx].copy()


def run_tuning(config_path: Path):
    cfg = _load_yaml(config_path)

    input_h5ad = Path(cfg["input_h5ad"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(output_dir / "tuning.log")
    logger.info(f"Config: {config_path}")
    logger.info(f"Input: {input_h5ad}")
    logger.info(f"Output dir: {output_dir}")

    batch_key = cfg.get("batch_key", "source-chemistry")
    age_key = cfg.get("age_key", "age_years")
    cell_type_key = cfg.get("cell_type_key", "cell_class")
    counts_layer = cfg.get("counts_layer", "counts")

    tuning_cfg = cfg.get("tuning", {})
    random_seed = int(tuning_cfg.get("random_seed", 42))
    n_trials = int(tuning_cfg.get("n_trials", 16))
    max_hours = float(tuning_cfg.get("max_hours", 12.0))
    max_cells = int(tuning_cfg.get("max_cells", 120000))

    training_cfg = cfg.get("training", {})
    max_epochs = int(training_cfg.get("max_epochs_scvi", 120))
    early_stopping = bool(training_cfg.get("early_stopping", True))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 15))
    train_size = float(training_cfg.get("train_size", 0.9))

    hvg_cfg = cfg.get("hvg", {})
    n_top_genes = int(hvg_cfg.get("n_top_genes", 10000))
    hvg_flavor = hvg_cfg.get("hvg_flavor", "seurat_v3")
    hvg_batch_key = hvg_cfg.get("hvg_batch_key", batch_key)

    metric_cfg = cfg.get("metric", {})
    if "age_bin_edges" not in metric_cfg:
        raise ValueError("metric.age_bin_edges is required")

    search_space = cfg.get("search_space", {})

    scvi.settings.seed = random_seed
    scvi.settings.num_threads = int(tuning_cfg.get("num_workers", 4))

    logger.info("Loading AnnData...")
    adata = sc.read_h5ad(str(input_h5ad))

    for key in [batch_key, age_key]:
        if key not in adata.obs.columns:
            raise ValueError(f"Required obs column missing: {key}")
    if counts_layer not in adata.layers:
        logger.info(f"{counts_layer!r} missing; copying .X")
        adata.layers[counts_layer] = adata.X.copy()

    logger.info("Selecting HVGs once before trials...")
    if hvg_flavor == "pearson_residuals":
        kwargs = {"n_top_genes": n_top_genes, "layer": counts_layer}
        sc.experimental.pp.highly_variable_genes(adata, **kwargs)
    else:
        kwargs = {"n_top_genes": n_top_genes, "flavor": hvg_flavor}
        if hvg_flavor == "seurat_v3":
            kwargs["layer"] = counts_layer
        if hvg_batch_key:
            kwargs["batch_key"] = hvg_batch_key
        sc.pp.highly_variable_genes(adata, **kwargs)

    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info(f"Prepared HVG AnnData: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    age_bins_full = _make_age_bins(adata.obs, age_key, metric_cfg["age_bin_edges"])
    adata.obs["_tune_age_bin"] = age_bins_full.astype(str)
    adata = _subset_for_tuning(
        adata,
        max_cells=max_cells,
        stratify_cols=[batch_key, "_tune_age_bin"],
        seed=random_seed,
        logger=logger,
    )

    covariate_kwargs = {}
    if cfg.get("continuous_covariate_keys"):
        covariate_kwargs["continuous_covariate_keys"] = cfg["continuous_covariate_keys"]
    if cfg.get("categorical_covariate_keys"):
        covariate_kwargs["categorical_covariate_keys"] = cfg["categorical_covariate_keys"]

    logger.info(f"Setting up anndata for scVI with covariates: {covariate_kwargs}")
    SCVI.setup_anndata(adata, layer=counts_layer, batch_key=batch_key, **covariate_kwargs)

    accel = {"accelerator": "gpu", "devices": 1} if torch.cuda.is_available() else {"accelerator": "cpu", "devices": 1}
    logger.info(f"Trainer accelerator: {accel}")

    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    start = time.time()
    rng = np.random.default_rng(random_seed)

    for trial in range(1, n_trials + 1):
        elapsed_h = (time.time() - start) / 3600.0
        if elapsed_h >= max_hours:
            logger.info(f"Reached max_hours={max_hours}; stopping before trial {trial}")
            break

        params = _sample_trial_params(search_space, rng)
        trial_seed = random_seed + trial
        scvi.settings.seed = trial_seed

        logger.info("-" * 80)
        logger.info(f"Trial {trial}/{n_trials} | elapsed={elapsed_h:.2f}h | params={params}")

        trial_t0 = time.time()
        try:
            model = SCVI(
                adata,
                n_latent=params["n_latent"],
                n_hidden=params["n_hidden"],
                n_layers=params["n_layers"],
                dropout_rate=params["dropout_rate"],
                gene_likelihood=params["gene_likelihood"],
            )
            train_kwargs = {
                "max_epochs": max_epochs,
                "early_stopping": early_stopping,
                "train_size": train_size,
                "validation_size": 1.0 - train_size,
                "batch_size": params["batch_size"],
                "enable_progress_bar": True,
                "plan_kwargs": {
                    "lr": params["lr"],
                    "weight_decay": params["weight_decay"],
                },
                **accel,
            }
            if early_stopping:
                train_kwargs["early_stopping_patience"] = early_stopping_patience
            model.train(**train_kwargs)

            adata.obsm["X_scVI"] = model.get_latent_representation()
            metrics = _evaluate_trial(
                adata=adata,
                batch_key=batch_key,
                age_key=age_key,
                cell_type_key=cell_type_key,
                metric_cfg=metric_cfg,
            )
            status = "ok"
            error = ""
        except Exception as e:  # keep running remaining trials
            status = "failed"
            error = repr(e)
            metrics = {
                "objective": float("-inf"),
                "age_batch_score": float("nan"),
                "celltype_silhouette_score_norm": float("nan"),
                "age_bin_scores": {},
            }
            logger.exception(f"Trial {trial} failed")

        duration_min = (time.time() - trial_t0) / 60.0
        row = {
            "trial": trial,
            "status": status,
            "error": error,
            "duration_min": round(duration_min, 2),
            **params,
            "objective": metrics["objective"],
            "age_batch_score": metrics["age_batch_score"],
            "celltype_silhouette_score_norm": metrics["celltype_silhouette_score_norm"],
            "age_bin_scores_json": json.dumps(metrics["age_bin_scores"], sort_keys=True),
        }
        results.append(row)

        if status == "ok" and (best is None or row["objective"] > best["objective"]):
            best = row
            logger.info(
                f"New best trial={trial} objective={row['objective']:.4f} "
                f"(age={row['age_batch_score']:.4f}, silhouette={row['celltype_silhouette_score_norm']:.4f})"
            )

        pd.DataFrame(results).to_csv(output_dir / "trial_results.csv", index=False)

    if not results:
        raise RuntimeError("No trials were executed.")

    df = pd.DataFrame(results).sort_values("objective", ascending=False)
    df.to_csv(output_dir / "trial_results.csv", index=False)

    if best is None:
        raise RuntimeError("All trials failed; no best hyperparameters available.")

    best_params = {
        "scvi": {
            "n_latent": int(best["n_latent"]),
            "n_hidden": int(best["n_hidden"]),
            "n_layers": int(best["n_layers"]),
            "batch_size": int(best["batch_size"]),
            "max_epochs_scvi": max_epochs,
            "early_stopping": early_stopping,
            "early_stopping_patience": early_stopping_patience,
        },
        "training_plan": {
            "dropout_rate": float(best["dropout_rate"]),
            "gene_likelihood": str(best["gene_likelihood"]),
            "lr": float(best["lr"]),
            "weight_decay": float(best["weight_decay"]),
        },
        "best_metrics": {
            "objective": float(best["objective"]),
            "age_batch_score": float(best["age_batch_score"]),
            "celltype_silhouette_score_norm": float(best["celltype_silhouette_score_norm"]),
            "age_bin_scores": json.loads(best["age_bin_scores_json"]),
        },
    }

    with open(output_dir / "best_hyperparameters.yaml", "w") as f:
        yaml.safe_dump(best_params, f, sort_keys=False)

    logger.info("=" * 80)
    logger.info(f"Tuning complete. Trials executed: {len(results)}")
    logger.info(f"Best objective: {best['objective']:.4f} (trial {best['trial']})")
    logger.info(f"Wrote: {output_dir / 'trial_results.csv'}")
    logger.info(f"Wrote: {output_dir / 'best_hyperparameters.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Tune scVI hyperparameters for batch correction")
    parser.add_argument("--config", required=True, help="Path to tuning YAML config")
    args = parser.parse_args()

    run_tuning(Path(args.config))


if __name__ == "__main__":
    main()
