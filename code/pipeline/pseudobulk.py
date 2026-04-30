"""
Pseudobulk aggregation: aggregate expression per donor (× optional cell type).

Reads the integrated h5ad in backed mode and, for each group defined in the
pipeline config's ``pseudobulk.groups`` section, sums or averages the requested
layers.  One output .h5ad is written per group.

Default layers aggregated (if present in the input):
  - counts          → sum  (raw integer counts for DESeq2 / edgeR)
  - scvi_normalized → mean (batch-corrected for continuous analyses)
  - scanvi_normalized → mean

Usage (invoked automatically by run_pipeline.py --steps pseudobulk):
    PYTHONPATH=code python -m pipeline.pseudobulk \\
        --input  /path/to/scvi_output/integrated.h5ad \\
        --output /path/to/pseudobulk_output/ \\
        --config /path/to/pipeline_config.yaml
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import yaml


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_LAYERS = [
    {"name": "counts",             "aggregation": "sum"},
    {"name": "scvi_normalized",    "aggregation": "mean"},
    {"name": "scanvi_normalized",  "aggregation": "mean"},
]

DEFAULT_GROUPS = [
    {
        "name": "all_cells_by_donor",
        "group_cols": ["individual"],
    },
    {
        "name": "by_cell_class",
        "group_cols": ["individual", "cell_class"],
    },
]

# Donor-level metadata columns carried into every pseudobulk obs
OBS_META_COLS = ["individual", "age_years", "sex", "source", "dataset", "region", "chemistry"]

DEFAULT_CHUNK_SIZE = 50_000
DEFAULT_MIN_CELLS = 10


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logger(log_path: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("pseudobulk")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_path:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── Core logic ────────────────────────────────────────────────────────────────

def _apply_filter(obs: pd.DataFrame, filter_dict: dict) -> np.ndarray:
    """Return boolean mask of cells passing all filter conditions."""
    mask = np.ones(len(obs), dtype=bool)
    for col, val in filter_dict.items():
        if col not in obs.columns:
            raise ValueError(
                f"Filter column '{col}' not in obs. "
                f"Available columns: {sorted(obs.columns)}"
            )
        if isinstance(val, list):
            mask &= obs[col].isin(val).values
        else:
            mask &= (obs[col].astype(str) == str(val)).values
    return mask


def _validate_group_cfg(group_cfg: dict, obs: pd.DataFrame) -> list[str]:
    """Return list of problems with a group config; empty list means OK."""
    problems = []
    for col in group_cfg.get("group_cols", []):
        if col not in obs.columns:
            problems.append(f"group_col '{col}' not in obs")
    for col in group_cfg.get("filter", {}).keys():
        if col not in obs.columns:
            problems.append(f"filter col '{col}' not in obs")
    return problems


def _aggregate_chunks(
    adata_backed: ad.AnnData,
    sorted_idx: np.ndarray,
    sorted_codes: np.ndarray,
    n_groups: int,
    layers_to_do: list,
    chunk_size: int,
    logger: logging.Logger,
) -> tuple[dict, np.ndarray]:
    """
    Memory-efficient aggregation in sequential HDF5 chunks.

    Parameters
    ----------
    adata_backed : AnnData opened in backed mode
    sorted_idx   : cell positions into adata.obs, sorted ascending (for seq. HDF5 reads)
    sorted_codes : integer group code for each position in sorted_idx
    n_groups     : total number of unique groups
    layers_to_do : list of {'name': str, 'aggregation': 'sum'|'mean'}
    chunk_size   : cells per read

    Returns
    -------
    accum  : {layer_name: np.ndarray(n_groups, n_genes)}
    ncells : np.ndarray(n_groups,) — cells per group
    """
    n_genes = adata_backed.shape[1]
    accum = {
        lc["name"]: np.zeros((n_groups, n_genes), dtype=np.float64)
        for lc in layers_to_do
    }
    ncells = np.zeros(n_groups, dtype=np.int64)

    n_total = len(sorted_idx)
    n_chunks = max(1, (n_total + chunk_size - 1) // chunk_size)

    for i_chunk in range(n_chunks):
        start = i_chunk * chunk_size
        end = min(start + chunk_size, n_total)
        chunk_idx = sorted_idx[start:end]
        chunk_codes = sorted_codes[start:end]

        # Sparse indicator matrix: (n_groups × chunk_size)
        # indicator[g, j] = 1  iff cell j belongs to group g
        indicator = sp.csr_matrix(
            (
                np.ones(len(chunk_idx), dtype=np.float32),
                (chunk_codes, np.arange(len(chunk_idx))),
            ),
            shape=(n_groups, len(chunk_idx)),
        )

        # Load each layer for this chunk; adata[chunk_idx] triggers backed HDF5 read
        chunk_adata = adata_backed[chunk_idx]
        for lc in layers_to_do:
            raw = chunk_adata.layers[lc["name"]]
            if sp.issparse(raw):
                raw = raw.toarray().astype(np.float32)
            else:
                raw = np.asarray(raw, dtype=np.float32)
            # Vectorised grouped sum: result is (n_groups, n_genes)
            accum[lc["name"]] += indicator.dot(raw)

        np.add.at(ncells, chunk_codes, 1)

        if (i_chunk + 1) % 10 == 0 or (i_chunk + 1) == n_chunks:
            logger.info(
                f"    Chunk {i_chunk + 1}/{n_chunks}  ({end}/{n_total} cells)"
            )

    # Convert sums → means where requested
    for lc in layers_to_do:
        if lc.get("aggregation", "sum") == "mean":
            safe_n = np.where(ncells > 0, ncells, 1).astype(np.float64)
            accum[lc["name"]] /= safe_n[:, np.newaxis]

    return accum, ncells


def pseudobulk_one(
    adata_backed: ad.AnnData,
    group_cfg: dict,
    layers_cfg: list,
    min_cells: int,
    chunk_size: int,
    input_path: str,
    logger: logging.Logger,
) -> Optional[ad.AnnData]:
    """
    Build one pseudobulk AnnData for a single group config.

    Parameters
    ----------
    group_cfg : {
        name       : output file stem
        group_cols : list of obs columns to group by (e.g. ['individual', 'cell_type_aligned'])
        filter     : optional {col: val | [vals]} to pre-filter cells
    }
    """
    name       = group_cfg["name"]
    group_cols = group_cfg["group_cols"]
    filter_dict = group_cfg.get("filter", {})

    logger.info(f"  [{name}]  group_cols={group_cols}  filter={filter_dict or '(none)'}")

    obs = adata_backed.obs

    # --- Validate config ---
    problems = _validate_group_cfg(group_cfg, obs)
    if problems:
        for p in problems:
            logger.warning(f"    Skipping {name}: {p}")
        return None

    # --- Filter cells ---
    if filter_dict:
        mask = _apply_filter(obs, filter_dict)
        cell_indices = np.where(mask)[0]
        logger.info(f"    {len(cell_indices):,} / {len(obs):,} cells pass filter")
    else:
        cell_indices = np.arange(len(obs))

    if len(cell_indices) == 0:
        logger.warning(f"    No cells after filter — skipping {name}")
        return None

    # --- Compute group codes ---
    obs_filt = obs.iloc[cell_indices]
    if len(group_cols) == 1:
        group_keys = obs_filt[group_cols[0]].astype(str)
    else:
        group_keys = obs_filt[group_cols].astype(str).agg("|".join, axis=1)

    cat = pd.Categorical(group_keys)
    group_codes = np.asarray(cat.codes)          # int array, same length as cell_indices
    n_groups = len(cat.categories)
    logger.info(f"    {n_groups:,} groups from {len(cell_indices):,} cells")

    # --- Determine available layers ---
    available = set(adata_backed.layers.keys())
    layers_to_do = [lc for lc in layers_cfg if lc["name"] in available]
    skipped = [lc["name"] for lc in layers_cfg if lc["name"] not in available]
    if skipped:
        logger.warning(f"    Layers not found (skipped): {skipped}")
    if not layers_to_do:
        logger.error(f"    No valid layers to aggregate — skipping {name}")
        return None

    # --- Sort indices for sequential HDF5 reads ---
    sort_order = np.argsort(cell_indices, kind="stable")
    sorted_idx   = cell_indices[sort_order]
    sorted_codes = group_codes[sort_order]

    # --- Aggregate ---
    accum, ncells = _aggregate_chunks(
        adata_backed, sorted_idx, sorted_codes, n_groups,
        layers_to_do, chunk_size, logger,
    )

    # --- Apply min_cells filter ---
    keep = ncells >= min_cells
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.warning(f"    Dropping {n_dropped:,} groups with < {min_cells} cells")
    keep_idx = np.where(keep)[0]
    if len(keep_idx) == 0:
        logger.warning(f"    All groups dropped by min_cells={min_cells} — skipping {name}")
        return None

    # --- Build obs metadata ---
    # Take the first cell from each group as the representative (donor-level
    # metadata is identical for all cells from the same donor).
    meta_cols = [c for c in OBS_META_COLS if c in obs_filt.columns]
    out_cols = list(dict.fromkeys(group_cols + meta_cols))  # preserve order, no dups

    obs_with_code = obs_filt[out_cols].copy()
    obs_with_code["_gc"] = group_codes
    pb_obs = (
        obs_with_code.groupby("_gc", sort=True)[out_cols]
        .first()
        .loc[keep_idx]
        .reset_index(drop=True)
    )
    pb_obs["n_cells"] = ncells[keep_idx].astype(int)

    # obs_names: join group_col values with __ (unique within a group)
    if len(group_cols) == 1:
        obs_names = pb_obs[group_cols[0]].astype(str).tolist()
    else:
        obs_names = pb_obs[group_cols].astype(str).agg("__".join, axis=1).tolist()

    # --- Build output layers (keep_idx rows only, as sparse float32) ---
    out_layers = {}
    for lc in layers_to_do:
        out_layers[lc["name"]] = sp.csr_matrix(
            accum[lc["name"]][keep_idx].astype(np.float32)
        )

    # X = counts if present (integer pseudobulk), else first available layer
    X = out_layers.get("counts", next(iter(out_layers.values())))

    # --- Assemble AnnData ---
    pb = ad.AnnData(
        X=X.copy(),
        obs=pb_obs,
        var=adata_backed.var.copy(),
        layers=out_layers,
        uns={
            "pseudobulk": {
                "input_path": str(input_path),
                "group_cols": group_cols,
                "filter": filter_dict,
                "layers": {
                    lc["name"]: lc.get("aggregation", "sum")
                    for lc in layers_to_do
                },
                "min_cells": min_cells,
                "date": datetime.now().isoformat(),
            }
        },
    )
    pb.obs_names = obs_names

    logger.info(
        f"    Output: {pb.shape[0]:,} samples × {pb.shape[1]:,} genes  "
        f"(layers: {list(out_layers)})"
    )
    return pb


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    input_path: str,
    output_dir: str,
    pb_cfg: dict,
    logger: logging.Logger,
    overwrite: bool = False,
) -> None:
    """Run pseudobulk aggregation for all groups defined in pb_cfg."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers_cfg = pb_cfg.get("layers", DEFAULT_LAYERS)
    min_cells  = pb_cfg.get("min_cells", DEFAULT_MIN_CELLS)
    chunk_size = pb_cfg.get("chunk_size", DEFAULT_CHUNK_SIZE)
    groups     = pb_cfg.get("groups", DEFAULT_GROUPS)

    logger.info("=" * 60)
    logger.info("Pseudobulk Aggregation")
    logger.info("=" * 60)
    logger.info(f"Input:      {input_path}")
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Groups:     {[g['name'] for g in groups]}")
    logger.info(f"Layers:     {[lc['name'] for lc in layers_cfg]}")
    logger.info(f"min_cells:  {min_cells}   chunk_size: {chunk_size:,}")

    logger.info("\nOpening input in backed mode …")
    adata_backed = sc.read_h5ad(input_path, backed="r")
    logger.info(
        f"  {adata_backed.shape[0]:,} cells × {adata_backed.shape[1]:,} genes"
    )
    logger.info(f"  Available layers: {list(adata_backed.layers.keys())}")
    logger.info(f"  obs columns: {sorted(adata_backed.obs.columns.tolist())}")

    for group_cfg in groups:
        name     = group_cfg["name"]
        out_path = out_dir / f"{name}.h5ad"

        logger.info(f"\nProcessing: {name}")
        pb = pseudobulk_one(
            adata_backed=adata_backed,
            group_cfg=group_cfg,
            layers_cfg=layers_cfg,
            min_cells=min_cells,
            chunk_size=chunk_size,
            input_path=input_path,
            logger=logger,
        )
        if pb is None:
            logger.warning(f"  [{name}] produced no output, skipping write")
            continue

        logger.info(f"  Writing → {out_path}")
        pb.write_h5ad(str(out_path))
        logger.info(f"  [{name}] saved: {pb.shape}")

    try:
        adata_backed.file.close()
    except Exception:
        pass

    logger.info("\n" + "=" * 60)
    logger.info("Pseudobulk complete.")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Pseudobulk aggregation: aggregate expression by donor (× cell type)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",  required=True, help="Path to integrated.h5ad")
    parser.add_argument("--output", required=True, help="Output directory for pseudobulk h5ads")
    parser.add_argument("--config", required=True,
                        help="Pipeline config YAML (reads pseudobulk: section)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pb_cfg = cfg.get("pseudobulk", {})

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(str(out_dir / "pseudobulk.log"))

    run(args.input, args.output, pb_cfg, logger, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
