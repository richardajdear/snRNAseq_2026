"""GPU-accelerated UMAP computation for integrated.h5ad subsets.

Computes UMAP embeddings on GPU using rapids_singlecell for configurable
cell subsets (e.g. all cells, excitatory only, all neurons).  Produces a
plot grid per subset: rows = embeddings, columns = colour variables.

Usage
-----
    PYTHONPATH=code python3 -m pipeline.gpu_umaps \
        --config  code/pipeline/configs/Vel_prepost_noage_tuning5.yaml \
        --input   /path/to/scvi_output/integrated.h5ad \
        --output_dir /path/to/plots \
        [--n_cells 5000]

Called by run_pipeline.py (step_gpu_umaps) and step5_umaps_gpu.sh.
Step6 (step6_diagnostics.sh) runs scANVI diagnostics separately on CPU.

Config section (diagnostic_umaps)
-----------------------------------
    diagnostic_umaps:
      embeddings:
        - key: X_pca          # computed on-the-fly via rsc.pp.pca
          compute_pca: true
          n_pcs: 50
        - key: X_scVI         # read from obsm
        - key: X_scANVI       # read from obsm
      n_neighbors: 30
      min_dist: 0.3
      subsets:
        - name: all_cells
        - name: excitatory
          filter:
            cell_class: Excitatory
        - name: neurons
          filter:
            cell_class: [Excitatory, Inhibitory]
      color_by:
        - cell_class
        - cell_type_aligned
        - source
        - age_years
        - cell_type_aligned_confidence

Outputs (per subset, inside output_dir/<subset_name>/)
------------------------------------------------------
    umap_grid.png              — full grid: rows=embeddings, cols=color_by
    umap_<embedding>_<col>.png — individual panels for quick inspection
"""

import argparse
import gc
import os
import resource
import sys
import warnings
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

# ── GPU backend ───────────────────────────────────────────────────────────────
try:
    import rapids_singlecell as rsc
    import cupy
    _GPU = True
    print("rapids_singlecell available — using GPU backend.")
except ImportError:
    _GPU = False
    print("WARNING: rapids_singlecell not importable — falling back to CPU "
          "(sc.pp.neighbors + sc.tl.umap).", file=sys.stderr)

# ── Colour palettes (reuse canonical palettes from diagnostics module) ────────
_CLASS_PALETTE = {
    'Excitatory':  '#E41A1C',
    'Inhibitory':  '#377EB8',
    'Astrocytes':  '#4DAF4A',
    'Oligos':      '#984EA3',
    'OPC':         '#FF7F00',
    'Microglia':   '#008080',
    'Endothelial': '#778899',
    'Glia':        '#2CA25F',
    'Other':       '#BDBDBD',
}

_FALLBACK_COLORS = plt.cm.tab20.colors  # for unknown categorical values


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _log_memory(label: str) -> None:
    msg = f"  [mem] {label}: RSS={_rss_mb():.0f} MB"
    if _GPU:
        try:
            pool = cupy.get_default_memory_pool()
            msg += f"  GPU-used={pool.used_bytes() / 1e6:.0f} MB"
        except Exception:
            pass
    print(msg)


def _free_gpu() -> None:
    if _GPU:
        try:
            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


# ── Subset filtering ──────────────────────────────────────────────────────────

def apply_subset_filter(obs: pd.DataFrame, filter_cfg: Optional[dict]) -> np.ndarray:
    """Return boolean index array for cells matching filter_cfg.

    filter_cfg may be None (all cells) or a dict like:
        {cell_class: Excitatory}         → scalar equality
        {cell_class: [Excitatory, Inhibitory]} → isin list
    """
    mask = np.ones(len(obs), dtype=bool)
    if not filter_cfg:
        return mask
    for col, val in filter_cfg.items():
        if col not in obs.columns:
            raise ValueError(f"Filter column '{col}' not found in obs. "
                             f"Available: {sorted(obs.columns)}")
        if isinstance(val, list):
            mask &= obs[col].isin(val).values
        else:
            mask &= (obs[col] == val).values
    return mask


# ── UMAP computation ──────────────────────────────────────────────────────────

def compute_subset_umaps(
    adata_sub,
    embeddings_cfg: List[dict],
    n_neighbors: int,
    min_dist: float,
) -> Dict[str, np.ndarray]:
    """Compute UMAP for each configured embedding on a materialised subset.

    Returns a dict {key: (n_cells, 2) float32 array}.

    For keys with compute_pca=True, PCA is computed from adata_sub.X before
    building the neighbour graph.  For all other keys the embedding is read
    from adata_sub.obsm.

    All GPU objects are moved back to CPU before returning; GPU memory is freed.
    """
    results = {}

    for emb in embeddings_cfg:
        key = emb['key']
        compute_pca = emb.get('compute_pca', False)
        n_pcs = emb.get('n_pcs', 50)

        # ── prepare embedding ────────────────────────────────────────────────
        if compute_pca:
            print(f"    PCA ({key}, n_pcs={n_pcs}) …")
            if _GPU:
                rsc.get.anndata_to_GPU(adata_sub)
                rsc.pp.pca(adata_sub, n_comps=min(n_pcs, adata_sub.n_vars - 1))
                rsc.get.anndata_to_CPU(adata_sub)
            else:
                sc.pp.pca(adata_sub, n_comps=min(n_pcs, adata_sub.n_vars - 1),
                          zero_center=False)
            use_rep = 'X_pca'
        else:
            if key not in adata_sub.obsm:
                print(f"    SKIP {key}: not found in obsm.")
                continue
            use_rep = key

        # ── neighbors + UMAP ─────────────────────────────────────────────────
        print(f"    neighbors + UMAP from {use_rep} …")
        if _GPU:
            rsc.get.anndata_to_GPU(adata_sub)
            rsc.pp.neighbors(adata_sub, use_rep=use_rep, n_neighbors=n_neighbors)
            rsc.tl.umap(adata_sub, min_dist=min_dist)
            coords = np.array(adata_sub.obsm['X_umap'], dtype=np.float32)
            rsc.get.anndata_to_CPU(adata_sub)
            _free_gpu()
        else:
            sc.pp.neighbors(adata_sub, use_rep=use_rep, n_neighbors=n_neighbors)
            sc.tl.umap(adata_sub, min_dist=min_dist)
            coords = np.array(adata_sub.obsm['X_umap'], dtype=np.float32)

        results[key] = coords
        _log_memory(f"after UMAP {key}")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def _categorical_palette(values: pd.Series) -> dict:
    """Build a colour dict for a categorical series, using canonical colours where known."""
    cats = sorted(values.dropna().unique().tolist(), key=str)
    palette = {}
    fallback_i = 0
    for c in cats:
        if c in _CLASS_PALETTE:
            palette[c] = _CLASS_PALETTE[c]
        else:
            palette[c] = mcolors.to_hex(_FALLBACK_COLORS[fallback_i % len(_FALLBACK_COLORS)])
            fallback_i += 1
    return palette


def _adaptive_size(n: int) -> float:
    return float(np.clip(400.0 / max(n, 1) ** 0.5, 0.5, 8.0))


def _plot_panel(ax, xy: np.ndarray, obs_col: pd.Series, col_name: str) -> None:
    """Render a single scatter panel on ax."""
    s = _adaptive_size(len(xy))
    is_numeric = pd.api.types.is_numeric_dtype(obs_col)

    if is_numeric:
        vals = obs_col.values.astype(float)
        finite = vals[np.isfinite(vals)]
        vmin, vmax = (finite.min(), finite.max()) if len(finite) else (0, 1)
        sc_obj = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap='viridis',
                            vmin=vmin, vmax=vmax,
                            s=s, alpha=0.5, linewidths=0, rasterized=True)
        plt.colorbar(sc_obj, ax=ax, shrink=0.7, pad=0.02,
                     label=col_name, fraction=0.046)
    else:
        pal = _categorical_palette(obs_col)
        cats = sorted(obs_col.dropna().unique().tolist(), key=str)
        col_arr = obs_col.values
        # Shuffle points so no single category always renders on top
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(xy))
        xy_p = xy[perm]
        col_p = col_arr[perm]
        for cat in cats:
            m = col_p == cat
            if not m.any():
                continue
            ax.scatter(xy_p[m, 0], xy_p[m, 1],
                       c=pal.get(cat, '#BDBDBD'),
                       s=s, alpha=0.5, linewidths=0, rasterized=True, label=cat)
        n_cats = len(cats)
        ax.legend(loc='lower right', fontsize=max(4, 7 - n_cats // 8),
                  framealpha=0.8, handletextpad=0.3,
                  ncol=max(1, n_cats // 15),
                  markerscale=1.5)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(col_name, fontsize=9)


def plot_umap_grid(
    coords_dict: Dict[str, np.ndarray],
    obs_df: pd.DataFrame,
    color_by: List[str],
    out_dir: str,
    subset_name: str,
) -> None:
    """Save a grid PNG (rows=embeddings, cols=color_by) plus individual PNGs."""
    os.makedirs(out_dir, exist_ok=True)

    emb_keys = list(coords_dict.keys())
    n_rows = len(emb_keys)
    n_cols = len(color_by)

    # ── combined grid ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    for r, ekey in enumerate(emb_keys):
        xy = coords_dict[ekey]
        axes[r, 0].set_ylabel(ekey, fontsize=9, rotation=90, labelpad=4)
        for c, col in enumerate(color_by):
            ax = axes[r, c]
            if col not in obs_df.columns:
                ax.axis('off')
                ax.set_title(f'{col}\n(missing)', fontsize=8)
                continue
            _plot_panel(ax, xy, obs_df[col], col)

    fig.suptitle(f'Subset: {subset_name}  (n={len(obs_df):,})',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    grid_path = os.path.join(out_dir, 'umap_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {grid_path}")

    # ── individual PNGs ──────────────────────────────────────────────────────
    for ekey in emb_keys:
        xy = coords_dict[ekey]
        safe_ekey = ekey[2:].lower() if ekey.startswith('X_') else ekey.lower()
        for col in color_by:
            if col not in obs_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(5, 4.5))
            _plot_panel(ax, xy, obs_df[col], col)
            ax.set_title(f'{subset_name} — {ekey}\n{col}  (n={len(obs_df):,})',
                         fontsize=10)
            plt.tight_layout()
            fname = f'umap_{safe_ekey}_{col}.png'
            plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_gpu_umaps(config_path: str, input_path: str, output_dir: str,
                  n_cells: Optional[int] = None) -> None:
    """Core logic, importable for testing."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    umap_cfg = cfg.get('diagnostic_umaps', {})
    embeddings_cfg = umap_cfg.get('embeddings', [
        {'key': 'X_scANVI'},
    ])
    n_neighbors = int(umap_cfg.get('n_neighbors', 30))
    min_dist = float(umap_cfg.get('min_dist', 0.3))
    color_by = umap_cfg.get('color_by', [
        'cell_class', 'cell_type_aligned', 'source', 'age_years',
        'cell_type_aligned_confidence',
    ])
    subsets = umap_cfg.get('subsets', [{'name': 'all_cells'}])

    print(f"\nLoading obs metadata from {input_path} (backed) …")
    adata_backed = sc.read_h5ad(input_path, backed='r')
    total_cells = adata_backed.n_obs
    print(f"  {total_cells:,} cells × {adata_backed.n_vars:,} genes")
    _log_memory("after backed open")

    # Global random subsample for testing
    if n_cells is not None and n_cells < total_cells:
        rng = np.random.default_rng(42)
        global_idx = np.sort(rng.choice(total_cells, size=n_cells, replace=False))
        print(f"  Downsampling to {n_cells:,} cells for testing.")
    else:
        global_idx = np.arange(total_cells)

    # Read obs once (small, always fits in RAM)
    obs_global = adata_backed.obs.iloc[global_idx].copy()

    for subset_cfg in subsets:
        sname = subset_cfg['name']
        filter_cfg = subset_cfg.get('filter')

        print(f"\n{'='*60}\nSubset: {sname}\n{'='*60}")

        # Identify h5ad row indices for this subset within the global (possibly
        # downsampled) pool.  sub_mask is boolean over obs_global rows.
        sub_mask = apply_subset_filter(obs_global, filter_cfg)
        sub_positions = global_idx[sub_mask]  # actual row indices into the h5ad file
        n_sub = len(sub_positions)
        print(f"  {n_sub:,} cells match filter {filter_cfg!r}")

        if n_sub < 10:
            print(f"  SKIP: too few cells ({n_sub}).")
            continue

        # Materialise ONLY this subset from disk
        print(f"  Loading subset from disk …")
        adata_sub = adata_backed[sub_positions].to_memory()
        _log_memory(f"after loading subset '{sname}'")

        obs_sub = adata_sub.obs.copy()

        out_dir = os.path.join(output_dir, sname)
        os.makedirs(out_dir, exist_ok=True)

        try:
            coords_dict = compute_subset_umaps(
                adata_sub, embeddings_cfg, n_neighbors, min_dist)
        except Exception as e:
            print(f"  ERROR computing UMAPs for subset '{sname}': {e}", file=sys.stderr)
            del adata_sub; gc.collect(); _free_gpu()
            continue

        if not coords_dict:
            print(f"  No UMAP coords produced for '{sname}' — check embedding keys.")
            del adata_sub; gc.collect()
            continue

        plot_umap_grid(coords_dict, obs_sub, color_by, out_dir, sname)

        del adata_sub, coords_dict, obs_sub
        gc.collect()
        _free_gpu()
        _log_memory(f"after clearing subset '{sname}'")

    adata_backed.file.close()
    print("\nDone.")


def main():
    p = argparse.ArgumentParser(description='GPU UMAP computation for integrated.h5ad subsets')
    p.add_argument('--config', required=True,
                   help='Pipeline config YAML (must contain diagnostic_umaps section)')
    p.add_argument('--input', required=True,
                   help='Path to integrated.h5ad')
    p.add_argument('--output_dir', required=True,
                   help='Output directory; subset plots land in <output_dir>/<subset_name>/')
    p.add_argument('--n_cells', type=int, default=None,
                   help='Randomly subsample to this many cells (for testing)')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_gpu_umaps(args.config, args.input, args.output_dir, args.n_cells)


if __name__ == '__main__':
    main()
