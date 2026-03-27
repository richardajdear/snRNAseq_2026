"""
Standalone diagnostic: compare three UMAP embeddings side-by-side.

  Column 1 — Raw:          raw counts -> log-CPM -> PCA -> UMAP
  Column 2 — X_scVI:       scVI latent space -> UMAP  (batch correction via encoder)
  Column 3 — scVI expr:    scvi_normalized (transform_batch=VELMESHEV) -> log-CPM -> PCA -> UMAP

Output: 2-row × 3-col figure, rows coloured by `source` and `cell_class`.

The key question: does batch-corrected gene-expression PCA (col 3) show different
structure to the latent-space UMAP (col 2)?  Both should be batch-integrated, but
via different mechanisms (linear gene-space vs nonlinear latent space).
"""

import sys
from pathlib import Path
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.sparse import issparse

# ── Config ────────────────────────────────────────────────────────────────────

DATA_FILE = (
    "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
    "/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld"
    "/scvi_output/integrated.h5ad"
)
OUTPUT_FILE = (
    "/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs"
    "/umap_comparison.png"
)

N_NEIGHBORS  = 30
MIN_DIST     = 0.3
N_PCA_COMPS  = 50
RANDOM_SEED  = 42

# ── Palettes ──────────────────────────────────────────────────────────────────

# Set1 (colourblind-friendly for up to 8 categories)
SET1 = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
    "#FF7F00", "#A65628", "#F781BF", "#999999",
]
# tab20 for cell classes (more categories possible)
TAB20 = plt.cm.tab20.colors


def palette_for(values):
    """Return (cats, colour_map) for a list/array of categorical values."""
    cats = sorted(set(values))
    base = SET1 if len(cats) <= 8 else TAB20
    cmap = {c: base[i % len(base)] for i, c in enumerate(cats)}
    return cats, cmap


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_dense(mat):
    return mat.toarray() if issparse(mat) else np.asarray(mat)


def log_cpm(counts: np.ndarray) -> np.ndarray:
    size = counts.sum(axis=1, keepdims=True)
    size = np.maximum(size, 1.0)
    return np.log1p(counts / size * 1e6)


def compute_pca_umap(X: np.ndarray, label: str,
                     n_pca: int = N_PCA_COMPS,
                     n_neighbors: int = N_NEIGHBORS,
                     min_dist: float = MIN_DIST) -> np.ndarray:
    """PCA → neighbors → UMAP on a cell × gene matrix. Returns (n_cells, 2)."""
    print(f"  PCA ({n_pca} comps) …", flush=True)
    tmp = ad.AnnData(X)
    sc.pp.pca(tmp, n_comps=n_pca, random_state=RANDOM_SEED)
    print(f"  Neighbors (k={n_neighbors}) …", flush=True)
    sc.pp.neighbors(tmp, n_neighbors=n_neighbors,
                    use_rep="X_pca", key_added="nn", random_state=RANDOM_SEED)
    print(f"  UMAP …", flush=True)
    sc.tl.umap(tmp, min_dist=min_dist, neighbors_key="nn",
               random_state=RANDOM_SEED)
    coords = tmp.obsm["X_umap"].copy()
    del tmp
    print(f"  {label}: done. shape={coords.shape}", flush=True)
    return coords


def compute_latent_umap(Z: np.ndarray,
                        n_neighbors: int = N_NEIGHBORS,
                        min_dist: float = MIN_DIST) -> np.ndarray:
    """neighbors → UMAP directly on a latent matrix Z. Returns (n_cells, 2)."""
    tmp = ad.AnnData(Z)
    tmp.obsm["X_latent"] = Z
    sc.pp.neighbors(tmp, n_neighbors=n_neighbors,
                    use_rep="X_latent", key_added="nn", random_state=RANDOM_SEED)
    sc.tl.umap(tmp, min_dist=min_dist, neighbors_key="nn",
               random_state=RANDOM_SEED)
    coords = tmp.obsm["X_umap"].copy()
    del tmp
    return coords


# ── Load ──────────────────────────────────────────────────────────────────────

print(f"Loading {DATA_FILE} …", flush=True)
adata = sc.read_h5ad(DATA_FILE)
print(f"Shape : {adata.shape}", flush=True)
print(f"Layers: {list(adata.layers.keys())}", flush=True)
print(f"obsm  : {list(adata.obsm.keys())}", flush=True)

source     = np.array(adata.obs["source"])
cell_class = np.array(adata.obs["cell_class"])

hvg_mask = None
if "highly_variable" in adata.var.columns:
    hvg_mask = adata.var["highly_variable"].values.astype(bool)
    print(f"HVGs  : {hvg_mask.sum():,}", flush=True)

# ── UMAP 1 — Raw log-CPM → PCA ───────────────────────────────────────────────

print("\n=== UMAP 1: Raw counts → log-CPM → PCA ===", flush=True)
raw = to_dense(adata.layers["counts"]).astype(np.float32)
if hvg_mask is not None:
    raw = raw[:, hvg_mask]
raw = log_cpm(raw)
umap_raw = compute_pca_umap(raw, label="raw")
del raw

# ── UMAP 2 — X_scVI latent space ─────────────────────────────────────────────

print("\n=== UMAP 2: X_scVI latent → UMAP ===", flush=True)
if "X_scVI" not in adata.obsm:
    print("ERROR: X_scVI not found in obsm. "
          "Run the scVI pipeline with train_scvi step first.", flush=True)
    sys.exit(1)
Z_scvi = adata.obsm["X_scVI"].astype(np.float32)
print(f"  X_scVI shape: {Z_scvi.shape}", flush=True)
umap_scvi_latent = compute_latent_umap(Z_scvi)
del Z_scvi

# ── UMAP 3 — scvi_normalized → log-CPM → PCA ─────────────────────────────────

print("\n=== UMAP 3: scvi_normalized → log-CPM → PCA ===", flush=True)
if "scvi_normalized" not in adata.layers:
    print("ERROR: scvi_normalized layer not found.", flush=True)
    sys.exit(1)
scvi_expr = to_dense(adata.layers["scvi_normalized"]).astype(np.float32)
if hvg_mask is not None:
    scvi_expr = scvi_expr[:, hvg_mask]
scvi_expr = log_cpm(scvi_expr)
umap_scvi_expr = compute_pca_umap(scvi_expr, label="scvi_normalized→PCA")
del scvi_expr

# ── Plot ──────────────────────────────────────────────────────────────────────

print("\nPlotting …", flush=True)

umaps  = [umap_raw, umap_scvi_latent, umap_scvi_expr]
titles = ["Raw (log-CPM → PCA)", "scVI latent (X_scVI)", "scVI expr → PCA\n(transform_batch=VELMESHEV)"]

row_vars  = [source, cell_class]
row_names = ["source", "cell_class"]

# Build colour maps
cats_source, cmap_source     = palette_for(source)
cats_class,  cmap_class      = palette_for(cell_class)
cats_list  = [cats_source, cats_class]
cmap_list  = [cmap_source,  cmap_class]

POINT_SIZE = 0.3
ALPHA      = 0.4
PANEL      = 4.0   # inches per panel
ANNO_W     = 2.0   # legend column width

fig, axes = plt.subplots(
    2, 3,
    figsize=(PANEL * 3 + ANNO_W, PANEL * 2 + 0.5),
    squeeze=False,
)
fig.subplots_adjust(left=0.02, right=0.78, top=0.94, bottom=0.04,
                    hspace=0.12, wspace=0.06)

rng   = np.random.RandomState(RANDOM_SEED)
order = rng.permutation(adata.n_obs)

for row, (values, row_name, cats, cmap) in enumerate(
        zip(row_vars, row_names, cats_list, cmap_list)):

    point_colors = np.array([cmap[v] for v in values])[order]

    for col, (coords, title) in enumerate(zip(umaps, titles)):
        ax = axes[row, col]
        ax.set_facecolor("white")
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#AAAAAA")
            sp.set_linewidth(0.6)

        ax.scatter(
            coords[order, 0], coords[order, 1],
            c=list(point_colors), s=POINT_SIZE, alpha=ALPHA,
            rasterized=True, linewidths=0,
        )

        if row == 0:
            ax.set_title(title, fontsize=9, pad=4)
        if col == 0:
            ax.set_ylabel(row_name, fontsize=9, labelpad=4)

    # Shared legend to the right of each row
    handles = [
        mpatches.Patch(color=cmap[c], label=c)
        for c in cats
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.795, 0.75 - row * 0.5),
        fontsize=7,
        frameon=False,
        title=row_name,
        title_fontsize=8,
        ncol=1,
    )

Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_FILE}", flush=True)
plt.close(fig)

print("\nDone.", flush=True)
