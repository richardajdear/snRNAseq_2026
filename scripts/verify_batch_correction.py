"""
Verify that scVI batch correction (transform_batch=VELMESHEV) has worked.

Compares PCA of log-normalized raw counts vs scVI-corrected expression.
Expectation: source (batch) explains substantially more variance in raw PCA
than in the scVI-corrected PCA.

Outputs a table of R² of `source` on each of the top 20 PCs for both matrices,
plus per-source mean expression of top PC loadings as a sanity check.
"""
import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.decomposition import PCA

DATA_FILE = (
    "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
    "/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld"
    "/scvi_output/integrated.h5ad"
)
N_PCS = 20
SUBSAMPLE = 50_000  # cells to subsample for PCA (memory-efficient)
RANDOM_SEED = 42

# ── helpers ───────────────────────────────────────────────────────────────────

def batch_r2(pcs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute the fraction of variance explained by batch label for each PC.
    Uses one-way ANOVA R² = SS_between / SS_total.
    """
    grand_mean = pcs.mean(axis=0)
    ss_total = ((pcs - grand_mean) ** 2).sum(axis=0)
    ss_between = np.zeros(pcs.shape[1])
    for lab in np.unique(labels):
        mask = labels == lab
        group_mean = pcs[mask].mean(axis=0)
        ss_between += mask.sum() * (group_mean - grand_mean) ** 2
    return ss_between / np.maximum(ss_total, 1e-12)


def dense(mat):
    return mat.toarray() if issparse(mat) else np.array(mat)


def log_cpm(counts: np.ndarray) -> np.ndarray:
    """Log1p-CPM normalization."""
    size = counts.sum(axis=1, keepdims=True)
    size = np.maximum(size, 1)
    return np.log1p(counts / size * 1e6)


# ── load ──────────────────────────────────────────────────────────────────────

print(f"Loading {DATA_FILE} …", flush=True)
adata = sc.read_h5ad(DATA_FILE)
print(f"Shape: {adata.shape}")
print(f"Layers: {list(adata.layers.keys())}")
print(f"obsm keys: {list(adata.obsm.keys())}")

required = ["counts", "scvi_normalized"]
for layer in required:
    if layer not in adata.layers:
        print(f"ERROR: layer '{layer}' not found. Available: {list(adata.layers.keys())}")
        sys.exit(1)

source = np.array(adata.obs["source"])
print(f"\nBatches: {dict(zip(*np.unique(source, return_counts=True)))}")

# ── subsample ─────────────────────────────────────────────────────────────────

rng = np.random.RandomState(RANDOM_SEED)
n = adata.n_obs
if n > SUBSAMPLE:
    idx = rng.choice(n, SUBSAMPLE, replace=False)
    idx.sort()
    print(f"\nSubsampling {SUBSAMPLE}/{n} cells for PCA")
else:
    idx = np.arange(n)
    print(f"\nUsing all {n} cells")

source_sub = source[idx]

# ── HVG mask ──────────────────────────────────────────────────────────────────

hvg_mask = None
if "highly_variable" in adata.var.columns:
    hvg_mask = adata.var["highly_variable"].values
    print(f"Using {hvg_mask.sum()} HVGs for PCA")
else:
    print("No 'highly_variable' column; using all genes")

# ── PCA on raw counts ─────────────────────────────────────────────────────────

print("\n--- Raw counts (log-CPM) PCA ---")
raw = dense(adata.layers["counts"])[idx]
if hvg_mask is not None:
    raw = raw[:, hvg_mask]
raw_logcpm = log_cpm(raw)
del raw

pca_raw = PCA(n_components=N_PCS, random_state=RANDOM_SEED)
pcs_raw = pca_raw.fit_transform(raw_logcpm)
del raw_logcpm

var_raw = pca_raw.explained_variance_ratio_
r2_raw = batch_r2(pcs_raw, source_sub)
print(f"  Variance explained (top 5): {var_raw[:5].round(4)}")
print(f"  R²(batch) top 5 PCs:        {r2_raw[:5].round(4)}")

# ── PCA on scVI-corrected expression ─────────────────────────────────────────

print("\n--- scVI-corrected expression (log-CPM) PCA ---")
scvi_expr = dense(adata.layers["scvi_normalized"])[idx]
if hvg_mask is not None:
    scvi_expr = scvi_expr[:, hvg_mask]
scvi_logcpm = log_cpm(scvi_expr)
del scvi_expr

pca_scvi = PCA(n_components=N_PCS, random_state=RANDOM_SEED)
pcs_scvi = pca_scvi.fit_transform(scvi_logcpm)
del scvi_logcpm

var_scvi = pca_scvi.explained_variance_ratio_
r2_scvi = batch_r2(pcs_scvi, source_sub)
print(f"  Variance explained (top 5): {var_scvi[:5].round(4)}")
print(f"  R²(batch) top 5 PCs:        {r2_scvi[:5].round(4)}")

# ── Summary table ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY: Batch (source) R² by PC")
print("=" * 70)
df = pd.DataFrame({
    "PC":       [f"PC{i+1}" for i in range(N_PCS)],
    "Var_raw":  var_raw.round(4),
    "R2_raw":   r2_raw.round(4),
    "Var_scVI": var_scvi.round(4),
    "R2_scVI":  r2_scvi.round(4),
    "Delta_R2": (r2_raw - r2_scvi).round(4),
})
print(df.to_string(index=False))

print(f"\nMean R²(batch) raw  : {r2_raw.mean():.4f}")
print(f"Mean R²(batch) scVI : {r2_scvi.mean():.4f}")
print(f"Reduction           : {(1 - r2_scvi.mean()/max(r2_raw.mean(), 1e-9))*100:.1f}%")
print("\nExpected: R²_raw >> R²_scVI (batch explains less variance after correction)")

# ── Per-source mean on PC1 (sanity check) ─────────────────────────────────────

print("\n--- Per-source mean on PC1 (raw vs scVI) ---")
for lab in np.unique(source_sub):
    mask = source_sub == lab
    print(f"  {lab:12s}  raw PC1={pcs_raw[mask, 0].mean():+.3f}  scVI PC1={pcs_scvi[mask, 0].mean():+.3f}")

print("\nDone.")
