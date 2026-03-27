"""
Diagnose why scVI batch-correction increases PC1 R²(batch).

Hypothesis: after removing technical noise, PC1 captures developmental age,
which is confounded with batch (Velmeshev/Wang ≈ young; Aging/HBCC ≈ adult).

Strategy:
  1. Print age × batch distribution to find overlapping windows.
  2. Re-run PCA in three conditions:
       A. Full dataset (raw)
       B. Full dataset (scVI-corrected)
       C. Age-restricted subset with all four batches represented (scVI-corrected)
  3. For each, report R²(batch) and R²(age) per PC.
  4. After age-restriction, the batch R² on PC1 should drop substantially
     in the scVI condition if the elevation was purely due to age confounding.
  5. Also partial-out age from PCs (residuals of PC ~ age) and re-compute batch R²
     to show what batch R² would look like with age variance removed.
"""

import sys
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

DATA_FILE = (
    "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
    "/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld"
    "/scvi_output/integrated.h5ad"
)

N_PCS = 20
SUBSAMPLE_PER_BATCH = 10_000   # cells per batch for balanced subsampling
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ── helpers ───────────────────────────────────────────────────────────────────

def dense(mat):
    return mat.toarray() if issparse(mat) else np.array(mat)


def log_cpm(counts: np.ndarray) -> np.ndarray:
    size = counts.sum(axis=1, keepdims=True)
    size = np.maximum(size, 1)
    return np.log1p(counts / size * 1e6)


def r2_variable(pcs: np.ndarray, var: np.ndarray, categorical: bool = True) -> np.ndarray:
    """R² of `var` (batch label or continuous age) on each PC."""
    if categorical:
        grand_mean = pcs.mean(axis=0)
        ss_total = ((pcs - grand_mean) ** 2).sum(axis=0)
        ss_between = np.zeros(pcs.shape[1])
        for lab in np.unique(var):
            mask = var == lab
            group_mean = pcs[mask].mean(axis=0)
            ss_between += mask.sum() * (group_mean - grand_mean) ** 2
        return ss_between / np.maximum(ss_total, 1e-12)
    else:
        # continuous: R² of simple linear regression
        r2 = np.zeros(pcs.shape[1])
        v = var.astype(float)
        v = v - v.mean()
        for i in range(pcs.shape[1]):
            pc = pcs[:, i]
            ss_tot = ((pc - pc.mean()) ** 2).sum()
            coef = np.dot(v, pc) / (np.dot(v, v) + 1e-12)
            pred = coef * v
            ss_res = ((pc - pred.mean() - (pc.mean())) ** 2).sum()  # residuals
            # simpler: pearson r²
            r2[i] = pearsonr(v, pc)[0] ** 2
        return r2


def partial_out_age(pcs: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Return residuals of PCs after regressing out age."""
    age_c = age.reshape(-1, 1).astype(float)
    residuals = np.zeros_like(pcs)
    for i in range(pcs.shape[1]):
        reg = LinearRegression().fit(age_c, pcs[:, i])
        residuals[:, i] = pcs[:, i] - reg.predict(age_c)
    return residuals


def run_pca_analysis(X, source, age, hvg_mask, label, subsample_idx=None):
    """Compute PCA, print R²(batch) and R²(age) per PC."""
    if sparse.issparse(X):
        X = dense(X)
    if subsample_idx is not None:
        X = X[subsample_idx]
        source = source[subsample_idx]
        age = age[subsample_idx]

    if hvg_mask is not None:
        X = X[:, hvg_mask]

    pca = PCA(n_components=N_PCS, random_state=RANDOM_SEED)
    pcs = pca.fit_transform(X)

    r2_batch = r2_variable(pcs, source, categorical=True)
    r2_age   = r2_variable(pcs, age,    categorical=False)

    # partial-out age then re-compute batch R²
    pcs_resid = partial_out_age(pcs, age)
    r2_batch_resid = r2_variable(pcs_resid, source, categorical=True)

    print(f"\n{'=' * 70}")
    print(f"  {label}  (n={len(source):,})")
    print(f"  Source counts: { {s: (source==s).sum() for s in np.unique(source)} }")
    print(f"  Age range: {age.min():.1f}–{age.max():.1f} yr  "
          f"(mean {age.mean():.1f})")
    print(f"{'=' * 70}")
    print(f"  {'PC':>4}  {'var_exp':>8}  {'R²_batch':>10}  {'R²_age':>8}  "
          f"{'R²_batch|age':>14}")
    print(f"  {'-' * 52}")
    for i in range(N_PCS):
        print(
            f"  PC{i+1:>2}  {pca.explained_variance_ratio_[i]:>8.4f}"
            f"  {r2_batch[i]:>10.4f}  {r2_age[i]:>8.4f}"
            f"  {r2_batch_resid[i]:>14.4f}"
        )

    wt = pca.explained_variance_ratio_
    print(f"\n  Var-weighted R²(batch):        {np.sum(r2_batch * wt) / wt.sum():.4f}")
    print(f"  Var-weighted R²(age):          {np.sum(r2_age   * wt) / wt.sum():.4f}")
    print(f"  Var-weighted R²(batch|age):    {np.sum(r2_batch_resid * wt) / wt.sum():.4f}")

    per_source_pc1 = {s: pcs[source == s, 0].mean() for s in np.unique(source)}
    print(f"\n  Per-source mean PC1: {per_source_pc1}")
    sys.stdout.flush()
    return pcs, pca


# ── load ──────────────────────────────────────────────────────────────────────

from scipy import sparse

print(f"Loading {DATA_FILE} …", flush=True)
adata = sc.read_h5ad(DATA_FILE)
print(f"Shape: {adata.shape}")
print(f"Layers: {list(adata.layers.keys())}")

source = np.array(adata.obs["source"])
age    = np.array(adata.obs["age_years"], dtype=float)
hvg_mask = (adata.var["highly_variable"].values
            if "highly_variable" in adata.var.columns else None)

# ── Age × batch distribution ──────────────────────────────────────────────────

print("\n\n=== AGE × BATCH DISTRIBUTION ===")
print(f"{'Source':12s}  {'N':>7}  {'Age_min':>8}  {'Age_p10':>8}  "
      f"{'Age_p50':>8}  {'Age_p90':>8}  {'Age_max':>8}")
for src in np.unique(source):
    a = age[source == src]
    print(f"{src:12s}  {len(a):>7,}  {a.min():>8.1f}  "
          f"{np.percentile(a,10):>8.1f}  {np.percentile(a,50):>8.1f}  "
          f"{np.percentile(a,90):>8.1f}  {a.max():>8.1f}")

# Age histogram per batch in bins — include prenatal (<0) explicitly
bins = [-1, 0, 1, 3, 6, 10, 15, 20, 25, 30, 40, 100]
labels_b = ["prenatal(<0)", "0-1", "1-3", "3-6", "6-10",
            "10-15", "15-20", "20-25", "25-30", "30-40", "40-100"]
print("\nCells per age bin × source (prenatal = age < 0):")
df_age = pd.DataFrame({"source": source, "age": age})
df_age["bin"] = pd.cut(age, bins=bins, labels=labels_b, right=False)
ct = df_age.groupby(["bin", "source"], observed=True).size().unstack(fill_value=0)
print(ct.to_string())

# Also print raw count of prenatal cells per batch
print("\nPrenatal cells (age < 0) per batch:")
for src in np.unique(source):
    n_prenatal = ((source == src) & (age < 0)).sum()
    n_total    = (source == src).sum()
    print(f"  {src:12s}: {n_prenatal:>6,} / {n_total:>7,}  ({100*n_prenatal/n_total:.1f}%)")
sys.stdout.flush()

# ── Choose overlapping age window ────────────────────────────────────────────

# Find age range where all 4 batches have ≥500 cells.
# Search from -1 to include prenatal, using 0.5-yr steps near zero.
batches = np.unique(source)
best_window = None
best_n = 0
# Use half-year steps near birth to properly capture prenatal overlap
search_starts = [-1, -0.5, 0, 2, 4, 6, 8, 10, 12]
search_ends   = [s + w for s in search_starts
                 for w in range(4, 42, 2)
                 if s + w <= 40]
for lo in search_starts:
    for width in range(4, 42, 2):
        hi = lo + width
        if hi > 40:
            break
        mask = (age >= lo) & (age < hi)
        counts = {b: ((source == b) & mask).sum() for b in batches}
        if all(v >= 500 for v in counts.values()):
            n_total = sum(counts.values())
            if n_total > best_n:
                best_n = n_total
                best_window = (lo, hi, counts)

if best_window is None:
    print("\nWARNING: No age window with ≥500 cells per batch found; "
          "relaxing to ≥200 cells")
    for lo in search_starts:
        for width in range(4, 42, 2):
            hi = lo + width
            if hi > 40:
                break
            mask = (age >= lo) & (age < hi)
            counts = {b: ((source == b) & mask).sum() for b in batches}
            if all(v >= 200 for v in counts.values()):
                n_total = sum(counts.values())
                if n_total > best_n:
                    best_n = n_total
                    best_window = (lo, hi, counts)

if best_window:
    age_lo, age_hi, age_counts = best_window
    print(f"\nSelected overlap window: age [{age_lo}, {age_hi}) yr")
    print(f"  Cells per batch: {age_counts}")
else:
    print("\nERROR: Could not find overlapping age window. Exiting.")
    sys.exit(1)

sys.stdout.flush()

# ── Subsample: balanced across batches ───────────────────────────────────────

rng = np.random.RandomState(RANDOM_SEED)

def balanced_subsample(source, n_per_batch, extra_mask=None):
    idx = []
    for b in np.unique(source):
        cand = np.where(source == b)[0]
        if extra_mask is not None:
            cand = cand[extra_mask[cand]]
        n = min(n_per_batch, len(cand))
        idx.append(rng.choice(cand, n, replace=False))
    return np.sort(np.concatenate(idx))

full_idx  = balanced_subsample(source, SUBSAMPLE_PER_BATCH)
age_mask  = (age >= age_lo) & (age < age_hi)
age_idx   = balanced_subsample(source, SUBSAMPLE_PER_BATCH, extra_mask=age_mask)

print(f"\nFull balanced subsample: {len(full_idx):,} cells")
print(f"Age-restricted subsample [{age_lo},{age_hi}): {len(age_idx):,} cells")
sys.stdout.flush()

# ── Prepare matrices ──────────────────────────────────────────────────────────

print("\nPreparing raw log-CPM matrix …", flush=True)
raw = dense(adata.layers["counts"])
raw_logcpm = log_cpm(raw)
del raw

print("Preparing scVI-normalized log-CPM matrix …", flush=True)
scvi_expr   = dense(adata.layers["scvi_normalized"])
scvi_logcpm = log_cpm(scvi_expr)
del scvi_expr

# ── PCA analyses ─────────────────────────────────────────────────────────────

run_pca_analysis(
    raw_logcpm, source, age, hvg_mask,
    label="A. Raw counts (log-CPM) — full balanced subsample",
    subsample_idx=full_idx,
)

run_pca_analysis(
    scvi_logcpm, source, age, hvg_mask,
    label="B. scVI-corrected (log-CPM) — full balanced subsample",
    subsample_idx=full_idx,
)

run_pca_analysis(
    scvi_logcpm, source, age, hvg_mask,
    label=f"C. scVI-corrected (log-CPM) — age [{age_lo},{age_hi}) yr only",
    subsample_idx=age_idx,
)

print("\n\nDone.", flush=True)
