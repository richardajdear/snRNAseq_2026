"""Shared helpers for the C3-maturation investigation.

Extends scripts/grn_dev_diagnostics/_lib.py (which only built the *C3+*
positive-weight table) with the **signed** full-C3 weight vector and a small
set of candidate scoring functions whose depth-robustness we test in Step 0.

Gene space throughout is Ensembl IDs (the pseudobulk var_names). The AHBA
weights CSV is indexed by gene symbol; we map symbol->Ensembl using the
`gene_symbol` column carried in the pseudobulk var, so no mygene lookup is
needed.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

ROOT = Path("/home/rajd2/rds/hpc-work/snRNAseq_2026")
REF_WEIGHTS = ROOT / "reference/ahba_dme_hcp_top8kgenes_weights.csv"
OUT_DIR = ROOT / "output/c3_maturation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

B = Path("/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated")

# Velmeshev carries both V2 and V3 chemistries -> the natural depth testbed.
PB = {
    "Vel_ExN_by_donor":    B / "Vel_prepost_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
    "Vel_by_cell_class":   B / "Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class.h5ad",
    "Vel_exc_by_celltype": B / "Vel_prepost_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad",
    "PsychAD_ExN_by_donor": B / "PsychAD_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
}


# ---------------------------------------------------------------------------
# AHBA component weights, mapped to Ensembl
# ---------------------------------------------------------------------------

AHBA_ENS_CACHE = OUT_DIR / "ahba_weights_ensembl.parquet"
# Velmeshev pseudobulk carries both gene_symbol and Ensembl var_names ->
# canonical symbol->Ensembl mapper, shared by all (Ensembl-indexed) datasets.
_MAPPER_PB = PB["Vel_ExN_by_donor"]


def _build_ahba_ensembl_cache() -> pd.DataFrame:
    ahba = pd.read_csv(REF_WEIGHTS, index_col=0)[["C1", "C2", "C3"]]  # index=symbol
    ref = ad.read_h5ad(_MAPPER_PB)
    sym_col = "gene_symbol" if "gene_symbol" in ref.var.columns else "feature_name"
    sym2ens = (ref.var.reset_index().rename(columns={"index": "ensembl"})
               .dropna(subset=[sym_col]).drop_duplicates(sym_col)
               .set_index(sym_col)["ensembl"])
    common = ahba.index.intersection(sym2ens.index)
    out = ahba.loc[common].copy()
    out.index = sym2ens.loc[common].values
    out = out[~out.index.duplicated(keep="first")]
    out.index.name = "ensembl"
    out.to_parquet(AHBA_ENS_CACHE)
    return out


def ahba_weights_ensembl(components=("C1", "C2", "C3")) -> pd.DataFrame:
    """Signed AHBA loadings indexed by Ensembl ID (dataset-independent).

    Built once from the AHBA CSV + Velmeshev symbol->Ensembl mapping and cached;
    all datasets here use Ensembl var_names so the same vector applies.
    """
    df = pd.read_parquet(AHBA_ENS_CACHE) if AHBA_ENS_CACHE.exists() else _build_ahba_ensembl_cache()
    return df[list(components)]


def c3_signed(ref_adata: ad.AnnData | None = None) -> pd.Series:
    """Signed C3 weight vector indexed by Ensembl (positive = C3+ pole).

    `ref_adata` is ignored (kept for call-site compatibility); weights come from
    the shared Ensembl cache so they are identical across datasets.
    """
    return ahba_weights_ensembl(("C3",))["C3"]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def cpm_matrix(adata: ad.AnnData, target_sum: float = 1e6) -> np.ndarray:
    X = adata.layers["counts"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float64)
    tot = X.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    return X * (target_sum / tot)


def depth_metrics(adata: ad.AnnData) -> pd.DataFrame:
    """Per-sample depth proxies from the counts layer."""
    X = adata.layers["counts"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    tot = X.sum(axis=1)
    n_cells = adata.obs.get("n_cells", pd.Series(np.nan, index=adata.obs_names)).values
    return pd.DataFrame({
        "total_counts": tot,
        "log10_total": np.log10(np.where(tot > 0, tot, 1)),
        "n_genes_det": (X > 0).sum(axis=1),
        "counts_per_cell": np.where(n_cells > 0, tot / n_cells, np.nan),
    }, index=adata.obs_names)


# ---------------------------------------------------------------------------
# Candidate C3 scores (all return a per-sample np.array aligned to adata)
# ---------------------------------------------------------------------------

def _align(weights: pd.Series, var_names) -> tuple[np.ndarray, np.ndarray]:
    common = var_names.intersection(weights.index)
    idx = np.array([var_names.get_loc(g) for g in common])
    w = weights.loc[common].values.astype(float)
    return idx, w


def score_weighted_cpm(cpm: np.ndarray, var_names, weights: pd.Series,
                       log1p: bool = False, pos_only: bool = False) -> np.ndarray:
    """Sum_g w_g * (log1p) cpm_g.  Signed weights => C3+ minus C3- contrast.
    pos_only restricts to positive weights (reproduces the prior depth-biased
    C3+ aggregate)."""
    w = weights.copy()
    if pos_only:
        w = w[w > 0]
    idx, wv = _align(w, var_names)
    M = cpm[:, idx]
    if log1p:
        M = np.log1p(M)
    return M @ wv


def score_rank_contrast(cpm: np.ndarray, var_names, weights: pd.Series) -> np.ndarray:
    """Scale-invariant AUCell-style contrast.

    Per sample, percentile-rank all genes by CPM, then take the
    |w|-weighted mean percentile of C3+ genes minus that of C3- genes.
    Bounded in [-1, 1]; invariant to any monotone rescaling of a sample,
    so global depth / library-size shifts cancel.
    """
    pos = weights[weights > 0]
    neg = weights[weights < 0]
    ip, wp = _align(pos, var_names)
    ing, wn = _align(-neg, var_names)  # use |w| for negatives
    # percentile rank within each sample (row)
    order = cpm.argsort(axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    n = cpm.shape[1]
    rows = np.arange(cpm.shape[0])[:, None]
    ranks[rows, order] = np.arange(n)[None, :]
    pct = ranks / (n - 1)
    pos_score = (pct[:, ip] * wp).sum(1) / wp.sum()
    neg_score = (pct[:, ing] * wn).sum(1) / wn.sum()
    return pos_score - neg_score


def matched_random_weights(weights: pd.Series, var_names,
                           ref_expr: np.ndarray, seed: int = 0,
                           _cache: dict = {}) -> pd.Series:
    """Build a null weight vector: same number of +/- genes as C3, drawn to
    match the mean-expression distribution of the real C3+/C3- gene sets,
    with the real |weights| reassigned. Used as a negative control.

    Decile->pool-gene index lists are cached across calls (keyed by var_names
    id) so repeated null draws are fast.
    """
    rng = np.random.default_rng(seed)
    key = (id(var_names), id(ref_expr))
    if key not in _cache:
        expr_rank = pd.Series(ref_expr, index=var_names).rank()
        deciles = pd.qcut(expr_rank, 10, labels=False, duplicates="drop")
        pool = var_names.difference(weights.index)
        pool_dec = deciles.reindex(pool).values
        by_dec = {d: pool[pool_dec == d].values for d in range(10)}
        gene_dec = deciles.to_dict()
        _cache[key] = (by_dec, gene_dec)
    by_dec, gene_dec = _cache[key]

    def sample_like(genes, sign):
        out = {}
        for g in genes:
            d = gene_dec.get(g, 0)
            cand = by_dec.get(d, None)
            if cand is None or len(cand) == 0:
                cand = np.concatenate(list(by_dec.values()))
            out[rng.choice(cand)] = abs(weights[g]) * sign
        return out

    common = var_names.intersection(weights.index)
    pos = weights[weights > 0].index.intersection(common)
    neg = weights[weights < 0].index.intersection(common)
    nullw = {}
    nullw.update(sample_like(pos, +1))
    nullw.update(sample_like(neg, -1))
    return pd.Series(nullw)


def downsample_counts(adata: ad.AnnData, target_total: float, seed: int = 0) -> np.ndarray:
    """Binomial-thin each sample's counts to <= target_total, return CPM of the
    thinned matrix. Samples already below target are left unchanged."""
    rng = np.random.default_rng(seed)
    X = adata.layers["counts"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float64)
    tot = X.sum(axis=1)
    p = np.where(tot > target_total, target_total / tot, 1.0)
    thin = rng.binomial(X.astype(np.int64), p[:, None]).astype(np.float64)
    t = thin.sum(axis=1, keepdims=True)
    t[t == 0] = 1.0
    return thin * (1e6 / t)
