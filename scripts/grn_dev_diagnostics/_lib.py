"""
Shared helpers for grn_dev_diagnostics scripts.

All scripts operate on the two ExN_manual_by_donor pseudobulks
(PsychAD_noage_tuning5 and Vel_prepost_noage_tuning5), the AHBA C3+
GRN weights, and the same fixed child/adolescent age windows used by
the main grn_dev_multi notebook.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats

# Make `code/` importable so we can reuse setup_grn / get_ahba_GRN
ROOT_REPO = Path("/home/rajd2/rds/hpc-work/snRNAseq_2026")
if str(ROOT_REPO / "code") not in sys.path:
    sys.path.insert(0, str(ROOT_REPO / "code"))

PSEUDOBULK = {
    "PsychAD":   Path("/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad"),
    "Velmeshev": Path("/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad"),
}

REF_DIR = ROOT_REPO / "reference"
OUT_DIR = ROOT_REPO / "output" / "grn_dev_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHILD = (1.0, 9.0)
ADOL  = (9.0, 25.0)

# Fuzzy childhood/adolescence boundary — boundary age varies over this set,
# and the reported d is the MEAN across boundaries. This makes the cross-
# cohort comparison robust to where the (genuinely fuzzy) developmental
# transition between childhood and adolescence is drawn.
FUZZY_BOUNDARIES = (8, 9, 10, 11, 12)
AGE_LO = 1.0
AGE_HI = 25.0


def fuzzy_d_from_donor_scores(donor_age: np.ndarray,
                              donor_score: np.ndarray,
                              boundaries=FUZZY_BOUNDARIES,
                              age_lo: float = AGE_LO,
                              age_hi: float = AGE_HI,
                              ) -> dict:
    """For donor-level scores, compute Cohen's d (child vs adol) at each
    boundary in `boundaries` (with child=[age_lo, b), adol=[b, age_hi)) and
    return the mean. Returns a dict with the mean, the per-boundary d's,
    and the n's per stage at each boundary."""
    donor_age = np.asarray(donor_age, dtype=float)
    donor_score = np.asarray(donor_score, dtype=float)
    keep = (donor_age >= age_lo) & (donor_age < age_hi) & np.isfinite(donor_score)
    age = donor_age[keep]
    s   = donor_score[keep]
    per_b = []
    for b in boundaries:
        c_mask = age < b
        a_mask = age >= b
        c = s[c_mask]; a = s[a_mask]
        d = cohens_d(c, a) if (len(c) >= 2 and len(a) >= 2) else np.nan
        per_b.append({"boundary": float(b),
                       "n_child": int(c_mask.sum()),
                       "n_adol":  int(a_mask.sum()),
                       "mean_child": float(np.mean(c)) if len(c) else np.nan,
                       "mean_adol":  float(np.mean(a)) if len(a) else np.nan,
                       "cohens_d":   float(d)})
    ds = [r["cohens_d"] for r in per_b if np.isfinite(r["cohens_d"])]
    mean_d = float(np.mean(ds)) if ds else np.nan
    return {"mean_d": mean_d, "per_boundary": per_b}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pseudobulk(name: str) -> ad.AnnData:
    """Read a pseudobulk h5ad."""
    return ad.read_h5ad(PSEUDOBULK[name])


def cpm_from_counts(adata: ad.AnnData, target_sum: float = 1e6) -> ad.AnnData:
    """Return a copy of adata with X replaced by CPM-normalised counts.

    Reads from layers['counts'] (raw integer pseudobulk sums).
    """
    a = adata.copy()
    X = a.layers["counts"]
    if sp.issparse(X):
        X = X.toarray().astype(np.float64)
    else:
        X = X.astype(np.float64)
    tot = X.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    a.X = X * (target_sum / tot)
    return a


def subset_age_window(adata: ad.AnnData,
                      child: Tuple[float, float] = CHILD,
                      adol:  Tuple[float, float] = ADOL,
                      extra_mask: Optional[np.ndarray] = None
                      ) -> ad.AnnData:
    """Restrict to donors whose age falls in childhood OR adolescence
    and tag a 'stage' column."""
    age = adata.obs["age_years"].values
    in_child = (age >= child[0]) & (age < child[1])
    in_adol  = (age >= adol[0])  & (age < adol[1])
    mask = in_child | in_adol
    if extra_mask is not None:
        mask = mask & extra_mask
    out = adata[mask].copy()
    stage = np.where(out.obs["age_years"] < child[1], "child", "adol")
    out.obs["stage"] = pd.Categorical(stage, categories=["child", "adol"])
    return out


# ---------------------------------------------------------------------------
# GRN loading + caching
# ---------------------------------------------------------------------------

GRN_CACHE = OUT_DIR / "ahba_c3plus_ensembl.parquet"


def build_c3plus_table(force: bool = False) -> pd.DataFrame:
    """Return DataFrame with columns ['gene_symbol', 'ensembl_id', 'weight'].

    Caches mygene lookup to disk; both pseudobulks use Ensembl IDs as
    var_names so any GRN consumer can `.merge` on ensembl_id.
    """
    if GRN_CACHE.exists() and not force:
        return pd.read_parquet(GRN_CACHE)

    from regulons import get_ahba_GRN
    from gene_mapping import map_grn_symbols_to_ensembl

    grn = get_ahba_GRN(
        path_to_ahba_weights=str(REF_DIR / "ahba_dme_hcp_top8kgenes_weights.csv"),
        use_weights=True,
    )
    c3 = grn[grn["Network"] == "C3+"].rename(columns={"Gene": "gene_symbol",
                                                       "Importance": "weight"})
    c3 = c3[c3["weight"] > 0].copy()

    # Use the Velmeshev pseudobulk for the lookup since it carries
    # feature_name in var, sparing thousands of mygene queries.
    vel = load_pseudobulk("Velmeshev")
    # map_grn_symbols_to_ensembl expects 'Gene'
    tmp = c3.rename(columns={"gene_symbol": "Gene", "weight": "Importance"})
    tmp_mapped = map_grn_symbols_to_ensembl(tmp, vel)
    tmp_mapped = (tmp_mapped.rename(columns={"Gene": "ensembl_id",
                                              "Importance": "weight"})
                            [["ensembl_id", "weight"]])

    # Recover original symbol via the symbol→ensembl mapping
    sym_to_ens = dict(zip(c3["gene_symbol"], c3["gene_symbol"]))  # placeholder
    sym_to_ens = {}
    if "feature_name" in vel.var.columns:
        for ens, sym in zip(vel.var_names, vel.var["feature_name"]):
            if sym not in sym_to_ens:
                sym_to_ens[sym] = ens
    inv = {v: k for k, v in sym_to_ens.items()}
    tmp_mapped["gene_symbol"] = tmp_mapped["ensembl_id"].map(inv)

    out = tmp_mapped[["gene_symbol", "ensembl_id", "weight"]].dropna(
        subset=["ensembl_id"]).copy()
    out = out.drop_duplicates(subset=["ensembl_id"], keep="first")
    out.to_parquet(GRN_CACHE)
    print(f"Cached {len(out)} C3+ genes → {GRN_CACHE}")
    return out


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges-like pooled-SD Cohen's d (no small-sample correction).

    Positive d means x > y. The grn_dev_multi convention is
    d > 0 when childhood > adolescence — pass (child, adol).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp_ = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp_ == 0 or not np.isfinite(sp_):
        return np.nan
    return (x.mean() - y.mean()) / sp_


def per_gene_child_vs_adol(adata_cpm: ad.AnnData,
                            ensembl_ids: Optional[np.ndarray] = None
                            ) -> pd.DataFrame:
    """For each gene (Ensembl ID), compute donor-level Cohen's d for
    childhood (x) vs adolescence (y).

    Expects `adata_cpm.obs['stage']` already set ('child' / 'adol').
    Operates on adata_cpm.X (assumed CPM, dense).
    """
    if ensembl_ids is None:
        ensembl_ids = adata_cpm.var_names.values
    keep = adata_cpm.var_names.isin(ensembl_ids)
    sub = adata_cpm[:, keep]
    X = sub.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    child_mask = (sub.obs["stage"].values == "child")
    adol_mask  = (sub.obs["stage"].values == "adol")
    if child_mask.sum() < 2 or adol_mask.sum() < 2:
        raise ValueError(f"need >=2 donors per stage; child={child_mask.sum()} adol={adol_mask.sum()}")

    Xc, Xa = X[child_mask], X[adol_mask]
    mc, ma = Xc.mean(0), Xa.mean(0)
    vc, va = Xc.var(0, ddof=1), Xa.var(0, ddof=1)
    nc, na = len(Xc), len(Xa)
    sp_ = np.sqrt(((nc - 1) * vc + (na - 1) * va) / (nc + na - 2))
    d = np.where(sp_ > 0, (mc - ma) / sp_, np.nan)

    # Welch t-test for per-gene p
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(vc / nc + va / na)
        t = np.where(se > 0, (mc - ma) / se, np.nan)
        df = np.where(se > 0,
                      (vc / nc + va / na) ** 2 /
                      ((vc / nc) ** 2 / (nc - 1) + (va / na) ** 2 / (na - 1)),
                      np.nan)
    p = 2 * stats.t.sf(np.abs(t), df)

    return pd.DataFrame({
        "ensembl_id":   sub.var_names.values,
        "n_child":      nc,
        "n_adol":       na,
        "mean_child":   mc,
        "mean_adol":    ma,
        "d":            d,
        "p":            p,
    })


def project_score(adata_cpm: ad.AnnData, weights: pd.Series) -> pd.Series:
    """Donor-level weighted GRN score: sum_g w_g * cpm_g."""
    common = adata_cpm.var_names.intersection(weights.index)
    if len(common) == 0:
        return pd.Series(dtype=float)
    sub = adata_cpm[:, common]
    X = sub.X
    if sp.issparse(X):
        X = X.toarray()
    w = weights.reindex(common).values.astype(float)
    s = np.asarray(X, dtype=np.float64) @ w
    return pd.Series(s, index=sub.obs_names, name="score")
