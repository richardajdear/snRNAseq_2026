#!/usr/bin/env python3
"""
Group D1 + D3 — shared gene universe + unweighted top-1000 sanity.

The PsychAD pseudobulk has 34,176 var_names; Velmeshev has 17,663. CPM
normalises by total counts across all var_names, so even when the same
genes are being measured the *per-gene CPM* values are scaled
differently. This script:

  D1  Intersect var_names → restrict both pseudobulks to a shared
      gene universe → re-CPM from layers['counts'] → re-project the
      AHBA C3+ weighted score → report new Cohen's d on aggregate
      AND per-gene.
  D3  Same exercise using the unweighted top-1000 C3+ gene sum (the
      §4.2 sanity score in grn_dev_multi.md).

Also reports the "top contributing genes" to the aggregate weighted
score in each dataset, to test whether the apparent disagreement is
driven by a handful of high-CPM, high-weight genes that differ
between datasets.

Outputs:
    d1_shared_universe_summary.csv
    d1_per_gene_psychad_vs_velmeshev_shared.parquet
    d3_top1000_unweighted.csv
    d_top_contributors.csv      what genes drive the aggregate
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (
    load_pseudobulk, cpm_from_counts, subset_age_window,
    per_gene_child_vs_adol, project_score, build_c3plus_table,
    cohens_d, OUT_DIR,
)


def shared_universe(psy, vel):
    """Intersect var_names and return aligned subsets of both adatas."""
    common = psy.var_names.intersection(vel.var_names)
    print(f"  shared var_names: {len(common)} "
          f"(PsychAD {psy.n_vars}, Velmeshev {vel.n_vars})")
    return psy[:, common].copy(), vel[:, common].copy()


def aggregate_d(adata_cpm, weights, label, extra=None):
    s = project_score(adata_cpm, weights)
    df = adata_cpm.obs[["stage"]].join(s)
    d = cohens_d(df[df["stage"] == "child"]["score"].values,
                 df[df["stage"] == "adol"]["score"].values)
    out = {
        "subset": label,
        "n_child": int((df["stage"] == "child").sum()),
        "n_adol":  int((df["stage"] == "adol").sum()),
        "mean_child": float(df[df["stage"] == "child"]["score"].mean()),
        "mean_adol":  float(df[df["stage"] == "adol"]["score"].mean()),
        "cohens_d": float(d),
    }
    if extra:
        out.update(extra)
    return out


def top_contributors(adata_cpm, weights, n=20):
    """For each gene, contribution to aggregate = weight × mean_CPM. Returns
    top-N genes with their child-vs-adol d."""
    common = adata_cpm.var_names.intersection(weights.index)
    sub = adata_cpm[:, common]
    X = sub.X
    if sp.issparse(X):
        X = X.toarray()
    mean_cpm = np.asarray(X, dtype=np.float64).mean(0)
    w = weights.reindex(common).values
    contrib = mean_cpm * w
    order = np.argsort(contrib)[::-1][:n]
    g = pd.DataFrame({
        "ensembl_id": common[order],
        "weight":     w[order],
        "mean_cpm":   mean_cpm[order],
        "contribution": contrib[order],
    })
    # add per-gene d
    pg = per_gene_child_vs_adol(adata_cpm, ensembl_ids=g["ensembl_id"].values)
    g = g.merge(pg[["ensembl_id", "d", "p"]], on="ensembl_id")
    return g


def main():
    weights_df = build_c3plus_table()
    weights = weights_df.set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes")

    psy_raw = load_pseudobulk("PsychAD")
    vel_raw = load_pseudobulk("Velmeshev")

    rows = []

    # ===== Baseline: each dataset on its own var universe (matches grn_dev_multi) =====
    psy_cpm_full = subset_age_window(cpm_from_counts(psy_raw))
    vel_cpm_full = subset_age_window(cpm_from_counts(vel_raw))
    rows.append(aggregate_d(psy_cpm_full, weights, "PsychAD_full_var"))
    rows.append(aggregate_d(vel_cpm_full, weights, "Velmeshev_full_var"))

    # ===== D1: shared gene universe =====
    print("\n=== D1: shared gene universe ===")
    psy_shared, vel_shared = shared_universe(psy_raw, vel_raw)
    psy_cpm = subset_age_window(cpm_from_counts(psy_shared))
    vel_cpm = subset_age_window(cpm_from_counts(vel_shared))
    rows.append(aggregate_d(psy_cpm, weights, "PsychAD_shared"))
    rows.append(aggregate_d(vel_cpm, weights, "Velmeshev_shared"))

    # also Vel V2/V3 with shared universe
    for chem in ("V2", "V3"):
        vsub = vel_shared[vel_shared.obs["chemistry"] == chem].copy()
        vsub = subset_age_window(cpm_from_counts(vsub))
        rows.append(aggregate_d(vsub, weights, f"Velmeshev_shared_{chem}"))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "d1_shared_universe_summary.csv", index=False)
    print(summary.to_string(index=False))

    # ===== D1 per-gene table on the shared universe =====
    print("\n=== D1: per-gene on shared universe ===")
    c3_ens = weights.index.values
    pg_psy = per_gene_child_vs_adol(psy_cpm, ensembl_ids=c3_ens)
    pg_vel = per_gene_child_vs_adol(vel_cpm, ensembl_ids=c3_ens)
    merged = (pg_psy.add_suffix("_psy")
              .rename(columns={"ensembl_id_psy": "ensembl_id"})
              .merge(pg_vel.add_suffix("_vel")
                     .rename(columns={"ensembl_id_vel": "ensembl_id"}),
                     on="ensembl_id"))
    merged = merged.merge(weights.rename("weight"),
                          left_on="ensembl_id", right_index=True)
    merged.to_parquet(OUT_DIR / "d1_per_gene_psychad_vs_velmeshev_shared.parquet")
    print(f"  shared per-gene table: {len(merged)} rows")

    # Concordance on shared universe
    from scipy import stats
    sub = merged.dropna(subset=["d_psy", "d_vel"])
    r_p = stats.pearsonr(sub["d_psy"], sub["d_vel"])[0]
    r_s = stats.spearmanr(sub["d_psy"], sub["d_vel"])[0]
    # weighted
    w = sub["weight"].values
    wm = lambda v: float(np.sum(w * v) / np.sum(w))
    print(f"  Pearson r (shared universe): {r_p:.3f}   Spearman r: {r_s:.3f}")
    print(f"  weighted-mean d_psychad: {wm(sub['d_psy']):.3f}")
    print(f"  weighted-mean d_velmeshev: {wm(sub['d_vel']):.3f}")
    print(f"  frac pos d_psy: {(sub['d_psy'] > 0).mean():.3f}")
    print(f"  frac pos d_vel: {(sub['d_vel'] > 0).mean():.3f}")

    # ===== D3: unweighted top-1000 sanity =====
    print("\n=== D3: unweighted top-1000 sanity ===")
    top1000 = weights.sort_values(ascending=False).head(1000)
    flat = pd.Series(1.0, index=top1000.index)
    d3_rows = []
    for label, ad_ in [
        ("PsychAD_full_var",   psy_cpm_full),
        ("Velmeshev_full_var", vel_cpm_full),
        ("PsychAD_shared",     psy_cpm),
        ("Velmeshev_shared",   vel_cpm),
    ]:
        d3_rows.append(aggregate_d(ad_, flat, label,
                                   extra={"score_type": "unweighted_top1000_sum"}))
    pd.DataFrame(d3_rows).to_csv(OUT_DIR / "d3_top1000_unweighted.csv", index=False)
    print(pd.DataFrame(d3_rows).to_string(index=False))

    # ===== Top contributors per dataset =====
    print("\n=== Top contributors to aggregate weighted score ===")
    tc_psy = top_contributors(psy_cpm_full, weights).assign(dataset="PsychAD")
    tc_vel = top_contributors(vel_cpm_full, weights).assign(dataset="Velmeshev")
    tc = pd.concat([tc_psy, tc_vel], ignore_index=True)
    # gene symbol if cache has it
    sym = weights_df.set_index("ensembl_id")["gene_symbol"]
    tc["gene_symbol"] = tc["ensembl_id"].map(sym)
    tc = tc[["dataset", "gene_symbol", "ensembl_id", "weight",
             "mean_cpm", "contribution", "d", "p"]]
    tc.to_csv(OUT_DIR / "d_top_contributors.csv", index=False)
    print("PsychAD top 10 contributors:")
    print(tc[tc["dataset"] == "PsychAD"].head(10).to_string(index=False))
    print("\nVelmeshev top 10 contributors:")
    print(tc[tc["dataset"] == "Velmeshev"].head(10).to_string(index=False))

    # Overlap between top-20 contributors of the two datasets
    set_psy = set(tc[tc["dataset"] == "PsychAD"]["ensembl_id"])
    set_vel = set(tc[tc["dataset"] == "Velmeshev"]["ensembl_id"])
    overlap = set_psy & set_vel
    print(f"\nOverlap in top-20 contributors: {len(overlap)} / 20")

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
