#!/usr/bin/env python3
"""Step 1A confound check — is the postnatal within-mature-EN C3 decrease real,
or a Velmeshev sub-dataset/chemistry batch effect confounded with age?

Velmeshev = 3 sub-sources (U01/Herring/Ramos) with different chemistries and
age coverage. We (1) tabulate age x chemistry x dataset, (2) refit the pooled
mature-EN age slope ADDING sub-dataset fixed effects, and (3) report the age
correlation WITHIN each sub-dataset. If the slope collapses once dataset is
controlled / is inconsistent within datasets, the postnatal trend is batch, not
biology, and the proper postnatal test must come from PsychAD V3.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L
from s01a_within_celltype_trajectory import (
    build_df, cluster_ols, partial_spearman, MATURE_EN, AGE_LO, AGE_HI,
)


def main():
    a = ad.read_h5ad(L.PB["Vel_exc_by_celltype"])
    df = build_df(a)
    df["dataset"] = a.obs["dataset"].astype(str).values
    df = df[(df["age_years"] >= AGE_LO) & (df["age_years"] < AGE_HI)
            & (df["n_cells"] >= 20) & df["subtype"].isin(MATURE_EN)].copy()
    print(f"postnatal mature-EN rows: {len(df)}  donors: {df['individual'].nunique()}")

    print("\n--- donors per dataset x chemistry ---")
    dd = df.drop_duplicates("individual")
    print(pd.crosstab(dd["dataset"], dd["chemistry"]).to_string())
    print("\n--- age range per dataset (donors) ---")
    print(dd.groupby("dataset")["age_years"].agg(["min", "median", "max", "size"]).round(2).to_string())

    # pooled slope WITHOUT dataset FE (subtype + depth only)
    def slope(d, with_dataset):
        sts = sorted(d["subtype"].unique())[1:]
        cols = [np.ones(len(d)), d["age_years"].values, d["log10_total"].values]
        names = ["const", "age", "log10_total"]
        for s in sts:
            cols.append((d["subtype"] == s).astype(float).values); names.append(f"st[{s}]")
        if with_dataset:
            for s in sorted(d["dataset"].unique())[1:]:
                cols.append((d["dataset"] == s).astype(float).values); names.append(f"ds[{s}]")
        X = np.column_stack(cols)
        beta, se, t, p = cluster_ols(d["score"].values, X, d["individual"].values)
        ai = names.index("age")
        return beta[ai], se[ai], t[ai], p[ai]

    b0, s0, t0, p0 = slope(df, False)
    b1, s1, t1, p1 = slope(df, True)
    print("\n--- pooled mature-EN age slope ---")
    print(f"  subtype+depth only        : slope={b0:+.4f} t={t0:+.2f} p={p0:.2e}")
    print(f"  + sub-dataset fixed effects: slope={b1:+.4f} t={t1:+.2f} p={p1:.2e}")

    print("\n--- age corr (partial on depth) WITHIN each sub-dataset ---")
    for ds, sub in df.groupby("dataset"):
        if len(sub) < 6:
            print(f"  {ds}: n={len(sub)} (too few)"); continue
        r = partial_spearman(sub["score"].values, sub["age_years"].values, sub["log10_total"].values)
        print(f"  {ds}: n={len(sub)} donors={sub['individual'].nunique()} "
              f"age range=[{sub['age_years'].min():.1f},{sub['age_years'].max():.1f}] "
              f"rho_age|depth={r:+.3f}")


if __name__ == "__main__":
    main()
