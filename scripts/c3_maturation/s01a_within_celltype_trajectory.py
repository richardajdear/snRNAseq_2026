#!/usr/bin/env python3
"""C3-maturation Step 1A — within-cell-type C3 trajectory (THE discriminating test).

If C3 is only neuron-vs-glia composition, then conditioning on a single mature
cell type the C3 score should be FLAT across age. If C3 also encodes a
maturational program, then within mature excitatory-neuron (EN) subtypes the
depth-robust C3 score (signed_logcpm, chosen in Step 0) should keep rising
across postnatal development, after differentiation is complete.

Uses the Velmeshev excitatory_by_celltype pseudobulk (donor x EN subtype).
Headline estimate: age slope of the C3 score pooled across MATURE EN subtypes,
controlling subtype identity + sequencing depth, with donor-clustered robust SE
(donors recur across subtypes). Compared against:
  - the progenitor pool (differentiation axis; expected steep)
  - the all-ExN aggregate (includes progenitor->mature composition)
  - V3-only (artefact replication)

Inline-safe. Run via singularity (see CLAUDE.md).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L

MATURE_EN = ["L2-3", "L4", "L5", "L6", "L5-6-IT"]
AGE_LO, AGE_HI = 1.0, 30.0   # postnatal, "after differentiation" window


def cluster_ols(y, X, groups):
    """OLS with cluster-robust (by `groups`) covariance. Returns beta, SE, t, p
    aligned to X columns. df from #clusters."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    G = len(np.unique(groups))
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        m = groups == g
        Xg = X[m]; ug = u[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    scale = (G / (G - 1)) * ((n - 1) / (n - k))
    V = scale * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), G - 1)
    return beta, se, t, p


def design(df, subtype_dummies=True):
    """Build [1, age, log10_total, subtype dummies...] design matrix."""
    cols = {"const": np.ones(len(df)), "age": df["age_years"].values,
            "log10_total": df["log10_total"].values}
    names = ["const", "age", "log10_total"]
    if subtype_dummies:
        sts = sorted(df["subtype"].unique())[1:]  # drop first as reference
        for s in sts:
            cols[f"st[{s}]"] = (df["subtype"] == s).astype(float).values
            names.append(f"st[{s}]")
    X = np.column_stack([cols[n] for n in names])
    return X, names


def partial_spearman(x, y, z):
    def resid(a, b):
        ra, rb = stats.rankdata(a), stats.rankdata(b)
        rb = np.c_[np.ones_like(rb), rb]
        beta, *_ = np.linalg.lstsq(rb, ra, rcond=None)
        return ra - rb @ beta
    return stats.spearmanr(resid(x, z), resid(y, z)).statistic


def build_df(a):
    w = L.c3_signed(a)
    cpm = L.cpm_matrix(a)
    score = L.score_weighted_cpm(cpm, a.var_names, w, log1p=True)  # signed_logcpm
    dm = L.depth_metrics(a)
    df = pd.DataFrame({
        "individual": a.obs["individual"].astype(str).values,
        "subtype": a.obs["cell_type_aligned"].astype(str).values,
        "age_years": pd.to_numeric(a.obs["age_years"], errors="coerce").values,
        "chemistry": a.obs["chemistry"].astype(str).values,
        "n_cells": a.obs["n_cells"].values,
        "score": score,
        "log10_total": dm["log10_total"].values,
    })
    return df


def main():
    print("Loading Velmeshev excitatory_by_celltype pseudobulk ...", flush=True)
    a = ad.read_h5ad(L.PB["Vel_exc_by_celltype"])
    df = build_df(a)
    print("  subtypes:", dict(df["subtype"].value_counts()), flush=True)

    # postnatal window, require enough cells per pseudobulk
    df = df[(df["age_years"] >= AGE_LO) & (df["age_years"] < AGE_HI)
            & (df["n_cells"] >= 20)].copy()
    print(f"  postnatal [{AGE_LO},{AGE_HI}) rows with >=20 cells: {len(df)}", flush=True)

    # ---------- per-subtype partial age corr ----------
    print("\n--- per-subtype partial Spearman(score, age | depth) ---")
    rows = []
    for st, sub in df.groupby("subtype"):
        if len(sub) < 6:
            continue
        r = partial_spearman(sub["score"].values, sub["age_years"].values,
                             sub["log10_total"].values)
        sub3 = sub[sub["chemistry"] == "V3"]
        r3 = (partial_spearman(sub3["score"].values, sub3["age_years"].values,
                               sub3["log10_total"].values)
              if len(sub3) >= 6 else np.nan)
        rows.append({"subtype": st, "n": len(sub), "n_donors": sub["individual"].nunique(),
                     "rho_age_partial": r, "rho_age_partial_V3": r3})
    per_st = pd.DataFrame(rows).sort_values("subtype")
    print(per_st.round(3).to_string(index=False))
    per_st.to_csv(L.OUT_DIR / "s01a_per_subtype_age_corr.csv", index=False)

    # ---------- headline: pooled mature-EN age slope, controlling subtype+depth ----------
    def pooled_slope(d, label):
        X, names = design(d, subtype_dummies=True)
        beta, se, t, p = cluster_ols(d["score"].values, X, d["individual"].values)
        ai = names.index("age")
        print(f"  [{label}] n={len(d)} donors={d['individual'].nunique()}  "
              f"age slope={beta[ai]:+.4f}  SE={se[ai]:.4f}  t={t[ai]:+.2f}  p={p[ai]:.2e}")
        return {"set": label, "n": len(d), "n_donors": int(d["individual"].nunique()),
                "age_slope": beta[ai], "se": se[ai], "t": t[ai], "p": p[ai]}

    print("\n--- HEADLINE: pooled MATURE-EN age slope (controls subtype identity + depth, "
          "donor-clustered SE) ---")
    mature = df[df["subtype"].isin(MATURE_EN)].copy()
    head = []
    head.append(pooled_slope(mature, "mature_EN_all"))
    head.append(pooled_slope(mature[mature["chemistry"] == "V3"], "mature_EN_V3only"))
    # contrast: progenitors (differentiation axis) if present
    prog = df[df["subtype"].str.contains("Progenitor", case=False)].copy()
    if len(prog) >= 6:
        Xp = np.column_stack([np.ones(len(prog)), prog["age_years"].values,
                              prog["log10_total"].values])
        bp, sep, tp, pp = cluster_ols(prog["score"].values, Xp, prog["individual"].values)
        print(f"  [progenitors] n={len(prog)}  age slope={bp[1]:+.4f}  t={tp[1]:+.2f}  p={pp[1]:.2e}")
        head.append({"set": "progenitors", "n": len(prog),
                     "n_donors": int(prog["individual"].nunique()),
                     "age_slope": bp[1], "se": sep[1], "t": tp[1], "p": pp[1]})
    head_df = pd.DataFrame(head)
    head_df.to_csv(L.OUT_DIR / "s01a_pooled_slopes.csv", index=False)

    # ---------- aggregate-vs-within comparison (composition contribution) ----------
    agg = ad.read_h5ad(L.PB["Vel_ExN_by_donor"])
    adf = build_df_donor(agg)
    adf = adf[(adf["age_years"] >= AGE_LO) & (adf["age_years"] < AGE_HI)]
    rho_agg = partial_spearman(adf["score"].values, adf["age_years"].values,
                               adf["log10_total"].values)
    rho_within = partial_spearman(mature["score"].values, mature["age_years"].values,
                                  mature["log10_total"].values)
    print(f"\n--- composition vs program (postnatal) ---")
    print(f"  all-ExN aggregate (incl. progenitor->mature comp): rho_age|depth = {rho_agg:+.3f}")
    print(f"  within mature-EN subtypes (comp removed):          rho_age|depth = {rho_within:+.3f}")
    print(f"  => within-type program retains {100*rho_within/rho_agg:.0f}% of the aggregate corr"
          if rho_agg != 0 else "")

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, st in zip(axes.ravel(), MATURE_EN + ["Progenitors"]):
        sub = df[df["subtype"].str.fullmatch(st, case=False) |
                 df["subtype"].str.contains(st, case=False)]
        sub = sub[sub["subtype"].isin([st]) | sub["subtype"].str.contains("Progenitor")] if st == "Progenitors" else sub[sub["subtype"] == st]
        for ch, c in [("V2", "C3"), ("V3", "C0")]:
            m = sub["chemistry"] == ch
            ax.scatter(sub["age_years"][m], sub["score"][m], s=28, alpha=.75, color=c, label=ch)
        if len(sub) >= 4:
            z = np.polyfit(sub["age_years"], sub["score"], 1)
            xs = np.linspace(sub["age_years"].min(), sub["age_years"].max(), 20)
            ax.plot(xs, np.polyval(z, xs), "k--", lw=1.5)
        r = per_st.set_index("subtype")["rho_age_partial"].get(st, np.nan)
        ax.set_title(f"{st}  (rho_age|depth={r:+.2f}, n={len(sub)})", fontsize=10)
        ax.set_xlabel("age (years)"); ax.set_ylabel("C3 signed_logcpm")
        ax.legend(fontsize=7)
    fig.suptitle("Step 1A: within-cell-type C3 trajectory (Velmeshev, postnatal)\n"
                 "flat => composition-only; rising within mature EN => maturation program", y=1.01)
    fig.tight_layout()
    fig.savefig(L.OUT_DIR / "s01a_within_celltype_trajectory.png", dpi=140, bbox_inches="tight")
    print(f"\nOutputs -> {L.OUT_DIR}")


def build_df_donor(a):
    """Same as build_df but for an object without subtype (all-ExN by donor)."""
    w = L.c3_signed(a)
    cpm = L.cpm_matrix(a)
    score = L.score_weighted_cpm(cpm, a.var_names, w, log1p=True)
    dm = L.depth_metrics(a)
    return pd.DataFrame({
        "individual": a.obs["individual"].astype(str).values,
        "age_years": pd.to_numeric(a.obs["age_years"], errors="coerce").values,
        "chemistry": a.obs["chemistry"].astype(str).values,
        "score": score, "log10_total": dm["log10_total"].values,
    })


if __name__ == "__main__":
    main()
