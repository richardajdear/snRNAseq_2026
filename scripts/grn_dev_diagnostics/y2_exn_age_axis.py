#!/usr/bin/env python3
"""
Y2 — re-run the late-maturation test on the ExN-ONLY joint V3 embedding.

The W-stage (w_age_axis.py) used the shipped ALL-cell X_scVI. This re-runs the
two key tests on the fresh ExN-only latent (VelPsychAD_V3_ExN_dev30), where the
full latent capacity encodes within-ExN maturation:

  PART 1 — child vs adolescent separability in the ExN latent (grouped-CV AUC +
           max|rho(latent dim, age)|), per cohort. Compare to the all-cell
           baseline (PsychAD 0.62, Vel-V3 0.67).
  PART 2 — the data-driven late-maturation axis: fit a child->adol direction in
           the ExN latent (donor-grouped LogisticRegression), score every cell,
           and test whether the EXTERNAL C3+ programme aligns with it (does C3+
           rise toward the immature/childhood end?). Cross-check vs the 9-gene
           early-diff module and vs rho(age-axis, age) (is it really an age
           axis?) and rho(age-axis, module) (over-correction guard).

Per-cell C3+ and the maturity module are computed from the integrated HVG
counts (var_names = Ensembl); C3+ x HVG coverage (gene count + weight mass) is
reported so the partial-coverage caveat is explicit.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=02:00:00 --mem=200G --cpus-per-task=16 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y2_exn_age_axis.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR, build_c3plus_table

EXN_INTEG = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_ExN_dev30/scvi_output/integrated.h5ad"
ALLCELL_INTEG = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"
GROUPS = ["PsychAD-V3", "Velmeshev-V3"]
SRC = {"PsychAD-V3": "PSYCHAD", "Velmeshev-V3": "VELMESHEV"}
COLORS = {"PsychAD-V3": "#C0392B", "Velmeshev-V3": "#2980B9"}
AGE_LO, AGE_HI, BOUND = 1.0, 25.0, 10.0
MODULE_ENS = ["ENSG00000171532", "ENSG00000127152", "ENSG00000119042",
              "ENSG00000081189", "ENSG00000104722", "ENSG00000100285",
              "ENSG00000067715", "ENSG00000132639", "ENSG00000078018"]


def grouped_auc(Z, y, groups, max_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    n_pos = len(np.unique(groups[y == 1])); n_neg = len(np.unique(groups[y == 0]))
    if n_pos < 2 or n_neg < 2:
        return np.nan, 0
    ns = min(max_splits, n_pos, n_neg)
    gkf = GroupKFold(n_splits=ns); aucs = []
    for tr, te in gkf.split(Z, y, groups=groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000).fit(Z[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Z[te])[:, 1]))
    return (float(np.mean(aucs)) if aucs else np.nan), len(aucs)


def per_cell_c3_and_module(adata):
    """Return per-cell C3+ score (CP1e6 over C3+∩HVG) and 9-gene module (mean log1p CP10k)."""
    c3 = build_c3plus_table().set_index("ensembl_id")["weight"]
    var_ens = pd.Index(adata.var_names.astype(str))
    counts = sp.csr_matrix(adata.layers["counts"] if "counts" in adata.layers else adata.X)
    tot = np.asarray(counts.sum(1)).ravel()
    inv = 1.0 / np.where(tot > 0, tot, 1)
    # C3+
    hit = var_ens.intersection(c3.index)
    w = c3.reindex(hit).values
    cov_frac = float(c3.reindex(hit).sum() / c3.sum())
    cidx = [var_ens.get_loc(g) for g in hit]
    c3_raw = np.asarray(counts[:, cidx].multiply(inv[:, None]).dot(w)).ravel() * 1e6
    # module: mean log1p CP10k of the 9 markers present
    midx = [var_ens.get_loc(g) for g in MODULE_ENS if g in var_ens]
    cp10k = counts[:, midx].multiply(inv[:, None]) * 1e4
    mod = np.asarray(np.log1p(cp10k.todense())).mean(1)
    print(f"  C3+ coverage: {len(hit)}/{len(c3)} genes in HVG "
          f"({100*cov_frac:.0f}% of weight mass); module markers: {len(midx)}/9")
    return c3_raw, mod, len(hit), cov_frac


def main():
    print(f"Loading ExN embedding {EXN_INTEG} (backed) ...")
    a = ad.read_h5ad(EXN_INTEG, backed="r")
    obs = a.obs.copy()
    Z = np.asarray(a.obsm["X_scVI"][:])
    counts = sp.csr_matrix(a.X[:])
    adata = ad.AnnData(X=counts, obs=obs, var=a.var.copy())
    adata.layers["counts"] = counts
    adata.obsm["X_scVI"] = Z
    del a
    print(f"  {adata.n_obs:,} ExN cells, latent {Z.shape[1]}")

    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    src = obs["source"].astype(str).values
    don = obs["individual"].astype(str).values if "individual" in obs.columns else obs.index.astype(str)
    c3, mod, n_c3, cov = per_cell_c3_and_module(adata)
    adata.obs["per_cell_c3"] = c3
    adata.obs["module9"] = mod
    adata.obs["age_axis"] = np.nan

    # ===================== PART 1: separability =====================
    print("\n" + "=" * 64 + "\nPART 1 — child vs adol separability on ExN latent\n" + "=" * 64)
    rows = []
    for g in GROUPS:
        s = SRC[g]
        m = (src == s) & (age >= AGE_LO) & (age < AGE_HI) & np.isfinite(age)
        Zg, ag, dg = Z[m], age[m], don[m]
        y = (ag < BOUND).astype(int)
        auc, nf = grouped_auc(Zg, y, dg)
        # donor-mean latent vs age (best dim)
        dm = pd.DataFrame(Zg, index=dg).groupby(level=0).mean()
        da = pd.Series(ag, index=dg).groupby(level=0).first().reindex(dm.index)
        best_rho = float(np.nanmax([abs(stats.spearmanr(dm[c], da)[0]) for c in dm.columns]))
        nd = len(np.unique(dg)); ncd = len(np.unique(dg[y == 1]))
        rows.append({"group": g, "n_cells": int(m.sum()), "n_donors": nd,
                     "n_child_donors": ncd, "cv_auc": auc, "folds": nf,
                     "max_abs_rho_latent_age": best_rho})
        print(f"{g}: {m.sum():,} cells, {nd} donors ({ncd} child); "
              f"AUC={auc:.3f}; max|rho(dim,age)|={best_rho:.2f}")
    sep = pd.DataFrame(rows); sep.to_csv(OUT_DIR / "y2_exn_separability.csv", index=False)

    # ===================== PART 2: data-driven age axis & C3+ alignment =====================
    print("\n" + "=" * 64 + "\nPART 2 — data-driven late-maturation axis vs external C3+\n" + "=" * 64)
    from sklearn.linear_model import LogisticRegression
    # fit child->adol axis on POOLED postnatal-window cells (both cohorts)
    mpool = (age >= AGE_LO) & (age < AGE_HI) & np.isfinite(age) & np.isin(src, list(SRC.values()))
    yall = (age[mpool] < BOUND).astype(int)
    clf = LogisticRegression(max_iter=2000).fit(Z[mpool], yall)
    # axis score: orient so HIGHER = older (adolescent) -> use -decision (since y=1 is child)
    axis_full = -(Z @ clf.coef_.ravel())
    adata.obs["age_axis"] = axis_full

    summ = []
    for g in ["POOLED"] + GROUPS:
        if g == "POOLED":
            m = mpool
        else:
            m = mpool & (src == SRC[g])
        ax, cc, mm, aa = axis_full[m], c3[m], mod[m], age[m]
        r_ax_age = stats.spearmanr(ax, aa).correlation
        r_ax_c3 = stats.spearmanr(ax, cc).correlation
        r_ax_mod = stats.spearmanr(ax, mm).correlation
        r_age_c3 = stats.spearmanr(aa, cc).correlation
        summ.append({"group": g, "n_cells": int(m.sum()),
                     "rho_axis_age": r_ax_age, "rho_axis_c3": r_ax_c3,
                     "rho_axis_module": r_ax_mod, "rho_age_c3": r_age_c3})
        print(f"{g}: n={m.sum():,}  rho(axis,age)={r_ax_age:+.2f}  "
              f"rho(axis,C3+)={r_ax_c3:+.2f}  rho(axis,module)={r_ax_mod:+.2f}  "
              f"rho(age,C3+)={r_age_c3:+.2f}")
    s2 = pd.DataFrame(summ)
    s2.attrs = {}
    s2.to_csv(OUT_DIR / "y2_age_axis_c3.csv", index=False)
    pd.DataFrame({"c3_hvg_genes": [n_c3], "c3_weight_coverage": [cov]}).to_csv(
        OUT_DIR / "y2_c3_coverage.csv", index=False)

    # ===================== figure =====================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    # (a) AUC bars: all-cell baseline vs ExN
    base = {"PsychAD-V3": 0.62, "Velmeshev-V3": 0.67}
    x = np.arange(len(GROUPS)); wbar = 0.38
    axes[0].bar(x - wbar/2, [base[g] for g in GROUPS], wbar, label="all-cell latent", color="#BDC3C7")
    axes[0].bar(x + wbar/2, [sep.set_index("group").loc[g, "cv_auc"] for g in GROUPS], wbar,
                label="ExN-only latent", color=[COLORS[g] for g in GROUPS])
    axes[0].axhline(0.5, color="k", lw=0.6, ls="--")
    axes[0].set_xticks(x); axes[0].set_xticklabels(GROUPS); axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel("child-vs-adol grouped-CV AUC")
    axes[0].set_title("PART 1 — separability: ExN-only vs all-cell latent")
    axes[0].legend(fontsize=8)
    # (b) C3+ vs age-axis decile, per cohort
    for g in GROUPS:
        m = mpool & (src == SRC[g])
        df = pd.DataFrame({"ax": axis_full[m], "c3": c3[m]})
        df["q"] = pd.qcut(df["ax"], 10, labels=False, duplicates="drop")
        bm = df.groupby("q").agg(ax=("ax", "median"), c3=("c3", "mean"))
        axes[1].plot(bm["ax"], bm["c3"], "o-", color=COLORS[g], label=g)
    axes[1].set_xlabel("data-driven age axis (→ older/adolescent)")
    axes[1].set_ylabel("mean per-cell C3+ (CP1e6)")
    axes[1].set_title("PART 2 — external C3+ vs the data-driven axis")
    axes[1].legend(fontsize=8)
    # (c) module vs age-axis decile (over-correction / comparison)
    for g in GROUPS:
        m = mpool & (src == SRC[g])
        df = pd.DataFrame({"ax": axis_full[m], "mod": mod[m]})
        df["q"] = pd.qcut(df["ax"], 10, labels=False, duplicates="drop")
        bm = df.groupby("q").agg(ax=("ax", "median"), mod=("mod", "mean"))
        axes[2].plot(bm["ax"], bm["mod"], "s--", color=COLORS[g], label=g)
    axes[2].set_xlabel("data-driven age axis (→ older/adolescent)")
    axes[2].set_ylabel("mean 9-gene early-diff module")
    axes[2].set_title("9-gene module vs the axis (early-diff, expect flat)")
    axes[2].legend(fontsize=8)
    fig.suptitle("Y2 — ExN-only embedding: does external C3+ align with the data-driven late-maturation axis?",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "y2_exn_age_axis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved y2_exn_separability.csv, y2_age_axis_c3.csv, y2_c3_coverage.csv, "
          f"y2_exn_age_axis.png in {OUT_DIR}")


if __name__ == "__main__":
    main()
