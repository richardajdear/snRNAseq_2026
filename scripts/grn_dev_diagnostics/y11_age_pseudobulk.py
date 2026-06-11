#!/usr/bin/env python3
"""
Y11 — how convincing is the donor-level C3+/module vs AGE decline?

Donor pseudobulk (ALL cells, 1-25y, good embedding). 2x2 grid:
  cols = {PsychAD-V3, Velmeshev-V3}, rows = {module vs age, C3+ vs age}
Each point = one donor. OLS fit line + 95% CI band; annotate Spearman r (p),
Pearson r (p), n donors, age range. Dumps per-donor table to CSV.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, scipy.stats as stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR, build_c3plus_table

GOOD = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_V3only_tuning5/scvi_output/integrated.h5ad"
MODULE_ENS = ["ENSG00000171532", "ENSG00000127152", "ENSG00000119042",
              "ENSG00000081189", "ENSG00000104722", "ENSG00000100285",
              "ENSG00000067715", "ENSG00000132639", "ENSG00000078018"]
COHORTS = {"PsychAD-V3": "PSYCHAD-V3", "Velmeshev-V3": "VELMESHEV-V3"}
COL = {"PsychAD-V3": "#C0392B", "Velmeshev-V3": "#2980B9"}


def fit_band(ax, x, y, color):
    xs = np.linspace(x.min(), x.max(), 100)
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; dof = max(len(x) - 2, 1)
    s2 = (resid @ resid) / dof
    Xs = np.vstack([np.ones_like(xs), xs]).T
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xs, cov, Xs))
    yh = Xs @ beta
    ax.plot(xs, yh, color=color, lw=2)
    ax.fill_between(xs, yh - 1.96 * se, yh + 1.96 * se, color=color, alpha=0.15)


def main():
    a = ad.read_h5ad(GOOD, backed="r")
    obs = a.obs
    bk = "source-chemistry" if "source-chemistry" in obs else "source"
    batch = obs[bk].astype(str).values
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    don = obs["individual"].astype(str).values
    counts = sp.csr_matrix(a.X[:])
    tot = np.asarray(counts.sum(1)).ravel(); inv = 1.0 / np.where(tot > 0, tot, 1)
    var = pd.Index(a.var_names.astype(str))
    c3w = build_c3plus_table().set_index("ensembl_id")["weight"]
    hit = var.intersection(c3w.index)
    c3 = np.asarray(counts[:, [var.get_loc(g) for g in hit]].multiply(inv[:, None])
                    .dot(c3w.reindex(hit).values)).ravel() * 1e6
    midx = [var.get_loc(g) for g in MODULE_ENS if g in var]
    module = np.asarray(np.log1p(counts[:, midx].multiply(inv[:, None]) * 1e4).todense()).mean(1)

    fig, axes = plt.subplots(2, len(COHORTS), figsize=(7 * len(COHORTS), 11))
    tbl = []
    for j, (lab, tag) in enumerate(COHORTS.items()):
        m = (batch == tag) & (age >= 1) & (age < 25) & np.isfinite(age)
        df = pd.DataFrame({"c3": c3[m], "mo": module[m], "age": age[m], "d": don[m]})
        pb = df.groupby("d").agg(c3=("c3", "mean"), mo=("mo", "mean"),
                                 age=("age", "mean"), n=("c3", "size")).reset_index()
        pb["cohort"] = lab; tbl.append(pb)
        for r, (yc, ylab) in enumerate([("mo", "donor-mean maturity module (logCP10K)"),
                                        ("c3", "donor-mean C3+ (CPM-wt)")]):
            ax = axes[r, j]
            ax.scatter(pb["age"], pb[yc], s=np.sqrt(pb["n"]) * 2.5, color=COL[lab],
                       edgecolors="k", linewidths=0.4, alpha=0.85)
            fit_band(ax, pb["age"].values, pb[yc].values, COL[lab])
            sr = stats.spearmanr(pb["age"], pb[yc]); pr = stats.pearsonr(pb["age"], pb[yc])
            ax.set_title(f"{lab}  (n={len(pb)} donors, age {pb['age'].min():.1f}-{pb['age'].max():.1f}y)\n"
                         f"Spearman r={sr.correlation:+.2f} p={sr.pvalue:.3f} | "
                         f"Pearson r={pr.statistic:+.2f} p={pr.pvalue:.3f}", fontsize=10)
            ax.set_xlabel("donor age (years)"); ax.set_ylabel(ylab)
    fig.suptitle("Y11: donor-pseudobulk module & C3+ vs age (ALL cells, 1-25y, good embedding)\n"
                 "point size ~ sqrt(n cells); band = 95% CI of OLS fit", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "y11_age_pseudobulk.png", dpi=150, bbox_inches="tight")
    out = pd.concat(tbl, ignore_index=True)
    out.to_csv(OUT_DIR / "y11_donor_pseudobulk.csv", index=False)
    for lab in COHORTS:
        s = out[out.cohort == lab]
        print(f"\n{lab}: n={len(s)} donors, ages={sorted(s['age'].round(1).tolist())}")
    print("DONE")


if __name__ == "__main__":
    main()
