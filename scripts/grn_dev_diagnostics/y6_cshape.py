#!/usr/bin/env python3
"""
Y6 — what does C3+ do across childhood/adolescence, and what is the x-axis?

Disambiguates the three candidate x-axes on the PRINCIPLED ExN set (from the
all-cell joint V3 embedding):
  (1) ACTUAL donor age (years)            — developmental TIME (the clean axis)
  (2) data-driven age axis                — supervised LogReg(child<10 vs adol>=10)
                                            direction through X_scVI (a latent
                                            projection that correlates with age)
  (3) maturation module / quantile        — single-cell maturation STATE
and tests the user's "adolescent dip" idea: is C3+ U-shaped in AGE, with a
trough in adolescence flanked by higher childhood and adult values?

Outputs:
  y6_c3_vs_age.png         donor-mean C3+ vs ACTUAL age (per cohort; all-ExN & q0),
                           quadratic fit + trough age (the adolescent-dip test)
  y6_triangulation.png     C3+ vs {age | age-axis | maturation} side by side
  y6_c3_age_stats.csv      quadratic fits, trough ages, per-axis correlations
  y6_percell_principled.parquet   per-cell cache (c3, module, age, donor, source, age_axis)

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=01:00:00 --mem=200G --cpus-per-task=8 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y6_cshape.py
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

ALLCELL = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"
GROUPS = ["PsychAD-V3", "Velmeshev-V3"]
SRC = {"PsychAD-V3": "PSYCHAD", "Velmeshev-V3": "VELMESHEV"}
COLORS = {"PsychAD-V3": "#C0392B", "Velmeshev-V3": "#2980B9"}
MODULE_ENS = ["ENSG00000171532", "ENSG00000127152", "ENSG00000119042",
              "ENSG00000081189", "ENSG00000104722", "ENSG00000100285",
              "ENSG00000067715", "ENSG00000132639", "ENSG00000078018"]
BOUND = 10.0


def quad_fit(age, y):
    """C3 ~ b0+b1*age+b2*age^2; return b2, p(b2), trough age (if U)."""
    m = np.isfinite(age) & np.isfinite(y)
    a, yy = age[m], y[m]
    if len(a) < 8:
        return np.nan, np.nan, np.nan
    X = np.vstack([np.ones_like(a), a, a**2]).T
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    dof = len(a) - 3
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    b2 = beta[2]; se2 = np.sqrt(cov[2, 2])
    p2 = 2 * stats.t.sf(abs(b2 / se2), dof)
    trough = -beta[1] / (2 * beta[2]) if beta[2] != 0 else np.nan
    return b2, p2, trough


def main():
    a = ad.read_h5ad(ALLCELL, backed="r")
    obs = a.obs
    Z = np.asarray(a.obsm["X_scVI"][:])
    counts = sp.csr_matrix(a.X[:])
    tot = np.asarray(counts.sum(1)).ravel(); inv = 1.0 / np.where(tot > 0, tot, 1)
    var = pd.Index(a.var_names.astype(str))
    c3w = build_c3plus_table().set_index("ensembl_id")["weight"]
    hit = var.intersection(c3w.index); cidx = [var.get_loc(g) for g in hit]
    c3 = np.asarray(counts[:, cidx].multiply(inv[:, None]).dot(c3w.reindex(hit).values)).ravel() * 1e6
    midx = [var.get_loc(g) for g in MODULE_ENS if g in var]
    module = np.asarray(np.log1p(counts[:, midx].multiply(inv[:, None]) * 1e4).todense()).mean(1)
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    src = obs["source"].astype(str).values
    don = obs["individual"].astype(str).values
    names = obs.index.astype(str)
    exn = np.asarray(names.isin(
        set(pd.read_parquet(OUT_DIR / "y4_principledexn_cellids_VELMESHEV.parquet").index.astype(str)) |
        set(pd.read_parquet(OUT_DIR / "y4_principledexn_cellids_PSYCHAD.parquet").index.astype(str))))

    # data-driven age axis (postnatal 1-25 fit)
    from sklearn.linear_model import LogisticRegression
    mp = exn & (age >= 1) & (age < 25) & np.isfinite(age) & np.isin(src, list(SRC.values()))
    clf = LogisticRegression(max_iter=2000).fit(Z[mp], (age[mp] < BOUND).astype(int))
    age_axis = -(Z @ clf.coef_.ravel())

    df = pd.DataFrame({"c3": c3, "module": module, "age": age, "don": don,
                       "src": src, "age_axis": age_axis, "exn": exn})
    df[df.exn].to_parquet(OUT_DIR / "y6_percell_principled.parquet")

    # quintile of maturation per cohort (over ExN)
    df["q0"] = False
    for s in SRC.values():
        m = df.exn & (df.src == s) & np.isfinite(df.age)
        thr = df.loc[m, "module"].quantile(0.2)
        df.loc[m & (df.module <= thr), "q0"] = True

    stat_rows = []

    # ---- FIG 1: donor-mean C3+ vs ACTUAL age (adolescent-dip test) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, stratum, lab in [(axes[0], df.exn, "all principled ExN"),
                             (axes[1], df.exn & df.q0, "least-mature quintile (q0)")]:
        for g in GROUPS:
            s = SRC[g]
            sub = df[stratum & (df.src == s) & np.isfinite(df.age)]
            dd = sub.groupby("don").agg(c3=("c3", "mean"), age=("age", "first"), n=("c3", "size"))
            dd = dd[dd.n >= 5]
            ax.scatter(dd.age, dd.c3, s=18, color=COLORS[g], alpha=0.6, label=f"{g} (n={len(dd)})")
            # quadratic over postnatal 0-30
            post = dd[(dd.age >= 0) & (dd.age < 30)]
            b2, p2, trough = quad_fit(post.age.values, post.c3.values)
            if np.isfinite(trough) and 0 < trough < 30 and b2 > 0:
                xs = np.linspace(0, 30, 100)
                # refit for line
                X = np.vstack([np.ones(len(post)), post.age, post.age**2]).T
                beta = np.linalg.lstsq(X, post.c3.values, rcond=None)[0]
                ax.plot(xs, beta[0] + beta[1]*xs + beta[2]*xs**2, color=COLORS[g], lw=1.5)
            stat_rows.append({"stratum": lab, "cohort": g, "n_donors": len(dd),
                              "quad_b2": b2, "quad_p": p2, "trough_age": trough,
                              "shape": ("U (dip)" if (np.isfinite(b2) and b2 > 0) else "inverted-U/none")})
        ax.axvspan(0, 1, color="grey", alpha=0.06); ax.axvspan(10, 18, color="gold", alpha=0.10)
        ax.set_xlabel("donor age (years)  [gold = adolescence 10-18y]")
        ax.set_ylabel("donor-mean C3+ (CP1e6)")
        ax.set_title(f"C3+ vs actual age — {lab}")
        ax.legend(fontsize=8)
    fig.suptitle("Y6 — Is there an adolescent C3+ dip? (donor-mean C3+ vs ACTUAL age; principled ExN)\n"
                 "Velmeshev-V3 prenatal=Ramos, postnatal=Herring", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "y6_c3_vs_age.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- FIG 2: triangulation — C3+ vs {age, age-axis, maturation} ----
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    post = df[df.exn & (df.age >= 0) & (df.age < 30)]
    for g in GROUPS:
        s = SRC[g]; sub = post[post.src == s]
        # (a) vs actual age: cell-level 1.5y bins
        sub2 = sub.copy(); sub2["b"] = pd.cut(sub2.age, np.arange(0, 31, 1.5))
        bm = sub2.groupby("b").agg(x=("age", "median"), y=("c3", "mean"))
        axes[0].plot(bm.x, bm.y, "o-", color=COLORS[g], ms=3, label=g)
        # (b) vs age axis decile
        sub3 = sub.copy(); sub3["q"] = pd.qcut(sub3.age_axis, 10, labels=False, duplicates="drop")
        bm = sub3.groupby("q").agg(x=("age_axis", "median"), y=("c3", "mean"))
        axes[1].plot(bm.x, bm.y, "o-", color=COLORS[g], ms=3, label=g)
        # (c) vs maturation module decile
        sub4 = sub.copy(); sub4["q"] = pd.qcut(sub4.module, 10, labels=False, duplicates="drop")
        bm = sub4.groupby("q").agg(x=("module", "median"), y=("c3", "mean"))
        axes[2].plot(bm.x, bm.y, "o-", color=COLORS[g], ms=3, label=g)
        for axx, xcol in [(axes[0], "age"), (axes[1], "age_axis"), (axes[2], "module")]:
            rho = stats.spearmanr(sub[xcol], sub["c3"]).correlation
            stat_rows.append({"stratum": f"triangulation rho({xcol},C3+)", "cohort": g,
                              "n_donors": np.nan, "quad_b2": np.nan, "quad_p": np.nan,
                              "trough_age": np.nan, "shape": round(rho, 3)})
    axes[0].set_xlabel("ACTUAL age (years)"); axes[0].set_title("(1) C3+ vs developmental TIME")
    axes[1].set_xlabel("data-driven age axis (→ older)"); axes[1].set_title("(2) C3+ vs supervised age axis")
    axes[2].set_xlabel("maturation module (→ mature)"); axes[2].set_title("(3) C3+ vs maturation STATE")
    for ax in axes:
        ax.set_ylabel("mean C3+ (CP1e6)"); ax.legend(fontsize=8)
    fig.suptitle("Y6 — what is the x-axis? C3+ against three different axes (principled ExN, cell-level)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "y6_triangulation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    sdf = pd.DataFrame(stat_rows)
    sdf.to_csv(OUT_DIR / "y6_c3_age_stats.csv", index=False)
    print(sdf.to_string(index=False))
    print(f"\nsaved y6_c3_vs_age.png, y6_triangulation.png, y6_c3_age_stats.csv, "
          f"y6_percell_principled.parquet in {OUT_DIR}")


if __name__ == "__main__":
    main()
