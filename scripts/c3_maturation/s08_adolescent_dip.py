#!/usr/bin/env python3
"""Is there a nonlinear 'adolescent dip' in C3 — high in childhood, low in
adolescence, recovering in adulthood — and is it real or a V2 depth artefact?

We test a U-shape (quadratic convexity in age) at THREE aggregation levels, per
cohort, under a depth-ROBUST metric (signed_logcpm) and a depth-BIASED one
(C3+ linear CPM = the original artefact-prone score):

  level 'all_cells'  : all_cells_by_donor pseudobulk      (composition + neurons)
  level 'ExN_agg'    : ExN_manual_by_donor (what s00b showed; ExN, subtype-mixed)
  level 'within_EN'  : per-donor MEAN over mature-EN subtypes (composition removed)

For each level x cohort x metric: fit score ~ age_c + age_c^2 + log10_depth
(age centered), HC3-robust SE. A positive, significant age^2 = U-shape (dip); the
minimum sits at age* = mean_age - b1/(2 b2). A real dip should (a) survive the
depth-robust metric, (b) replicate across the two independent V3 cohorts
(PsychAD-V3, Herring-V3), not just V2.

Inline-safe (small pseudobulks).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L
from s04_within_en_cohorts import VEL_MATURE, PSY_MATURE

AGE_LO, AGE_HI = 1.0, 45.0
PSY_LO = 5.0


def score_df(path, metric, within_mature=None):
    """Return per-pseudobulk-row df with score, age, depth, cohort, individual.
    metric in {'signed_log','pos_lin'}. If within_mature given, restrict to those
    subtypes (cell_type_aligned)."""
    a = ad.read_h5ad(path)
    w = L.c3_signed()
    cpm = L.cpm_matrix(a)
    if metric == "signed_log":
        s = L.score_weighted_cpm(cpm, a.var_names, w, log1p=True)
    else:  # pos_lin: depth-biased original
        s = L.score_weighted_cpm(cpm, a.var_names, w, log1p=False, pos_only=True)
    dm = L.depth_metrics(a)
    df = pd.DataFrame({
        "individual": a.obs["individual"].astype(str).values,
        "age": pd.to_numeric(a.obs["age_years"], errors="coerce").values,
        "score": s, "log10_total": dm["log10_total"].values,
        "dataset": a.obs["dataset"].astype(str).values if "dataset" in a.obs else "?",
        "chemistry": a.obs["chemistry"].astype(str).values if "chemistry" in a.obs else "?",
        "n_cells": a.obs["n_cells"].values if "n_cells" in a.obs else np.nan,
    })
    if "cell_type_aligned" in a.obs:
        df["subtype"] = a.obs["cell_type_aligned"].astype(str).values
    if within_mature is not None:
        df = df[df["subtype"].isin(within_mature) & (df["n_cells"] >= 20)]
        # collapse to per-donor mean over subtypes (equal weight -> remove subtype mix)
        df = (df.groupby(["individual"], as_index=False)
              .agg(score=("score", "mean"), age=("age", "first"),
                   log10_total=("log10_total", "mean"),
                   dataset=("dataset", "first"), chemistry=("chemistry", "first")))
    return df


def cohort_of(df, psychad):
    if psychad:
        return pd.Series("PsychAD-V3", index=df.index)
    return df["dataset"].map({"Herring": "Herring-V3", "U01": "U01-V2", "Ramos": "Ramos-V3"})


def quad_fit(d):
    """OLS score ~ age_c + age_c^2 + depth; HC3 SE. Return dict with age^2 coef/p,
    convexity, min-age."""
    d = d.dropna(subset=["score", "age", "log10_total"])
    if len(d) < 8:
        return None
    ac = d["age"].values - d["age"].mean()
    X = sm.add_constant(np.column_stack([ac, ac**2, d["log10_total"].values]))
    m = sm.OLS(d["score"].values, X).fit(cov_type="HC3")
    b1, b2 = m.params[1], m.params[2]
    p2 = m.pvalues[2]
    minage = d["age"].mean() - b1 / (2 * b2) if b2 != 0 else np.nan
    return dict(n=len(d), b_age=b1, b_age2=b2, p_age2=p2,
                convex=(b2 > 0), min_age=minage)


def main():
    vel_all = L.B / "Vel_prepost_noage_tuning5/pseudobulk_output/all_cells_by_donor.h5ad"
    vel_exn = L.B / "Vel_prepost_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad"
    vel_exc = L.B / "Vel_prepost_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad"
    psy_all = L.B / "PsychAD_noage_tuning5/pseudobulk_output/all_cells_by_donor.h5ad"
    psy_exn = L.B / "PsychAD_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad"
    psy_exc = L.B / "PsychAD_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad"

    levels = {
        "all_cells": (dict(path=vel_all), dict(path=psy_all)),
        "ExN_agg":   (dict(path=vel_exn), dict(path=psy_exn)),
        "within_EN": (dict(path=vel_exc, within_mature=VEL_MATURE),
                      dict(path=psy_exc, within_mature=PSY_MATURE)),
    }

    rows = []
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    for ri, (lvl, (velcfg, psycfg)) in enumerate(levels.items()):
        for mi, metric in enumerate(["signed_log", "pos_lin"]):
            ax = axes[ri, mi]
            # build per-cohort frames
            vd = score_df(velcfg["path"], metric, velcfg.get("within_mature"))
            vd["cohort"] = cohort_of(vd, False)
            pd_ = score_df(psycfg["path"], metric, psycfg.get("within_mature"))
            pd_["cohort"] = "PsychAD-V3"
            d = pd.concat([vd, pd_], ignore_index=True)
            lo = PSY_LO if lvl == "within_EN" else AGE_LO
            d = d[(d["age"] >= 1.0) & (d["age"] < AGE_HI)]
            for coh, c in [("PsychAD-V3", "C0"), ("Herring-V3", "C2"), ("U01-V2", "C3")]:
                s = d[d["cohort"] == coh]
                s = s[s["age"] >= (PSY_LO if (coh == "PsychAD-V3" and lvl == "within_EN") else 1.0)]
                if len(s) < 6:
                    continue
                ax.scatter(s["age"], s["score"], s=14, alpha=.45, color=c)
                lw = lowess(s["score"], s["age"], frac=0.7, return_sorted=True)
                ax.plot(lw[:, 0], lw[:, 1], color=c, lw=2, label=f"{coh} (n={len(s)})")
                q = quad_fit(s)
                if q:
                    q.update(level=lvl, metric=metric, cohort=coh)
                    rows.append(q)
            ax.set_title(f"{lvl} | {metric}", fontsize=11)
            ax.set_xlabel("age (years)"); ax.set_ylabel("C3 score"); ax.legend(fontsize=7)
    fig.suptitle("Adolescent-dip test: C3 vs age (LOWESS) by level x metric x cohort\n"
                 "depth-robust 'signed_log' vs depth-biased 'pos_lin'", y=1.005)
    fig.tight_layout()
    fig.savefig(L.OUT_DIR / "s08_adolescent_dip.png", dpi=140, bbox_inches="tight")

    tab = pd.DataFrame(rows)[["level", "metric", "cohort", "n", "b_age", "b_age2",
                              "p_age2", "convex", "min_age"]]
    tab.to_csv(L.OUT_DIR / "s08_dip_quadratic.csv", index=False)

    # ===== POOLED cross-cohort test (cohort FE + depth adjusted), signed_log =====
    print("\n===== POOLED across cohorts (cohort FE + depth), signed_log =====")
    fig2, ax2s = plt.subplots(1, 3, figsize=(18, 5))
    pooled_rows = []
    for ax2, (lvl, (velcfg, psycfg)) in zip(ax2s, levels.items()):
        vd = score_df(velcfg["path"], "signed_log", velcfg.get("within_mature"))
        vd["cohort"] = cohort_of(vd, False)
        pq = score_df(psycfg["path"], "signed_log", psycfg.get("within_mature"))
        pq["cohort"] = "PsychAD-V3"
        d = pd.concat([vd, pq], ignore_index=True)
        d = d[d["cohort"].isin(["PsychAD-V3", "Herring-V3", "U01-V2"])]
        d = d[(d["age"] >= 1.0) & (d["age"] < AGE_HI)].dropna(subset=["score", "age", "log10_total"])
        # design: cohort dummies + depth + age_c + age_c^2
        ac = d["age"].values - d["age"].mean()
        cohd = pd.get_dummies(d["cohort"], drop_first=True).values.astype(float)
        X = np.column_stack([np.ones(len(d)), cohd, d["log10_total"].values, ac, ac**2])
        m = sm.OLS(d["score"].values, X).fit(cov_type="HC3")
        b1, b2 = m.params[-2], m.params[-1]
        p2 = m.pvalues[-1]
        minage = d["age"].mean() - b1 / (2 * b2) if b2 != 0 else np.nan
        pooled_rows.append(dict(level=lvl, n=len(d), b_age2=b2, p_age2=p2,
                                convex=(b2 > 0), min_age=minage))
        # partial residuals: score - (cohort FE + depth), plotted vs age
        fitted_nuisance = X[:, :-2] @ m.params[:-2]
        resid = d["score"].values - fitted_nuisance + m.params[0]
        for coh, c in [("PsychAD-V3", "C0"), ("Herring-V3", "C2"), ("U01-V2", "C3")]:
            mk = d["cohort"].values == coh
            ax2.scatter(d["age"].values[mk], resid[mk], s=14, alpha=.4, color=c, label=coh)
        lw = lowess(resid, d["age"].values, frac=0.6, return_sorted=True)
        ax2.plot(lw[:, 0], lw[:, 1], "k-", lw=2.5, label="pooled LOWESS")
        ax2.set_title(f"{lvl}: age^2 p={p2:.3f} {'(U-dip, min %.0fy)'%minage if b2>0 else '(concave)'}",
                      fontsize=11)
        ax2.set_xlabel("age (years)"); ax2.set_ylabel("C3 signed_log (cohort+depth adj.)")
        ax2.legend(fontsize=8)
    fig2.suptitle("Pooled cross-cohort C3 age-shape (cohort FE + depth adjusted)", y=1.02)
    fig2.tight_layout()
    fig2.savefig(L.OUT_DIR / "s08_dip_pooled.png", dpi=140, bbox_inches="tight")
    ptab = pd.DataFrame(pooled_rows)
    print(ptab.round(3).to_string(index=False))
    ptab.to_csv(L.OUT_DIR / "s08_dip_pooled.csv", index=False)

    # age coverage per cohort
    print("\n--- donor age coverage per cohort (all_cells level) ---")
    vd = score_df(vel_all, "signed_log"); vd["cohort"] = cohort_of(vd, False)
    pq = score_df(psy_all, "signed_log"); pq["cohort"] = "PsychAD-V3"
    allc = pd.concat([vd, pq], ignore_index=True)
    allc["agebin"] = pd.cut(allc["age"], [-1, 1, 5, 10, 15, 20, 30, 45, 200],
                            labels=["<1", "1-5", "5-10", "10-15", "15-20", "20-30", "30-45", "45+"])
    print(pd.crosstab(allc["cohort"], allc["agebin"]).to_string())
    pd.set_option("display.width", 200)
    print("\n===== U-shape (quadratic) test: age^2 coef (>0 & sig = dip) =====")
    print(tab.round(3).to_string(index=False))
    print("\nA real dip should be convex (age^2>0) under the DEPTH-ROBUST metric "
          "(signed_log) AND replicate in both V3 cohorts (PsychAD-V3, Herring-V3).")
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
