#!/usr/bin/env python3
"""Step 1 consolidated — within-EN C3 trajectory across THREE cohorts.

Treats the data as three independent cohorts (per user instruction; V3 = Herring
is a different source from U01-V2 anyway):
    PsychAD-V3   (postnatal, many donors, all V3)
    Herring-V3   (Velmeshev 'Herring' sub-source, V3)   -- cleanest, but small
    U01-V2       (Velmeshev 'U01' sub-source, V2)
Ramos (Velmeshev, V3) is ~all prenatal -> excluded from the postnatal test.

For each cohort, the depth-robust C3 score (signed_logcpm, Step 0) is regressed
on age WITHIN mature EN subtypes, controlling subtype identity + sequencing depth,
with donor-clustered robust SE. We also split UPPER vs DEEP layers, because the
synaptic-pruning interpretation predicts an upper-layer-biased postnatal decline.

Mature-EN subtype definitions (cell_type_aligned; EN purity validated in s03B):
  Velmeshev: L2-3, L4, L5, L5-6-IT, L6   (upper: L2-3,L4; deep: L5,L5-6-IT,L6)
  PsychAD:   EN_L2_3_IT, EN_L3_5_IT_1/2/3, EN_L6_IT_1/2, EN_L6_CT, EN_L6B, EN_L5_6_NP
             (upper: L2_3_IT + L3_5_IT_*; deep: L6_* + L5_6_NP)

Inline-safe. Run via singularity (CLAUDE.md).
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
from s01a_within_celltype_trajectory import cluster_ols, partial_spearman

AGE_LO_POST, AGE_HI = 1.0, 30.0     # Velmeshev postnatal window
AGE_LO_PSY = 5.0                     # PsychAD: <5y labels unreliable

VEL_MATURE = ["L2-3", "L4", "L5", "L5-6-IT", "L6"]
VEL_UPPER = {"L2-3", "L4"}
PSY_MATURE = ["EN_L2_3_IT", "EN_L3_5_IT_1", "EN_L3_5_IT_2", "EN_L3_5_IT_3",
              "EN_L6_IT_1", "EN_L6_IT_2", "EN_L6_CT", "EN_L6B", "EN_L5_6_NP"]
PSY_UPPER = {"EN_L2_3_IT", "EN_L3_5_IT_1", "EN_L3_5_IT_2", "EN_L3_5_IT_3"}


def build(path, mature, upper):
    a = ad.read_h5ad(path)
    w = L.c3_signed()
    cpm = L.cpm_matrix(a)
    score = L.score_weighted_cpm(cpm, a.var_names, w, log1p=True)
    dm = L.depth_metrics(a)
    df = pd.DataFrame({
        "individual": a.obs["individual"].astype(str).values,
        "subtype": a.obs["cell_type_aligned"].astype(str).values,
        "dataset": a.obs["dataset"].astype(str).values,
        "chemistry": a.obs["chemistry"].astype(str).values,
        "age_years": pd.to_numeric(a.obs["age_years"], errors="coerce").values,
        "n_cells": a.obs["n_cells"].values,
        "score": score, "log10_total": dm["log10_total"].values,
    })
    df = df[df["subtype"].isin(mature) & (df["n_cells"] >= 20)].copy()
    df["layer"] = np.where(df["subtype"].isin(upper), "upper", "deep")
    return df


def pooled_slope(d):
    """within-EN age slope controlling subtype + depth; donor-clustered SE."""
    if d["individual"].nunique() < 6 or d["subtype"].nunique() < 1:
        return dict(n=len(d), donors=d["individual"].nunique(), slope=np.nan,
                    se=np.nan, t=np.nan, p=np.nan, rho=np.nan)
    sts = sorted(d["subtype"].unique())[1:]
    cols = [np.ones(len(d)), d["age_years"].values, d["log10_total"].values]
    names = ["const", "age", "log10_total"]
    for s in sts:
        cols.append((d["subtype"] == s).astype(float).values); names.append(s)
    X = np.column_stack(cols)
    beta, se, t, p = cluster_ols(d["score"].values, X, d["individual"].values)
    ai = names.index("age")
    rho = partial_spearman(d["score"].values, d["age_years"].values, d["log10_total"].values)
    return dict(n=len(d), donors=int(d["individual"].nunique()),
                slope=beta[ai], se=se[ai], t=t[ai], p=p[ai], rho=rho)


def main():
    vel = build(L.B / "Vel_prepost_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad",
                VEL_MATURE, VEL_UPPER)
    vel = vel[(vel["age_years"] >= AGE_LO_POST) & (vel["age_years"] < AGE_HI)]
    psy = build(L.B / "PsychAD_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad",
                PSY_MATURE, PSY_UPPER)
    psy = psy[(psy["age_years"] >= AGE_LO_PSY) & (psy["age_years"] < AGE_HI)]

    cohorts = {
        "PsychAD-V3": psy,
        "Herring-V3": vel[vel["dataset"] == "Herring"],
        "U01-V2":     vel[vel["dataset"] == "U01"],
    }

    rows = []
    for coh, d in cohorts.items():
        for layer, dd in [("all", d), ("upper", d[d["layer"] == "upper"]),
                          ("deep", d[d["layer"] == "deep"])]:
            r = pooled_slope(dd); r.update(cohort=coh, layer=layer)
            rows.append(r)
    tab = pd.DataFrame(rows)[["cohort", "layer", "n", "donors", "slope", "se", "t", "p", "rho"]]
    pd.set_option("display.width", 200)
    print("\n========== Within-EN C3 age trend by cohort x layer ==========")
    print(tab.round(3).to_string(index=False))
    tab.to_csv(L.OUT_DIR / "s04_cohort_layer_slopes.csv", index=False)

    # inverse-variance meta-analysis of the three 'all mature EN' slopes
    m = tab[(tab.layer == "all") & tab["se"].notna() & (tab["se"] > 0)]
    wts = 1.0 / m["se"].values ** 2
    b = np.sum(wts * m["slope"].values) / np.sum(wts)
    se = np.sqrt(1.0 / np.sum(wts))
    z = b / se
    p = 2 * stats.norm.sf(abs(z))
    print(f"\nMeta (inverse-variance, 3 cohorts, all mature EN): "
          f"slope={b:+.3f} SE={se:.3f} z={z:+.2f} p={p:.3f}")
    Q = np.sum(wts * (m["slope"].values - b) ** 2)
    print(f"  heterogeneity Q={Q:.2f} (df=2)  per-cohort slopes={list(m['slope'].round(2))}")

    # ---- figure: 3 cohorts (cols) x [all EN; upper vs deep] ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, (coh, d) in enumerate(cohorts.items()):
        ax = axes[0, j]
        ax.scatter(d["age_years"], d["score"], s=22, alpha=.6, c="C0")
        if len(d) > 3:
            z = np.polyfit(d["age_years"], d["score"], 1)
            xs = np.linspace(d["age_years"].min(), d["age_years"].max(), 20)
            ax.plot(xs, np.polyval(z, xs), "k--", lw=1.6)
        r = tab[(tab.cohort == coh) & (tab.layer == "all")].iloc[0]
        ax.set_title(f"{coh}  all mature EN\n"
                     f"rho_age|depth={r['rho']:+.2f}  slope p={r['p']:.2g}  (donors={r['donors']})",
                     fontsize=10)
        ax.set_xlabel("age (years)"); ax.set_ylabel("C3 signed_logcpm")

        ax2 = axes[1, j]
        for layer, c in [("upper", "C2"), ("deep", "C1")]:
            s = d[d["layer"] == layer]
            ax2.scatter(s["age_years"], s["score"], s=20, alpha=.6, c=c, label=layer)
            if len(s) > 3:
                z = np.polyfit(s["age_years"], s["score"], 1)
                xs = np.linspace(s["age_years"].min(), s["age_years"].max(), 20)
                ax2.plot(xs, np.polyval(z, xs), "--", c=c, lw=1.6)
        ru = tab[(tab.cohort == coh) & (tab.layer == "upper")].iloc[0]
        rd = tab[(tab.cohort == coh) & (tab.layer == "deep")].iloc[0]
        ax2.set_title(f"{coh}  upper rho={ru['rho']:+.2f} (p={ru['p']:.2g}) | "
                      f"deep rho={rd['rho']:+.2f} (p={rd['p']:.2g})", fontsize=9)
        ax2.set_xlabel("age (years)"); ax2.set_ylabel("C3 signed_logcpm"); ax2.legend(fontsize=8)
    fig.suptitle("Step 1: within-mature-EN C3 trajectory by cohort (top) and layer (bottom)\n"
                 "depth-robust signed_logcpm; postnatal", y=1.01)
    fig.tight_layout()
    fig.savefig(L.OUT_DIR / "s04_within_en_cohorts.png", dpi=140, bbox_inches="tight")
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
