#!/usr/bin/env python3
"""
S — combined-dataset maturity result + depth↔maturity figures for FINAL_REPORT.

Builds the artefacts the revised FINAL_REPORT needs to present MATURITY
(not depth) as the key result, with a single combined-cohort estimate and
the split-cohort numbers as robustness checks.

Inputs: outputs/r_per_cell_cache_v4.parquet (built by r_immature_investigation.py)

Outputs:
  s1_depth_maturity_scatter.png   — per-cohort hexbin log10(UMI) vs mature
                                     module, Spearman r (depth↔maturity link)
  s1_stage_shift.png / .csv       — median depth & median maturity by age
                                     stage per cohort (the FANS-shift evidence)
  s2_combined_estimate.csv        — per-cohort, V3-pooled (centred), all-three
                                     (z-scored), and meta-analytic combined d
  s2_combined_forest.png          — forest plot of the above
  s2_combined_trajectory.png      — pooled (cohort-centred) q0 donor score vs age
  s3_psychad_depth_x_layer_maturity.png — R4 (depth×mat) | R5 (layer×mat) side by side

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=00:30:00 --mem=64G \
     scripts/run_script.sh scripts/grn_dev_diagnostics/s_combined_maturity.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (OUT_DIR, fuzzy_d_from_donor_scores, FUZZY_BOUNDARIES,
                   AGE_LO, AGE_HI)

CACHE_V4 = OUT_DIR / "r_per_cell_cache_v4.parquet"
GROUPS = ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]
COLORS = {"PsychAD-V3": "#C0392B", "Velmeshev-V2": "#27AE60",
          "Velmeshev-V3": "#2980B9"}
MODULE_MATURE = ["NEUROD2", "BCL11B", "SATB2", "MEF2C", "NEFM", "NEFH",
                 "SYT1", "SNAP25", "MAP2"]   # NEFL absent from var → dropped
EXCLUDE_DONORS = {"Donor_1400"}
MIN_CELLS = 5
N_Q = 5
CENTRAL_B = 10  # central boundary for SE / meta-analysis


def add_module(df):
    cp = [f"cp_{m}" for m in MODULE_MATURE if f"cp_{m}" in df.columns]
    df = df.copy()
    df["mature_module"] = df[cp].mean(axis=1)
    return df


def donor_table(cells):
    don = (cells.groupby("individual", observed=True)
                .agg(score=("per_cell_c3", "mean"),
                     n_cells=("per_cell_c3", "size"),
                     age_years=("age_years", "first"),
                     group=("group", "first"))
                .reset_index())
    don = don[(don["n_cells"] >= MIN_CELLS)
              & (don["age_years"] >= AGE_LO)
              & (don["age_years"] < AGE_HI)].dropna(subset=["score"])
    return don


def fuzzy_d_table(don):
    if len(don) < 4:
        return np.nan
    return fuzzy_d_from_donor_scores(don["age_years"].values,
                                     don["score"].values)["mean_d"]


def cohens_d_se(don, b):
    """SE of Cohen's d at a single boundary b (Hedges approx)."""
    c = don[don["age_years"] < b]["score"].values
    a = don[don["age_years"] >= b]["score"].values
    n1, n2 = len(c), len(a)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    sp = np.sqrt(((n1-1)*c.std(ddof=1)**2 + (n2-1)*a.std(ddof=1)**2)
                 / (n1+n2-2))
    if sp == 0:
        return np.nan, np.nan
    d = (c.mean() - a.mean()) / sp
    se = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))
    return d, se


# ---------------------------------------------------------------------------
# S1 — depth ↔ maturity link
# ---------------------------------------------------------------------------

def s1_depth_maturity(df):
    print("\n=== S1: depth ↔ maturity ===", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    rho_rows = []
    for ax, g in zip(axes, GROUPS):
        sub = add_module(df[df["group"] == g])
        x = np.log10(sub["total_umi"].clip(lower=1))
        y = sub["mature_module"].values
        rho, p = stats.spearmanr(sub["total_umi"], y)
        rho_rows.append({"group": g, "spearman_rho_umi_vs_maturity": rho,
                         "p": p, "n_cells": len(sub)})
        hb = ax.hexbin(x, y, gridsize=45, cmap="magma", bins="log", mincnt=1)
        ax.set_title(f"{g}\nSpearman ρ(UMI, maturity) = {rho:+.2f}",
                     fontsize=10)
        ax.set_xlabel("log10(total UMI per cell)")
        if g == GROUPS[0]:
            ax.set_ylabel("mature-module score\n(mean log1p CP10k, 9 markers)")
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="cells (log)")
    fig.suptitle("S1 — per-cell library depth is positively correlated with "
                 "the mature-module score in every cohort\n"
                 "(deeper cells are more mature; shallow ≈ immature — the "
                 "basis of the depth↔maturity confound)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "s1_depth_maturity_scatter.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rho_rows).to_csv(OUT_DIR / "s1_spearman.csv", index=False)
    print(pd.DataFrame(rho_rows).to_string(index=False))

    # stage shift: median depth & median maturity by age stage (b=10) per cohort
    rows = []
    for g in GROUPS:
        sub = add_module(df[df["group"] == g])
        sub["stage"] = np.where(sub["age_years"] < CENTRAL_B, "child", "adol")
        for stg, s2 in sub.groupby("stage"):
            rows.append({"group": g, "stage": stg, "n_cells": len(s2),
                         "median_umi": s2["total_umi"].median(),
                         "median_maturity": s2["mature_module"].median(),
                         "frac_immature_binary":
                             (s2["marker_annotation"] == "ExN_immature").mean()})
    shift = pd.DataFrame(rows)
    shift.to_csv(OUT_DIR / "s1_stage_shift.csv", index=False)
    print("\n--- median depth & maturity by stage ---")
    print(shift.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, ttl in zip(
            axes,
            ["median_umi", "median_maturity", "frac_immature_binary"],
            ["median UMI / cell", "median mature-module",
             "fraction ExN_immature (DCX+RBFOX3-)"]):
        w = 0.35
        for j, g in enumerate(GROUPS):
            s = shift[shift["group"] == g].set_index("stage").reindex(
                ["child", "adol"])
            ax.bar(np.arange(2) + j*w - w, s[metric].values, w,
                   color=COLORS[g], label=g if metric == "median_umi" else None)
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(["child\n(<10y)", "adol\n(10-25y)"])
        ax.set_title(ttl, fontsize=10)
        if metric == "median_umi":
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("S1b — in PsychAD-V3 childhood cells are shallower AND less "
                 "mature AND more immature; since depth↔maturity are tightly "
                 "coupled (S1), the age-related depth shift confounds the "
                 "all-cell C3+ aggregate", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "s1_stage_shift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved s1 figures")


# ---------------------------------------------------------------------------
# S2 — combined estimate
# ---------------------------------------------------------------------------

def s2_combined(df):
    print("\n=== S2: combined estimate ===", flush=True)
    # per-cohort maturity-q0 donor tables
    q0_donors = {}
    per_cohort_d = {}
    for g in GROUPS:
        sub = add_module(df[df["group"] == g])
        sub["mat_q"] = pd.qcut(sub["mature_module"], N_Q, labels=False,
                               duplicates="drop")
        don = donor_table(sub[sub["mat_q"] == 0])
        q0_donors[g] = don
        per_cohort_d[g] = fuzzy_d_table(don)

    rows = []
    for g in GROUPS:
        rows.append({"estimate": f"per_cohort: {g}",
                     "n_donors": len(q0_donors[g]),
                     "fuzzy_d": per_cohort_d[g]})

    # (a) V3-pair pooled, cohort-centred
    v3 = []
    for g in ["PsychAD-V3", "Velmeshev-V3"]:
        d = q0_donors[g].copy()
        d["score"] = d["score"] - d["score"].mean()   # centre within cohort
        v3.append(d)
    v3 = pd.concat(v3, ignore_index=True)
    d_v3 = fuzzy_d_table(v3)
    rows.append({"estimate": "COMBINED V3-pair (PsychAD-V3 + Vel-V3), "
                 "cohort-centred", "n_donors": len(v3), "fuzzy_d": d_v3})

    # (b) all-three pooled, z-scored within cohort
    allz = []
    for g in GROUPS:
        d = q0_donors[g].copy()
        s = d["score"]
        d["score"] = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1)
        allz.append(d)
    allz = pd.concat(allz, ignore_index=True)
    d_allz = fuzzy_d_table(allz)
    rows.append({"estimate": "COMBINED all-3, z-scored within cohort",
                 "n_donors": len(allz), "fuzzy_d": d_allz})

    # (c) meta-analysis (fixed + random) of per-cohort d at central boundary
    md = []
    for g in GROUPS:
        d, se = cohens_d_se(q0_donors[g], CENTRAL_B)
        md.append({"group": g, "d": d, "se": se})
    md = pd.DataFrame(md).dropna()
    w = 1 / md["se"]**2
    d_fixed = (w * md["d"]).sum() / w.sum()
    se_fixed = np.sqrt(1 / w.sum())
    # DerSimonian-Laird tau^2
    Q = (w * (md["d"] - d_fixed)**2).sum()
    dfree = len(md) - 1
    C = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0, (Q - dfree) / C) if C > 0 else 0
    wr = 1 / (md["se"]**2 + tau2)
    d_random = (wr * md["d"]).sum() / wr.sum()
    se_random = np.sqrt(1 / wr.sum())
    rows.append({"estimate": f"COMBINED meta fixed-effect (b={CENTRAL_B})",
                 "n_donors": int(md.shape[0]), "fuzzy_d": d_fixed,
                 "se": se_fixed})
    rows.append({"estimate": f"COMBINED meta random-effect (b={CENTRAL_B}, "
                 f"tau2={tau2:.2f})", "n_donors": int(md.shape[0]),
                 "fuzzy_d": d_random, "se": se_random})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "s2_combined_estimate.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nmeta per-cohort d at b={CENTRAL_B}:")
    print(md.to_string(index=False))

    # forest plot
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = list(out["estimate"])
    y = np.arange(len(labels))[::-1]
    for yi, (_, r) in zip(y, out.iterrows()):
        col = "#444"
        if r["estimate"].startswith("per_cohort"):
            for g in GROUPS:
                if g in r["estimate"]:
                    col = COLORS[g]
        elif "COMBINED" in r["estimate"]:
            col = "#000"
        se = r.get("se", np.nan)
        if pd.notna(se):
            ax.errorbar(r["fuzzy_d"], yi, xerr=1.96*se, fmt="o", color=col,
                        capsize=3)
        else:
            ax.plot(r["fuzzy_d"], yi, "o", color=col)
        ax.text(r["fuzzy_d"], yi+0.18, f"{r['fuzzy_d']:+.2f}", ha="center",
                fontsize=8, color=col)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("fuzzy Cohen's d (child→adol), maturity-q0 cells")
    ax.set_title("S2 — combined-cohort maturity result with split-cohort "
                 "robustness\n(d>0 = C3+ drops with age)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "s2_combined_forest.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # combined trajectory: cohort-centred q0 donor score vs age, all 3 cohorts
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for g in GROUPS:
        d = q0_donors[g].copy()
        d["score_c"] = d["score"] - d["score"].mean()
        ax.scatter(d["age_years"], d["score_c"], s=24, color=COLORS[g],
                   alpha=0.7, label=f"{g} (d={per_cohort_d[g]:+.2f})")
        # lowess-ish: simple age-bin mean
        if len(d) >= 6:
            order = d.sort_values("age_years")
            ax.plot(order["age_years"].rolling(5, center=True, min_periods=3).mean(),
                    order["score_c"].rolling(5, center=True, min_periods=3).mean(),
                    color=COLORS[g], lw=1.5, alpha=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(CENTRAL_B, color="grey", ls=":", lw=1)
    ax.set_xlabel("donor age (years)")
    ax.set_ylabel("cohort-centred mean C3+ (maturity-q0 cells)")
    ax.set_title(f"S2 — combined maturity-q0 trajectory "
                 f"(V3-pair pooled d = {d_v3:+.2f}, all-3 z-scored d = "
                 f"{d_allz:+.2f})", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "s2_combined_trajectory.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("saved s2 figures")


# ---------------------------------------------------------------------------
# S3 — R4 (depth×maturity) | R5 (layer×maturity) side by side, PsychAD-V3
# ---------------------------------------------------------------------------

def _heat(ax, piv, title, xlabel):
    M = piv.values.astype(float)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto",
                   origin="lower")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=20, fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"matQ{int(i)}" for i in piv.index], fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                        fontsize=7)
    return im


def s3_side_by_side():
    print("\n=== S3: R4|R5 side by side ===", flush=True)
    r4 = pd.read_csv(OUT_DIR / "r4_depth_x_module.csv")
    r5 = pd.read_csv(OUT_DIR / "r5_layer_x_module.csv")
    r4 = r4[r4["group"] == "PsychAD-V3"]
    r5 = r5[r5["group"] == "PsychAD-V3"]
    p4 = r4.pivot_table(index="mat_bin", columns="x_bin", values="fuzzy_d")
    p4.columns = [f"depthQ{int(c)}" for c in p4.columns]
    layer_order = [c for c in ["upper", "L5_ET", "L6_IT", "L6_CT", "ambiguous"]
                   if c in r5["x_bin"].unique()]
    p5 = r5.pivot_table(index="mat_bin", columns="x_bin", values="fuzzy_d")
    p5 = p5[layer_order]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = _heat(axes[0], p4, "R4 — depth × maturity",
               "depth quartile (Q0 shallow → Q3 deep)")
    _heat(axes[1], p5, "R5 — cortical layer × maturity", "layer (TF argmax)")
    axes[0].set_ylabel("maturity bin (0 = least mature)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                 label="fuzzy Cohen's d")
    fig.suptitle("PsychAD-V3 — the immature-cell C3+ drop (top rows, blue→red "
                 "positive) is present across depth AND across layers; depth "
                 "and maturity are entangled, layer is not the driver",
                 fontweight="bold", fontsize=11)
    fig.savefig(OUT_DIR / "s3_psychad_depth_x_layer_maturity.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("saved s3 figure")


def main():
    df = pd.read_parquet(CACHE_V4)
    df = df[~df["individual"].isin(EXCLUDE_DONORS)].reset_index(drop=True)
    print(f"loaded v4 cache: {len(df):,} cells (Donor_1400 excluded)")
    s1_depth_maturity(df)
    s2_combined(df)
    s3_side_by_side()
    print(f"\nAll S outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
