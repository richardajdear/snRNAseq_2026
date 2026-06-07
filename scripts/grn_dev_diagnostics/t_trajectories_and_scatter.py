#!/usr/bin/env python3
"""
T — headline maturity-quintile trajectories + C3+ vs maturity scatter
    with a compositional-vs-within-state decomposition of the drop.

Motivation (user reflection):
  Stratifying to maturity-q0 uses only 1/5 of cells. If the C3+ drop is
  *because* cells mature (a q0 child cell becomes a q2/q3 adolescent cell),
  then conditioning on maturity REMOVES the compositional part of the drop
  and keeps only the within-state (same-maturity) age decline. We must
  therefore (a) show the raw per-quintile trajectories, and (b) test
  whether C3+ is purely a function of maturity (curves overlap across age →
  drop is all compositional) or also declines within a maturity bin (curves
  shift down with age → genuine within-state effect). A Kitagawa
  decomposition quantifies the split per cohort.

Outputs:
  t1_headline_trajectories.png      — left: all 5 maturity quintiles (raw
                                       donor C3+ vs age); right: q0 only.
                                       Fuzzy boundary band (8–12 y) shaded.
  t2_c3_vs_maturity.png             — per cohort: C3+ vs maturity, child vs
                                       adolescent binned curves over hexbin.
  t2_c3_vs_maturity_binned.csv      — decile means (cohort × maturity × age)
  t2_decomposition.csv              — Kitagawa: composition vs within-state

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=00:30:00 --mem=64G \
     scripts/run_script.sh scripts/grn_dev_diagnostics/t_trajectories_and_scatter.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR, AGE_LO, AGE_HI

CACHE_V4 = OUT_DIR / "r_per_cell_cache_v4.parquet"
GROUPS = ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]
COLORS = {"PsychAD-V3": "#C0392B", "Velmeshev-V2": "#27AE60",
          "Velmeshev-V3": "#2980B9"}
MODULE_MATURE = ["NEUROD2", "BCL11B", "SATB2", "MEF2C", "NEFM", "NEFH",
                 "SYT1", "SNAP25", "MAP2"]
EXCLUDE_DONORS = {"Donor_1400"}
MIN_CELLS = 5
FUZZY_LO, FUZZY_HI = 8, 12
BOUND = 10           # single child/adol split for the age-stratified scatter
N_Q = 5
N_DEC = 10
QCOLORS = plt.cm.viridis(np.linspace(0, 0.92, N_Q))


def add_module(df):
    cp = [f"cp_{m}" for m in MODULE_MATURE if f"cp_{m}" in df.columns]
    df = df.copy()
    df["mature_module"] = df[cp].mean(axis=1)
    return df


GRID = np.arange(2, 24.01, 1.0)          # fixed age grid for averaging


def grid_means(age, y, bin_w=3.0, min_n=3):
    """Moving-average of y over the fixed GRID; NaN where <min_n donors."""
    age = np.asarray(age); y = np.asarray(y)
    out = np.full(len(GRID), np.nan)
    for i, c in enumerate(GRID):
        m = (age >= c - bin_w/2) & (age < c + bin_w/2)
        if m.sum() >= min_n:
            out[i] = np.mean(y[m])
    return out


def binned_line(ax, age, y, color, lw=2.2, bin_w=3.0, label=None):
    """Moving-average line for a single series (used in the per-cohort q0 panel)."""
    ys = grid_means(age, y, bin_w=bin_w)
    ok = ~np.isnan(ys)
    if ok.any():
        ax.plot(GRID[ok], ys[ok], "-", color=color, lw=lw, label=label, zorder=5)


# ---------------------------------------------------------------------------
# PART A — headline trajectory pair
# ---------------------------------------------------------------------------

def part_a(df):
    print("\n=== PART A: headline trajectories ===", flush=True)
    # donor-level table per (cohort, quintile)
    rows = []
    for g in GROUPS:
        sub = add_module(df[df["group"] == g])
        sub["mat_q"] = pd.qcut(sub["mature_module"], N_Q, labels=False,
                               duplicates="drop")
        for q in range(N_Q):
            cells = sub[sub["mat_q"] == q]
            don = (cells.groupby("individual", observed=True)
                        .agg(score=("per_cell_c3", "mean"),
                             n=("per_cell_c3", "size"),
                             age=("age_years", "first")).reset_index())
            don = don[(don["n"] >= MIN_CELLS) & (don["age"] >= AGE_LO)
                      & (don["age"] < AGE_HI)]
            don["group"] = g; don["mat_q"] = q
            rows.append(don)
    D = pd.concat(rows, ignore_index=True)

    # raw scale check
    print("raw donor C3+ by cohort (q0):")
    print(D[D.mat_q == 0].groupby("group")["score"]
          .describe()[["mean", "min", "max"]].to_string())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    # LEFT — all quintiles, raw, cohorts weighted EQUALLY per age bin
    # (averaging each cohort's local mean avoids the cohort×age composition
    #  artdefact that naive raw pooling would introduce, since Velmeshev
    #  skews young / high-baseline and PsychAD skews older / low-baseline).
    for q in range(N_Q):
        d = D[D.mat_q == q]
        # per-cohort grid means, then average across cohorts (equal weight)
        cohort_lines = []
        for g in GROUPS:
            dg = d[d.group == g]
            cohort_lines.append(grid_means(dg["age"], dg["score"]))
        bal = np.nanmean(np.vstack(cohort_lines), axis=0)
        ok = ~np.isnan(bal)
        lab = f"q{q}" + (" (least mature)" if q == 0 else
                         " (most mature)" if q == N_Q-1 else "")
        axL.plot(GRID[ok], bal[ok], "-o", color=QCOLORS[q], lw=2.3, ms=4,
                 label=lab, zorder=5)
    axL.axvspan(FUZZY_LO, FUZZY_HI, color="grey", alpha=0.13, zorder=0)
    axL.set_title("All maturity quintiles (raw donor C3+, cohorts equal-weighted)",
                  fontsize=11)
    axL.set_xlabel("donor age (years)")
    axL.set_ylabel("mean per-cell C3+ (raw, CPM units)")
    axL.legend(frameon=False, fontsize=8, title="maturity quintile")
    axL.set_xlim(0, 25)

    # RIGHT — q0 only, by cohort, raw
    d0 = D[D.mat_q == 0]
    for g in GROUPS:
        dg = d0[d0.group == g]
        axR.scatter(dg["age"], dg["score"], s=20, color=COLORS[g], alpha=0.6,
                    edgecolors="none", label=g)
        binned_line(axR, dg["age"], dg["score"], COLORS[g])
    axR.axvspan(FUZZY_LO, FUZZY_HI, color="grey", alpha=0.13, zorder=0)
    axR.set_title("Least-mature quintile (q0) only — by cohort (raw)",
                  fontsize=11)
    axR.set_xlabel("donor age (years)")
    axR.legend(frameon=False, fontsize=8)
    axR.set_xlim(0, 25)
    # annotate the shaded band once
    axR.text(np.mean([FUZZY_LO, FUZZY_HI]), axR.get_ylim()[1],
             "fuzzy child/adol\nboundary (8–12 y)", ha="center", va="top",
             fontsize=7, color="dimgrey")
    fig.suptitle("Headline — the C3+ childhood peak is carried by the "
                 "least-mature quintile (q0); higher quintiles are flat",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "t1_headline_trajectories.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("saved t1_headline_trajectories.png")
    D.to_csv(OUT_DIR / "t1_quintile_donor_trajectories.csv", index=False)


# ---------------------------------------------------------------------------
# PART B — C3+ vs maturity scatter + decomposition
# ---------------------------------------------------------------------------

def part_b(df):
    print("\n=== PART B: C3+ vs maturity + decomposition ===", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=False)
    binned_rows, decomp_rows = [], []
    for ax, g in zip(axes, GROUPS):
        sub = add_module(df[df["group"] == g]).copy()
        sub["stage"] = np.where(sub["age_years"] < BOUND, "child", "adol")
        # maturity deciles defined on ALL ages within cohort
        sub["mdec"] = pd.qcut(sub["mature_module"], N_DEC, labels=False,
                              duplicates="drop")
        # faint hexbin background (all cells)
        ax.hexbin(sub["mature_module"], sub["per_cell_c3"], gridsize=40,
                  cmap="Greys", bins="log", mincnt=1, alpha=0.45)
        # spearman overall
        rho, _ = stats.spearmanr(sub["mature_module"], sub["per_cell_c3"])
        # child vs adol decile-mean curves
        for stg, col in [("child", "#2166AC"), ("adol", "#B2182B")]:
            s2 = sub[sub["stage"] == stg]
            gb = s2.groupby("mdec").agg(
                mat=("mature_module", "median"),
                c3=("per_cell_c3", "mean"),
                n=("per_cell_c3", "size")).reset_index()
            ax.plot(gb["mat"], gb["c3"], "o-", color=col, lw=2, ms=5,
                    label=f"{stg} (<{BOUND}y)" if stg == "child"
                    else f"{stg} (≥{BOUND}y)")
            for _, r in gb.iterrows():
                binned_rows.append({"group": g, "stage": stg,
                                    "mdec": int(r["mdec"]), "mat_med": r["mat"],
                                    "mean_c3": r["c3"], "n": int(r["n"])})
        ax.set_title(f"{g}\nSpearman ρ(maturity, C3+) = {rho:+.2f}", fontsize=10)
        ax.set_xlabel("mature-module score (per cell)")
        if g == GROUPS[0]:
            ax.set_ylabel("per-cell C3+ (raw, CPM units)")
        ax.legend(frameon=False, fontsize=8)

        # ---- Kitagawa decomposition of child→adol aggregate C3+ change ----
        piv = (sub.groupby(["mdec", "stage"])
                  .agg(c3=("per_cell_c3", "mean"),
                       n=("per_cell_c3", "size")).reset_index())
        wide = piv.pivot(index="mdec", columns="stage",
                         values=["c3", "n"]).fillna(0)
        c_ch = wide[("c3", "child")].values
        c_ad = wide[("c3", "adol")].values
        n_ch = wide[("n", "child")].values
        n_ad = wide[("n", "adol")].values
        f_ch = n_ch / n_ch.sum()
        f_ad = n_ad / n_ad.sum()
        Cbar_ch = (f_ch * c_ch).sum()
        Cbar_ad = (f_ad * c_ad).sum()
        d_total = Cbar_ch - Cbar_ad
        comp = ((f_ch - f_ad) * (c_ch + c_ad) / 2).sum()      # composition
        within = ((c_ch - c_ad) * (f_ch + f_ad) / 2).sum()    # within-state
        decomp_rows.append({
            "group": g, "C3_child": Cbar_ch, "C3_adol": Cbar_ad,
            "delta_total": d_total,
            "composition": comp, "within_state": within,
            "pct_composition": 100*comp/d_total if d_total else np.nan,
            "pct_within": 100*within/d_total if d_total else np.nan,
            "spearman_mat_c3": rho})

    fig.suptitle("C3+ vs neuronal maturity, split by age. Overlapping "
                 "child/adol curves ⇒ drop is compositional (maturation); a "
                 "downward child→adol shift ⇒ within-state age decline",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "t2_c3_vs_maturity.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("saved t2_c3_vs_maturity.png")

    pd.DataFrame(binned_rows).to_csv(
        OUT_DIR / "t2_c3_vs_maturity_binned.csv", index=False)
    dec = pd.DataFrame(decomp_rows)
    dec.to_csv(OUT_DIR / "t2_decomposition.csv", index=False)
    print("\n--- Kitagawa decomposition of the child→adol C3+ drop ---")
    print("(delta_total>0 = childhood higher; composition = maturation shift,")
    print(" within_state = same-maturity age decline)")
    print(dec.round(1).to_string(index=False))


def main():
    df = pd.read_parquet(CACHE_V4)
    df = df[~df["individual"].isin(EXCLUDE_DONORS)].reset_index(drop=True)
    print(f"loaded v4 cache: {len(df):,} cells")
    part_a(df)
    part_b(df)
    print(f"\nAll T outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
