#!/usr/bin/env python3
"""
T — headline trajectories (all-ExN | q0) with linear trend tests,
    C3+ vs maturity scatter + Kitagawa decomposition (§3.5),
    and maturity-vs-age distribution (§3.6).

Headline pair (per cohort, NOT pooled):
  left  = all ExN cells (no maturity filter)   — PsychAD masked, Vel drops
  right = least-mature quintile (q0)           — all cohorts drop
Both: donor-mean C3+ vs age, raw units, per-cohort OLS fit + significance.
Continuous beta (C3+ per year, and %/yr) reported alongside the fuzzy d.

§3.5  t2_c3_vs_maturity         — C3+ vs maturity, child/adol split + decomp
§3.6  t3_maturity_vs_age        — maturity distribution vs age per cohort

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

sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR, AGE_LO, AGE_HI, fuzzy_d_from_donor_scores

CACHE_V4 = OUT_DIR / "r_per_cell_cache_v4.parquet"
GROUPS = ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]
COLORS = {"PsychAD-V3": "#C0392B", "Velmeshev-V2": "#27AE60",
          "Velmeshev-V3": "#2980B9"}
MODULE_MATURE = ["NEUROD2", "BCL11B", "SATB2", "MEF2C", "NEFM", "NEFH",
                 "SYT1", "SNAP25", "MAP2"]
EXCLUDE_DONORS = {"Donor_1400"}
MIN_CELLS = 5
FUZZY_LO, FUZZY_HI = 8, 12
BOUND = 10
N_Q = 5
N_DEC = 10


def add_module(df):
    cp = [f"cp_{m}" for m in MODULE_MATURE if f"cp_{m}" in df.columns]
    df = df.copy()
    df["mature_module"] = df[cp].mean(axis=1)
    return df


def donor_agg(cells):
    don = (cells.groupby("individual", observed=True)
                .agg(score=("per_cell_c3", "mean"),
                     n=("per_cell_c3", "size"),
                     age=("age_years", "first")).reset_index())
    don = don[(don["n"] >= MIN_CELLS) & (don["age"] >= AGE_LO)
              & (don["age"] < AGE_HI)].dropna(subset=["score"])
    return don


def linfit(don):
    r = stats.linregress(don["age"], don["score"])
    mean = don["score"].mean()
    return dict(beta=r.slope, p=r.pvalue, r2=r.rvalue**2,
                pct_per_yr=100*r.slope/mean if mean else np.nan,
                intercept=r.intercept, n=len(don))


def fuzzy_d(don):
    if len(don) < 4:
        return np.nan
    return fuzzy_d_from_donor_scores(don["age"].values,
                                     don["score"].values)["mean_d"]


# ---------------------------------------------------------------------------
# PART A — headline trajectory pair (all-ExN | q0), per cohort, linear fits
# ---------------------------------------------------------------------------

def _panel(ax, donor_dict, fits, title):
    for g in GROUPS:
        d = donor_dict[g]
        ax.scatter(d["age"], d["score"], s=20, color=COLORS[g], alpha=0.55,
                   edgecolors="none")
        f = fits[g]
        xs = np.array([d["age"].min(), d["age"].max()])
        ax.plot(xs, f["intercept"] + f["beta"]*xs, "-", color=COLORS[g], lw=2.4,
                zorder=5,
                label=(f"{g}: β={f['pct_per_yr']:+.1f}%/yr, "
                       f"p={f['p']:.1e}" + ("*" if f["p"] < 0.05 else "")))
    ax.axvspan(FUZZY_LO, FUZZY_HI, color="grey", alpha=0.13, zorder=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("donor age (years)")
    ax.set_xlim(0, 25)
    ax.legend(frameon=False, fontsize=8)


def part_a(df):
    print("\n=== PART A: headline trajectories + linear trends ===", flush=True)
    donor_all, donor_q0, fits_all, fits_q0 = {}, {}, {}, {}
    trend_rows = []
    for g in GROUPS:
        sub = add_module(df[df["group"] == g])
        sub["mat_q"] = pd.qcut(sub["mature_module"], N_Q, labels=False,
                               duplicates="drop")
        da = donor_agg(sub)
        d0 = donor_agg(sub[sub["mat_q"] == 0])
        donor_all[g], donor_q0[g] = da, d0
        fa, f0 = linfit(da), linfit(d0)
        fits_all[g], fits_q0[g] = fa, f0
        for strat, don, f in [("all_ExN", da, fa), ("q0_least_mature", d0, f0)]:
            trend_rows.append({
                "group": g, "stratum": strat, "n_donors": f["n"],
                "fuzzy_d": round(fuzzy_d(don), 3),
                "beta_per_yr": f["beta"], "pct_per_yr": round(f["pct_per_yr"], 2),
                "p_value": f["p"], "r2": round(f["r2"], 3),
                "significant_0.05": f["p"] < 0.05})
    trends = pd.DataFrame(trend_rows)
    trends.to_csv(OUT_DIR / "t1_linear_trends.csv", index=False)
    print(trends.to_string(index=False))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.6), sharey=False)
    _panel(axL, donor_all, fits_all, "All ExN cells (no maturity filter)")
    _panel(axR, donor_q0, fits_q0, "Least-mature quintile (q0) only")
    axL.set_ylabel("mean per-cell C3+ (raw, CPM units)")
    axR.text(np.mean([FUZZY_LO, FUZZY_HI]), axR.get_ylim()[1],
             "fuzzy child/adol\nboundary (8–12 y)", ha="center", va="top",
             fontsize=7, color="dimgrey")
    fig.suptitle("Headline — per-cohort C3+ trajectories with linear trend. "
                 "Left (all ExN): PsychAD flat/masked; right (q0): all cohorts "
                 "decline. Raw donor means, OLS fit, β = % change per year",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "t1_headline_trajectories.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("saved t1_headline_trajectories.png")


# ---------------------------------------------------------------------------
# PART B — C3+ vs maturity scatter + Kitagawa decomposition (§3.5)
# ---------------------------------------------------------------------------

def part_b(df):
    print("\n=== PART B: C3+ vs maturity + decomposition ===", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=False)
    binned_rows, decomp_rows = [], []
    for ax, g in zip(axes, GROUPS):
        sub = add_module(df[df["group"] == g]).copy()
        sub["stage"] = np.where(sub["age_years"] < BOUND, "child", "adol")
        sub["mdec"] = pd.qcut(sub["mature_module"], N_DEC, labels=False,
                              duplicates="drop")
        ax.hexbin(sub["mature_module"], sub["per_cell_c3"], gridsize=40,
                  cmap="Greys", bins="log", mincnt=1, alpha=0.45)
        rho, _ = stats.spearmanr(sub["mature_module"], sub["per_cell_c3"])
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

        piv = (sub.groupby(["mdec", "stage"])
                  .agg(c3=("per_cell_c3", "mean"),
                       n=("per_cell_c3", "size")).reset_index())
        wide = piv.pivot(index="mdec", columns="stage",
                         values=["c3", "n"]).fillna(0)
        c_ch = wide[("c3", "child")].values; c_ad = wide[("c3", "adol")].values
        n_ch = wide[("n", "child")].values; n_ad = wide[("n", "adol")].values
        f_ch = n_ch / n_ch.sum(); f_ad = n_ad / n_ad.sum()
        Cbar_ch = (f_ch * c_ch).sum(); Cbar_ad = (f_ad * c_ad).sum()
        d_total = Cbar_ch - Cbar_ad
        comp = ((f_ch - f_ad) * (c_ch + c_ad) / 2).sum()
        within = ((c_ch - c_ad) * (f_ch + f_ad) / 2).sum()
        decomp_rows.append({
            "group": g, "C3_child": Cbar_ch, "C3_adol": Cbar_ad,
            "delta_total": d_total, "composition": comp, "within_state": within,
            "pct_composition": 100*comp/d_total if d_total else np.nan,
            "pct_within": 100*within/d_total if d_total else np.nan,
            "spearman_mat_c3": rho})

    fig.suptitle("C3+ vs neuronal maturity, split by age. Overlapping "
                 "child/adol curves ⇒ drop is compositional (maturation); a "
                 "downward child→adol shift ⇒ within-state age decline",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "t2_c3_vs_maturity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(binned_rows).to_csv(
        OUT_DIR / "t2_c3_vs_maturity_binned.csv", index=False)
    dec = pd.DataFrame(decomp_rows)
    dec.to_csv(OUT_DIR / "t2_decomposition.csv", index=False)
    print("\n--- Kitagawa decomposition of the child→adol C3+ drop ---")
    print(dec.round(1).to_string(index=False))


# ---------------------------------------------------------------------------
# PART C — maturity distribution vs age (§3.6)
# ---------------------------------------------------------------------------

def part_c(df):
    print("\n=== PART C: maturity vs age (§3.6) ===", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
    rows = []
    for ax, g in zip(axes, GROUPS):
        sub = add_module(df[df["group"] == g])
        ax.hexbin(sub["age_years"], sub["mature_module"], gridsize=40,
                  cmap="magma", bins="log", mincnt=1)
        # median maturity per 2-y age bin
        bins = np.arange(0, 26, 2.0)
        sub = sub.copy()
        sub["abin"] = pd.cut(sub["age_years"], bins,
                             labels=(bins[:-1] + 1))
        med = sub.groupby("abin", observed=True)["mature_module"].median()
        ax.plot(med.index.astype(float), med.values, "o-", color="cyan",
                lw=2, ms=4, label="median maturity / 2 y")
        rho_a, p_a = stats.spearmanr(sub["age_years"], sub["mature_module"])
        # cell-level slope (maturity per year) for reference
        lr = stats.linregress(sub["age_years"], sub["mature_module"])
        ax.axvspan(FUZZY_LO, FUZZY_HI, color="white", alpha=0.12, zorder=0)
        ax.set_title(f"{g}\nSpearman ρ(age, maturity) = {rho_a:+.2f}  "
                     f"(slope {lr.slope:+.3f}/yr)", fontsize=10)
        ax.set_xlabel("donor age (years)")
        ax.set_xlim(0, 25)
        if g == GROUPS[0]:
            ax.set_ylabel("mature-module score (per cell)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        rows.append({"group": g, "spearman_age_maturity": rho_a, "p": p_a,
                     "slope_per_yr": lr.slope,
                     "median_maturity_child":
                         sub[sub["age_years"] < BOUND]["mature_module"].median(),
                     "median_maturity_adol":
                         sub[sub["age_years"] >= BOUND]["mature_module"].median()})
    fig.suptitle("§3.6 — distribution of cell maturity vs age (density). The "
                 "maturity composition shifts only weakly with age within "
                 "1–25 y; cells are mostly already mature by childhood",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "t3_maturity_vs_age.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st = pd.DataFrame(rows)
    st.to_csv(OUT_DIR / "t3_maturity_age_stats.csv", index=False)
    print(st.round(3).to_string(index=False))


def main():
    df = pd.read_parquet(CACHE_V4)
    df = df[~df["individual"].isin(EXCLUDE_DONORS)].reset_index(drop=True)
    print(f"loaded v4 cache: {len(df):,} cells")
    part_a(df)
    part_b(df)
    part_c(df)
    print(f"\nAll T outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
