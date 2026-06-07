#!/usr/bin/env python3
"""
H — within-PsychAD maturity baseline.

User asked: before applying complex methods (lineage tracing, label
transfer), is there a clear baseline INSIDE PsychAD-V3 that shows the
marker-based subtypes (ExN_mature / immature / weak) capture real
developmental biology?

This script uses the per-cell marker parquet from G (already produced)
to:

  H1. Validate that ExN subtype proportions shift with donor age inside
      PsychAD-V3. If they don't, the classifier carries no maturity
      information beyond its depth artifact.
  H2. Build a CONTINUOUS, depth-normalised maturity score per cell
      (mature_score - immature_score, both log1p CP10K), entirely
      within PsychAD-V3.
  H3. Compare the continuous score to the discrete marker_annotation.
      Do "ExN_mature" cells systematically score higher than
      "ExN_immature" cells? Is there overlap that exposes the
      threshold artifact?
  H4. Per-donor median score vs donor age — does mean cell maturity
      track donor age across the 71 PsychAD-V3 developmental donors?

Outputs:
  h_psychad_subtype_vs_age.csv          per-donor ExN composition + age
  h_continuous_maturity_per_cell.parquet  per-cell continuous score
  h_continuous_vs_discrete.csv          subtype mean/SD of cont. score
  h_donor_maturity_vs_age.png           plot
  h_subtype_score_distributions.png     plot
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR

CELLS = OUT_DIR / "g_per_cell_markers.parquet"


def cp10k(x, tot):
    return np.log1p(np.where(tot > 0, x / tot * 1e4, 0.0))


def main():
    df = pd.read_parquet(CELLS)
    print(f"loaded {len(df):,} cells; columns: {list(df.columns)}")

    psy = df[df["group"] == "PsychAD-V3"].copy()
    exn = psy[psy["marker_annotation"].isin(["ExN_mature", "ExN_immature",
                                              "ExN_weak"])].copy()
    print(f"PsychAD-V3 ExN cells (age 1-25 y): {len(exn):,}")

    # ---------- H2/H3: continuous maturity per cell ----------
    exn["RBFOX3_cp10k_log1p"] = cp10k(exn["RBFOX3"].values, exn["total_umi"].values)
    exn["DCX_cp10k_log1p"]    = cp10k(exn["DCX"].values,    exn["total_umi"].values)
    exn["RBFOX1_cp10k_log1p"] = cp10k(exn["RBFOX1"].values, exn["total_umi"].values)
    exn["maturity_score"] = exn["RBFOX3_cp10k_log1p"] - exn["DCX_cp10k_log1p"]
    exn.to_parquet(OUT_DIR / "h_continuous_maturity_per_cell.parquet")

    cont = (exn.groupby("marker_annotation")
                .agg(n_cells=("maturity_score", "size"),
                     mean_score=("maturity_score", "mean"),
                     median_score=("maturity_score", "median"),
                     sd_score=("maturity_score", "std"),
                     mean_RBFOX3_cp10k=("RBFOX3_cp10k_log1p", "mean"),
                     mean_DCX_cp10k=("DCX_cp10k_log1p", "mean"))
                .round(4))
    cont.to_csv(OUT_DIR / "h_continuous_vs_discrete.csv")
    print("\n=== H3: continuous maturity (mature - immature, log1p CP10K) by discrete subtype ===")
    print(cont.to_string())

    # ---------- H4: per-donor age trend ----------
    donor = (exn.groupby("age_years")
                 .agg(n_cells=("maturity_score", "size"),
                      median_score=("maturity_score", "median"),
                      mean_score=("maturity_score", "mean"))
                 .reset_index())
    print("\n=== H4: per-age maturity score (PsychAD-V3) ===")
    print(donor.head(40).to_string(index=False))
    # Correlation
    r_age = stats.spearmanr(exn["age_years"], exn["maturity_score"])
    print(f"\nspearman(age_years, maturity_score) over cells: rho={r_age.statistic:+.3f}, p={r_age.pvalue:.2e}")

    # also using "individual" (donor) median score
    if "obs_name" in exn.columns:
        donor_med = (exn.groupby(["age_years"]).median(numeric_only=True)
                          .reset_index()[["age_years", "maturity_score"]])
        r_dn = stats.spearmanr(donor_med["age_years"], donor_med["maturity_score"])
        print(f"spearman over per-age medians: rho={r_dn.statistic:+.3f}, p={r_dn.pvalue:.2e}")

    # ---------- H1: discrete subtype mix per donor age ----------
    mix = (exn.groupby(["age_years", "marker_annotation"], observed=True)
                .size().unstack(fill_value=0))
    mix = mix.div(mix.sum(axis=1), axis=0).reset_index()
    mix.to_csv(OUT_DIR / "h_psychad_subtype_vs_age.csv", index=False)
    print("\n=== H1: ExN subtype mix vs donor age (PsychAD-V3, frac of ExN) ===")
    print(mix.to_string(index=False))

    # ---------- plots ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls, color in [("ExN_mature", "C0"),
                        ("ExN_immature", "C1"),
                        ("ExN_weak",   "C2")]:
        sub = exn[exn["marker_annotation"] == cls]["maturity_score"]
        ax.hist(sub, bins=80, alpha=0.55, color=color,
                 label=f"{cls} (n={len(sub):,}, mean={sub.mean():+.2f})")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("continuous maturity score  (log1p RBFOX3_CP10K − log1p DCX_CP10K)")
    ax.set_ylabel("# cells")
    ax.set_title("H3: continuous depth-normalised maturity by discrete subtype\n"
                 "(PsychAD-V3, age 1-25 y)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "h_subtype_score_distributions.png", dpi=150)
    plt.close(fig)

    # subtype mix vs age
    fig, ax = plt.subplots(figsize=(9, 5))
    for cls, color in [("ExN_mature",   "C0"),
                        ("ExN_immature", "C1"),
                        ("ExN_weak",     "C2")]:
        if cls not in mix.columns:
            continue
        ax.plot(mix["age_years"], mix[cls], "-o", label=cls, color=color)
    ax.set_xlabel("donor age (years)")
    ax.set_ylabel("fraction of ExN cells")
    ax.set_title("H1: PsychAD-V3 ExN subtype composition vs donor age")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "h_subtype_mix_vs_age.png", dpi=150)
    plt.close(fig)

    # per-age median continuous score vs age
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(donor["age_years"], donor["median_score"],
                s=donor["n_cells"]/30, alpha=0.7)
    ax.set_xlabel("donor age (years)")
    ax.set_ylabel("median continuous maturity score (per age)")
    ax.set_title(f"H4: PsychAD-V3 per-age median maturity (rho={r_age.statistic:+.3f})\n"
                 "circle size = # cells at that age")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "h_donor_maturity_vs_age.png", dpi=150)
    plt.close(fig)

    # ---------- H sanity: at matched depth bin, what is the subtype mix? ----------
    umi_bins = [0, 1000, 3000, 5000, 8000, 12000, 20000, 100000]
    exn["umi_bin"] = pd.cut(exn["total_umi"], umi_bins)
    matched = (exn.groupby(["umi_bin", "marker_annotation"], observed=True)
                    .size().unstack(fill_value=0))
    matched = matched.div(matched.sum(axis=1), axis=0).round(3)
    matched.to_csv(OUT_DIR / "h_psychad_subtype_at_matched_depth.csv")
    print("\n=== Subtype mix at matched UMI depth (PsychAD-V3 only) ===")
    print(matched.to_string())

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
