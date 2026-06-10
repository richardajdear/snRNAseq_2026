#!/usr/bin/env python3
"""C3-maturation Step 0 — depth-robustness GATE.

The prior "adolescent dip" in the C3+ aggregate was a UMI-depth artefact in V2
Velmeshev. Before trusting ANY age trend we must pick a C3 score that is robust
to sequencing depth / chemistry. This script compares candidate scores on the
Velmeshev ExN-per-donor pseudobulk (which carries both V2 and V3 donors) and
reports, for each:

  (A) depth sensitivity   — Spearman(score, log10 total counts), within adults
                            and age-partialled overall  (want ~0)
  (B) chemistry effect    — V2-vs-V3 effect at matched age (OLS coef; want ~0)
  (C) downsample concord. — Spearman(full, thinned-to-common-depth) (want ~1)
  (D) negative control    — null band from expression-matched random weights

Candidate scores (code/_lib_c3.py):
  pos_cpm       Sum w+ * cpm                (reproduces the depth-biased prior)
  signed_cpm    Sum w  * cpm                (C3+ minus C3- contrast)
  signed_logcpm Sum w  * log1p(cpm)         (compresses high-expression dominance)
  rank_contrast |w|-weighted percentile contrast (scale-invariant)

Inline-safe (small pseudobulk). Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  singularity exec --pwd $PWD ... python3 scripts/c3_maturation/s00b_depth_harness.py
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

ADULT_MIN = 18.0
N_NULL = 50


def partial_spearman(x, y, z):
    """Spearman corr of x,y after rank-regressing out z from both."""
    def resid(a, b):
        ra, rb = stats.rankdata(a), stats.rankdata(b)
        rb = np.c_[np.ones_like(rb), rb]
        beta, *_ = np.linalg.lstsq(rb, ra, rcond=None)
        return ra - rb @ beta
    rx = resid(x, z); ry = resid(y, z)
    return stats.spearmanr(rx, ry).statistic


def compute_scores(adata, w, cpm):
    vn = adata.var_names
    return {
        "pos_cpm":       L.score_weighted_cpm(cpm, vn, w, pos_only=True),
        "signed_cpm":    L.score_weighted_cpm(cpm, vn, w),
        "signed_logcpm": L.score_weighted_cpm(cpm, vn, w, log1p=True),
        "rank_contrast": L.score_rank_contrast(cpm, vn, w),
    }


def main():
    print("Loading Velmeshev ExN-by-donor pseudobulk ...", flush=True)
    a = ad.read_h5ad(L.PB["Vel_ExN_by_donor"])
    w = L.c3_signed(a)
    print(f"  C3 signed weights mapped to Ensembl: {len(w)} "
          f"({(w>0).sum()} +, {(w<0).sum()} -)", flush=True)

    cpm = L.cpm_matrix(a)
    dm = L.depth_metrics(a)
    age = pd.to_numeric(a.obs["age_years"], errors="coerce").values
    chem = a.obs["chemistry"].astype(str).values

    scores = compute_scores(a, w, cpm)
    df = pd.DataFrame(scores, index=a.obs_names)
    df["age_years"] = age
    df["chemistry"] = chem
    df["dataset"] = a.obs["dataset"].astype(str).values
    df = pd.concat([df, dm], axis=1)
    df.to_csv(L.OUT_DIR / "s00b_donor_scores.csv")

    adult = age >= ADULT_MIN
    # matched-age band for chemistry test = overlap of V2 and V3 age ranges
    v2a, v3a = age[chem == "V2"], age[chem == "V3"]
    lo, hi = max(v2a.min(), v3a.min()), min(v2a.max(), v3a.max())
    band = (age >= lo) & (age <= hi)
    print(f"  matched-age band for chemistry test: [{lo:.2f}, {hi:.2f}] "
          f"(V2 n={int(((chem=='V2')&band).sum())}, V3 n={int(((chem=='V3')&band).sum())})",
          flush=True)

    # ----- null band (D): expression-matched random weights -----
    ref_expr = cpm.mean(0)
    null_depth_r, null_age_r = {k: [] for k in scores}, {k: [] for k in scores}
    for s in range(N_NULL):
        nw = L.matched_random_weights(w, a.var_names, ref_expr, seed=s)
        nsc = {
            "signed_cpm":    L.score_weighted_cpm(cpm, a.var_names, nw),
            "signed_logcpm": L.score_weighted_cpm(cpm, a.var_names, nw, log1p=True),
            "rank_contrast": L.score_rank_contrast(cpm, a.var_names, nw),
            "pos_cpm":       L.score_weighted_cpm(cpm, a.var_names, nw, pos_only=True),
        }
        for k, v in nsc.items():
            null_depth_r[k].append(stats.spearmanr(v[adult], dm["log10_total"].values[adult]).statistic)
            null_age_r[k].append(stats.spearmanr(v, age).statistic)

    # downsample once (C), score all candidates on the thinned matrix
    target = np.percentile(dm["total_counts"].values, 10)
    cpm_ds = L.downsample_counts(a, target_total=target, seed=0)
    scores_ds = compute_scores(a, w, cpm_ds)

    # ----- summary table -----
    rows = []
    for k, v in scores.items():
        r_depth_adult = stats.spearmanr(v[adult], dm["log10_total"].values[adult]).statistic
        r_depth_part = partial_spearman(v, dm["log10_total"].values, age)
        # chemistry effect at matched age: OLS score ~ age + 1[V3]
        X = np.c_[np.ones(band.sum()), age[band], (chem[band] == "V3").astype(float)]
        beta, *_ = np.linalg.lstsq(X, v[band], rcond=None)
        resid = v[band] - X @ beta
        sd = resid.std(ddof=X.shape[1])
        chem_coef = beta[2] / (np.abs(v[band]).mean() + 1e-12)  # normalised
        r_age = stats.spearmanr(v, age).statistic
        r_age_v3 = stats.spearmanr(v[chem == "V3"], age[chem == "V3"]).statistic
        # downsample concordance (C)
        r_ds = stats.spearmanr(v, scores_ds[k]).statistic
        rows.append({
            "score": k,
            "r_depth_adult": r_depth_adult,
            "r_depth_partialled": r_depth_part,
            "chem_coef_norm": chem_coef,
            "r_downsample_concord": r_ds,
            "r_age_all": r_age,
            "r_age_V3only": r_age_v3,
            "null_depth_r_p95": np.nanpercentile(np.abs(null_depth_r[k]), 95),
        })
    summary = pd.DataFrame(rows).set_index("score")
    summary.to_csv(L.OUT_DIR / "s00b_robustness_summary.csv")
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n========== Step 0 robustness summary ==========")
    print(summary.round(3).to_string())

    # decision rule: a depth-robust score must (i) have ~0 age-partialled
    # depth correlation, (ii) ~0 chemistry effect at matched age, (iii) give
    # the SAME age trend in V3-only as in V2+V3 (the direct artefact test —
    # the prior C3+ dip showed up precisely as a V2-vs-V3 disagreement), and
    # (iv) be stable under downsampling. The within-adult depth correlation is
    # kept in the table for info but excluded here (few adult donors -> noisy).
    summary["age_trend_chem_gap"] = (summary["r_age_all"] - summary["r_age_V3only"]).abs()
    summary["badness"] = (summary["r_depth_partialled"].abs()
                          + summary["chem_coef_norm"].abs()
                          + summary["age_trend_chem_gap"]
                          + (1 - summary["r_downsample_concord"]))
    best = summary["badness"].idxmin()
    print(f"\nRecommended depth-robust score: **{best}**  "
          f"(badness={summary.loc[best,'badness']:.3f})")
    print("  Use this score for all downstream age analyses (Steps 1-3).")

    # ----- figure -----
    fig, axes = plt.subplots(len(scores), 2, figsize=(11, 3.2 * len(scores)))
    for i, (k, v) in enumerate(scores.items()):
        for ch, c in [("V2", "C3"), ("V3", "C0")]:
            m = chem == ch
            axes[i, 0].scatter(age[m], v[m], s=22, alpha=.7, color=c, label=ch)
            axes[i, 1].scatter(dm["log10_total"].values[m], v[m], s=22, alpha=.7, color=c, label=ch)
        axes[i, 0].set_title(f"{k}: vs age  (rho_all={summary.loc[k,'r_age_all']:+.2f}, "
                             f"V3={summary.loc[k,'r_age_V3only']:+.2f})", fontsize=9)
        axes[i, 0].set_xlabel("age (years)"); axes[i, 0].set_ylabel(k)
        axes[i, 1].set_title(f"{k}: vs depth  (rho_adult={summary.loc[k,'r_depth_adult']:+.2f}, "
                             f"partial={summary.loc[k,'r_depth_partialled']:+.2f})", fontsize=9)
        axes[i, 1].set_xlabel("log10 total counts")
        axes[i, 0].legend(fontsize=7); axes[i, 1].legend(fontsize=7)
    fig.suptitle("Step 0: candidate C3 scores vs age and depth (Velmeshev ExN, V2+V3)", y=1.005)
    fig.tight_layout()
    fig.savefig(L.OUT_DIR / "s00b_depth_harness.png", dpi=140, bbox_inches="tight")
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
