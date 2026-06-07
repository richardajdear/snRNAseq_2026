#!/usr/bin/env python3
"""
F2 — biological deep-dive on the "same genes, opposite stories" finding.

F1 showed:
  - Top 90 of 3,331 shared C3+ genes carry 50% of |signal|
  - PsychAD high-weight genes go NEGATIVE (mean weight-pct for d<-0.5 = 0.61)
  - Velmeshev high-weight genes go POSITIVE (mean weight-pct for d>+0.5 = 0.54)

We need to understand: which genes ARE these, what do they do, and is the
divergence dataset-wide or carried by a specific functional module of C3+?

Outputs:
  f2_top_contributors_annotated.csv         top contributors w/ symbols + delta
  f2_concordance_by_weight_quintile.csv     d_psy vs d_vel grouped by C3+ weight quintile
  f2_dataset_disagreement_genes.csv         genes where d_psy<-0.5 AND d_vel>+0.5 (or vice versa)
  f2_panels_per_quintile.png                scatter of d_psy vs d_vel per weight quintile
  f2_weight_vs_d.png                        signed d vs weight, both datasets overlaid
  f2_log_directional_ratio.png              fraction of high-weight genes going up vs down
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
from _lib import OUT_DIR, build_c3plus_table, load_pseudobulk

CONTRIB = OUT_DIR / "f_contribution_per_gene.parquet"


def add_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Attach gene_symbol via the cached C3+ table (built from Vel var)."""
    c3 = build_c3plus_table()[["ensembl_id", "gene_symbol"]]
    df = df.merge(c3, on="ensembl_id", how="left")
    return df


def main():
    print(f"Reading {CONTRIB}")
    j = pd.read_parquet(CONTRIB)
    print(f"  {len(j)} C3+ genes  cols: {list(j.columns)}")

    j = add_symbols(j)
    # symbols are missing for some genes — fill with ENSG
    j["gene_symbol"] = j["gene_symbol"].fillna(j["ensembl_id"])

    # ----- 1. annotate top contributors in each dataset -----
    cols_keep = ["gene_symbol", "ensembl_id", "weight", "weight_pct",
                 "mc_A", "ma_A", "delta_cpm_psy", "d_psy", "contrib_psy",
                 "mc_B", "ma_B", "delta_cpm_vel", "d_vel", "contrib_vel"]
    top_v_drop = (j.sort_values("contrib_vel", ascending=False)
                   .head(40)[cols_keep].copy())
    top_v_rise = (j.sort_values("contrib_vel", ascending=True)
                   .head(40)[cols_keep].copy())
    top_p_drop = (j.sort_values("contrib_psy", ascending=False)
                   .head(40)[cols_keep].copy())
    top_p_rise = (j.sort_values("contrib_psy", ascending=True)
                   .head(40)[cols_keep].copy())
    top_v_drop["bucket"] = "Vel_top_drop"   # contributes positively to Vel agg
    top_v_rise["bucket"] = "Vel_top_rise"   # contributes negatively
    top_p_drop["bucket"] = "Psy_top_drop"
    top_p_rise["bucket"] = "Psy_top_rise"
    out = pd.concat([top_v_drop, top_v_rise, top_p_drop, top_p_rise],
                    ignore_index=True)
    out.to_csv(OUT_DIR / "f2_top_contributors_annotated.csv", index=False)

    print("\n=== Top 25 Vel drop carriers (annotated) ===")
    print(top_v_drop.head(25)[["gene_symbol", "weight", "d_psy", "d_vel",
                                "contrib_psy", "contrib_vel"]].to_string(index=False))

    print("\n=== Top 25 PsychAD 'rise' (i.e. adol > child) drivers ===")
    print(top_p_rise.head(25)[["gene_symbol", "weight", "d_psy", "d_vel",
                                "contrib_psy", "contrib_vel"]].to_string(index=False))

    # ----- 2. concordance grouped by weight quintile -----
    j["weight_q"] = pd.qcut(j["weight"], 5,
                             labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_top"])
    rows = []
    for q, sub in j.groupby("weight_q", observed=True):
        sub = sub.dropna(subset=["d_psy", "d_vel"])
        r_p = stats.pearsonr(sub["d_psy"], sub["d_vel"])[0]
        r_s = stats.spearmanr(sub["d_psy"], sub["d_vel"])[0]
        rows.append({
            "weight_quintile": q,
            "n_genes": len(sub),
            "weight_min": float(sub["weight"].min()),
            "weight_max": float(sub["weight"].max()),
            "median_d_psy": float(sub["d_psy"].median()),
            "median_d_vel": float(sub["d_vel"].median()),
            "mean_d_psy":   float(sub["d_psy"].mean()),
            "mean_d_vel":   float(sub["d_vel"].mean()),
            "pearson_r":    float(r_p),
            "spearman_r":   float(r_s),
            "frac_pos_psy": float((sub["d_psy"] > 0).mean()),
            "frac_pos_vel": float((sub["d_vel"] > 0).mean()),
            "frac_both_pos":  float(((sub["d_psy"] > 0.3) & (sub["d_vel"] > 0.3)).mean()),
            "frac_both_neg":  float(((sub["d_psy"] < -0.3) & (sub["d_vel"] < -0.3)).mean()),
            "frac_psyneg_velpos": float(((sub["d_psy"] < -0.3) & (sub["d_vel"] > 0.3)).mean()),
            "frac_psypos_velneg": float(((sub["d_psy"] > 0.3) & (sub["d_vel"] < -0.3)).mean()),
        })
    qsum = pd.DataFrame(rows)
    qsum.to_csv(OUT_DIR / "f2_concordance_by_weight_quintile.csv", index=False)
    print("\n=== Concordance by C3+ weight quintile ===")
    print(qsum.to_string(index=False))

    # ----- 3. disagreement genes: opposite-sign cases -----
    flip_v_up_p_dn = j[(j["d_vel"] > 0.5) & (j["d_psy"] < -0.5)].sort_values(
        "contrib_vel", ascending=False)
    flip_v_dn_p_up = j[(j["d_vel"] < -0.5) & (j["d_psy"] > 0.5)].sort_values(
        "contrib_psy", ascending=False)
    print(f"\n=== Genes where Vel drops (d>+0.5) AND PsychAD rises (d<-0.5): {len(flip_v_up_p_dn)} ===")
    print(flip_v_up_p_dn.head(20)[["gene_symbol", "weight", "weight_pct",
                                     "d_psy", "d_vel", "mc_A", "ma_A",
                                     "mc_B", "ma_B"]].to_string(index=False))
    flips = pd.concat([
        flip_v_up_p_dn.assign(disagreement="vel_up_psy_down"),
        flip_v_dn_p_up.assign(disagreement="vel_down_psy_up"),
    ], ignore_index=True)
    flips[["gene_symbol", "ensembl_id", "weight", "weight_pct", "d_psy", "d_vel",
           "mc_A", "ma_A", "mc_B", "ma_B", "disagreement"]].to_csv(
        OUT_DIR / "f2_dataset_disagreement_genes.csv", index=False)

    # how many of these have weight in the top quintile?
    n_top = (flips["weight_q"] == "Q5_top").sum() if "weight_q" in flips.columns else None
    print(f"  of these flips, {n_top} are top-quintile-weight genes")

    # ----- 4. weight x d scatter overlay -----
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(j["weight"], j["d_vel"], s=4, c="C3", alpha=0.4,
                label=f"Velmeshev  (mean d = {j['d_vel'].mean():+.2f})")
    ax.scatter(j["weight"], j["d_psy"], s=4, c="C0", alpha=0.4,
                label=f"PsychAD    (mean d = {j['d_psy'].mean():+.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.5, color="grey", lw=0.4, ls=":")
    ax.axhline(-0.5, color="grey", lw=0.4, ls=":")
    ax.set_xlabel("C3+ gene importance weight")
    ax.set_ylabel("per-gene Cohen's d (child vs adol)")
    ax.set_title("Signed per-gene d vs C3+ weight\nhigher-weight genes split: Vel positive, PsychAD negative")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "f2_weight_vs_d_overlay.png", dpi=150)
    plt.close(fig)

    # ----- 5. weight-binned mean d  -----
    j["weight_decile"] = pd.qcut(j["weight"], 10, labels=False)
    bins = (j.groupby("weight_decile")
              .agg(median_weight=("weight", "median"),
                   mean_d_psy=("d_psy", "mean"),
                   sem_d_psy=("d_psy", lambda x: x.std()/np.sqrt(len(x))),
                   mean_d_vel=("d_vel", "mean"),
                   sem_d_vel=("d_vel", lambda x: x.std()/np.sqrt(len(x))),
                   n=("d_psy", "size"))
              .reset_index())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(bins["median_weight"], bins["mean_d_psy"], yerr=bins["sem_d_psy"],
                 fmt="-o", label="PsychAD", color="C0")
    ax.errorbar(bins["median_weight"], bins["mean_d_vel"], yerr=bins["sem_d_vel"],
                 fmt="-o", label="Velmeshev", color="C3")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("C3+ weight (decile median)")
    ax.set_ylabel("mean per-gene d (± SEM)")
    ax.set_title("Mean per-gene d vs C3+ weight decile\n"
                 "the divergence GROWS at higher weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "f2_mean_d_by_weight_decile.png", dpi=150)
    plt.close(fig)
    bins.to_csv(OUT_DIR / "f2_mean_d_by_weight_decile.csv", index=False)
    print("\n=== Mean per-gene d by weight decile ===")
    print(bins.to_string(index=False))

    # ----- 6. concordance within weight quintile - scatter panel  -----
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharex=True, sharey=True)
    for ax, (q, sub) in zip(axes, j.groupby("weight_q", observed=True)):
        sub = sub.dropna(subset=["d_psy", "d_vel"])
        ax.scatter(sub["d_psy"], sub["d_vel"], s=5, alpha=0.4)
        lim = 3
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.4)
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("d (PsychAD)"); ax.set_ylabel("d (Velmeshev)")
        r = stats.pearsonr(sub["d_psy"], sub["d_vel"])[0]
        ax.set_title(f"{q}\nn={len(sub)}, r={r:.2f}\n"
                     f"med d_psy={sub['d_psy'].median():+.2f}  "
                     f"med d_vel={sub['d_vel'].median():+.2f}")
    fig.suptitle("Per-gene d concordance within C3+ weight quintile", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "f2_concordance_per_quintile.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig)

    # ----- 7. summary table of the "story" -----
    summary_rows = []
    for name, mask in [
        ("All genes",                        np.ones(len(j), dtype=bool)),
        ("|d_psy|<0.3 & |d_vel|<0.3 (flat)",
            (j["d_psy"].abs() < 0.3) & (j["d_vel"].abs() < 0.3)),
        ("Both drop (d>+0.3 in both)",
            (j["d_psy"] > 0.3) & (j["d_vel"] > 0.3)),
        ("Both rise (d<-0.3 in both)",
            (j["d_psy"] < -0.3) & (j["d_vel"] < -0.3)),
        ("Vel drops, PsychAD rises",
            (j["d_vel"] > 0.3) & (j["d_psy"] < -0.3)),
        ("Vel rises, PsychAD drops",
            (j["d_vel"] < -0.3) & (j["d_psy"] > 0.3)),
    ]:
        sub = j[mask]
        if len(sub) == 0: continue
        summary_rows.append({
            "bucket":     name,
            "n_genes":    int(len(sub)),
            "frac":       round(len(sub)/len(j), 4),
            "mean_weight":   round(sub["weight"].mean(), 4),
            "mean_weight_pct": round(sub["weight"].rank(pct=True).mean(), 3) if len(sub) > 1 else None,
            "weight_pct_of_full":
                round(j["weight"].rank(pct=True).loc[mask].mean(), 3),
            "total_contrib_psy": round(sub["contrib_psy"].sum(), 2),
            "total_contrib_vel": round(sub["contrib_vel"].sum(), 2),
        })
    story = pd.DataFrame(summary_rows)
    story.to_csv(OUT_DIR / "f2_story_summary.csv", index=False)
    print("\n=== Story summary (concordant vs flipped genes) ===")
    print(story.to_string(index=False))

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
