#!/usr/bin/env python3
"""
Group C — per-gene concordance for AHBA C3+.

For each gene in the C3+ network, compute donor-level Cohen's d
(childhood vs adolescence; positive d means child > adol, matching the
sign convention used in grn_dev_multi.md). Compare:

    Pair 1: PsychAD vs Velmeshev (all)
    Pair 2: Velmeshev-V2 vs Velmeshev-V3

Outputs (all in scripts/grn_dev_diagnostics/outputs/):
    c_per_gene_psychad_vs_velmeshev.parquet
    c_per_gene_velV2_vs_velV3.parquet
    c_concordance_summary.csv
    c_scatter_psychad_vs_velmeshev.png
    c_scatter_velV2_vs_velV3.png
    c_aggregate_score.csv   (sanity-check vs grn_dev_multi §6.1)
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
from _lib import (
    load_pseudobulk, cpm_from_counts, subset_age_window,
    per_gene_child_vs_adol, project_score, build_c3plus_table,
    cohens_d, OUT_DIR, CHILD, ADOL,
)

# detection floor: gene must reach this min mean CPM in BOTH compared
# subsets to be eligible — keeps the concordance scatter from being
# dominated by detection-floor noise
MIN_MEAN_CPM = 0.5


def build_split(name: str, chemistry_filter: str = None):
    """Return (cpm_adata, scored_age_window) for a (sub)dataset."""
    a = load_pseudobulk(name)
    if chemistry_filter is not None:
        a = a[a.obs["chemistry"] == chemistry_filter].copy()
    a = cpm_from_counts(a)
    return subset_age_window(a)


def per_gene_table(adata_window, ensembl_ids):
    return per_gene_child_vs_adol(adata_window, ensembl_ids=ensembl_ids)


def concordance_stats(df_a, df_b, weights, label_a, label_b,
                      eligible_mask=None):
    """Per-gene concordance summary between two child-vs-adol d tables.

    weights: ensembl_id → C3+ weight
    eligible_mask: bool array aligned to df_a (already merged on ensembl_id)
    """
    j = (df_a[["ensembl_id", "d", "p", "mean_child", "mean_adol"]]
         .rename(columns={"d": "d_A", "p": "p_A",
                          "mean_child": "mc_A", "mean_adol": "ma_A"})
         .merge(df_b[["ensembl_id", "d", "p", "mean_child", "mean_adol"]]
                .rename(columns={"d": "d_B", "p": "p_B",
                                 "mean_child": "mc_B", "mean_adol": "ma_B"}),
                on="ensembl_id", how="inner"))
    j = j.merge(weights.rename("weight"), left_on="ensembl_id", right_index=True)
    j["eligible"] = ((j["mc_A"] >= MIN_MEAN_CPM) & (j["ma_A"] >= MIN_MEAN_CPM) &
                     (j["mc_B"] >= MIN_MEAN_CPM) & (j["ma_B"] >= MIN_MEAN_CPM))

    rows = []
    for eligible_only in (False, True):
        sub = j[j["eligible"]] if eligible_only else j
        sub = sub.dropna(subset=["d_A", "d_B"])
        if len(sub) == 0: continue
        rho_p, _ = stats.pearsonr(sub["d_A"], sub["d_B"])
        rho_s, _ = stats.spearmanr(sub["d_A"], sub["d_B"])
        # weighted Pearson (by GRN weight)
        w = sub["weight"].values
        x, y = sub["d_A"].values, sub["d_B"].values
        wmean = lambda v: np.sum(w * v) / np.sum(w)
        wmx, wmy = wmean(x), wmean(y)
        wcov = np.sum(w * (x - wmx) * (y - wmy)) / np.sum(w)
        wvx  = np.sum(w * (x - wmx) ** 2) / np.sum(w)
        wvy  = np.sum(w * (y - wmy) ** 2) / np.sum(w)
        rho_w = wcov / np.sqrt(wvx * wvy) if wvx > 0 and wvy > 0 else np.nan
        # sign concordance
        sig_a = (sub["p_A"] < 0.05) & (sub["d_A"].abs() > 0.3)
        sig_b = (sub["p_B"] < 0.05) & (sub["d_B"].abs() > 0.3)
        both_sig = sig_a & sig_b
        same_sign = (np.sign(sub["d_A"]) == np.sign(sub["d_B"]))
        rows.append({
            "comparison": f"{label_a} vs {label_b}",
            "eligible_only": eligible_only,
            "n_genes": len(sub),
            "pearson_r": rho_p,
            "spearman_r": rho_s,
            "weighted_pearson_r": rho_w,
            "n_sig_A": int(sig_a.sum()),
            "n_sig_B": int(sig_b.sum()),
            "n_sig_both_same_sign": int((both_sig & same_sign).sum()),
            "n_sig_both_diff_sign": int((both_sig & ~same_sign).sum()),
            "median_d_A": float(sub["d_A"].median()),
            "median_d_B": float(sub["d_B"].median()),
            "weighted_mean_d_A": float(wmean(x)),
            "weighted_mean_d_B": float(wmean(y)),
            "frac_pos_d_A": float((sub["d_A"] > 0).mean()),
            "frac_pos_d_B": float((sub["d_B"] > 0).mean()),
        })
    return j, pd.DataFrame(rows)


def scatter(j, label_a, label_b, out_path, weight_min_for_label=0.4):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, eligible in zip(axes, [False, True]):
        sub = j[j["eligible"]] if eligible else j
        sub = sub.dropna(subset=["d_A", "d_B"])
        s = (sub["weight"] * 50).clip(2, 80)
        sc = ax.scatter(sub["d_A"], sub["d_B"], s=s, c=sub["weight"],
                        cmap="viridis", alpha=0.6, edgecolor="none")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        lim = max(abs(sub["d_A"]).max(), abs(sub["d_B"]).max(), 1.5)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5, alpha=0.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"Cohen's d  (child vs adol)  — {label_a}")
        ax.set_ylabel(f"Cohen's d  (child vs adol)  — {label_b}")
        title = f"{label_a} vs {label_b}"
        if eligible:
            title += f"\neligible only (mean CPM ≥ {MIN_MEAN_CPM} in all groups)"
        else:
            title += "\nall C3+ genes present in both"
        ax.set_title(title)
        plt.colorbar(sc, ax=ax, label="GRN weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def aggregate_scores(adata_window, weights, label):
    """Aggregate weighted GRN score and Cohen's d per pseudobulk subset."""
    s = project_score(adata_window, weights)
    df = adata_window.obs[["age_years", "stage", "chemistry"]].join(s)
    d = cohens_d(df[df["stage"] == "child"]["score"].values,
                 df[df["stage"] == "adol"]["score"].values)
    return {
        "subset": label,
        "n_child": int((df["stage"] == "child").sum()),
        "n_adol":  int((df["stage"] == "adol").sum()),
        "mean_child": float(df[df["stage"] == "child"]["score"].mean()),
        "mean_adol":  float(df[df["stage"] == "adol"]["score"].mean()),
        "cohens_d_aggregate_score": float(d),
    }


def main():
    weights_df = build_c3plus_table()
    weights = weights_df.set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes")

    # Build the four (sub)dataset windows
    print("Building per-dataset CPM-CPMd age windows ...")
    splits = {
        "PsychAD":     build_split("PsychAD"),
        "Velmeshev":   build_split("Velmeshev"),
        "Velmeshev_V2": build_split("Velmeshev", "V2"),
        "Velmeshev_V3": build_split("Velmeshev", "V3"),
    }
    for k, v in splits.items():
        nc = int((v.obs["stage"] == "child").sum())
        na = int((v.obs["stage"] == "adol").sum())
        print(f"  {k}: child={nc} adol={na}")

    # ----- Aggregate sanity check
    print("\n=== Aggregate weighted GRN score (sanity check) ===")
    agg = pd.DataFrame([aggregate_scores(splits[k], weights, k)
                        for k in splits])
    agg.to_csv(OUT_DIR / "c_aggregate_score.csv", index=False)
    print(agg.to_string(index=False))

    # ----- Per-gene tables (restricted to C3+ Ensembl IDs)
    c3_ens = weights.index.values
    per_gene = {k: per_gene_table(v, c3_ens) for k, v in splits.items()}
    print("\nPer-gene tables computed for all splits")

    # ----- Concordance Pair 1: PsychAD vs Velmeshev (all)
    j1, s1 = concordance_stats(per_gene["PsychAD"],
                               per_gene["Velmeshev"],
                               weights,
                               "PsychAD", "Velmeshev")
    j1.to_parquet(OUT_DIR / "c_per_gene_psychad_vs_velmeshev.parquet")
    scatter(j1, "PsychAD", "Velmeshev",
            OUT_DIR / "c_scatter_psychad_vs_velmeshev.png")

    # ----- Concordance Pair 2: Vel-V2 vs Vel-V3
    j2, s2 = concordance_stats(per_gene["Velmeshev_V2"],
                               per_gene["Velmeshev_V3"],
                               weights,
                               "Velmeshev_V2", "Velmeshev_V3")
    j2.to_parquet(OUT_DIR / "c_per_gene_velV2_vs_velV3.parquet")
    scatter(j2, "Velmeshev_V2", "Velmeshev_V3",
            OUT_DIR / "c_scatter_velV2_vs_velV3.png")

    # ----- Combined summary
    summary = pd.concat([s1, s2], ignore_index=True)
    summary.to_csv(OUT_DIR / "c_concordance_summary.csv", index=False)
    print("\n=== Per-gene concordance summary ===")
    print(summary.to_string(index=False))

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
