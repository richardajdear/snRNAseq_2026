#!/usr/bin/env python3
"""
Group F — gene-level dominance of the Velmeshev child→adolescence drop.

User question: is the +1.28 aggregate d in Velmeshev (vs -0.44 in PsychAD)
carried by a few high-weight high-CPM genes, or is it broadly distributed?
And do those drop-carriers have high C3+ weights, or only middling weights
with high CPM?

Reads c_per_gene_psychad_vs_velmeshev.parquet (per-gene Cohen's d in each
dataset, plus C3+ weight and mean CPM in each stage).

For each dataset (PsychAD, Velmeshev):
  contribution_g = weight_g * (mean_child_g - mean_adol_g)
  positive contribution = pulls aggregate up = supports child > adol
  negative contribution = pulls aggregate down

Outputs:
    f_contribution_per_gene.parquet   per-gene contribution + cumulative
    f_dominance_summary.csv           top-N for {50, 80, 95, 99}% of |signal|
    f_scatter_d_vs_weight.png         d vs C3+ weight, color by mean CPM
    f_weight_percentile_high_drop.csv weight-percentile of |d|>0.5 genes
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR

PER_GENE = OUT_DIR / "c_per_gene_psychad_vs_velmeshev.parquet"


# concordance file uses suffix _A (PsychAD) and _B (Velmeshev)
_SUFFIX_MAP = {"psy": "A", "vel": "B"}


def add_contribution(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """contribution = weight * (mean_child_CPM - mean_adol_CPM)."""
    s = _SUFFIX_MAP[suffix]
    mc = df[f"mc_{s}"]
    ma = df[f"ma_{s}"]
    df[f"delta_cpm_{suffix}"] = mc - ma
    df[f"contrib_{suffix}"]   = df["weight"] * (mc - ma)
    df[f"abs_contrib_{suffix}"] = df[f"contrib_{suffix}"].abs()
    df[f"mean_cpm_{suffix}"]   = (mc + ma) / 2.0
    df[f"d_{suffix}"]          = df[f"d_{s}"]
    df[f"p_{suffix}"]          = df[f"p_{s}"]
    return df


def dominance_rows(df: pd.DataFrame, dataset: str, suffix: str) -> list:
    """How many top genes for X% of total |contribution|, AND signed totals."""
    s_pos = df[f"contrib_{suffix}"].clip(lower=0).sum()
    s_neg = df[f"contrib_{suffix}"].clip(upper=0).sum()
    s_net = s_pos + s_neg
    abs_sum = df[f"abs_contrib_{suffix}"].sum()
    sorted_abs = df.sort_values(f"abs_contrib_{suffix}", ascending=False)
    cum = sorted_abs[f"abs_contrib_{suffix}"].cumsum()
    rows = [{
        "dataset": dataset, "metric": "n_genes_total", "value": int(len(df)),
    }, {
        "dataset": dataset, "metric": "sum_pos_contrib", "value": float(s_pos),
    }, {
        "dataset": dataset, "metric": "sum_neg_contrib", "value": float(s_neg),
    }, {
        "dataset": dataset, "metric": "sum_net_contrib", "value": float(s_net),
    }, {
        "dataset": dataset, "metric": "sum_abs_contrib", "value": float(abs_sum),
    }]
    for q in (0.5, 0.8, 0.95, 0.99):
        thresh = q * abs_sum
        n = int((cum < thresh).sum() + 1)
        rows.append({"dataset": dataset, "metric": f"n_genes_for_{int(q*100)}pct_abs",
                     "value": n})
    # also signed: how many genes for 80% of positive signal
    sorted_pos = df.sort_values(f"contrib_{suffix}", ascending=False)
    cum_pos = sorted_pos[f"contrib_{suffix}"].clip(lower=0).cumsum()
    rows.append({"dataset": dataset, "metric": "n_genes_for_80pct_positive",
                 "value": int((cum_pos < 0.8 * s_pos).sum() + 1) if s_pos > 0 else 0})
    sorted_neg = df.sort_values(f"contrib_{suffix}", ascending=True)
    cum_neg = sorted_neg[f"contrib_{suffix}"].clip(upper=0).cumsum()
    rows.append({"dataset": dataset, "metric": "n_genes_for_80pct_negative",
                 "value": int((cum_neg > 0.8 * s_neg).sum() + 1) if s_neg < 0 else 0})
    return rows


def scatter_panel(ax, df: pd.DataFrame, suffix: str, label: str):
    d_col = f"d_{suffix}"
    mean_cpm = df[f"mean_cpm_{suffix}"]
    log_cpm = np.log10(mean_cpm.clip(lower=0.01) + 1)
    sc = ax.scatter(df["weight"], df[d_col], s=4, c=log_cpm,
                    cmap="viridis", alpha=0.6, edgecolor="none")
    ax.axhline(0, color="k", lw=0.4)
    ax.axhline(0.5, color="r", lw=0.4, ls="--")
    ax.axhline(-0.5, color="r", lw=0.4, ls="--")
    ax.set_xlabel("C3+ weight (importance)")
    ax.set_ylabel(f"Cohen's d (child vs adol) — {label}")
    ax.set_title(f"{label}: per-gene d vs C3+ weight")
    plt.colorbar(sc, ax=ax, label="log10(mean CPM + 1)")


def weight_pct_summary(df: pd.DataFrame, suffix: str, label: str) -> list:
    """For genes with |d|>0.5 in this dataset, what is the percentile of
    their C3+ weight (within the C3+ network)?"""
    s = _SUFFIX_MAP[suffix]
    w_pct = df["weight"].rank(pct=True)
    rows = []
    for sign, mask in [("up   (d>+0.5)", df[f"d_{suffix}"] > 0.5),
                       ("down (d<-0.5)", df[f"d_{suffix}"] < -0.5),
                       ("flat (|d|<0.5)", df[f"d_{suffix}"].abs() < 0.5)]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        rows.append({
            "dataset": label,
            "bucket":  sign,
            "n_genes": int(len(sub)),
            "mean_weight":          float(sub["weight"].mean()),
            "median_weight":        float(sub["weight"].median()),
            "mean_weight_pct":      float(w_pct[mask].mean()),
            "median_weight_pct":    float(w_pct[mask].median()),
            "mean_meanCPM_child":   float(sub[f"mc_{s}"].mean()),
            "mean_meanCPM_adol":    float(sub[f"ma_{s}"].mean()),
        })
    return rows


def main():
    print(f"Reading {PER_GENE}")
    j = pd.read_parquet(PER_GENE)
    print(f"  {len(j)} C3+ genes present in both datasets")

    j = add_contribution(j, "psy")
    j = add_contribution(j, "vel")
    j["weight_pct"] = j["weight"].rank(pct=True)
    j.sort_values("abs_contrib_vel", ascending=False).to_parquet(
        OUT_DIR / "f_contribution_per_gene.parquet")

    # ---------- dominance summary ----------
    dom_rows = []
    dom_rows.extend(dominance_rows(j, "PsychAD", "psy"))
    dom_rows.extend(dominance_rows(j, "Velmeshev", "vel"))
    dom = pd.DataFrame(dom_rows)
    dom.to_csv(OUT_DIR / "f_dominance_summary.csv", index=False)
    print("\n=== Dominance summary (how many genes carry the signal) ===")
    print(dom.to_string(index=False))

    # ---------- weight-percentile of high-drop / high-rise genes ----------
    wp_rows = []
    wp_rows.extend(weight_pct_summary(j, "psy", "PsychAD"))
    wp_rows.extend(weight_pct_summary(j, "vel", "Velmeshev"))
    wp = pd.DataFrame(wp_rows)
    wp.to_csv(OUT_DIR / "f_weight_percentile_high_drop.csv", index=False)
    print("\n=== Weight-percentile of high-d genes ===")
    print(wp.to_string(index=False))

    # ---------- top-N gene tables ----------
    top_vel_drop = (j.sort_values("contrib_vel", ascending=False)
                     .head(25)
                     [["ensembl_id", "weight", "weight_pct", "mc_B", "ma_B",
                       "d_psy", "d_vel", "contrib_vel", "contrib_psy"]]
                     .rename(columns={"mc_B": "mc_vel", "ma_B": "ma_vel"}))
    top_vel_drop.to_csv(OUT_DIR / "f_top25_velmeshev_drop_carriers.csv", index=False)
    print("\n=== Top 25 genes carrying the Velmeshev drop ===")
    print(top_vel_drop.to_string(index=False))

    top_psy_neg = (j.sort_values("contrib_psy", ascending=True)
                    .head(15)
                    [["ensembl_id", "weight", "weight_pct", "mc_A", "ma_A",
                      "d_psy", "d_vel", "contrib_psy", "contrib_vel"]]
                    .rename(columns={"mc_A": "mc_psy", "ma_A": "ma_psy"}))
    top_psy_neg.to_csv(OUT_DIR / "f_top15_psychad_negative_drivers.csv", index=False)
    print("\n=== Top 15 genes pulling PsychAD net DOWN (i.e. adol > child) ===")
    print(top_psy_neg.to_string(index=False))

    # ---------- scatter ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    scatter_panel(axes[0], j, "psy", "PsychAD")
    scatter_panel(axes[1], j, "vel", "Velmeshev")
    fig.suptitle(
        "C3+ per-gene child→adolescent Cohen's d vs gene importance weight\n"
        "(red dashed = ±0.5)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "f_scatter_d_vs_weight.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Lorenz curve of cumulative |contribution| ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    for suffix, label, color in [("psy", "PsychAD", "C0"),
                                  ("vel", "Velmeshev", "C3")]:
        sorted_abs = j[f"abs_contrib_{suffix}"].sort_values(ascending=False).values
        cum = np.cumsum(sorted_abs)
        cum /= cum[-1]
        x = np.arange(1, len(cum) + 1) / len(cum)
        ax.plot(x, cum, lw=1.8, label=label, color=color)
    for q in (0.5, 0.8, 0.95):
        ax.axhline(q, ls=":", color="grey", lw=0.5)
    ax.set_xlabel("Top-fraction of genes (sorted by |contribution|)")
    ax.set_ylabel("Cumulative |contribution| / total")
    ax.set_title("Lorenz curve: gene dominance of aggregate signal")
    ax.legend()
    ax.set_xlim(0, 0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "f_lorenz_curve.png", dpi=150)
    plt.close(fig)

    # ---------- distribution histogram of d ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, suffix, label, color in [
        (axes[0], "psy", "PsychAD", "C0"),
        (axes[1], "vel", "Velmeshev", "C3"),
    ]:
        d_col = f"d_{suffix}"
        ax.hist(j[d_col].dropna(), bins=60, color=color, alpha=0.85)
        ax.axvline(0, color="k", lw=0.5)
        ax.axvline(j[d_col].median(), color="orange", lw=1.5,
                   label=f"median = {j[d_col].median():.2f}")
        ax.axvline(j[d_col].mean(), color="red", lw=1.5, ls="--",
                   label=f"mean   = {j[d_col].mean():.2f}")
        ax.set_xlabel(f"per-gene Cohen's d ({label})")
        ax.set_title(f"{label}: distribution of per-gene d")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "f_d_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
