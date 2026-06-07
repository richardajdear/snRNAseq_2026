#!/usr/bin/env python3
"""
P — disentangle depth vs maturity as the driver of the C3+ developmental drop.

User hypothesis: the "depth effect" is in fact a "maturity effect".
Immature cells are small / shallow → they all show the C3+ drop because
they ARE in the developmentally-falling state. PsychAD's FANS prep
preferentially recovers large mature cells → masks the trend. Restricting
PsychAD-V3 to shallow cells indirectly restricts to less-mature cells,
which is why the depth window flips the sign.

Tests:
  P1. Median UMI per (group × marker_annotation subtype). Check that
      ExN_immature < ExN_mature in BOTH cohorts (= biology, not just
      Vel/PsychAD difference).
  P2. Per-cell Spearman(maturity_score, total_umi) within each group.
      Maturity_score = RBFOX3_cp10k_log1p − DCX_cp10k_log1p (same as H).
  P3. Within ExN_mature only, fuzzy d per per-cell-UMI quartile.
      Does the drop reappear at the shallow end of mature cells?
  P4. Within all ExN, fuzzy d per maturity-quintile (cells binned by
      maturity_score, then per-donor mean of those cells' per_cell_c3).
  P5. 2D table: cells per (depth_q, maturity_q) × group, and fuzzy d
      in each non-empty cell.

Produces:
  - extended per-cell cache: m_per_cell_cache_v2.parquet
  - p1_subtype_depth.csv / png
  - p2_maturity_vs_depth.csv
  - p3_mature_depth_q_d.csv / png
  - p4_maturity_q_d.csv / png
  - p5_depth_x_maturity_d.csv / png

Run via:
    cd /home/rajd2/rds/hpc-work/snRNAseq_2026
    sbatch --time=01:00:00 --mem=200G \
       scripts/run_script.sh scripts/grn_dev_diagnostics/p_depth_vs_maturity.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (OUT_DIR, fuzzy_d_from_donor_scores, build_c3plus_table,
                   FUZZY_BOUNDARIES, AGE_LO, AGE_HI)

GROUPS = ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]
GROUP_TO_DATASET = {
    "PsychAD-V3":   ("PsychAD",   "V3"),
    "Velmeshev-V2": ("Velmeshev", "V2"),
    "Velmeshev-V3": ("Velmeshev", "V3"),
}
INPUTS = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/scvi_output/integrated.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad",
}
MANUAL = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
}
MARKERS = {
    "RBFOX3": "ENSG00000167281",
    "DCX":    "ENSG00000077279",
    "RBFOX1": "ENSG00000078328",
}

CACHE_V2 = OUT_DIR / "m_per_cell_cache_v2.parquet"
EXCLUDE_DONORS = {"Donor_1400"}
MIN_CELLS = 5
COLORS = {"PsychAD-V3": "#C0392B", "Velmeshev-V2": "#27AE60",
           "Velmeshev-V3": "#2980B9"}


# ---------------------------------------------------------------------------
# Build extended cache (marker UMIs + maturity + per_cell_c3 + layer)
# ---------------------------------------------------------------------------

def build_cache():
    if CACHE_V2.exists():
        print(f"Using cached: {CACHE_V2}", flush=True)
        return pd.read_parquet(CACHE_V2)

    weights = build_c3plus_table().set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes", flush=True)

    frames = []
    for g in GROUPS:
        dataset, chem = GROUP_TO_DATASET[g]
        print(f"\n=== {g} ({dataset}, {chem}) — loading integrated h5ad",
              flush=True)
        a = ad.read_h5ad(INPUTS[dataset], backed="r")
        obs = a.obs.copy()
        if "chemistry" not in obs.columns and "source-chemistry" in obs.columns:
            obs["chemistry"] = obs["source-chemistry"].astype(str).str.extract(
                r"(V2|V3)")[0].fillna("unknown")
        obs["chemistry"] = obs["chemistry"].astype(str)
        ma = pd.read_parquet(MANUAL[dataset])
        obs = obs.join(ma, how="left")
        age = pd.to_numeric(obs["age_years"], errors="coerce")
        mask = ((age >= AGE_LO) & (age < AGE_HI)
                & obs["marker_annotation"].isin(
                    ["ExN_mature", "ExN_immature", "ExN_weak"])
                & (obs["chemistry"].values == chem))
        obs_idx = np.where(mask.values)[0]
        print(f"  ExN cells: {len(obs_idx):,}", flush=True)
        counts = sp.csr_matrix(a.layers["counts"])[obs_idx, :]
        obs = obs.iloc[obs_idx].copy().reset_index(drop=True)
        obs["age_years"] = age.values[obs_idx]
        obs["individual"] = obs.get("individual", obs.get("donor_id"))
        obs["total_umi"] = np.asarray(counts.sum(axis=1)).ravel()
        var_names = a.var_names.values
        var_pos = {v: i for i, v in enumerate(var_names)}

        # marker UMIs
        for mname, ens in MARKERS.items():
            if ens in var_pos:
                obs[mname] = np.asarray(counts[:, var_pos[ens]].todense()).ravel()
            else:
                obs[mname] = 0

        # maturity score (same definition as H)
        total = obs["total_umi"].values.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rb_cp = np.where(total > 0, obs["RBFOX3"] / total * 1e4, 0.0)
            dc_cp = np.where(total > 0, obs["DCX"]    / total * 1e4, 0.0)
        obs["RBFOX3_log1p_cp10k"] = np.log1p(rb_cp)
        obs["DCX_log1p_cp10k"]    = np.log1p(dc_cp)
        obs["maturity_score"]     = obs["RBFOX3_log1p_cp10k"] - obs["DCX_log1p_cp10k"]

        # per-cell C3+
        present = [g_ for g_ in weights.index if g_ in var_pos]
        grn_cols = np.array([var_pos[g_] for g_ in present], dtype=np.int64)
        grn_w = np.array([weights[g_] for g_ in present], dtype=np.float64)
        raw_dot = np.asarray(counts[:, grn_cols] @ grn_w).ravel()
        obs["per_cell_c3"] = np.where(total > 0, raw_dot / total * 1e6, 0.0)

        keep_cols = ["individual", "age_years", "chemistry", "total_umi",
                     "marker_annotation",
                     "RBFOX3", "DCX", "RBFOX1",
                     "RBFOX3_log1p_cp10k", "DCX_log1p_cp10k", "maturity_score",
                     "per_cell_c3"]
        out = obs[keep_cols].copy()
        out["group"] = g
        frames.append(out)
        print(f"  done.", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(CACHE_V2)
    print(f"\nSaved extended cache: {CACHE_V2} ({len(df):,} rows)", flush=True)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fuzzy_d_from_per_cell(per_cell, score_col="per_cell_c3"):
    don = (per_cell.groupby("individual", observed=True)
              .agg(score=(score_col, "mean"),
                   n_cells=(score_col, "size"),
                   age_years=("age_years", "first"))
              .reset_index())
    don = don[don["n_cells"] >= MIN_CELLS]
    don = don.dropna(subset=["score", "age_years"])
    don = don[(don["age_years"] >= AGE_LO) & (don["age_years"] < AGE_HI)]
    if len(don) < 4:
        return np.nan, 0
    r = fuzzy_d_from_donor_scores(don["age_years"].values,
                                    don["score"].values)
    return r["mean_d"], len(don)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def p1_subtype_depth(df):
    print("\n" + "="*72)
    print("P1 — median UMI per (group × marker_annotation subtype)")
    print("="*72)
    rows = []
    for g in GROUPS:
        sub = df[df["group"] == g]
        for sty in ["ExN_mature", "ExN_immature", "ExN_weak"]:
            ss = sub[sub["marker_annotation"] == sty]
            if len(ss) == 0: continue
            rows.append({"group": g, "subtype": sty,
                          "n_cells": int(len(ss)),
                          "median_umi": float(ss["total_umi"].median()),
                          "mean_umi":   float(ss["total_umi"].mean()),
                          "p25_umi":    float(ss["total_umi"].quantile(0.25)),
                          "p75_umi":    float(ss["total_umi"].quantile(0.75))})
    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT_DIR / "p1_subtype_depth.csv", index=False)
    print(pdf.to_string(index=False))

    # plot: violins (log UMI) per group × subtype
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, g in zip(axes, GROUPS):
        sub = df[df["group"] == g]
        data = [np.log10(sub[sub["marker_annotation"] == s]["total_umi"]
                         .clip(lower=1))
                for s in ["ExN_mature", "ExN_immature", "ExN_weak"]]
        parts = ax.violinplot(data, showmedians=True, widths=0.7)
        for pc in parts['bodies']: pc.set_facecolor(COLORS[g]); pc.set_alpha(0.5)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["ExN_mature", "ExN_immature", "ExN_weak"], rotation=15)
        ax.set_title(g, color=COLORS[g])
        if g == GROUPS[0]: ax.set_ylabel("log10(per-cell UMI)")
        # human-readable y-ticks
        ax.set_yticks(np.log10([300, 1000, 3000, 10000, 30000, 100000]))
        ax.set_yticklabels(["300", "1k", "3k", "10k", "30k", "100k"])
    fig.suptitle("P1: UMI distribution by ExN subtype × group "
                  "(ExN_immature systematically shallower)",
                  fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p1_subtype_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved p1_subtype_depth.png")


def p2_maturity_vs_depth(df):
    print("\n" + "="*72)
    print("P2 — per-cell Spearman(maturity_score, total_umi) per group")
    print("="*72)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, g in zip(axes, GROUPS):
        sub = df[df["group"] == g]
        r = stats.spearmanr(sub["maturity_score"], sub["total_umi"]).statistic
        rows.append({"group": g, "n_cells": int(len(sub)),
                      "spearman_maturity_vs_umi": float(r)})
        # sample to 10k for plotting
        s = sub.sample(n=min(10000, len(sub)), random_state=0)
        ax.hexbin(np.log10(s["total_umi"].clip(lower=1)),
                   s["maturity_score"],
                   gridsize=40, cmap="viridis",
                   mincnt=1, bins="log")
        ax.set_xticks(np.log10([300, 1000, 3000, 10000, 30000, 100000]))
        ax.set_xticklabels(["300", "1k", "3k", "10k", "30k", "100k"])
        ax.set_xlabel("log10(per-cell UMI)")
        ax.set_title(f"{g}\n rho = {r:+.3f}", color=COLORS[g])
        if g == GROUPS[0]: ax.set_ylabel("maturity_score (RBFOX3 − DCX)")
    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT_DIR / "p2_maturity_vs_depth.csv", index=False)
    print(pdf.to_string(index=False))
    fig.suptitle("P2: per-cell maturity vs depth — strong positive within each group?",
                  fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p2_maturity_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved p2_maturity_vs_depth.png")


def p3_mature_depth_q(df):
    print("\n" + "="*72)
    print("P3 — within ExN_mature, fuzzy d per per-cell-UMI quartile")
    print("="*72)
    rows = []
    for g in GROUPS:
        sub = df[(df["group"] == g)
                  & (df["marker_annotation"] == "ExN_mature")].copy()
        if len(sub) < 100: continue
        sub["depth_q"] = pd.qcut(sub["total_umi"], 4, labels=False,
                                   duplicates="drop")
        for q in sorted(sub["depth_q"].dropna().unique()):
            s2 = sub[sub["depth_q"] == q]
            d, n = fuzzy_d_from_per_cell(s2)
            rows.append({"group": g, "depth_q": int(q),
                          "n_cells": int(len(s2)),
                          "median_umi": float(s2["total_umi"].median()),
                          "n_donors": n, "fuzzy_d": d})
    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT_DIR / "p3_mature_depth_q_d.csv", index=False)
    print(pdf.to_string(index=False))
    pv = pdf.pivot_table(index="depth_q", columns="group", values="fuzzy_d").round(3)
    print("\nfuzzy d pivot:")
    print(pv.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(4); bar_w = 0.27
    for i, g in enumerate(GROUPS):
        vals = [pv.loc[q, g] if (q in pv.index and g in pv.columns) else np.nan
                for q in range(4)]
        ax.bar(x + (i-1)*bar_w, vals, bar_w, color=COLORS[g], label=g)
        for j_, v in enumerate(vals):
            if np.isnan(v): continue
            ax.text(j_ + (i-1)*bar_w, v + (0.04 if v >= 0 else -0.06),
                     f"{v:+.2f}", ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontsize=8, fontweight="bold", color=COLORS[g])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q0 shallow", "Q1", "Q2", "Q3 deep"])
    ax.set_ylabel("Cohen's d (fuzzy boundary)")
    ax.set_title("P3: within ExN_mature, fuzzy d by per-cell UMI quartile\n"
                  "(if hypothesis: shallow Q0/Q1 mature cells show drop; "
                  "deep Q3 do not)")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p3_mature_depth_q_d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved p3_mature_depth_q_d.png")


def p4_maturity_q(df):
    print("\n" + "="*72)
    print("P4 — within all ExN, fuzzy d per maturity-quintile (per group)")
    print("="*72)
    rows = []
    for g in GROUPS:
        sub = df[df["group"] == g].copy()
        sub["mat_q"] = pd.qcut(sub["maturity_score"], 5,
                                labels=False, duplicates="drop")
        for q in sorted(sub["mat_q"].dropna().unique()):
            s2 = sub[sub["mat_q"] == q]
            d, n = fuzzy_d_from_per_cell(s2)
            rows.append({"group": g, "mat_q": int(q),
                          "n_cells": int(len(s2)),
                          "median_maturity": float(s2["maturity_score"].median()),
                          "median_umi": float(s2["total_umi"].median()),
                          "n_donors": n, "fuzzy_d": d})
    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT_DIR / "p4_maturity_q_d.csv", index=False)
    print(pdf.to_string(index=False))
    pv = pdf.pivot_table(index="mat_q", columns="group", values="fuzzy_d").round(3)
    print("\nfuzzy d pivot:")
    print(pv.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(5); bar_w = 0.27
    for i, g in enumerate(GROUPS):
        vals = [pv.loc[q, g] if (q in pv.index and g in pv.columns) else np.nan
                for q in range(5)]
        ax.bar(x + (i-1)*bar_w, vals, bar_w, color=COLORS[g], label=g)
        for j_, v in enumerate(vals):
            if np.isnan(v): continue
            ax.text(j_ + (i-1)*bar_w, v + (0.04 if v >= 0 else -0.06),
                     f"{v:+.2f}", ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontsize=7.5, fontweight="bold", color=COLORS[g])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q0\nmost immature", "Q1", "Q2", "Q3", "Q4\nmost mature"])
    ax.set_ylabel("Cohen's d (fuzzy boundary)")
    ax.set_title("P4: fuzzy d by maturity-score quintile\n"
                  "(if hypothesis: low-maturity Q0/Q1 show drop in ALL cohorts; "
                  "high-maturity Q4 flat/anti in PsychAD)")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p4_maturity_q_d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved p4_maturity_q_d.png")


def p5_depth_x_maturity(df):
    print("\n" + "="*72)
    print("P5 — 2D fuzzy d by (depth_q × maturity_q), per group")
    print("="*72)
    all_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, g in zip(axes, GROUPS):
        sub = df[df["group"] == g].copy()
        sub["depth_q"] = pd.qcut(sub["total_umi"], 4, labels=False,
                                   duplicates="drop")
        sub["mat_q"]   = pd.qcut(sub["maturity_score"], 4, labels=False,
                                   duplicates="drop")
        rows = []
        for dq in range(4):
            for mq in range(4):
                s2 = sub[(sub["depth_q"] == dq) & (sub["mat_q"] == mq)]
                if len(s2) < 20:
                    rows.append({"depth_q": dq, "mat_q": mq, "n_cells": int(len(s2)),
                                  "n_donors": 0, "fuzzy_d": np.nan})
                    continue
                d, n = fuzzy_d_from_per_cell(s2)
                rows.append({"depth_q": dq, "mat_q": mq, "n_cells": int(len(s2)),
                              "n_donors": n, "fuzzy_d": d})
        rdf = pd.DataFrame(rows)
        rdf["group"] = g; all_rows.append(rdf)
        mat = rdf.pivot(index="depth_q", columns="mat_q", values="fuzzy_d")
        nmat = rdf.pivot(index="depth_q", columns="mat_q", values="n_donors")
        im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        for i_ in range(mat.shape[0]):
            for j_ in range(mat.shape[1]):
                v = mat.values[i_, j_]
                n = nmat.values[i_, j_]
                if np.isnan(v): continue
                ax.text(j_, i_, f"{v:+.2f}\nn={n}",
                         ha="center", va="center",
                         color="white" if abs(v) > 1 else "k",
                         fontsize=7.5)
        ax.set_xticks([0,1,2,3])
        ax.set_xticklabels(["Q0\nimm","Q1","Q2","Q3\nmature"])
        ax.set_yticks([0,1,2,3])
        ax.set_yticklabels(["Q0 shallow","Q1","Q2","Q3 deep"])
        ax.set_title(g, color=COLORS[g])
        plt.colorbar(im, ax=ax, label="fuzzy d")
    fig.suptitle("P5: fuzzy d by (depth_q × maturity_q). "
                  "Reading: where IS the drop? Is it carried by low-maturity or shallow cells?",
                  y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "p5_depth_x_maturity_d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved p5_depth_x_maturity_d.png")
    pd.concat(all_rows, ignore_index=True).to_csv(
        OUT_DIR / "p5_depth_x_maturity_d.csv", index=False)


def main():
    df = build_cache()
    # apply donor exclusion
    df = df[~df["individual"].isin(EXCLUDE_DONORS)].reset_index(drop=True)

    p1_subtype_depth(df)
    p2_maturity_vs_depth(df)
    p3_mature_depth_q(df)
    p4_maturity_q(df)
    p5_depth_x_maturity(df)
    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
