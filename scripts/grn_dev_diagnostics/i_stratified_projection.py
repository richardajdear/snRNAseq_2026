#!/usr/bin/env python3
"""
I — re-project the AHBA C3+ GRN with PER-CELL stratification.

We have shown (G/H) that the marker-based classifier confounds
sequencing depth with biological maturity. Now we redo the C3+
projection using per-cell stratification by:

  (1) continuous depth-normalised MATURITY QUARTILE
      score = log1p(RBFOX3 / total_UMI × 1e4) − log1p(DCX / total_UMI × 1e4)
      (computed within each dataset to avoid depth confounding the quartile)
  (2) per-cell DEPTH (UMI) QUARTILE
      (computed within each dataset)
  (3) joint maturity × depth × age stratification, per donor

For each stratum we pseudobulk (sum raw counts within donor × stratum),
CPM-normalise, project the AHBA C3+ weighted score, and compute child
vs adolescent Cohen's d. This both:
  - cleans the classifier artifact (use continuous quartile rather than
    threshold bin)
  - controls for depth (compare matched quartiles across datasets)

Also re-runs everything WITHOUT Donor_1400 to bound the outlier
contribution to the PsychAD child mean.

Submit via:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=01:00:00 --mem=160G \
    scripts/run_script.sh scripts/grn_dev_diagnostics/i_stratified_projection.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (
    cpm_from_counts, build_c3plus_table, cohens_d, OUT_DIR,
    CHILD, ADOL,
)

INPUTS = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/scvi_output/integrated.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad",
}
MANUAL = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
}

EXCLUDE_DONORS = {"Donor_1400"}   # for the F3 outlier sensitivity check

# Marker gene Ensembl IDs (present in both datasets)
RBFOX3 = "ENSG00000167281"
DCX    = "ENSG00000077279"


def load_dataset(name: str):
    """Return (counts CSR, obs DataFrame, var_names, marker col idx).

    Restricted to ExN cells (marker_annotation in {ExN_mature, ExN_immature,
    ExN_weak}) in the developmental window (age 1-25 y, PFC).
    """
    print(f"\n=== {name}: loading", flush=True)
    a = ad.read_h5ad(INPUTS[name], backed="r")
    print(f"  shape: {a.shape}", flush=True)

    obs = a.obs.copy()
    # chemistry
    if "chemistry" not in obs.columns and "source-chemistry" in obs.columns:
        obs["chemistry"] = obs["source-chemistry"].astype(str).str.extract(
            r"(V2|V3)")[0].fillna("unknown")
    # join marker annotation (per-cell parquet)
    ma = pd.read_parquet(MANUAL[name])
    obs = obs.join(ma, how="left")
    # filter
    age = pd.to_numeric(obs["age_years"], errors="coerce")
    mask = (age >= CHILD[0]) & (age < ADOL[1]) & \
           obs["marker_annotation"].isin(["ExN_mature", "ExN_immature", "ExN_weak"])
    obs_idx = np.where(mask.values)[0]
    print(f"  ExN cells in age window: {len(obs_idx):,}", flush=True)

    var_names = a.var_names.values
    col_rbfox3 = int(np.where(var_names == RBFOX3)[0][0])
    col_dcx    = int(np.where(var_names == DCX)[0][0])

    print("  loading counts CSR ...", flush=True)
    counts = a.layers["counts"]
    counts = sp.csr_matrix(counts)
    counts = counts[obs_idx, :]
    print(f"  counts shape: {counts.shape}, nnz: {counts.nnz:,}", flush=True)

    obs = obs.iloc[obs_idx].copy()
    obs["age_years"] = age.values[obs_idx]
    obs["chemistry"] = obs["chemistry"].astype(str)
    obs["stage"] = np.where(obs["age_years"] < CHILD[1], "child", "adol")
    obs["individual"] = obs.get("individual", obs.get("donor_id"))
    # per-cell totals + markers (raw counts)
    print("  computing per-cell totals + marker counts ...", flush=True)
    obs["total_umi"]   = np.asarray(counts.sum(axis=1)).ravel()
    obs["RBFOX3_raw"]  = np.asarray(counts[:, col_rbfox3].toarray()).ravel()
    obs["DCX_raw"]     = np.asarray(counts[:, col_dcx].toarray()).ravel()
    # continuous maturity score (depth-normalised)
    rb = np.log1p(np.where(obs["total_umi"] > 0,
                            obs["RBFOX3_raw"] / obs["total_umi"] * 1e4, 0))
    dc = np.log1p(np.where(obs["total_umi"] > 0,
                            obs["DCX_raw"] / obs["total_umi"] * 1e4, 0))
    obs["maturity_score"] = rb - dc
    print(f"  maturity score: min={obs['maturity_score'].min():.2f} "
          f"max={obs['maturity_score'].max():.2f} "
          f"mean={obs['maturity_score'].mean():.2f}", flush=True)
    return counts, obs.reset_index(drop=True), var_names


def quartile_within(series: pd.Series, n=4) -> pd.Series:
    """Compute integer quartile bin (0..n-1) within the given series."""
    try:
        return pd.qcut(series, n, labels=False, duplicates="drop")
    except Exception:
        # if uniformly zero etc.
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)


def aggregate_score(counts: sp.csr_matrix, obs: pd.DataFrame,
                     var_names: np.ndarray, weights: pd.Series,
                     stratum_cols: list) -> pd.DataFrame:
    """For each (individual, *stratum_cols) group, sum raw counts across
    cells, CPM-normalise, and project the GRN weighted score.

    Returns DataFrame with (individual, stratum_cols, n_cells, age_years,
    stage, score)."""
    # build C3+ column mask
    var_pos = {v: i for i, v in enumerate(var_names)}
    grn_cols = np.array([var_pos[g] for g in weights.index if g in var_pos],
                        dtype=np.int64)
    grn_weights = np.array([weights[g] for g in weights.index if g in var_pos],
                            dtype=np.float64)
    print(f"  GRN columns matched: {len(grn_cols)} / {len(weights)}", flush=True)

    groupby_cols = ["individual"] + stratum_cols
    # GroupBy keys
    keys = obs[groupby_cols].astype(str).agg("|".join, axis=1)
    uniq = keys.unique()
    print(f"  pseudobulk groups: {len(uniq):,}", flush=True)

    rows = []
    for grp_key in uniq:
        mask = (keys == grp_key).values
        idx = np.where(mask)[0]
        if len(idx) < 5:
            continue
        sub_counts = counts[idx, :]
        # per-cell totals (already known)
        total_per_cell = obs["total_umi"].values[idx]
        # bulk: sum then CPM
        bulk = np.asarray(sub_counts.sum(axis=0)).ravel().astype(np.float64)
        tot  = bulk.sum()
        if tot <= 0:
            continue
        cpm = bulk * (1e6 / tot)
        # project only on the GRN columns
        score = float(np.dot(cpm[grn_cols], grn_weights))
        obs_row = obs.iloc[idx[0]]
        row = {"n_cells": int(len(idx)), "score": score,
               "age_years": float(obs_row["age_years"]),
               "stage": obs_row["stage"],
               "chemistry": obs_row["chemistry"],
               "individual": obs_row["individual"]}
        for c in stratum_cols:
            row[c] = obs_row[c]
        rows.append(row)
    return pd.DataFrame(rows)


def child_vs_adol_d(scores: pd.DataFrame, stratum_cols: list,
                     extra_filter: pd.Series = None) -> pd.DataFrame:
    """Donor-level Cohen's d per stratum, child vs adol."""
    if extra_filter is not None:
        scores = scores[extra_filter.reindex(scores.index, fill_value=True)]
    rows = []
    for keys, sub in scores.groupby(stratum_cols, observed=True):
        c = sub[sub["stage"] == "child"]["score"].values
        a = sub[sub["stage"] == "adol"]["score"].values
        if len(c) < 2 or len(a) < 2:
            continue
        d = cohens_d(c, a)
        row = dict(zip(stratum_cols if isinstance(stratum_cols, list) else [stratum_cols],
                       keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "n_child": len(c),
            "n_adol": len(a),
            "mean_child": float(np.mean(c)),
            "mean_adol":  float(np.mean(a)),
            "cohens_d":   float(d),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    weights = build_c3plus_table().set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes", flush=True)

    all_donor_scores = []
    all_strat = {}
    for name in ["PsychAD", "Velmeshev"]:
        counts, obs, var_names = load_dataset(name)
        # quartiles WITHIN dataset (so each dataset has its own depth/maturity range)
        obs["maturity_q"] = quartile_within(obs["maturity_score"], 4)
        obs["depth_q"]    = quartile_within(obs["total_umi"], 4)
        obs["dataset"] = name
        print(f"\n  --- {name} per-cell strata ---", flush=True)
        print(f"  maturity_q range medians:")
        print(obs.groupby("maturity_q")["maturity_score"].agg(["min", "median", "max", "size"]).to_string())
        print(f"  depth_q range medians:")
        print(obs.groupby("depth_q")["total_umi"].agg(["min", "median", "max", "size"]).to_string())

        # ----- A. per-donor whole-ExN aggregate (sanity, matches grn_dev_multi) -----
        donor_scores = aggregate_score(counts, obs, var_names, weights, [])
        donor_scores["dataset"] = name
        all_donor_scores.append(donor_scores)
        print(f"\n  {name}: aggregate over all ExN cells per donor")
        d_all = child_vs_adol_d(donor_scores, ["dataset"])
        print(d_all.to_string(index=False))

        # ----- B. per-donor × maturity quartile -----
        agg_m = aggregate_score(counts, obs, var_names, weights, ["maturity_q"])
        agg_m["dataset"] = name
        all_strat.setdefault("by_maturity_q", []).append(agg_m)

        # ----- C. per-donor × depth quartile -----
        agg_d = aggregate_score(counts, obs, var_names, weights, ["depth_q"])
        agg_d["dataset"] = name
        all_strat.setdefault("by_depth_q", []).append(agg_d)

        # ----- D. per-donor × marker_annotation × depth_q (3-way E-style)
        agg_mc = aggregate_score(counts, obs, var_names, weights,
                                  ["marker_annotation", "depth_q"])
        agg_mc["dataset"] = name
        all_strat.setdefault("by_anno_x_depth", []).append(agg_mc)

    # ===================== save per-donor aggregates ============================
    donor_scores_df = pd.concat(all_donor_scores, ignore_index=True)
    donor_scores_df.to_csv(OUT_DIR / "i_donor_scores_full.csv", index=False)

    # ===================== I1: maturity-quartile stratification =================
    mat = pd.concat(all_strat["by_maturity_q"], ignore_index=True)
    mat.to_csv(OUT_DIR / "i1_donor_scores_by_maturity_q.csv", index=False)
    print("\n========== I1: child-vs-adol d, stratified by MATURITY quartile ==========")
    d_mat = child_vs_adol_d(mat, ["dataset", "maturity_q"])
    d_mat.to_csv(OUT_DIR / "i1_d_by_maturity_q.csv", index=False)
    print(d_mat.to_string(index=False))

    # ===================== I2: depth-quartile stratification ====================
    depth = pd.concat(all_strat["by_depth_q"], ignore_index=True)
    depth.to_csv(OUT_DIR / "i2_donor_scores_by_depth_q.csv", index=False)
    print("\n========== I2: child-vs-adol d, stratified by DEPTH quartile ==========")
    d_depth = child_vs_adol_d(depth, ["dataset", "depth_q"])
    d_depth.to_csv(OUT_DIR / "i2_d_by_depth_q.csv", index=False)
    print(d_depth.to_string(index=False))

    # ===================== I3: anno × depth (three-way) =========================
    md = pd.concat(all_strat["by_anno_x_depth"], ignore_index=True)
    md.to_csv(OUT_DIR / "i3_donor_scores_anno_x_depth.csv", index=False)
    print("\n========== I3: child-vs-adol d, by marker_annotation × depth_q ==========")
    d_md = child_vs_adol_d(md, ["dataset", "marker_annotation", "depth_q"])
    d_md.to_csv(OUT_DIR / "i3_d_anno_x_depth.csv", index=False)
    print(d_md.to_string(index=False))

    # ===================== I4: Donor_1400 sensitivity ===========================
    print("\n========== I4: drop Donor_1400 sensitivity check ==========")
    mask = ~donor_scores_df["individual"].isin(EXCLUDE_DONORS)
    d_full   = child_vs_adol_d(donor_scores_df, ["dataset"])
    d_nodrop = child_vs_adol_d(donor_scores_df[mask], ["dataset"])
    d_nodrop_mat = child_vs_adol_d(mat[~mat["individual"].isin(EXCLUDE_DONORS)],
                                     ["dataset", "maturity_q"])
    d_nodrop_dep = child_vs_adol_d(depth[~depth["individual"].isin(EXCLUDE_DONORS)],
                                     ["dataset", "depth_q"])
    print("== full ==");           print(d_full.to_string(index=False))
    print("\n== no Donor_1400 =="); print(d_nodrop.to_string(index=False))
    print("\n== no Donor_1400 × maturity_q =="); print(d_nodrop_mat.to_string(index=False))
    print("\n== no Donor_1400 × depth_q =="); print(d_nodrop_dep.to_string(index=False))
    d_nodrop.to_csv(OUT_DIR / "i4_d_full_nodrop.csv", index=False)
    d_nodrop_mat.to_csv(OUT_DIR / "i4_d_by_maturity_q_nodrop.csv", index=False)
    d_nodrop_dep.to_csv(OUT_DIR / "i4_d_by_depth_q_nodrop.csv", index=False)

    # ===================== I5: 3-way average score per stratum =================
    # Per (dataset, maturity_q, depth_q, stage) summary
    md_dep = pd.concat([
        pd.concat(all_strat["by_maturity_q"], ignore_index=True).assign(_kind="m"),
        pd.concat(all_strat["by_depth_q"],    ignore_index=True).assign(_kind="d"),
    ])

    # ===================== plots =================================================
    # I1 plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, ds in zip(axes, ["PsychAD", "Velmeshev"]):
        sub = mat[mat["dataset"] == ds].copy()
        sub["maturity_q"] = sub["maturity_q"].astype(int)
        for stage, color in [("child", "C3"), ("adol", "C0")]:
            ss = sub[sub["stage"] == stage]
            ax.scatter(ss["maturity_q"] + (0.1 if stage == "child" else -0.1),
                        ss["score"], color=color, alpha=0.5, s=25,
                        label=f"{stage} (n={len(ss)})")
        d_sub = d_mat[d_mat["dataset"] == ds]
        title = f"{ds}\nd by maturity quartile: " + ", ".join(
            f"Q{int(q)}={dd:+.2f}" for q, dd in zip(d_sub["maturity_q"], d_sub["cohens_d"]))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("maturity quartile (Q0 = least mature)")
        ax.set_ylabel("C3+ score (donor × quartile)")
        ax.legend()
    fig.suptitle("I1: C3+ score by maturity quartile (continuous, depth-normalised)", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "i1_by_maturity_q.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # I2 plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, ds in zip(axes, ["PsychAD", "Velmeshev"]):
        sub = depth[depth["dataset"] == ds].copy()
        sub["depth_q"] = sub["depth_q"].astype(int)
        for stage, color in [("child", "C3"), ("adol", "C0")]:
            ss = sub[sub["stage"] == stage]
            ax.scatter(ss["depth_q"] + (0.1 if stage == "child" else -0.1),
                        ss["score"], color=color, alpha=0.5, s=25,
                        label=f"{stage} (n={len(ss)})")
        d_sub = d_depth[d_depth["dataset"] == ds]
        title = f"{ds}\nd by depth quartile: " + ", ".join(
            f"Q{int(q)}={dd:+.2f}" for q, dd in zip(d_sub["depth_q"], d_sub["cohens_d"]))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("depth (UMI) quartile (Q0 = shallowest)")
        ax.set_ylabel("C3+ score")
        ax.legend()
    fig.suptitle("I2: C3+ score by per-cell depth quartile", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "i2_by_depth_q.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
