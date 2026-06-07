#!/usr/bin/env python3
"""
J — confirmation experiments for the depth-confound diagnosis.

User-requested analyses:
  (1) Per-cell-CPM averaging: instead of sum-then-CPM, compute per-cell
      CPM, then take the donor-level MEAN across cells (equal weight
      per cell regardless of UMI depth).
  (2) Downsampling to matched depth: for each per-cell UMI cap c in a
      range, multinomial-thin per-cell raw counts to c reads (cells
      with N < c untouched), then sum-then-CPM and project. Report the
      child-vs-adol d AND the % of cells in each donor×age group that
      hit the cap (lost data).

For both, results split by 3 groups: PsychAD-V3, Velmeshev-V2,
Velmeshev-V3.

Submit via:
    cd /home/rajd2/rds/hpc-work/snRNAseq_2026
    sbatch --time=01:00:00 --mem=160G \
      scripts/run_script.sh scripts/grn_dev_diagnostics/j_normalization_fixes.py
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
    build_c3plus_table, cohens_d, OUT_DIR, CHILD, ADOL,
)

INPUTS = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/scvi_output/integrated.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad",
}
MANUAL = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
}

CAPS = [500, 1000, 2000, 3000, 5000, 8000]
SEED = 42


def load_for_group(name: str, chemistry_filter=None):
    """Return (counts_csr_cells_x_genes, obs DataFrame, var_names np.ndarray).

    Restricted to ExN cells in [1, 25) y, PFC, optionally filtered by
    chemistry='V2'|'V3'.
    """
    print(f"\n=== {name}{(' chem='+chemistry_filter) if chemistry_filter else ''}: loading", flush=True)
    a = ad.read_h5ad(INPUTS[name], backed="r")
    print(f"  shape: {a.shape}", flush=True)
    obs = a.obs.copy()
    if "chemistry" not in obs.columns and "source-chemistry" in obs.columns:
        obs["chemistry"] = obs["source-chemistry"].astype(str).str.extract(
            r"(V2|V3)")[0].fillna("unknown")
    obs["chemistry"] = obs["chemistry"].astype(str)
    ma = pd.read_parquet(MANUAL[name])
    obs = obs.join(ma, how="left")
    age = pd.to_numeric(obs["age_years"], errors="coerce")
    mask = (age >= CHILD[0]) & (age < ADOL[1]) & \
           obs["marker_annotation"].isin(["ExN_mature", "ExN_immature", "ExN_weak"])
    if chemistry_filter is not None:
        mask &= (obs["chemistry"].values == chemistry_filter)
    obs_idx = np.where(mask.values)[0]
    print(f"  ExN cells in age window (chemistry={chemistry_filter or 'any'}): {len(obs_idx):,}", flush=True)

    print("  loading counts CSR...", flush=True)
    counts = sp.csr_matrix(a.layers["counts"])
    counts = counts[obs_idx, :]
    obs = obs.iloc[obs_idx].copy()
    obs["age_years"] = age.values[obs_idx]
    obs["individual"] = obs.get("individual", obs.get("donor_id"))
    obs["stage"] = np.where(obs["age_years"] < CHILD[1], "child", "adol")
    obs["total_umi"] = np.asarray(counts.sum(axis=1)).ravel()
    print(f"  counts shape: {counts.shape}, nnz: {counts.nnz:,}, "
          f"total UMI median: {np.median(obs['total_umi']):.0f}", flush=True)
    return counts, obs.reset_index(drop=True), a.var_names.values


def per_cell_score(counts: sp.csr_matrix, total_per_cell: np.ndarray,
                    grn_cols: np.ndarray, grn_weights: np.ndarray) -> np.ndarray:
    """Per-cell C3+ score = (count_ig . w_g) / total_i × 1e6."""
    sub = counts[:, grn_cols]
    raw_dot = sub @ grn_weights.astype(np.float64)
    # raw_dot shape (n_cells,)
    raw_dot = np.asarray(raw_dot).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        cpm_score = np.where(total_per_cell > 0,
                              raw_dot / total_per_cell * 1e6, 0.0)
    return cpm_score


def aggregate_donor_mean(per_cell_c3: np.ndarray, obs: pd.DataFrame) -> pd.DataFrame:
    """J1: per-donor MEAN of per-cell C3+ scores."""
    df = pd.DataFrame({
        "individual": obs["individual"].values,
        "score": per_cell_c3,
        "age_years": obs["age_years"].values,
        "stage": obs["stage"].values,
    })
    agg = (df.groupby("individual", observed=True)
              .agg(n_cells=("score", "size"),
                   score=("score", "mean"),
                   age_years=("age_years", "first"),
                   stage=("stage", "first"))
              .reset_index())
    return agg


def aggregate_sum_then_cpm(counts: sp.csr_matrix, obs: pd.DataFrame,
                            grn_cols: np.ndarray, grn_weights: np.ndarray) -> pd.DataFrame:
    """Original method: sum raw counts per donor, CPM, then project."""
    rows = []
    for donor, sub in obs.groupby("individual", observed=True):
        idx = sub.index.values
        sub_counts = counts[idx, :]
        bulk = np.asarray(sub_counts.sum(axis=0)).ravel().astype(np.float64)
        tot = bulk.sum()
        if tot <= 0:
            continue
        cpm = bulk * (1e6 / tot)
        score = float(np.dot(cpm[grn_cols], grn_weights))
        row0 = sub.iloc[0]
        rows.append({"individual": donor, "n_cells": int(len(idx)),
                     "score": score, "age_years": float(row0["age_years"]),
                     "stage": row0["stage"]})
    return pd.DataFrame(rows)


def downsample_csr(counts: sp.csr_matrix, total_per_cell: np.ndarray,
                    cap: int, rng: np.random.Generator) -> tuple:
    """Multinomial-thin per-cell raw counts to cap reads.
    Returns (downsampled_counts, hit_cap_mask).
    For cells with N <= cap, untouched. For cells with N > cap, each
    nonzero count k for that cell is thinned via binomial(k, cap/N).
    """
    counts = counts.tocsr()
    hit = total_per_cell > cap
    if not hit.any():
        return counts.copy(), hit
    new_data = counts.data.copy().astype(np.int64)
    indptr = counts.indptr
    for i in np.where(hit)[0]:
        s = int(indptr[i]); e = int(indptr[i + 1])
        if e <= s: continue
        p = cap / total_per_cell[i]
        new_data[s:e] = rng.binomial(counts.data[s:e].astype(np.int64), p)
    out = sp.csr_matrix((new_data, counts.indices.copy(), indptr.copy()),
                         shape=counts.shape, dtype=np.float64)
    out.eliminate_zeros()
    return out, hit


def cv_a_d_table(donor_df: pd.DataFrame, group_label: str, method: str,
                  extra: dict = None) -> dict:
    c = donor_df[donor_df["stage"] == "child"]["score"].values
    a = donor_df[donor_df["stage"] == "adol"]["score"].values
    if len(c) < 2 or len(a) < 2:
        return None
    row = {"group": group_label, "method": method,
           "n_child": len(c), "n_adol": len(a),
           "mean_child": float(np.mean(c)),
           "mean_adol":  float(np.mean(a)),
           "cohens_d":   float(cohens_d(c, a))}
    if extra: row.update(extra)
    return row


def main():
    rng = np.random.default_rng(SEED)
    weights_df = build_c3plus_table()
    weights = weights_df.set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes", flush=True)

    GROUPS = [("PsychAD-V3", "PsychAD", "V3"),
              ("Velmeshev-V2", "Velmeshev", "V2"),
              ("Velmeshev-V3", "Velmeshev", "V3")]

    all_d_rows = []
    all_donor_rows = []
    all_capping_rows = []

    for group_label, dataset, chem in GROUPS:
        counts, obs, var_names = load_for_group(dataset, chem)
        if len(obs) == 0:
            print(f"  no cells; skipping {group_label}")
            continue

        # match GRN cols
        var_pos = {v: i for i, v in enumerate(var_names)}
        present = [g for g in weights.index if g in var_pos]
        grn_cols = np.array([var_pos[g] for g in present], dtype=np.int64)
        grn_weights = np.array([weights[g] for g in present], dtype=np.float64)
        print(f"  GRN columns matched: {len(grn_cols)} / {len(weights)}", flush=True)

        total_per_cell = obs["total_umi"].values.astype(np.float64)

        # ----- ORIGINAL: sum-then-CPM (baseline) -----
        baseline_donor = aggregate_sum_then_cpm(counts, obs, grn_cols, grn_weights)
        baseline_donor["group"] = group_label
        baseline_donor["method"] = "sum_then_CPM"
        all_donor_rows.append(baseline_donor)
        row = cv_a_d_table(baseline_donor, group_label, "sum_then_CPM (baseline)")
        if row: all_d_rows.append(row)
        print(f"\n  baseline (sum-then-CPM): d={row['cohens_d']:+.3f} "
              f"child={row['mean_child']:.1f} adol={row['mean_adol']:.1f}")

        # ----- J1: per-cell-CPM averaging -----
        per_cell = per_cell_score(counts, total_per_cell, grn_cols, grn_weights)
        j1_donor = aggregate_donor_mean(per_cell, obs)
        j1_donor["group"] = group_label
        j1_donor["method"] = "per_cell_CPM_mean"
        all_donor_rows.append(j1_donor)
        row = cv_a_d_table(j1_donor, group_label, "per_cell_CPM_mean (J1)")
        if row: all_d_rows.append(row)
        print(f"  per-cell-CPM mean (J1): d={row['cohens_d']:+.3f} "
              f"child={row['mean_child']:.1f} adol={row['mean_adol']:.1f}")

        # ----- J2: downsample at each cap -----
        for cap in CAPS:
            print(f"\n  J2 downsample to {cap}:", flush=True)
            ds_counts, hit = downsample_csr(counts, total_per_cell, cap, rng)
            # capping % per donor × age
            obs_tmp = obs.copy()
            obs_tmp["hit_cap"] = hit
            cap_summary = (obs_tmp.groupby(["individual", "stage", "age_years"],
                                            observed=True)
                                  .agg(n_cells=("hit_cap", "size"),
                                       n_hit=("hit_cap", "sum"))
                                  .reset_index())
            cap_summary["pct_hit_cap"] = cap_summary["n_hit"] / cap_summary["n_cells"]
            cap_summary["group"] = group_label
            cap_summary["cap"] = cap
            all_capping_rows.append(cap_summary)
            stage_hit = (cap_summary.groupby("stage")
                            .agg(median_pct=("pct_hit_cap", "median"),
                                 mean_pct=("pct_hit_cap", "mean"))
                            .round(3))
            print(f"    %cells hit cap (median per donor): "
                  f"child={stage_hit.loc['child', 'median_pct']:.2%}, "
                  f"adol={stage_hit.loc['adol', 'median_pct']:.2%}")

            ds_donor = aggregate_sum_then_cpm(ds_counts, obs, grn_cols, grn_weights)
            ds_donor["group"] = group_label
            ds_donor["method"] = f"downsample_{cap}_sum_then_CPM"
            ds_donor["cap"] = cap
            all_donor_rows.append(ds_donor)
            row = cv_a_d_table(ds_donor, group_label,
                                f"downsample_{cap}_sum_then_CPM",
                                extra={"cap": cap})
            if row: all_d_rows.append(row)
            print(f"    d after downsample to {cap}: {row['cohens_d']:+.3f}")

    # save
    d_df = pd.DataFrame(all_d_rows)
    d_df.to_csv(OUT_DIR / "j_d_per_method.csv", index=False)
    print("\n========== child-vs-adol d per group × method ==========")
    print(d_df.to_string(index=False))

    don_df = pd.concat(all_donor_rows, ignore_index=True)
    don_df.to_csv(OUT_DIR / "j_donor_scores_per_method.csv", index=False)

    cap_df = pd.concat(all_capping_rows, ignore_index=True)
    cap_df.to_csv(OUT_DIR / "j_capping_per_donor.csv", index=False)

    # summary of % capping per group × cap × stage
    cap_summary_overall = (cap_df.groupby(["group", "cap", "stage"], observed=True)
                              .agg(n_donors=("individual", "nunique"),
                                   mean_pct_hit=("pct_hit_cap", "mean"),
                                   median_pct_hit=("pct_hit_cap", "median"))
                              .round(4)
                              .reset_index())
    cap_summary_overall.to_csv(OUT_DIR / "j_capping_summary.csv", index=False)
    print("\n========== % cells hitting cap per group × cap × stage ==========")
    print(cap_summary_overall.to_string(index=False))

    # plots
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"PsychAD-V3": "C0", "Velmeshev-V2": "C3", "Velmeshev-V3": "C2"}
    for group_label, sub in d_df.groupby("group", observed=True):
        # baseline + per-cell + caps
        base = sub[sub["method"].str.startswith("sum_then_CPM")]
        if len(base):
            ax.axhline(base["cohens_d"].iloc[0], color=colors[group_label], ls=":",
                        lw=0.7, alpha=0.5, label=f"{group_label} baseline d")
        pccpm = sub[sub["method"].str.startswith("per_cell")]
        if len(pccpm):
            ax.axhline(pccpm["cohens_d"].iloc[0], color=colors[group_label], ls="--",
                        lw=0.7, alpha=0.5, label=f"{group_label} per-cell CPM mean d")
        caps_sub = sub[sub["cap"].notna()].sort_values("cap")
        ax.plot(caps_sub["cap"], caps_sub["cohens_d"], "o-",
                 color=colors[group_label], label=f"{group_label} downsample")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("per-cell UMI downsampling cap (log)")
    ax.set_ylabel("child-vs-adol Cohen's d")
    ax.set_title("J: child vs adol C3+ d as a function of normalization\n"
                 "(baseline = sum-then-CPM, dashed = per-cell-CPM mean, points = downsample)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "j_d_vs_method.png", dpi=150)
    plt.close(fig)

    # plot capping % vs cap
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, group_label in zip(axes, ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]):
        sub = cap_summary_overall[cap_summary_overall["group"] == group_label]
        for stage, color in [("child", "C3"), ("adol", "C0")]:
            ss = sub[sub["stage"] == stage].sort_values("cap")
            ax.plot(ss["cap"], ss["mean_pct_hit"], "o-", color=color,
                     label=f"{stage} (n_donors={ss['n_donors'].iloc[0] if len(ss) else 0})")
        ax.set_xscale("log")
        ax.set_xlabel("per-cell UMI cap")
        ax.set_ylabel("mean % of cells hitting cap (per donor)")
        ax.set_title(group_label)
        ax.set_ylim(0, 1.05)
        ax.legend()
    fig.suptitle("J: data lost at each downsampling cap, per dataset × stage", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "j_capping_vs_cap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
