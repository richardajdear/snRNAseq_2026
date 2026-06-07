#!/usr/bin/env python3
"""
L — joint depth × layer stratification, with depth-matched cross-dataset
comparison.

User question: K2 showed PsychAD-V3 per-cell-CPM d goes +0.24 → -0.86
across depth quartiles, meaning depth tags REAL biological heterogeneity
beyond what per-cell-CPM normalisation can fix. K1 showed layer
stratification doesn't close the PsychAD-vs-Vel gap. But we haven't
checked: does depth × layer JOINT stratification close the gap? I.e.,
is the depth confound INSIDE each layer, or BETWEEN layers?

This script:
  L1. Per-layer depth distribution per dataset — does layer assignment
      itself correlate with depth?
  L2. Joint (layer × depth-quartile) child-vs-adol d per dataset, with
      per-cell-CPM mean aggregation.
  L3. Depth-matched analysis: restrict PsychAD-V3 to the UMI range of
      Velmeshev-V3 (overlapping depth window), recompute layer × age d.
      Same for layer composition. Does PsychAD-V3 at Vel-V3 depths look
      like Vel-V3?
  L4. Reverse: restrict Vel-V3 to PsychAD-V3 deep cells (Q3 depth range)
      and recompute. Does deep Vel-V3 show the anti-drop that deep
      PsychAD-V3 shows?

Submit via:
    cd /home/rajd2/rds/hpc-work/snRNAseq_2026
    sbatch --time=01:00:00 --mem=180G \
      scripts/run_script.sh scripts/grn_dev_diagnostics/l_depth_matched_layer.py
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

LAYER_MODULES = {
    "upper":  {"SATB2":"ENSG00000119042","CUX2":"ENSG00000111249",
                "CUX1":"ENSG00000257923","RORB":"ENSG00000198963"},
    "L5_ET":  {"FEZF2":"ENSG00000153266","BCL11B":"ENSG00000127152",
                "POU3F1":"ENSG00000185650"},
    "L6_CT":  {"TBR1":"ENSG00000136535","FOXP2":"ENSG00000128573",
                "TLE4":"ENSG00000106829","NXPH4":"ENSG00000182379",
                "SYT6":"ENSG00000147642"},
    "L6_IT":  {"SULF1":"ENSG00000137573","OPRK1":"ENSG00000082556"},
}

MIN_CELLS_PER_GROUP = 5


def load_for_group(name: str, chemistry_filter=None):
    print(f"\n=== {name}{(' chem='+chemistry_filter) if chemistry_filter else ''}: loading", flush=True)
    a = ad.read_h5ad(INPUTS[name], backed="r")
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
    counts = sp.csr_matrix(a.layers["counts"])
    counts = counts[obs_idx, :]
    obs = obs.iloc[obs_idx].copy().reset_index(drop=True)
    obs["age_years"] = age.values[obs_idx]
    obs["individual"] = obs.get("individual", obs.get("donor_id"))
    obs["stage"] = np.where(obs["age_years"] < CHILD[1], "child", "adol")
    obs["total_umi"] = np.asarray(counts.sum(axis=1)).ravel()
    print(f"  ExN cells: {len(obs):,}; total UMI median: {np.median(obs['total_umi']):.0f}", flush=True)
    return counts, obs, a.var_names.values


def per_cell_score(counts, total_per_cell, grn_cols, grn_weights):
    sub = counts[:, grn_cols]
    raw_dot = np.asarray(sub @ grn_weights.astype(np.float64)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total_per_cell > 0, raw_dot / total_per_cell * 1e6, 0.0)


def assign_layer(counts, total_per_cell, var_names):
    var_pos = {v: i for i, v in enumerate(var_names)}
    mod_scores = {}
    for mod_name, gene_map in LAYER_MODULES.items():
        cols = [var_pos[ens] for ens in gene_map.values() if ens in var_pos]
        cols = np.array(cols, dtype=np.int64)
        sub = counts[:, cols].toarray()
        with np.errstate(divide="ignore", invalid="ignore"):
            cp10k = np.where(total_per_cell[:, None] > 0,
                              sub / total_per_cell[:, None] * 1e4, 0.0)
        mod_scores[mod_name] = np.log1p(cp10k).mean(axis=1)
    df = pd.DataFrame(mod_scores)
    max_score = df.max(axis=1)
    layer = df.idxmax(axis=1)
    layer[max_score == 0] = "ambiguous"
    return layer.values


def aggregate_donor_mean(per_cell_c3, obs, stratum_cols=None):
    groupby_cols = ["individual"] + list(stratum_cols or [])
    df = obs[groupby_cols + ["age_years", "stage"]].copy()
    df["score"] = per_cell_c3
    agg = (df.groupby(groupby_cols, observed=True)
              .agg(n_cells=("score", "size"),
                   score=("score", "mean"),
                   age_years=("age_years", "first"),
                   stage=("stage", "first"))
              .reset_index())
    return agg[agg["n_cells"] >= MIN_CELLS_PER_GROUP]


def child_vs_adol_d(donor_df, group_cols):
    rows = []
    for keys, sub in donor_df.groupby(group_cols, observed=True):
        c = sub[sub["stage"] == "child"]["score"].values
        a = sub[sub["stage"] == "adol"]["score"].values
        if len(c) < 2 or len(a) < 2:
            continue
        d = cohens_d(c, a)
        row = dict(zip(group_cols if isinstance(group_cols, list) else [group_cols],
                       keys if isinstance(keys, tuple) else (keys,)))
        row.update({"n_child": len(c), "n_adol": len(a),
                    "mean_child": float(np.mean(c)),
                    "mean_adol":  float(np.mean(a)),
                    "cohens_d":   float(d)})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    weights = build_c3plus_table().set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes", flush=True)

    GROUPS = [("PsychAD-V3", "PsychAD", "V3"),
              ("Velmeshev-V3", "Velmeshev", "V3")]

    # ----- Load + annotate -----
    cache = {}
    for group_label, dataset, chem in GROUPS:
        counts, obs, var_names = load_for_group(dataset, chem)
        var_pos = {v: i for i, v in enumerate(var_names)}
        present = [g for g in weights.index if g in var_pos]
        grn_cols = np.array([var_pos[g] for g in present], dtype=np.int64)
        grn_weights = np.array([weights[g] for g in present], dtype=np.float64)
        total = obs["total_umi"].values.astype(np.float64)
        per_cell = per_cell_score(counts, total, grn_cols, grn_weights)
        layer = assign_layer(counts, total, var_names)
        obs["layer"] = layer
        obs["per_cell_c3"] = per_cell
        cache[group_label] = {"counts": counts, "obs": obs}
        print(f"  {group_label}: layer counts = {pd.Series(layer).value_counts().to_dict()}")

    # ===================== L1: per-layer depth distribution ====================
    print("\n========== L1: per-layer depth distribution per dataset ==========")
    l1_rows = []
    for group_label, ent in cache.items():
        obs = ent["obs"]
        for layer in ["upper", "L5_ET", "L6_CT", "L6_IT", "ambiguous"]:
            sub = obs[obs["layer"] == layer]
            if len(sub) == 0: continue
            l1_rows.append({
                "group": group_label, "layer": layer,
                "n_cells": int(len(sub)),
                "median_umi": float(sub["total_umi"].median()),
                "mean_umi":   float(sub["total_umi"].mean()),
                "p25_umi":    float(sub["total_umi"].quantile(0.25)),
                "p75_umi":    float(sub["total_umi"].quantile(0.75)),
            })
    l1_df = pd.DataFrame(l1_rows)
    l1_df.to_csv(OUT_DIR / "l1_layer_depth_distribution.csv", index=False)
    print(l1_df.to_string(index=False))

    # ===================== L2: joint depth-quartile × layer d =================
    # Compute depth quartiles WITHIN each dataset (so each dataset's depth_q is relative to itself)
    print("\n========== L2: joint (depth_q × layer) d per dataset ==========")
    l2_rows = []
    for group_label, ent in cache.items():
        obs = ent["obs"].copy()
        obs["depth_q"] = pd.qcut(obs["total_umi"], 4, labels=False, duplicates="drop")
        per_cell = obs["per_cell_c3"].values
        agg = aggregate_donor_mean(per_cell, obs, ["depth_q", "layer"])
        d = child_vs_adol_d(agg, ["depth_q", "layer"])
        d["group"] = group_label
        l2_rows.append(d)
    l2_df = pd.concat(l2_rows, ignore_index=True)
    l2_df = l2_df.sort_values(["group", "depth_q", "layer"])
    l2_df.to_csv(OUT_DIR / "l2_d_depth_q_x_layer.csv", index=False)
    print(l2_df.to_string(index=False))

    # ===================== L3: depth-matched comparison =======================
    # Determine common depth window: use Vel-V3 [p10, p90] as the reference
    psy_umi = cache["PsychAD-V3"]["obs"]["total_umi"].values
    vel_umi = cache["Velmeshev-V3"]["obs"]["total_umi"].values

    print("\n========== L3: depth-matched comparison ==========")
    print(f"  PsychAD-V3 UMI distribution: p10={np.percentile(psy_umi, 10):.0f}, "
          f"median={np.median(psy_umi):.0f}, p90={np.percentile(psy_umi, 90):.0f}")
    print(f"  Velmeshev-V3 UMI distribution: p10={np.percentile(vel_umi, 10):.0f}, "
          f"median={np.median(vel_umi):.0f}, p90={np.percentile(vel_umi, 90):.0f}")

    # Try multiple depth windows
    DEPTH_WINDOWS = [
        ("overlap_2k_15k",         2000, 15000),    # broad common range
        ("vel_v3_central_3k_12k",  3000, 12000),    # tighter central
        ("psy_v3_shallow_1k_8k",   1000, 8000),     # PsychAD's shallow half
        ("vel_v3_full_p5_p95",     int(np.percentile(vel_umi, 5)),
                                    int(np.percentile(vel_umi, 95))),
    ]

    l3_all_rows = []
    l3_comp_rows = []
    for window_label, lo, hi in DEPTH_WINDOWS:
        print(f"\n  --- depth window '{window_label}' = [{lo}, {hi}] UMI ---")
        for group_label, ent in cache.items():
            obs = ent["obs"]
            mask = (obs["total_umi"] >= lo) & (obs["total_umi"] < hi)
            sub_obs = obs[mask].reset_index(drop=True)
            per_cell = sub_obs["per_cell_c3"].values
            print(f"    {group_label}: {mask.sum():,} cells retained "
                  f"({mask.mean()*100:.1f}%), "
                  f"median UMI after filter: {sub_obs['total_umi'].median():.0f}")

            # Aggregate baseline (whole-ExN at matched depth)
            agg_base = aggregate_donor_mean(per_cell, sub_obs, None)
            c = agg_base[agg_base["stage"] == "child"]["score"].values
            a = agg_base[agg_base["stage"] == "adol"]["score"].values
            if len(c) >= 2 and len(a) >= 2:
                d_base = cohens_d(c, a)
                l3_all_rows.append({"window": window_label, "lo": lo, "hi": hi,
                                     "group": group_label, "layer": "ALL",
                                     "n_child": len(c), "n_adol": len(a),
                                     "mean_child": float(np.mean(c)),
                                     "mean_adol": float(np.mean(a)),
                                     "cohens_d": float(d_base)})
                print(f"      baseline d (all ExN at matched depth): {d_base:+.3f}")

            # Layer × age within matched depth
            agg_layer = aggregate_donor_mean(per_cell, sub_obs, ["layer"])
            d_layer = child_vs_adol_d(agg_layer, ["layer"])
            for _, r in d_layer.iterrows():
                l3_all_rows.append({"window": window_label, "lo": lo, "hi": hi,
                                     "group": group_label, "layer": r["layer"],
                                     "n_child": r["n_child"], "n_adol": r["n_adol"],
                                     "mean_child": r["mean_child"],
                                     "mean_adol": r["mean_adol"],
                                     "cohens_d": r["cohens_d"]})
            print(f"      per-layer d:")
            for _, r in d_layer.iterrows():
                print(f"        {r['layer']:10s} d={r['cohens_d']:+.3f}  "
                       f"(n_c={r['n_child']}, n_a={r['n_adol']})")

            # Layer composition shift at matched depth
            comp = (sub_obs.groupby(["individual", "stage", "layer"], observed=True)
                       .size().rename("n_cells").reset_index())
            tot = (comp.groupby(["individual", "stage"], observed=True)["n_cells"]
                       .sum().rename("total_cells").reset_index())
            comp = comp.merge(tot, on=["individual", "stage"])
            comp["frac"] = comp["n_cells"] / comp["total_cells"]
            comp["group"] = group_label
            comp["window"] = window_label
            l3_comp_rows.append(comp)

    l3_df = pd.DataFrame(l3_all_rows)
    l3_df.to_csv(OUT_DIR / "l3_d_depth_matched.csv", index=False)
    l3_comp = pd.concat(l3_comp_rows, ignore_index=True)
    l3_comp.to_csv(OUT_DIR / "l3_layer_composition_depth_matched.csv", index=False)

    # composition summary per window × group × stage × layer
    l3_comp_sum = (l3_comp.groupby(["window", "group", "stage", "layer"], observed=True)
                     .agg(mean_frac=("frac", "mean"),
                          median_frac=("frac", "median"),
                          n_donors=("individual", "nunique"))
                     .round(3).reset_index())
    l3_comp_sum.to_csv(OUT_DIR / "l3_layer_composition_summary.csv", index=False)
    print("\n========== L3 layer composition at matched depth ==========")
    print(l3_comp_sum.to_string(index=False))

    # ----- plots -----
    # L2 heatmap of d per (depth_q, layer) per dataset
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, group_label in zip(axes, ["PsychAD-V3", "Velmeshev-V3"]):
        sub = l2_df[l2_df["group"] == group_label]
        pivot = sub.pivot_table(index="depth_q", columns="layer", values="cohens_d")
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
        ax.set_xticks(np.arange(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=45)
        ax.set_yticks(np.arange(pivot.shape[0])); ax.set_yticklabels([f"Q{int(q)}" for q in pivot.index])
        ax.set_title(f"{group_label}\n(rows = depth_q, columns = layer; cells show d)")
        # annotate
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                             color="white" if abs(v) > 1 else "k", fontsize=9)
        plt.colorbar(im, ax=ax, label="Cohen's d (child vs adol)")
    fig.suptitle("L2: per-cell-CPM mean d by (depth_q × layer) per dataset", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "l2_d_depth_q_x_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # L3 d per window × layer × group
    fig, axes = plt.subplots(1, len(DEPTH_WINDOWS), figsize=(5 * len(DEPTH_WINDOWS), 4.5),
                              sharey=True)
    layers_order = ["upper", "L5_ET", "L6_CT", "L6_IT", "ambiguous", "ALL"]
    for ax, (window_label, lo, hi) in zip(axes, DEPTH_WINDOWS):
        sub = l3_df[l3_df["window"] == window_label]
        x = np.arange(len(layers_order))
        for i, group_label in enumerate(["PsychAD-V3", "Velmeshev-V3"]):
            ss = sub[sub["group"] == group_label].set_index("layer")
            vals = [ss.loc[l, "cohens_d"] if l in ss.index else np.nan
                    for l in layers_order]
            ax.bar(x + (i - 0.5) * 0.4, vals, width=0.4, label=group_label,
                    color="C0" if i == 0 else "C2")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(layers_order, rotation=45)
        ax.set_title(f"{window_label}\nUMI [{lo}, {hi}]")
        ax.legend()
    axes[0].set_ylabel("child-vs-adol C3+ Cohen's d")
    fig.suptitle("L3: depth-matched per-layer d (per-cell-CPM mean)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "l3_d_depth_matched.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
