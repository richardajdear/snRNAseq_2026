#!/usr/bin/env python3
"""
K — layer-marker module stratification (K1) + depth-quartile with per-cell-CPM
(K2), using the J-baseline (per-cell-CPM mean) as the aggregation rule.

K1: Within marker-annotated ExN cells only (so the GAD-ambient
subclass-label problem is sidestepped), score each cell on layer-
defining transcription-factor modules. Assign cell to argmax layer.
Pseudobulk per donor × layer using per-cell-CPM MEAN. Compute child-vs-
adol C3+ d per (group, layer).

Layer modules use postnatally-stable TFs that are established during
embryonic neurogenesis:
  upper (L2/3 IT, L4 IT):  SATB2, CUX2, CUX1, RORB
  L5 ET:                   FEZF2, BCL11B, POU3F1
  L6 CT:                   TBR1, FOXP2, TLE4, NXPH4, SYT6
  L6 IT:                   SULF1, OPRK1

K2: Within each group, compute per-cell depth quartile. Pseudobulk per
(donor × depth_q) using per-cell-CPM MEAN. Compute child-vs-adol d.
Compare to I2 (which used sum-then-CPM) — expectation is the depth
gradient shrinks but does not vanish, and PsychAD's shallow quartile
goes more positive than the I2 baseline of +0.26.

Submit via:
    cd /home/rajd2/rds/hpc-work/snRNAseq_2026
    sbatch --time=01:00:00 --mem=180G \
      scripts/run_script.sh scripts/grn_dev_diagnostics/k_layer_and_depth.py
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

# Layer-defining TFs (Ensembl IDs); postnatally stable.
LAYER_MODULES = {
    "upper":  {"SATB2":  "ENSG00000119042",
                "CUX2":   "ENSG00000111249",
                "CUX1":   "ENSG00000257923",
                "RORB":   "ENSG00000198963"},
    "L5_ET":  {"FEZF2":  "ENSG00000153266",
                "BCL11B": "ENSG00000127152",
                "POU3F1": "ENSG00000185650"},
    "L6_CT":  {"TBR1":   "ENSG00000136535",
                "FOXP2":  "ENSG00000128573",
                "TLE4":   "ENSG00000106829",
                "NXPH4":  "ENSG00000182379",
                "SYT6":   "ENSG00000147642"},
    "L6_IT":  {"SULF1":  "ENSG00000137573",
                "OPRK1":  "ENSG00000082556"},
}
ALL_LAYER_GENES = {sym: ens for mod in LAYER_MODULES.values() for sym, ens in mod.items()}

MIN_CELLS_PER_GROUP = 5  # per (donor × stratum), need at least this many


def load_for_group(name: str, chemistry_filter=None):
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
    obs = obs.iloc[obs_idx].copy().reset_index(drop=True)
    obs["age_years"] = age.values[obs_idx]
    obs["individual"] = obs.get("individual", obs.get("donor_id"))
    obs["stage"] = np.where(obs["age_years"] < CHILD[1], "child", "adol")
    obs["total_umi"] = np.asarray(counts.sum(axis=1)).ravel()
    print(f"  total UMI median: {np.median(obs['total_umi']):.0f}", flush=True)
    return counts, obs, a.var_names.values


def per_cell_score(counts, total_per_cell, grn_cols, grn_weights):
    sub = counts[:, grn_cols]
    raw_dot = np.asarray(sub @ grn_weights.astype(np.float64)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total_per_cell > 0, raw_dot / total_per_cell * 1e6, 0.0)


def aggregate_donor_mean(per_cell_c3, obs, stratum_cols=None):
    """Per-donor (× stratum) MEAN of per-cell C3+ scores; the J-baseline."""
    if stratum_cols is None:
        groupby_cols = ["individual"]
    else:
        groupby_cols = ["individual"] + list(stratum_cols)
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


def quartile_within(series, n=4):
    try:
        return pd.qcut(series, n, labels=False, duplicates="drop")
    except Exception:
        return pd.Series(np.zeros(len(series), dtype=int), index=series.index)


def assign_layer(counts, total_per_cell, var_names, log_cp_target=1e4):
    """For each cell compute log1p(CP10K) of each layer module's
    gene set, then assign cell to layer = argmax(mean module score).
    Returns (layer_label_per_cell, module_scores_df).
    """
    var_pos = {v: i for i, v in enumerate(var_names)}
    module_scores = {}
    found = {}
    for mod_name, gene_map in LAYER_MODULES.items():
        cols = []
        present_syms = []
        for sym, ens in gene_map.items():
            if ens in var_pos:
                cols.append(var_pos[ens])
                present_syms.append(sym)
            else:
                print(f"    WARNING: {sym} ({ens}) not in var_names")
        found[mod_name] = present_syms
        cols = np.array(cols, dtype=np.int64)
        sub = counts[:, cols].toarray()
        # per-cell CP10K then log1p
        with np.errstate(divide="ignore", invalid="ignore"):
            cp10k = np.where(total_per_cell[:, None] > 0,
                              sub / total_per_cell[:, None] * log_cp_target, 0.0)
        log_cp = np.log1p(cp10k)
        module_scores[mod_name] = log_cp.mean(axis=1)
    for mod, syms in found.items():
        print(f"    {mod}: genes used = {syms}")
    df = pd.DataFrame(module_scores)
    # Assign layer = argmax module score; if max is 0 (no markers detected), label "ambiguous"
    max_score = df.max(axis=1)
    layer = df.idxmax(axis=1)
    layer[max_score == 0] = "ambiguous"
    return layer.values, df


def main():
    weights_df = build_c3plus_table()
    weights = weights_df.set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes", flush=True)

    GROUPS = [("PsychAD-V3", "PsychAD", "V3"),
              ("Velmeshev-V2", "Velmeshev", "V2"),
              ("Velmeshev-V3", "Velmeshev", "V3")]

    k1_d = []
    k2_d = []
    layer_comp_rows = []
    layer_score_rows = []
    depth_score_rows = []

    for group_label, dataset, chem in GROUPS:
        counts, obs, var_names = load_for_group(dataset, chem)
        if len(obs) == 0:
            continue

        # match GRN cols
        var_pos = {v: i for i, v in enumerate(var_names)}
        present = [g for g in weights.index if g in var_pos]
        grn_cols = np.array([var_pos[g] for g in present], dtype=np.int64)
        grn_weights = np.array([weights[g] for g in present], dtype=np.float64)
        print(f"  GRN columns matched: {len(grn_cols)} / {len(weights)}", flush=True)

        total_per_cell = obs["total_umi"].values.astype(np.float64)
        per_cell = per_cell_score(counts, total_per_cell, grn_cols, grn_weights)

        # ----- K2: depth quartile + per-cell-CPM mean -----
        obs["depth_q"] = quartile_within(obs["total_umi"], 4)
        print(f"\n  K2 depth quartiles ({group_label}):", flush=True)
        depth_range = (obs.groupby("depth_q")["total_umi"]
                          .agg(["min", "median", "max", "size"]))
        print(depth_range.to_string())
        agg_depth = aggregate_donor_mean(per_cell, obs, ["depth_q"])
        agg_depth["group"] = group_label
        agg_depth["method"] = "per_cell_CPM_mean"
        depth_score_rows.append(agg_depth)
        d_depth = child_vs_adol_d(agg_depth, ["depth_q"])
        d_depth["group"] = group_label
        k2_d.append(d_depth)
        print(f"\n  K2: child-vs-adol d by depth_q ({group_label}, per-cell-CPM mean):")
        print(d_depth.to_string(index=False))

        # ----- K1: layer-marker module assignment -----
        print(f"\n  K1: assigning layer for {group_label}...")
        layer, module_scores = assign_layer(counts, total_per_cell, var_names)
        obs["layer"] = layer
        print(f"  layer counts: {pd.Series(layer).value_counts().to_dict()}")

        # capture per-cell module scores for QC
        for mod in LAYER_MODULES:
            row = module_scores[mod].values
            print(f"    module {mod}: mean log1p CP10K = {row.mean():.3f}, "
                   f"median = {np.median(row):.3f}, frac>0 = {(row>0).mean():.3f}")

        # composition: per donor × stage × layer  (fraction of ExN cells)
        comp = (obs.groupby(["individual", "stage", "age_years", "layer"], observed=True)
                    .size().rename("n_cells").reset_index())
        comp["group"] = group_label
        layer_comp_rows.append(comp)

        # ----- aggregate per donor × layer using per-cell-CPM mean -----
        agg_layer = aggregate_donor_mean(per_cell, obs, ["layer"])
        agg_layer["group"] = group_label
        agg_layer["method"] = "per_cell_CPM_mean"
        layer_score_rows.append(agg_layer)
        d_layer = child_vs_adol_d(agg_layer, ["layer"])
        d_layer["group"] = group_label
        k1_d.append(d_layer)
        print(f"\n  K1: child-vs-adol d by LAYER ({group_label}, per-cell-CPM mean):")
        print(d_layer.to_string(index=False))

        # also: per-donor whole-ExN aggregate for baseline comparison
        baseline = aggregate_donor_mean(per_cell, obs, None)
        baseline["group"] = group_label
        c = baseline[baseline["stage"] == "child"]["score"].values
        a = baseline[baseline["stage"] == "adol"]["score"].values
        d_base = cohens_d(c, a)
        print(f"  baseline per-cell-CPM mean d = {d_base:+.3f} "
               f"(child n={len(c)}, adol n={len(a)})")

    # ----- save -----
    k1_df = pd.concat(k1_d, ignore_index=True)
    k1_df.to_csv(OUT_DIR / "k1_d_by_layer.csv", index=False)
    k2_df = pd.concat(k2_d, ignore_index=True)
    k2_df.to_csv(OUT_DIR / "k2_d_by_depth_q_perCellCPM.csv", index=False)
    comp_df = pd.concat(layer_comp_rows, ignore_index=True)
    comp_df.to_csv(OUT_DIR / "k1_layer_composition_per_donor.csv", index=False)
    score_layer = pd.concat(layer_score_rows, ignore_index=True)
    score_layer.to_csv(OUT_DIR / "k1_donor_scores_by_layer.csv", index=False)
    score_depth = pd.concat(depth_score_rows, ignore_index=True)
    score_depth.to_csv(OUT_DIR / "k2_donor_scores_by_depth_q.csv", index=False)

    print("\n========== K1: child-vs-adol d by LAYER (per-cell-CPM mean) ==========")
    print(k1_df.to_string(index=False))
    print("\n========== K2: child-vs-adol d by DEPTH QUARTILE (per-cell-CPM mean) ==========")
    print(k2_df.to_string(index=False))

    # ----- layer composition summary across stages -----
    comp_summary = (comp_df.groupby(["group", "stage", "layer"], observed=True)
                       .agg(mean_n_cells=("n_cells", "mean"),
                            n_donors=("individual", "nunique"))
                       .reset_index())
    # also per-donor fraction
    tot = (comp_df.groupby(["group", "individual", "stage"], observed=True)["n_cells"]
              .sum().rename("total_cells").reset_index())
    comp_df_frac = comp_df.merge(tot, on=["group", "individual", "stage"])
    comp_df_frac["frac"] = comp_df_frac["n_cells"] / comp_df_frac["total_cells"]
    frac_summary = (comp_df_frac.groupby(["group", "stage", "layer"], observed=True)
                       .agg(mean_frac=("frac", "mean"),
                            median_frac=("frac", "median"),
                            n_donors=("individual", "nunique"))
                       .round(3)
                       .reset_index())
    frac_summary.to_csv(OUT_DIR / "k1_layer_composition_summary.csv", index=False)
    print("\n========== K1: layer composition (mean fraction per donor) ==========")
    print(frac_summary.to_string(index=False))

    # ----- plots -----
    # K1 d-by-layer
    fig, ax = plt.subplots(figsize=(11, 5))
    layers_order = ["upper", "L5_ET", "L6_CT", "L6_IT", "ambiguous"]
    width = 0.25
    x_base = np.arange(len(layers_order))
    for i, group in enumerate(["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]):
        sub = k1_df[k1_df["group"] == group].set_index("layer")
        vals = [sub.loc[l, "cohens_d"] if l in sub.index else np.nan
                for l in layers_order]
        ax.bar(x_base + (i - 1) * width, vals, width=width, label=group)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x_base); ax.set_xticklabels(layers_order)
    ax.set_ylabel("child-vs-adol C3+ Cohen's d (per-cell-CPM mean)")
    ax.set_title("K1: C3+ developmental d by cortical layer module assignment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k1_d_by_layer.png", dpi=150)
    plt.close(fig)

    # K2 d-by-depth-q under per-cell-CPM
    fig, ax = plt.subplots(figsize=(10, 5))
    for group, color in [("PsychAD-V3", "C0"),
                          ("Velmeshev-V2", "C3"),
                          ("Velmeshev-V3", "C2")]:
        sub = k2_df[k2_df["group"] == group].sort_values("depth_q")
        ax.plot(sub["depth_q"], sub["cohens_d"], "o-", color=color, label=group)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("depth quartile (0=shallow → 3=deep)")
    ax.set_ylabel("child-vs-adol C3+ Cohen's d (per-cell-CPM mean)")
    ax.set_title("K2: C3+ d by depth quartile, per-cell-CPM aggregation\n"
                 "(compare to I2 which used sum-then-CPM)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k2_d_by_depth_q_perCellCPM.png", dpi=150)
    plt.close(fig)

    # layer composition stacked bar per group × stage
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    layer_colors = {"upper": "#1f77b4", "L5_ET": "#ff7f0e", "L6_CT": "#2ca02c",
                     "L6_IT": "#d62728", "ambiguous": "#888888"}
    for ax, group in zip(axes, ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]):
        sub = frac_summary[frac_summary["group"] == group]
        stages = ["child", "adol"]
        bottom = np.zeros(2)
        for l in layers_order:
            vals = [sub[(sub["stage"] == s) & (sub["layer"] == l)]["mean_frac"].sum()
                    for s in stages]
            ax.bar(stages, vals, bottom=bottom, color=layer_colors[l], label=l)
            bottom = bottom + np.array(vals)
        ax.set_title(group); ax.set_ylim(0, 1.05)
        if ax is axes[0]: ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("K1: layer composition of ExN cells per group × stage", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k1_layer_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
