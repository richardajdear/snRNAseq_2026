#!/usr/bin/env python3
"""
M — generate publication-quality plots and tables for FINAL_REPORT.md.

Revised version (2026-06-06):
  * All d values use a FUZZY childhood/adolescence boundary, averaging
    Cohen's d across the boundary set {8, 9, 10, 11, 12} years. This
    avoids any single-boundary cherry-picking artefact.
  * Velmeshev-V2 included in every cross-cohort plot (m2, m5, m6, m7).
  * m6 trajectory smoothing is a moving average in age-year bins
    (window of 4 y) rather than over donor counts.
  * New m7: per-donor trajectories under several different depth windows.
  * New table: window-bounds × group × fuzzy-d.

Submit via:
    cd /home/rajd2/rds/hpc-work/snRNAseq_2026
    sbatch --time=02:30:00 --mem=200G \\
       scripts/run_script.sh scripts/grn_dev_diagnostics/m_report_plots.py
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
import matplotlib.ticker as mtick

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (
    build_c3plus_table, cohens_d, OUT_DIR, CHILD, ADOL,
    FUZZY_BOUNDARIES, AGE_LO, AGE_HI, fuzzy_d_from_donor_scores,
)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

INPUTS_INTEGRATED = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/scvi_output/integrated.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad",
}
INPUTS_MANUAL = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
}
INPUTS_BYCLASS = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/by_cell_class.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class.h5ad",
}
INPUTS_MARKER_PB = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
}

# groups in plotting order
GROUPS = ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]
GROUP_TO_DATASET = {
    "PsychAD-V3":   ("PsychAD",   "V3"),
    "Velmeshev-V2": ("Velmeshev", "V2"),
    "Velmeshev-V3": ("Velmeshev", "V3"),
}

# layer-defining TF modules (postnatally stable; established embryonically)
LAYER_MODULES = {
    "upper":  {"SATB2": "ENSG00000119042", "CUX2": "ENSG00000111249",
                "CUX1": "ENSG00000257923", "RORB": "ENSG00000198963"},
    "L5_ET":  {"FEZF2": "ENSG00000153266", "BCL11B": "ENSG00000127152",
                "POU3F1": "ENSG00000185650"},
    "L6_CT":  {"TBR1": "ENSG00000136535", "FOXP2": "ENSG00000128573",
                "TLE4": "ENSG00000106829", "NXPH4": "ENSG00000182379",
                "SYT6": "ENSG00000147642"},
    "L6_IT":  {"SULF1": "ENSG00000137573", "OPRK1": "ENSG00000082556"},
}

# colours
COLORS = {
    "PsychAD-V3":   "#C0392B",   # red
    "Velmeshev-V2": "#27AE60",   # green
    "Velmeshev-V3": "#2980B9",   # blue
}

# matplotlib style
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.titleweight": "bold",
})

PER_CELL_CACHE = OUT_DIR / "m_per_cell_cache.parquet"
MIN_CELLS_PER_DONOR = 5

# Donor(s) excluded from all PsychAD-V3 analyses below.
# Donor_1400 (3y, n=238 cells): z=+2.5 on ExN_immature % (38% immature —
# biologically implausible at 3 y, when cortical EN are largely mature),
# also previously flagged in F3 as a C3+ low outlier. Removing it shifts
# the main 3-12 k fuzzy d from +0.24 → +0.32.
EXCLUDE_DONORS = {"Donor_1400"}

# Depth windows used in §3 multi-window analysis, ordered narrow→wide
# by (lo, hi). "none" first as the no-filter baseline.
DEPTH_WINDOWS = [
    ("none",         None,  None),
    ("1k-8k",        1000,  8000),
    ("1k-12k",       1000, 12000),
    ("1k-15k",       1000, 15000),
    ("1k-35k",       1000, 35000),
    ("2k-15k",       2000, 15000),
    ("3k-12k",       3000, 12000),
    ("3k-15k",       3000, 15000),
    ("5k-25k",       5000, 25000),
]
MAIN_WINDOW = ("3k-12k", 3000, 12000)


# ---------------------------------------------------------------------------
# Per-cell C3+ score precomputation
# ---------------------------------------------------------------------------

def _per_cell_for_group(group_label, weights):
    dataset, chem = GROUP_TO_DATASET[group_label]
    print(f"\n=== {group_label} ({dataset}, {chem}) — loading integrated h5ad", flush=True)
    a = ad.read_h5ad(INPUTS_INTEGRATED[dataset], backed="r")
    obs = a.obs.copy()
    if "chemistry" not in obs.columns and "source-chemistry" in obs.columns:
        obs["chemistry"] = obs["source-chemistry"].astype(str).str.extract(
            r"(V2|V3)")[0].fillna("unknown")
    obs["chemistry"] = obs["chemistry"].astype(str)
    ma = pd.read_parquet(INPUTS_MANUAL[dataset])
    obs = obs.join(ma, how="left")
    age = pd.to_numeric(obs["age_years"], errors="coerce")
    mask = ((age >= AGE_LO) & (age < AGE_HI)
            & obs["marker_annotation"].isin(["ExN_mature", "ExN_immature", "ExN_weak"])
            & (obs["chemistry"].values == chem))
    obs_idx = np.where(mask.values)[0]
    print(f"  ExN cells after filter: {len(obs_idx):,}", flush=True)
    counts = sp.csr_matrix(a.layers["counts"])[obs_idx, :]
    obs = obs.iloc[obs_idx].copy().reset_index(drop=True)
    obs["age_years"] = age.values[obs_idx]
    obs["individual"] = obs.get("individual", obs.get("donor_id"))
    obs["total_umi"] = np.asarray(counts.sum(axis=1)).ravel()
    var_names = a.var_names.values

    # C3+ per-cell projection
    var_pos = {v: i for i, v in enumerate(var_names)}
    present = [g for g in weights.index if g in var_pos]
    grn_cols = np.array([var_pos[g] for g in present], dtype=np.int64)
    grn_weights = np.array([weights[g] for g in present], dtype=np.float64)
    raw_dot = np.asarray(counts[:, grn_cols] @ grn_weights).ravel()
    total = obs["total_umi"].values.astype(np.float64)
    per_cell_c3 = np.where(total > 0, raw_dot / total * 1e6, 0.0)

    # layer assignment via TF-module argmax
    mod_scores = {}
    for mod_name, gene_map in LAYER_MODULES.items():
        cols = [var_pos[ens] for ens in gene_map.values() if ens in var_pos]
        cols = np.array(cols, dtype=np.int64)
        sub = counts[:, cols].toarray()
        with np.errstate(divide="ignore", invalid="ignore"):
            cp10k = np.where(total[:, None] > 0,
                              sub / total[:, None] * 1e4, 0.0)
        mod_scores[mod_name] = np.log1p(cp10k).mean(axis=1)
    mdf = pd.DataFrame(mod_scores)
    max_score = mdf.max(axis=1)
    layer = mdf.idxmax(axis=1)
    layer[max_score == 0] = "ambiguous"

    out = pd.DataFrame({
        "group":      group_label,
        "individual": obs["individual"].astype(str).values,
        "age_years":  obs["age_years"].values,
        "chemistry":  obs["chemistry"].astype(str).values,
        "total_umi":  obs["total_umi"].values.astype(np.int32),
        "layer":      layer.values,
        "marker_annotation": obs["marker_annotation"].astype(str).values,
        "per_cell_c3": per_cell_c3.astype(np.float32),
    })
    print(f"  done: {len(out):,} cells, median UMI {int(out['total_umi'].median()):,}", flush=True)
    return out


def build_per_cell_cache(force=False):
    """Cache per-cell C3+ scores + layer for all 3 groups."""
    if PER_CELL_CACHE.exists() and not force:
        print(f"Using cached per-cell scores: {PER_CELL_CACHE}", flush=True)
        return pd.read_parquet(PER_CELL_CACHE)
    weights = build_c3plus_table().set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes", flush=True)
    frames = [_per_cell_for_group(g, weights) for g in GROUPS]
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(PER_CELL_CACHE)
    print(f"\nSaved per-cell cache: {PER_CELL_CACHE} ({len(out):,} rows)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Donor-level aggregation + fuzzy d
# ---------------------------------------------------------------------------

def aggregate_donors(per_cell: pd.DataFrame,
                     group: str,
                     umi_lo=None, umi_hi=None,
                     layer=None,
                     min_cells=MIN_CELLS_PER_DONOR) -> pd.DataFrame:
    """Mean per-cell C3+ per donor (within the given group/window/layer)."""
    df = per_cell[per_cell["group"] == group]
    if EXCLUDE_DONORS:
        df = df[~df["individual"].isin(EXCLUDE_DONORS)]
    if umi_lo is not None:
        df = df[df["total_umi"] >= umi_lo]
    if umi_hi is not None:
        df = df[df["total_umi"] < umi_hi]
    if layer is not None:
        df = df[df["layer"] == layer]
    if len(df) == 0:
        return df.iloc[0:0].copy()
    don = (df.groupby("individual", observed=True)
              .agg(score=("per_cell_c3", "mean"),
                   n_cells=("per_cell_c3", "size"),
                   age_years=("age_years", "first"),
                   chemistry=("chemistry", "first"))
              .reset_index())
    return don[don["n_cells"] >= min_cells].reset_index(drop=True)


def fuzzy_d(donor_df):
    """Wrapper to compute mean d across boundaries from a donor DataFrame."""
    if len(donor_df) == 0 or donor_df["score"].notna().sum() < 4:
        return np.nan, []
    res = fuzzy_d_from_donor_scores(donor_df["age_years"].values,
                                     donor_df["score"].values)
    return res["mean_d"], res["per_boundary"]


# ---------------------------------------------------------------------------
# Stage-A and stage-B donor projections (pseudobulk-level)
# ---------------------------------------------------------------------------

C3_WEIGHTS_SERIES = build_c3plus_table().set_index("ensembl_id")["weight"]


def project_native_cell_class_donor(name, value_filter):
    """Stage A: native cell_class label, sum-then-CPM, donor-level."""
    cell_class_col = "cell_class" if name == "PsychAD" else "cell_class_original"
    a = ad.read_h5ad(INPUTS_BYCLASS[name])
    mask = ((a.obs[cell_class_col] == value_filter)
            & (a.obs["age_years"] >= AGE_LO) & (a.obs["age_years"] < AGE_HI))
    a = a[mask].copy()
    X = a.layers["counts"]
    if sp.issparse(X):
        X = X.toarray().astype(np.float64)
    tot = X.sum(axis=1, keepdims=True); tot[tot == 0] = 1.0
    cpm = X * (1e6 / tot)
    var_pos = {v: i for i, v in enumerate(a.var_names)}
    common = [g for g in C3_WEIGHTS_SERIES.index if g in var_pos]
    cols = np.array([var_pos[g] for g in common], dtype=np.int64)
    w = np.array([C3_WEIGHTS_SERIES[g] for g in common], dtype=np.float64)
    scores = cpm[:, cols] @ w
    df = a.obs[["individual", "age_years", "chemistry"]].copy()
    df["score"] = scores
    df["dataset"] = name
    return df.reset_index(drop=True)


def project_marker_sum_cpm_donor(name):
    """Stage B: marker_annotation, sum-then-CPM, donor-level."""
    a = ad.read_h5ad(INPUTS_MARKER_PB[name])
    mask = (a.obs["age_years"] >= AGE_LO) & (a.obs["age_years"] < AGE_HI)
    a = a[mask].copy()
    X = a.layers["counts"]
    if sp.issparse(X):
        X = X.toarray().astype(np.float64)
    tot = X.sum(axis=1, keepdims=True); tot[tot == 0] = 1.0
    cpm = X * (1e6 / tot)
    var_pos = {v: i for i, v in enumerate(a.var_names)}
    common = [g for g in C3_WEIGHTS_SERIES.index if g in var_pos]
    cols = np.array([var_pos[g] for g in common], dtype=np.int64)
    w = np.array([C3_WEIGHTS_SERIES[g] for g in common], dtype=np.float64)
    scores = cpm[:, cols] @ w
    df = a.obs[["individual", "age_years", "chemistry"]].copy()
    df["score"] = scores
    df["dataset"] = name
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot 1 — first confound: cell-class labels
# ---------------------------------------------------------------------------

def plot_cell_class_problem():
    """SATB2 + other EN/IN markers across datasets in <1y cells."""
    df = pd.read_csv(Path("/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/relabel_comparison/no_age_run/marker_means_raw_cpm.csv"))
    sub = df[df["group"].isin(["PSYCHAD_under1y", "PSYCHAD_1_5y",
                                  "WANG_under1y", "VEL_V3_under1y"])]
    sub = sub[sub["marker"].isin(["SATB2", "SLC17A7", "NEUROD2", "RBFOX3",
                                    "GAD1", "GAD2"])]
    label_map = {"PSYCHAD_under1y": "PsychAD <1y",
                  "PSYCHAD_1_5y":    "PsychAD 1-5y",
                  "WANG_under1y":    "Wang <1y",
                  "VEL_V3_under1y":  "Vel-V3 <1y"}
    sub = sub.assign(group_label=sub["group"].map(label_map))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, mtype, title in [
        (axes[0], "EN", "EN markers: PsychAD shows 4-11× deficit"),
        (axes[1], "IN", "IN markers: PsychAD is elevated")
    ]:
        msub = sub[sub["marker_type"] == mtype]
        markers = msub["marker"].unique()
        groups = ["PsychAD <1y", "PsychAD 1-5y", "Wang <1y", "Vel-V3 <1y"]
        gc = {"PsychAD <1y": "#C0392B", "PsychAD 1-5y": "#E07B5A",
               "Wang <1y": "#2980B9", "Vel-V3 <1y": "#27AE60"}
        x = np.arange(len(markers)); bar_w = 0.2
        for i, g in enumerate(groups):
            vals = []
            for m in markers:
                rs = msub[(msub["marker"] == m) & (msub["group_label"] == g)]["frac_nonzero"].values
                vals.append(float(rs[0]) if len(rs) else 0.0)
            ax.bar(x + (i - 1.5) * bar_w, vals, bar_w, label=g, color=gc[g])
        ax.set_xticks(x); ax.set_xticklabels(markers, fontsize=9, fontstyle="italic")
        ax.set_ylabel("Fraction of cells with ≥1 UMI" if mtype == "EN" else "")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        if mtype == "EN":
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("Native cell-class labels are unreliable for PsychAD <1y: EN markers depressed, IN markers elevated",
                  fontsize=11, fontweight="bold", y=1.04)
    fig.text(0.5, -0.06,
              "PsychAD <1y mean EN detection = 12 %  vs  Wang 48 %  vs  Vel-V3 62 %.\n"
              "Library depth comparable or higher in PsychAD, ruling out CPM artefact.\n"
              "Marker-based annotation (used hereafter) bypasses this label confound.",
              ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m1_cell_class_problem.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved m1_cell_class_problem.png")


# ---------------------------------------------------------------------------
# Plot 2 — correction-stage progression (fuzzy d, all 3 groups, all 4 stages)
# ---------------------------------------------------------------------------

def plot_correction_progression(per_cell):
    """Bar chart: fuzzy d at four correction stages for all three groups.

    Stages:
      0. NATIVE  — upstream cell_class label, donor-level sum-then-CPM
      A. MARKER  — marker_annotation, donor-level sum-then-CPM
      B. PCC     — marker + per-cell-CPM mean (no depth window)
      C. PCC+DM  — marker + per-cell-CPM mean + 3-12 k UMI window
    """
    # Stage 0 — native cell_class
    psy_n   = project_native_cell_class_donor("PsychAD",   "Excitatory")
    vel_n   = project_native_cell_class_donor("Velmeshev", "Excitatory")
    psy_n["group"] = "PsychAD-V3"
    vel_v2_n = vel_n[vel_n["chemistry"] == "V2"].copy(); vel_v2_n["group"] = "Velmeshev-V2"
    vel_v3_n = vel_n[vel_n["chemistry"] == "V3"].copy(); vel_v3_n["group"] = "Velmeshev-V3"

    # Stage A — marker, sum-then-CPM
    psy_m   = project_marker_sum_cpm_donor("PsychAD")
    vel_m   = project_marker_sum_cpm_donor("Velmeshev")
    psy_m["group"] = "PsychAD-V3"
    vel_v2_m = vel_m[vel_m["chemistry"] == "V2"].copy(); vel_v2_m["group"] = "Velmeshev-V2"
    vel_v3_m = vel_m[vel_m["chemistry"] == "V3"].copy(); vel_v3_m["group"] = "Velmeshev-V3"

    # Stages B, C — from per-cell cache
    stage_b = {g: aggregate_donors(per_cell, g) for g in GROUPS}
    stage_c = {g: aggregate_donors(per_cell, g, umi_lo=MAIN_WINDOW[1],
                                    umi_hi=MAIN_WINDOW[2]) for g in GROUPS}

    stage_a_map = {"PsychAD-V3": psy_n, "Velmeshev-V2": vel_v2_n, "Velmeshev-V3": vel_v3_n}
    stage_b_map = {"PsychAD-V3": psy_m, "Velmeshev-V2": vel_v2_m, "Velmeshev-V3": vel_v3_m}

    # Apply donor exclusion to all stage maps
    if EXCLUDE_DONORS:
        for smap in (stage_a_map, stage_b_map):
            for g, df in smap.items():
                smap[g] = df[~df["individual"].astype(str).isin(EXCLUDE_DONORS)].reset_index(drop=True)

    stages = [
        ("0 native\ncell-class\n(sum-then-CPM)", stage_a_map),
        ("A marker\nannotation\n(sum-then-CPM)", stage_b_map),
        ("B + per-cell\nCPM mean",              stage_b),
        ("C + depth-matched\n(3-12 k UMI)",      stage_c),
    ]

    # Compute fuzzy d for each (stage, group)
    rows = []
    for stage_label, smap in stages:
        for g in GROUPS:
            df = smap[g]
            if df is None or len(df) < 4:
                d = np.nan
            else:
                d, _ = fuzzy_d(df)
            rows.append({"stage": stage_label, "group": g, "d": d,
                         "n_donors": int(len(df)) if df is not None else 0})
    df_bar = pd.DataFrame(rows)
    df_bar.to_csv(OUT_DIR / "m2_correction_progression_data.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    stages_x = [s[0] for s in stages]
    x = np.arange(len(stages_x))
    bar_w = 0.26
    for i, g in enumerate(GROUPS):
        vals = [df_bar[(df_bar["group"] == g) & (df_bar["stage"] == s)]["d"].iloc[0]
                for s in stages_x]
        bars = ax.bar(x + (i - 1) * bar_w, vals, bar_w, label=g, color=COLORS[g])
        for j_, v in enumerate(vals):
            if np.isnan(v): continue
            ax.text(x[j_] + (i - 1) * bar_w,
                     v + (0.04 if v >= 0 else -0.08),
                     f"{v:+.2f}", ha="center",
                     va="bottom" if v >= 0 else "top",
                     fontsize=8.5, fontweight="bold", color=COLORS[g])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(stages_x, fontsize=9)
    ax.set_ylabel("Cohen's d (childhood vs adolescence, fuzzy boundary)\n"
                   "positive = drops with age (expected biology)")
    ax.set_title(
        "C3+ child→adolescent Cohen's d at each correction stage\n"
        f"(d averaged across boundary ages {list(FUZZY_BOUNDARIES)} y)",
        fontsize=11)
    ax.legend(loc="upper left", frameon=False)

    # Robust annotation: locate the PsychAD-V3 bar in the final stage and
    # arrow to its top.
    psy_final = df_bar[(df_bar["group"] == "PsychAD-V3") & (df_bar["stage"] == stages_x[-1])]["d"].iloc[0]
    psy_native = df_bar[(df_bar["group"] == "PsychAD-V3") & (df_bar["stage"] == stages_x[0])]["d"].iloc[0]
    psy_marker = df_bar[(df_bar["group"] == "PsychAD-V3") & (df_bar["stage"] == stages_x[1])]["d"].iloc[0]
    psy_pcc = df_bar[(df_bar["group"] == "PsychAD-V3") & (df_bar["stage"] == stages_x[2])]["d"].iloc[0]
    # x-position of PsychAD-V3 bar at the final stage (i=0 → -1*bar_w)
    psy_x_final = (len(stages_x) - 1) + (0 - 1) * bar_w
    ymin = min(0, min(df_bar["d"].dropna())) - 0.2
    ymax = max(df_bar["d"].dropna()) + 0.4
    ax.annotate(
        f"PsychAD-V3 progresses:\n"
        f"{psy_native:+.2f} → {psy_marker:+.2f} → {psy_pcc:+.2f} → "
        f"{psy_final:+.2f}\n(sign flips at the last step)",
        xy=(psy_x_final, psy_final),
        xytext=(psy_x_final - 0.6, max(psy_final + 0.9, 1.2)),
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.3),
        fontsize=9, fontweight="bold", color="#C0392B", ha="right")
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m2_correction_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved m2_correction_progression.png")

    return {
        "stage_a": stage_a_map, "stage_b": stage_b_map,
        "stage_c": stage_b, "stage_d": stage_c,
    }


# ---------------------------------------------------------------------------
# Plot 3 — per-cell UMI distributions
# ---------------------------------------------------------------------------

def plot_depth_distributions(per_cell):
    """Two panels: log-scale x axis (with human-readable UMI tick labels)
    and linear-scale x axis."""
    lo, hi = MAIN_WINDOW[1], MAIN_WINDOW[2]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # --- panel 1: log scale, but show UMI numbers on x-axis ---
    ax = axes[0]
    bins = np.linspace(2, 5.5, 70)
    for g in GROUPS:
        sub = per_cell[per_cell["group"] == g]
        if EXCLUDE_DONORS:
            sub = sub[~sub["individual"].isin(EXCLUDE_DONORS)]
        if len(sub) == 0: continue
        log_u = np.log10(sub["total_umi"].clip(lower=1))
        ax.hist(log_u, bins=bins, alpha=0.45, color=COLORS[g], density=True,
                 label=f"{g}  (n={len(sub):,}; median={int(sub['total_umi'].median()):,} UMI)")
    ax.axvspan(np.log10(lo), np.log10(hi), color="grey", alpha=0.15)
    ax.axvline(np.log10(lo), color="grey", lw=1, ls=":",
                label=f"matched window: {lo}-{hi} UMI")
    ax.axvline(np.log10(hi), color="grey", lw=1, ls=":")
    # Human-readable UMI ticks on a log axis
    umi_ticks = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]
    tick_labels = ["100", "300", "1k", "3k", "10k", "30k", "100k", "300k"]
    ax.set_xticks(np.log10(umi_ticks))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("per-cell total UMI (log scale)")
    ax.set_ylabel("density")
    ax.set_title("Log-scale view")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_xlim(2, 5.5)

    # --- panel 2: linear scale (clipped to 0-60k for visibility) ---
    ax = axes[1]
    bins_lin = np.linspace(0, 60000, 80)
    for g in GROUPS:
        sub = per_cell[per_cell["group"] == g]
        if EXCLUDE_DONORS:
            sub = sub[~sub["individual"].isin(EXCLUDE_DONORS)]
        if len(sub) == 0: continue
        ax.hist(sub["total_umi"].clip(upper=60000), bins=bins_lin,
                 alpha=0.45, color=COLORS[g], density=True,
                 label=f"{g}  (median {int(sub['total_umi'].median()):,})")
    ax.axvspan(lo, hi, color="grey", alpha=0.15)
    ax.axvline(lo, color="grey", lw=1, ls=":")
    ax.axvline(hi, color="grey", lw=1, ls=":")
    ax.set_xlabel("per-cell total UMI (linear scale; clipped at 60k)")
    ax.set_ylabel("density")
    ax.set_title("Linear-scale view")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax.set_xlim(0, 60000)
    # nicer tick formatting
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))

    fig.suptitle("Per-cell UMI distribution differs by chemistry × dataset "
                  f"(shaded grey = main matched-depth window {lo}-{hi} UMI)",
                  y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m3_depth_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved m3_depth_distributions.png")


# ---------------------------------------------------------------------------
# Plot 4 — per-layer d, no-depth-match vs depth-matched
# ---------------------------------------------------------------------------

def plot_per_layer_d(per_cell):
    layers = ["upper", "L5_ET", "L6_CT", "L6_IT"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, g in zip(axes, GROUPS):
        before, after = [], []
        for L in layers:
            don_before = aggregate_donors(per_cell, g, layer=L)
            don_after  = aggregate_donors(per_cell, g,
                                            umi_lo=MAIN_WINDOW[1],
                                            umi_hi=MAIN_WINDOW[2], layer=L)
            d_b, _ = fuzzy_d(don_before)
            d_a, _ = fuzzy_d(don_after)
            before.append(d_b); after.append(d_a)
        x = np.arange(len(layers))
        ax.bar(x - 0.2, before, 0.4, color=COLORS[g], alpha=0.45,
                label="per-cell-CPM\n(no depth match)")
        ax.bar(x + 0.2, after, 0.4, color=COLORS[g],
                label="+ depth-matched\n(3-12 k UMI)")
        for j_, (b_, a_) in enumerate(zip(before, after)):
            if np.isfinite(b_):
                ax.text(j_ - 0.2, b_ + (0.04 if b_ >= 0 else -0.06),
                         f"{b_:+.2f}", ha="center", fontsize=8,
                         color=COLORS[g], alpha=0.7)
            if np.isfinite(a_):
                ax.text(j_ + 0.2, a_ + (0.04 if a_ >= 0 else -0.06),
                         f"{a_:+.2f}", ha="center", fontsize=8,
                         color=COLORS[g], fontweight="bold")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(layers)
        ax.set_title(g, color=COLORS[g])
        if g == GROUPS[0]:
            ax.set_ylabel("Cohen's d (fuzzy boundary)")
        ax.set_ylim(-1.4, 2.6)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Per-layer C3+ d: depth-matching flips PsychAD-V3 to agree with Velmeshev",
                  y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m4_per_layer_d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved m4_per_layer_d.png")


# ---------------------------------------------------------------------------
# Plot 5 — layer composition at matched depth (all 3 groups)
# ---------------------------------------------------------------------------

def plot_layer_composition(per_cell):
    """Compute composition fresh from cache (so Vel-V2 is included)."""
    lo, hi = MAIN_WINDOW[1], MAIN_WINDOW[2]
    sub = per_cell[(per_cell["total_umi"] >= lo) & (per_cell["total_umi"] < hi)].copy()
    # Use fuzzy MAIN boundary 10 for composition split (could average but
    # composition shifts are slower than d so a representative boundary
    # is fine).
    BOUND = 10
    sub["stage"] = np.where(sub["age_years"] < BOUND, "child", "adol")

    rows = []
    for g in GROUPS:
        gs = sub[sub["group"] == g]
        # per-donor fraction first → mean across donors
        per_donor = (gs.groupby(["individual", "stage"], observed=True)["layer"]
                       .value_counts(normalize=True).rename("frac").reset_index())
        comp = (per_donor.groupby(["stage", "layer"], observed=True)["frac"]
                          .mean().reset_index())
        comp["group"] = g
        rows.append(comp)
    comp_all = pd.concat(rows, ignore_index=True)
    comp_all.to_csv(OUT_DIR / "m5_layer_composition_data.csv", index=False)

    layers = ["upper", "L5_ET", "L6_CT", "L6_IT", "ambiguous"]
    layer_colors = {"upper": "#1f77b4", "L5_ET": "#ff7f0e",
                     "L6_CT": "#2ca02c", "L6_IT": "#d62728",
                     "ambiguous": "#7F7F7F"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, g in zip(axes, GROUPS):
        comp_g = comp_all[comp_all["group"] == g]
        stages_ = ["child", "adol"]
        bottom = np.zeros(len(stages_))
        for L in layers:
            vals = [float(comp_g[(comp_g["stage"] == s)
                                  & (comp_g["layer"] == L)]["frac"].sum())
                    for s in stages_]
            ax.bar(stages_, vals, bottom=bottom, color=layer_colors[L],
                    label=L if g == GROUPS[0] else None, width=0.6)
            for k_, v in enumerate(vals):
                if v > 0.05:
                    ax.text(k_, bottom[k_] + v/2, f"{v*100:.0f}%",
                             ha="center", va="center", fontsize=9,
                             color="white", fontweight="bold")
            bottom += np.array(vals)
        ax.set_title(g, color=COLORS[g])
        ax.set_ylim(0, 1.05)
        if g == GROUPS[0]:
            ax.set_ylabel("fraction of ExN cells")
            ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=5,
                       bbox_to_anchor=(0, -0.28))
    fig.suptitle(
        f"Layer composition at matched depth ({lo//1000}-{hi//1000} k UMI; "
        f"child=<{BOUND} y, adol=≥{BOUND} y)",
        y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m5_layer_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved m5_layer_composition.png")


# ---------------------------------------------------------------------------
# Plot 6 — continuous trends, 3 panels, age-bin smoothing
# ---------------------------------------------------------------------------

def age_bin_moving_average(age, score, bin_width=4.0, step=0.5,
                            age_lo=AGE_LO, age_hi=AGE_HI, min_donors=3):
    """Compute a moving average over age bins of width `bin_width` years,
    stepped at `step` years. Each output point at age x is the mean of
    donor scores whose age falls in [x - w/2, x + w/2]."""
    age = np.asarray(age); score = np.asarray(score)
    xs, ys = [], []
    x = age_lo + bin_width / 2
    while x <= age_hi - bin_width / 2 + 1e-9:
        m = (age >= x - bin_width / 2) & (age < x + bin_width / 2)
        if m.sum() >= min_donors:
            xs.append(x); ys.append(score[m].mean())
        x += step
    return np.array(xs), np.array(ys)


def plot_continuous_trends(stage_donors):
    """3-panel continuous trend plot showing donor scores at each
    correction stage. `stage_donors` is the dict returned by
    plot_correction_progression."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        ("(A) Original analysis\n(native cell_class = 'Excitatory', sum-then-CPM)",
         stage_donors["stage_a"]),
        ("(B) After marker-annotation fix\n(marker_annotation, sum-then-CPM)",
         stage_donors["stage_b"]),
        ("(C) After all critical fixes\n(marker + per-cell-CPM mean + 3-12 k UMI window)",
         stage_donors["stage_d"]),
    ]
    for ax, (title, smap) in zip(axes, panels):
        for g in GROUPS:
            df = smap.get(g)
            if df is None or len(df) == 0:
                continue
            ax.scatter(df["age_years"], df["score"], s=22, color=COLORS[g],
                        alpha=0.55, edgecolor="none",
                        label=f"{g}  (n={len(df)})")
            xs, ys = age_bin_moving_average(df["age_years"].values,
                                              df["score"].values,
                                              bin_width=4.0, step=0.5,
                                              min_donors=3)
            if len(xs) > 1:
                ax.plot(xs, ys, color=COLORS[g], lw=2.2, alpha=0.85)
        # mark fuzzy boundary range
        ax.axvspan(min(FUZZY_BOUNDARIES), max(FUZZY_BOUNDARIES),
                    color="grey", alpha=0.12,
                    label=f"fuzzy boundary range {list(FUZZY_BOUNDARIES)}")
        ax.set_xlabel("donor age (years)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.set_xlim(0.5, 26)
    axes[0].set_ylabel("C3+ score per donor")
    fig.suptitle(
        "C3+ developmental trajectory across cohorts and correction stages\n"
        "Trend lines = 4-year moving average; shaded band = fuzzy childhood/adolescence boundary range",
        fontsize=11, y=1.04, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "m6_continuous_trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved m6_continuous_trends.png")

    # save the per-donor data tables for traceability
    all_rows = []
    for label, key in [("A_native_cell_class", "stage_a"),
                        ("B_marker_sumCPM",    "stage_b"),
                        ("C_marker_perCellCPM_depthMatched", "stage_d")]:
        for g, df in stage_donors[key].items():
            if df is None or len(df) == 0: continue
            t = df.copy(); t["group"] = g; t["panel"] = label
            all_rows.append(t)
    pd.concat(all_rows, ignore_index=True).to_csv(
        OUT_DIR / "m6_continuous_trends_data.csv", index=False)


# ---------------------------------------------------------------------------
# Plot 7 — multi-window trajectories AND window-bounds d table
# ---------------------------------------------------------------------------

def plot_multi_window_trajectories(per_cell):
    """For each candidate depth window, show donor-level trajectories
    under per-cell-CPM mean projection (3×3 grid), then a barplot of
    fuzzy d across windows × groups below."""
    import matplotlib.gridspec as gridspec

    n_win = len(DEPTH_WINDOWS)
    ncol = 3
    nrow_traj = int(np.ceil(n_win / ncol))
    fig = plt.figure(figsize=(16, 4 * nrow_traj + 4.5))
    gs = gridspec.GridSpec(nrow_traj + 1, ncol, figure=fig,
                            height_ratios=[3] * nrow_traj + [2.2],
                            hspace=0.5, wspace=0.25)

    # Trajectory panels (3×3)
    traj_axes = []
    for i in range(n_win):
        r, c = divmod(i, ncol)
        ax = fig.add_subplot(gs[r, c])
        traj_axes.append(ax)

    # Window-bounds × group fuzzy d table
    rows = []
    score_lims = []
    for ax, (w_label, lo, hi) in zip(traj_axes, DEPTH_WINDOWS):
        for g in GROUPS:
            df = aggregate_donors(per_cell, g, umi_lo=lo, umi_hi=hi)
            if len(df) == 0:
                continue
            ax.scatter(df["age_years"], df["score"], s=22, color=COLORS[g],
                        alpha=0.55, edgecolor="none",
                        label=f"{g}  (n={len(df)})")
            xs, ys = age_bin_moving_average(df["age_years"].values,
                                              df["score"].values,
                                              bin_width=4.0, step=0.5,
                                              min_donors=3)
            if len(xs) > 1:
                ax.plot(xs, ys, color=COLORS[g], lw=2.0, alpha=0.85)
            d_fuzzy, per_b = fuzzy_d(df)
            rows.append({"window": w_label,
                         "lo": lo, "hi": hi, "group": g,
                         "n_donors": int(len(df)),
                         "mean_cells_per_donor": float(df["n_cells"].mean()),
                         "fuzzy_d": d_fuzzy,
                         **{f"d_b{int(r['boundary'])}": r["cohens_d"] for r in per_b}})
            score_lims.extend(df["score"].tolist())
        ax.axvspan(min(FUZZY_BOUNDARIES), max(FUZZY_BOUNDARIES),
                    color="grey", alpha=0.12)
        title = (f"window: {w_label}" if lo is not None
                  else "window: none (all cells)")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("donor age (years)")
        ax.legend(loc="best", fontsize=7.5, frameon=False)
        ax.set_xlim(0.5, 26)
    # share y-range across trajectory panels
    if score_lims:
        ylo, yhi = np.percentile(score_lims, [1, 99])
        for ax in traj_axes:
            ax.set_ylim(ylo - 0.05 * (yhi - ylo), yhi + 0.05 * (yhi - ylo))
    for i, ax in enumerate(traj_axes):
        if i % ncol == 0:
            ax.set_ylabel("C3+ score per donor")

    df_w = pd.DataFrame(rows)

    # ----- bottom row: barplot of fuzzy d × window × group -----
    bar_ax = fig.add_subplot(gs[nrow_traj, :])
    window_order = [w[0] for w in DEPTH_WINDOWS]
    x = np.arange(len(window_order))
    bar_w = 0.26
    for i, g in enumerate(GROUPS):
        vals = []
        for w in window_order:
            sub = df_w[(df_w["window"] == w) & (df_w["group"] == g)]
            vals.append(float(sub["fuzzy_d"].iloc[0]) if len(sub) else np.nan)
        bar_ax.bar(x + (i - 1) * bar_w, vals, bar_w, label=g, color=COLORS[g])
        for j_, v in enumerate(vals):
            if np.isnan(v): continue
            bar_ax.text(x[j_] + (i - 1) * bar_w,
                         v + (0.04 if v >= 0 else -0.08),
                         f"{v:+.2f}", ha="center",
                         va="bottom" if v >= 0 else "top",
                         fontsize=7.5, fontweight="bold", color=COLORS[g])
    bar_ax.axhline(0, color="k", lw=0.5)
    bar_ax.set_xticks(x); bar_ax.set_xticklabels(window_order, fontsize=9)
    bar_ax.set_ylabel("Cohen's d (fuzzy boundary)")
    bar_ax.set_title("Fuzzy d per depth window × cohort", fontsize=10)
    bar_ax.legend(loc="upper right", frameon=False)
    yvals = df_w["fuzzy_d"].dropna()
    if len(yvals):
        bar_ax.set_ylim(min(yvals.min() - 0.3, -0.5),
                          max(yvals.max() + 0.3, 1.0))

    fig.suptitle(
        "Per-donor C3+ trajectories under different per-cell UMI windows "
        f"(marker annotation + per-cell-CPM mean; "
        f"fuzzy boundary {list(FUZZY_BOUNDARIES)})",
        fontsize=11, y=0.995, fontweight="bold")
    fig.savefig(OUT_DIR / "m7_multi_window_trajectories.png", dpi=150,
                 bbox_inches="tight")
    plt.close(fig)
    print("saved m7_multi_window_trajectories.png")

    df_w.to_csv(OUT_DIR / "m_window_bounds_d.csv", index=False)
    print("saved m_window_bounds_d.csv")

    pivot = df_w.pivot_table(index=["window", "lo", "hi"],
                              columns="group", values="fuzzy_d").round(3)
    pivot.to_csv(OUT_DIR / "m_window_bounds_d_pivot.csv")
    print("saved m_window_bounds_d_pivot.csv")
    print("\nWindow-bounds fuzzy d table:")
    print(pivot.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    per_cell = build_per_cell_cache()
    plot_cell_class_problem()
    stage_donors = plot_correction_progression(per_cell)
    plot_depth_distributions(per_cell)
    plot_per_layer_d(per_cell)
    plot_layer_composition(per_cell)
    plot_continuous_trends(stage_donors)
    plot_multi_window_trajectories(per_cell)
    print("\nAll plots and tables saved to", OUT_DIR)


if __name__ == "__main__":
    main()
