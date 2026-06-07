#!/usr/bin/env python3
"""
Group G — depth dependence of marker-based classification.

For PsychAD-V3, Vel-V2 and Vel-V3 cells in the developmental window
(age 1-25 y, PFC-only), extract raw per-cell counts of:
  RBFOX3, DCX, RBFOX1   (the ExN classification gates)
  GAD1, GAD2, SLC32A1   (the InN gates)
plus total UMI per cell (proxy for sequencing depth).

The marker-based classifier in code/annotation_by_markers.py uses
absolute thresholds: RBFOX3≥1 → ExN_mature, DCX-only ≥1 → ExN_immature,
RBFOX1≥1 → ExN_weak (after glia screen). These thresholds are
inherently library-size dependent: in a shallow V2 cell, RBFOX3=0
is far more likely just from undersampling, so the cell is pushed
into the "immature" or "weak" bins even when its biological identity
is mature.

This script quantifies that bias:

  G1. Distribution of total UMI per cell in each group.
  G2. Distribution of raw counts of RBFOX3, DCX, RBFOX1.
  G3. P(RBFOX3 ≥ 1) and P(DCX ≥ 1) and P(RBFOX1 ≥ 1) as a function
      of total UMI bin — for each group separately.
  G4. Classification breakdown (ExN_mature / immature / weak) at
      matched UMI bins. If the breakdown shifts dramatically with
      UMI but not with biology, the classification is depth-biased.
  G5. PsychAD-V3 internal: within ExN_mature cells, distribution of
      RBFOX3 counts vs total UMI. Show that even within "mature"
      there is a continuum.

Inputs:
  Vel:     /home/rajd2/rds/.../Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad
  PsychAD: /home/rajd2/rds/.../PsychAD_noage_tuning5/scvi_output/integrated.h5ad

Outputs:
  g_per_cell_markers.parquet            cell × (markers, totals, group, age, label)
  g_marker_distributions.png            histograms G1+G2
  g_detection_vs_depth.png              G3 + G4
  g_classification_vs_depth.csv         G4 tabular
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
from _lib import OUT_DIR, CHILD, ADOL

INPUTS = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/scvi_output/integrated.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad",
}
MANUAL = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
}

# Ensembl IDs (the integrated.h5ads use Ensembl as var_names)
MARKERS = {
    "RBFOX3":  "ENSG00000167281",  # v98 stable id
    "DCX":     "ENSG00000077279",
    "RBFOX1":  "ENSG00000078328",
    "GAD1":    "ENSG00000128683",
    "GAD2":    "ENSG00000136750",
    "SLC32A1": "ENSG00000101438",
}

DEV_LOW, DEV_HIGH = CHILD[0], ADOL[1]   # 1-25 y


def resolve_marker_ids(adata: ad.AnnData) -> dict:
    """Map gene symbol → ensembl_id present in adata.var_names."""
    out = {}
    has_feature = "feature_name" in adata.var.columns
    if has_feature:
        sym_to_ens = dict(zip(adata.var["feature_name"].values, adata.var_names))
    else:
        sym_to_ens = {}
    for sym, default_ens in MARKERS.items():
        if default_ens in adata.var_names:
            out[sym] = default_ens
        elif sym in sym_to_ens:
            out[sym] = sym_to_ens[sym]
        else:
            print(f"  WARNING: {sym} not found (tried {default_ens})")
    return out


def extract_per_cell(name: str, h5ad_path: str,
                      manual_path: str) -> pd.DataFrame:
    print(f"\n=== {name}: loading {h5ad_path}")
    a = ad.read_h5ad(h5ad_path, backed="r")
    print(f"  shape: {a.shape}")
    print(f"  obs cols: {list(a.obs.columns)[:25]}")

    age_col = "age_years" if "age_years" in a.obs.columns else None
    chem_col = ("chemistry" if "chemistry" in a.obs.columns else
                "source-chemistry" if "source-chemistry" in a.obs.columns else
                None)
    print(f"  age_col: {age_col}   chem_col: {chem_col}")

    obs = a.obs.copy()
    if age_col and age_col != "age_years":
        obs["age_years"] = obs[age_col]
    if chem_col and chem_col != "chemistry":
        obs["chemistry"] = obs[chem_col]
    if "chemistry" not in obs.columns:
        obs["chemistry"] = "unknown"

    # restrict to developmental window for tractable plot density
    age = pd.to_numeric(obs["age_years"], errors="coerce")
    mask = (age >= DEV_LOW) & (age < DEV_HIGH)
    obs_idx = np.where(mask.values)[0]
    print(f"  cells in age window [{DEV_LOW}, {DEV_HIGH}): {len(obs_idx):,} / {a.n_obs:,}")

    # resolve marker ensembl ids
    found = resolve_marker_ids(a)
    sym_list = list(found.keys())
    col_idx = np.array([a.var_names.get_loc(found[s]) for s in sym_list],
                       dtype=np.int64)
    print(f"  resolved markers: {found}")

    # Pull raw counts for these columns from layers['counts']
    if "counts" in a.layers:
        layer = a.layers["counts"]
        print(f"  using layers['counts'] (type {type(layer).__name__})")
    else:
        layer = a.X
        print(f"  WARNING: no layers['counts'], using .X")

    # Use scipy sparse fancy indexing if sparse, else dense
    if sp.issparse(layer):
        # Backed CSR — read column slice; this also fully loads into mem for
        # the selected columns. With only 6 columns × ~700k cells = small.
        block = layer[:, col_idx]
        block = sp.csr_matrix(block)
        block = block[obs_idx, :].toarray()
    else:
        block = np.asarray(layer[obs_idx][:, col_idx])

    # Also total UMI per cell (over ALL genes)
    print("  computing per-cell total UMI ...")
    if "n_counts" in obs.columns:
        total_umi = obs["n_counts"].values[obs_idx]
    elif "total_counts" in obs.columns:
        total_umi = obs["total_counts"].values[obs_idx]
    else:
        # compute on the fly via chunked row sum
        if sp.issparse(layer):
            full = layer[obs_idx, :]
            total_umi = np.asarray(full.sum(axis=1)).ravel()
        else:
            total_umi = np.asarray(layer[obs_idx]).sum(axis=1)
    total_umi = np.asarray(total_umi).ravel().astype(np.float64)

    df = pd.DataFrame(block, columns=sym_list)
    df["obs_name"]   = a.obs_names.values[obs_idx]
    df["age_years"]  = age.values[obs_idx]
    df["chemistry"]  = obs["chemistry"].values[obs_idx]
    df["total_umi"]  = total_umi
    df["dataset"]    = name

    # Join the marker_annotation from the manual parquet
    ma = pd.read_parquet(manual_path)
    df = df.set_index("obs_name").join(ma, how="left").reset_index()
    df["chemistry"] = df["chemistry"].astype(str).str.upper().str.replace(
        ".*-?(V2|V3).*", lambda m: m.group(1), regex=True)
    # When chemistry is something like 'PSYCHAD-V3', collapse to V3
    df["chemistry"] = df["chemistry"].where(df["chemistry"].isin(["V2", "V3"]),
                                              "unknown")
    print(f"  per-cell rows: {len(df):,}  "
          f"chemistry counts: {df['chemistry'].value_counts().to_dict()}")
    print(f"  marker_annotation: "
          f"{df['marker_annotation'].value_counts().to_dict()}")
    return df


def main():
    parts = []
    for name in ("PsychAD", "Velmeshev"):
        parts.append(extract_per_cell(name, INPUTS[name], MANUAL[name]))
    df = pd.concat(parts, ignore_index=True)
    df["group"] = df["dataset"] + "-" + df["chemistry"]
    df = df[df["group"].isin(["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"])]
    print(f"\nFinal per-cell rows: {len(df):,}")
    print(df["group"].value_counts())

    df.to_parquet(OUT_DIR / "g_per_cell_markers.parquet", index=False)

    # ---------- G1: total UMI distribution ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, group in zip(axes, ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]):
        sub = df[df["group"] == group]
        ax.hist(np.log10(sub["total_umi"].clip(lower=1)), bins=60, color="C0",
                 alpha=0.8)
        med = sub["total_umi"].median()
        ax.axvline(np.log10(med), color="red", lw=1.5,
                    label=f"median = {int(med):,}")
        ax.set_title(f"{group}  (n={len(sub):,})")
        ax.set_xlabel("log10(total UMI)")
        ax.legend()
    axes[0].set_ylabel("# cells")
    fig.suptitle("G1: per-cell total UMI distribution (age 1-25 y)", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "g1_total_umi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- G2: marker count distributions ----------
    markers = ["RBFOX3", "DCX", "RBFOX1"]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharey="row")
    for col, group in enumerate(["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]):
        sub = df[df["group"] == group]
        for row, m in enumerate(markers):
            ax = axes[row, col]
            v = sub[m].clip(upper=6).values  # truncate tail
            ax.hist(v, bins=np.arange(-0.5, 7.5, 1), color="C0", alpha=0.8)
            ax.axvline(0.5, color="red", ls="--", lw=1,
                        label=f"≥1 thresh: {(sub[m]>=1).mean()*100:.1f}%")
            ax.set_title(f"{m}  in  {group}")
            ax.set_xlabel(f"raw count (≤6 shown)")
            ax.legend()
    fig.suptitle("G2: per-cell raw count of ExN gate markers", y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "g2_marker_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- G3: P(marker ≥ 1) vs total UMI bin ----------
    umi_bins = [0, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 40000, 100000]
    df["umi_bin"] = pd.cut(df["total_umi"], umi_bins, right=False)

    rows = []
    for group in ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]:
        sub = df[df["group"] == group]
        for m in markers:
            g = (sub.groupby("umi_bin", observed=True)
                    .agg(n_cells=(m, "size"),
                         frac_positive=(m, lambda x: (x >= 1).mean()))
                    .reset_index())
            g["group"]  = group
            g["marker"] = m
            rows.append(g)
    det_long = pd.concat(rows, ignore_index=True)
    det_long.to_csv(OUT_DIR / "g3_detection_vs_depth.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, m in zip(axes, markers):
        for group, color in [("PsychAD-V3", "C0"),
                              ("Velmeshev-V2", "C3"),
                              ("Velmeshev-V3", "C2")]:
            sub = det_long[(det_long["group"] == group) & (det_long["marker"] == m)]
            if sub.empty: continue
            x = sub["umi_bin"].astype(str).str.extract(r"\[(\d+),")[0].astype(float)
            ax.plot(x, sub["frac_positive"], "-o", color=color, label=group)
        ax.set_xscale("log")
        ax.set_xlabel("UMI bin lower edge")
        ax.set_ylabel(f"P({m} ≥ 1)")
        ax.set_title(f"P({m} ≥ 1) vs depth")
        ax.set_ylim(0, 1.05)
        ax.legend()
    fig.suptitle("G3: marker-detection probability is depth-driven", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "g3_detection_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- G4: classification mix per UMI bin ----------
    cls = ["ExN_mature", "ExN_immature", "ExN_weak"]
    df_exn = df[df["marker_annotation"].isin(cls)].copy()
    rows = []
    for group in ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]:
        sub = df_exn[df_exn["group"] == group]
        g = (sub.groupby(["umi_bin", "marker_annotation"], observed=True)
                .size().unstack(fill_value=0))
        g["total"] = g.sum(axis=1)
        for c in cls:
            if c not in g.columns:
                g[c] = 0
            g[f"frac_{c}"] = g[c] / g["total"].clip(lower=1)
        g = g.reset_index()
        g["group"] = group
        rows.append(g)
    cls_long = pd.concat(rows, ignore_index=True)
    cls_long.to_csv(OUT_DIR / "g4_classification_vs_depth.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    colors = {"ExN_mature": "#1f77b4", "ExN_immature": "#ff7f0e", "ExN_weak": "#2ca02c"}
    for ax, group in zip(axes, ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]):
        sub = cls_long[cls_long["group"] == group]
        x = sub["umi_bin"].astype(str).str.extract(r"\[(\d+),")[0].astype(float)
        bottom = np.zeros(len(sub))
        for c in cls:
            ax.bar(np.log10(x.values + 1), sub[f"frac_{c}"], bottom=bottom,
                    width=0.25, color=colors[c], label=c)
            bottom = bottom + sub[f"frac_{c}"].values
        ax.set_title(f"{group}")
        ax.set_xlabel("log10(UMI bin lower edge + 1)")
        ax.set_ylabel("fraction of ExN cells")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("G4: ExN subtype mix vs sequencing depth", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "g4_classification_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- G5: aggregate stats ----------
    summary = (df.groupby("group")
                 .agg(n_cells=("RBFOX3", "size"),
                      median_umi=("total_umi", "median"),
                      mean_umi=("total_umi", "mean"),
                      p_RBFOX3=("RBFOX3", lambda x: (x >= 1).mean()),
                      p_DCX=("DCX", lambda x: (x >= 1).mean()),
                      p_RBFOX1=("RBFOX1", lambda x: (x >= 1).mean()),
                      p_GAD1=("GAD1", lambda x: (x >= 1).mean()),
                      mean_RBFOX3=("RBFOX3", "mean"))
                 .round(4))
    summary.to_csv(OUT_DIR / "g_group_summary.csv")
    print("\n=== Group summary ===")
    print(summary.to_string())

    cls_mix = (df_exn.groupby("group")["marker_annotation"]
                 .value_counts(normalize=True).unstack().round(4))
    print("\n=== ExN subtype mix per group ===")
    print(cls_mix.to_string())
    cls_mix.to_csv(OUT_DIR / "g_exn_mix_per_group.csv")

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
