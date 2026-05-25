"""
Diagnostic script: compare PsychAD <1y label-transfer behaviour across two pipeline runs.

  age_run:    VelWangPsychAD_semisup3_age_tuning5   (age_years as scVI continuous covariate)
  no_age_run: VelWangPsychAD_semisup3_tuning5        (no age covariate; otherwise identical)

Hypotheses under test:
  H1 — age covariate erased developmental signal in z  [REJECTED by S1 test: no change]
  H2 — PsychAD-adult labels dominate scANVI supervision (460k/723k labelled cells)
  H5 — Vel Interneurons removal left IN class PsychAD-only
  H6 — PsychAD 5-18y developmental cells labelled adult-IN, corrupting classifier
  H7 — PsychAD subclass is IN-heavy by default

Run with:
  sbatch --mem=200G --time=01:00:00 scripts/run_script.sh scripts/diagnose_psychad_relabel.py

Outputs to scripts/relabel_comparison/.
"""
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

# ── Paths ─────────────────────────────────────────────────────────────────────
RDS = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
RUNS = {
    "age_run": f"{RDS}/Cam_snRNAseq/integrated/VelWangPsychAD_semisup3_age_tuning5",
    "no_age_run": f"{RDS}/Cam_snRNAseq/integrated/VelWangPsychAD_semisup3_tuning5",
}
OUTPUT_DIR = Path("/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/relabel_comparison")

# ── Markers ───────────────────────────────────────────────────────────────────
EN_MARKERS = ["SLC17A7", "NEUROD2", "NEUROD6", "RBFOX3", "SATB2", "TBR1"]
IN_MARKERS = ["GAD1", "GAD2", "SLC32A1", "DLX1", "DLX2", "LHX6"]
ALL_MARKERS = EN_MARKERS + IN_MARKERS

# Age bins matching composition_check.py (right=False → left-inclusive)
AGE_BINS   = [-np.inf, 0, 1, 5, 18, 30, 50, np.inf]
AGE_LABELS = ["prenatal", "<1y", "1-5y", "5-18y", "18-30y", "30-50y", "50+y"]

K_NEIGHBORS = 15

# ── Error accumulation ────────────────────────────────────────────────────────
_errors = []

def _safe(name, fn, *args, **kwargs):
    """Run fn; on exception accumulate error and return None."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        msg = f"[{name}] FAILED:\n{traceback.format_exc()}"
        _errors.append(msg)
        print(f"  WARNING: {name} failed — {e}")
        return None


# ── Cell masks ────────────────────────────────────────────────────────────────
def _age_bin(series):
    return pd.cut(series.astype(float), bins=AGE_BINS, labels=AGE_LABELS, right=False)

def _psychad_mask(obs):
    return (obs["source-chemistry"] == "PSYCHAD-V3").values

def _psychad_under1y(obs):
    """Match composition_check <1y bin: age_years in [0, 1)."""
    return _psychad_mask(obs) & (obs["age_years"].astype(float) >= 0).values & \
           (obs["age_years"].astype(float) < 1).values


# ── D1: Predictions by age bin ────────────────────────────────────────────────
def d1_predictions_by_age_bin(obs, out_csv):
    """What is scANVI predicting for each PsychAD-V3 age bin?"""
    obs2 = obs.copy()
    obs2["age_bin"] = _age_bin(obs2["age_years"])
    psychad = obs2[_psychad_mask(obs2)]

    rows = []
    for (age_bin, predicted), grp in psychad.groupby(
            ["age_bin", "cell_type_aligned"], observed=True):
        rows.append({
            "age_bin": str(age_bin),
            "predicted_label": predicted,
            "n_cells": len(grp),
            "mean_confidence": grp["cell_type_aligned_confidence"].mean(),
            "frac_low_conf": (grp["cell_type_aligned_confidence"] < 0.5).mean(),
        })
    df = pd.DataFrame(rows)
    totals = psychad.groupby("age_bin", observed=True).size().rename("bin_total").reset_index()
    totals["age_bin"] = totals["age_bin"].astype(str)
    df = df.merge(totals, on="age_bin", how="left")
    df["frac_of_bin"] = df["n_cells"] / df["bin_total"]
    df = df.sort_values(["age_bin", "n_cells"], ascending=[True, False])
    df.to_csv(out_csv, index=False)

    for ab in ["<1y", "1-5y", "5-18y"]:
        sub = df[df["age_bin"] == ab].head(10)
        if len(sub):
            print(f"\n  PSYCHAD-V3 {ab} top predictions:")
            print(sub[["predicted_label", "n_cells", "frac_of_bin",
                        "mean_confidence"]].to_string(index=False))
    return df


# ── D2: Confidence summary ────────────────────────────────────────────────────
def d2_confidence_summary(obs, out_csv):
    """Confidence quartiles per (source-chemistry × age_bin)."""
    obs2 = obs.copy()
    obs2["age_bin"] = _age_bin(obs2["age_years"])
    focus = obs2[obs2["source-chemistry"].isin(
        ["PSYCHAD-V3", "VELMESHEV-V3", "WANG-multiome"])]

    rows = []
    for (sc, ab), grp in focus.groupby(
            ["source-chemistry", "age_bin"], observed=True):
        conf = grp["cell_type_aligned_confidence"]
        rows.append({
            "source_chemistry": sc,
            "age_bin": str(ab),
            "n_cells": len(grp),
            "P10": conf.quantile(0.10),
            "P25": conf.quantile(0.25),
            "P50": conf.quantile(0.50),
            "P75": conf.quantile(0.75),
            "P90": conf.quantile(0.90),
            "frac_below_0.5": (conf < 0.5).mean(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n  Confidence summary — PSYCHAD-V3:")
    sub = df[df["source_chemistry"] == "PSYCHAD-V3"]
    print(sub[["age_bin", "n_cells", "P25", "P50", "P75",
               "frac_below_0.5"]].to_string(index=False))
    return df


# ── D3: Predicted vs raw crosstab ─────────────────────────────────────────────
def d3_predicted_vs_raw(obs, out_csv):
    """Crosstab scANVI prediction vs PsychAD raw subclass for <1y Unknown cells."""
    mask = _psychad_under1y(obs)
    sub = obs[mask]
    n = mask.sum()
    print(f"\n  PsychAD-V3 <1y cells: {n:,}")

    if "cell_type_raw" not in obs.columns:
        print("  WARNING: cell_type_raw not in obs — skipping D3")
        return None

    # Check how many were labelled Unknown vs supervised
    if "cell_type_for_scanvi" in obs.columns:
        vc_scanvi = sub["cell_type_for_scanvi"].value_counts()
        print(f"  cell_type_for_scanvi distribution:")
        print(f"    {vc_scanvi.to_string()}")

    ct = pd.crosstab(sub["cell_type_aligned"], sub["cell_type_raw"])
    ct.to_csv(out_csv)

    # Row marginals (predicted totals)
    row_totals = ct.sum(axis=1).sort_values(ascending=False)
    print(f"\n  Predicted cell_type_aligned distribution (PsychAD <1y):")
    for lbl, n in row_totals.head(12).items():
        print(f"    {lbl:<45s} {n:6d}")

    print(f"\n  Top 5 raw subclass labels (cell_type_raw) for PsychAD <1y:")
    col_totals = ct.sum(axis=0).sort_values(ascending=False)
    for lbl, n in col_totals.head(5).items():
        print(f"    {lbl:<45s} {n:6d}")
    return ct


# ── D4: Marker gene expression ────────────────────────────────────────────────
def d4_marker_means(adata, obs, out_csv):
    """Mean EN/IN marker expression (scanvi_normalized) per source × age_bin × cell_class.

    Reads marker columns from the backed h5ad using h5py for efficient column access.
    If scanvi_normalized is stored as CSR sparse, falls back to chunked row reads.
    """
    import h5py

    # Identify marker indices in var
    var_names = adata.var_names.tolist()
    marker_idx = {g: var_names.index(g) for g in ALL_MARKERS if g in var_names}
    missing = [g for g in ALL_MARKERS if g not in var_names]
    if missing:
        print(f"  Markers absent from var: {missing}")
    if not marker_idx:
        print("  No markers found in var — skipping D4")
        return None
    gene_list = list(marker_idx.keys())
    gene_positions = np.array(list(marker_idx.values()))
    print(f"  Using {len(gene_list)} markers: {gene_list}")

    obs2 = obs.copy()
    obs2["age_bin"] = _age_bin(obs2["age_years"])

    # Restrict to the three source-chemistry groups we care about
    focus_mask = obs2["source-chemistry"].isin(
        ["PSYCHAD-V3", "VELMESHEV-V3", "WANG-multiome"]).values
    focus_idx = np.where(focus_mask)[0]  # int positions into obs

    # Determine layer name
    layer_name = ("scanvi_normalized" if "scanvi_normalized" in adata.layers
                  else "counts")
    print(f"  Reading layer '{layer_name}' for {len(focus_idx):,} cells × {len(gene_list)} markers...")

    # Try fast column-slice via h5py (works for dense layers)
    expr = None
    h5_path = adata.filename  # path to the backing HDF5 file

    if h5_path:
        try:
            with h5py.File(str(h5_path), "r") as f:
                key = f"layers/{layer_name}"
                node = f.get(key)
                if isinstance(node, h5py.Dataset):
                    # Dense array: full column slice then row select
                    cols = node[:, gene_positions]  # (N, n_markers), reads only these cols
                    expr = cols[focus_idx]  # (n_focus, n_markers)
                    print(f"  Fast h5py column read: {expr.shape}")
        except Exception as e:
            print(f"  h5py fast read failed ({e}), falling back to chunked load")

    if expr is None:
        # Fallback: chunked row reads (works for both sparse and dense)
        sorted_order = np.argsort(focus_idx)
        sorted_focus = focus_idx[sorted_order]
        chunk_size = 50_000
        pieces = []
        for start in range(0, len(sorted_focus), chunk_size):
            chunk_idx = sorted_focus[start:start + chunk_size]
            chunk = adata[chunk_idx, :]
            raw = chunk.layers.get(layer_name, chunk.X)
            if sp.issparse(raw):
                raw = raw.toarray()
            pieces.append(np.asarray(raw, dtype=np.float32)[:, gene_positions])
        expr_sorted = np.concatenate(pieces, axis=0)
        # Unsort back to focus_idx order
        unsort = np.argsort(sorted_order)
        expr = expr_sorted[unsort]
        print(f"  Chunked read complete: {expr.shape}")

    # Build per-group summary.
    # Tag each row with its position (0-based) in expr so groupby can map back.
    focus_obs = obs2.iloc[focus_idx].copy()
    focus_obs["_row_in_expr"] = np.arange(len(focus_idx))
    rows = []
    for (sc, ab, cc), grp in focus_obs.groupby(
            ["source-chemistry", "age_bin", "cell_class"], observed=True):
        grp_rows = grp["_row_in_expr"].values
        grp_expr = expr[grp_rows]  # (n_grp, n_markers)
        for i, g in enumerate(gene_list):
            rows.append({
                "source_chemistry": sc,
                "age_bin": str(ab),
                "cell_class": cc,
                "gene": g,
                "marker_type": "EN" if g in EN_MARKERS else "IN",
                "mean_expr": float(grp_expr[:, i].mean()),
                "n_cells": len(grp_rows),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Print key comparison: PsychAD <1y predicted Excitatory vs Inhibitory
    for pred_class in ["Excitatory", "Inhibitory"]:
        sub = df[
            (df["source_chemistry"] == "PSYCHAD-V3") &
            (df["age_bin"] == "<1y") &
            (df["cell_class"] == pred_class)
        ]
        if len(sub):
            print(f"\n  PSYCHAD-V3 <1y predicted '{pred_class}': mean marker expression")
            for _, row in sub.iterrows():
                print(f"    {row['gene']:<12s} ({row['marker_type']})  {row['mean_expr']:.4f}")

    # Also print Wang <1y Excitatory as reference
    sub_ref = df[
        (df["source_chemistry"] == "WANG-multiome") &
        (df["age_bin"] == "<1y") &
        (df["cell_class"] == "Excitatory")
    ]
    if len(sub_ref):
        print(f"\n  WANG-multiome <1y Excitatory (reference):")
        for _, row in sub_ref.iterrows():
            print(f"    {row['gene']:<12s} ({row['marker_type']})  {row['mean_expr']:.4f}")
    return df


# ── D5: Classifier confusion on labelled cells ────────────────────────────────
def d5_scanvi_confusion(obs, out_csv):
    """Crosstab true label (cell_type_for_scanvi) vs predicted (cell_type_aligned)
    for all cells that were supervised (not Unknown)."""
    if "cell_type_for_scanvi" not in obs.columns:
        print("  WARNING: cell_type_for_scanvi not in obs — skipping D5")
        return None

    labelled = obs[obs["cell_type_for_scanvi"] != "Unknown"]
    n = len(labelled)
    ct = pd.crosstab(labelled["cell_type_for_scanvi"],
                     labelled["cell_type_aligned"],
                     rownames=["true_label"],
                     colnames=["predicted_label"])
    ct.to_csv(out_csv)

    n_correct = sum(ct.loc[r, r] for r in ct.index if r in ct.columns)
    print(f"\n  Classifier accuracy on labelled cells: {n_correct:,}/{n:,} = {n_correct/n:.1%}")

    en_rows = [l for l in ct.index if "EN" in l or "Excitatory" in l]
    if en_rows:
        en_total = ct.loc[en_rows].sum().sum()
        en_correct = sum(ct.loc[r, r] for r in en_rows if r in ct.columns)
        print(f"  EN class accuracy: {en_correct:,}/{en_total:,} = "
              f"{en_correct/en_total:.1%}" if en_total > 0 else "  No EN labelled cells")

        # EN → IN cross-prediction (H5/H6 smoking gun)
        in_cols = [c for c in ct.columns if "IN" in c or "Inhibitory" in c]
        if in_cols and en_rows:
            en_to_in = ct.loc[en_rows, in_cols].sum().sum()
            print(f"  EN cells predicted as IN: {en_to_in:,}/{en_total:,} = "
                  f"{en_to_in/en_total:.1%}")

    # Source breakdown of labelled cells
    if "source-chemistry" in obs.columns:
        src_counts = labelled.groupby("source-chemistry").size()
        print(f"\n  Labelled cell count by source:")
        for src, n_src in src_counts.items():
            print(f"    {src:<25s} {n_src:8,}")
    return ct


# ── D6: Latent neighbour composition ─────────────────────────────────────────
def d6_latent_neighbors(adata, obs, out_csv, k=K_NEIGHBORS):
    """For PsychAD <1y cells, find K=15 nearest neighbours in X_scANVI and
    report their source/age composition. Returns (df, X_latent) so X_latent
    can be reused in the cross-run latent-movement analysis."""
    from sklearn.neighbors import NearestNeighbors

    if "X_scANVI" not in adata.obsm:
        print(f"  WARNING: X_scANVI not in obsm {list(adata.obsm.keys())} — skipping D6")
        return None, None

    print(f"  Loading X_scANVI ({adata.n_obs:,} cells × "
          f"{adata.obsm['X_scANVI'].shape[1]} dims)...")
    X = np.array(adata.obsm["X_scANVI"], dtype=np.float32)  # (N, 50), ~200 MB

    obs2 = obs.copy()
    obs2["age_bin"] = _age_bin(obs2["age_years"])

    target_mask = _psychad_under1y(obs2)
    target_idx = np.where(target_mask)[0]
    print(f"  Fitting NearestNeighbors on {len(X):,} cells, "
          f"querying {len(target_idx):,} PsychAD-V3 <1y cells (k={k})...")

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn.fit(X)
    _, indices = nn.kneighbors(X[target_idx])
    indices = indices[:, 1:]  # drop self (index 0)

    def _neigh_cat(sc, age):
        if sc == "WANG-multiome" and age < 1:          return "Wang_under1y"
        if sc == "VELMESHEV-V3" and age < 1:           return "Vel_V3_under1y"
        if sc == "VELMESHEV-V2" and age < 1:           return "Vel_V2_under1y"
        if sc == "PSYCHAD-V3" and age < 5:             return "PsychAD_under5y"
        if sc == "PSYCHAD-V3":                         return "PsychAD_adult"
        if sc in ("WANG-multiome", "VELMESHEV-V3", "VELMESHEV-V2"):
            return "WangVel_older"
        return "other"

    CAT_COLS = ["Wang_under1y", "Vel_V3_under1y", "Vel_V2_under1y",
                "PsychAD_under5y", "PsychAD_adult", "WangVel_older", "other"]

    rows = []
    for i, cidx in enumerate(target_idx):
        neigh = obs2.iloc[indices[i]]
        cats = [_neigh_cat(s, a) for s, a in
                zip(neigh["source-chemistry"], neigh["age_years"].astype(float))]
        cat_counts = pd.Series(cats).value_counts()
        row = {
            "cell_id": obs2.index[cidx],
            "predicted_class": obs2.iloc[cidx]["cell_type_aligned"],
            "confidence": obs2.iloc[cidx]["cell_type_aligned_confidence"],
        }
        for cat in CAT_COLS:
            row[f"frac_{cat}"] = cat_counts.get(cat, 0) / k
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"\n  PsychAD <1y latent neighbour composition "
          f"(n={len(target_idx):,} cells, K={k}):")
    print(f"  {'Category':<25s}  median   mean")
    for cat in CAT_COLS:
        col = f"frac_{cat}"
        if col in df.columns:
            print(f"  {cat:<25s}  {df[col].median():.3f}   {df[col].mean():.3f}")

    return df, X


# ── D7: Confidence histogram ──────────────────────────────────────────────────
def d7_confidence_hist(obs, out_png):
    GROUPS = {
        "PsychAD-V3 <1y":    _psychad_under1y(obs),
        "PsychAD-V3 1-5y":   _psychad_mask(obs) & (obs["age_years"].between(1, 5)).values,
        "PsychAD-V3 5-18y":  _psychad_mask(obs) & (obs["age_years"].between(5, 18)).values,
        "Wang <1y":          (obs["source-chemistry"] == "WANG-multiome").values &
                             (obs["age_years"] < 1).values,
        "Vel-V3 <1y":        (obs["source-chemistry"] == "VELMESHEV-V3").values &
                             (obs["age_years"].between(0, 1)).values,
    }
    COLORS = ["#E41A1C", "#FF7F00", "#FFCC00", "#377EB8", "#4DAF4A"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for (label, mask), color in zip(GROUPS.items(), COLORS):
        conf = obs.loc[mask, "cell_type_aligned_confidence"]
        if len(conf) > 0:
            ax.hist(conf, bins=50, alpha=0.55, density=True,
                    label=f"{label} (n={len(conf):,})", color=color)

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="threshold 0.5")
    ax.set_xlabel("scANVI prediction confidence", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("scANVI confidence distribution: PsychAD vs Wang/Vel young")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  Saved: {out_png}")


# ── D9: UMAP ──────────────────────────────────────────────────────────────────
def d9_umap(adata, obs, out_png):
    """Reuse cached UMAP if available; otherwise compute from X_scANVI (subsampled)."""
    import scanpy as sc

    umap_key = next((k for k in ["X_umap_scANVI", "X_umap"] if k in adata.obsm), None)
    if umap_key is None:
        print(f"  No cached UMAP in obsm ({list(adata.obsm.keys())}); "
              "computing subsampled UMAP from X_scANVI...")
        if "X_scANVI" not in adata.obsm:
            print("  X_scANVI not found — skipping D9")
            return
        X = np.array(adata.obsm["X_scANVI"], dtype=np.float32)
        target_mask = _psychad_under1y(obs)
        target_idx = np.where(target_mask)[0]
        rng = np.random.default_rng(42)
        bg_idx = np.where(~target_mask)[0]
        bg_sample = rng.choice(bg_idx, size=min(100_000, len(bg_idx)), replace=False)
        sub_idx = np.sort(np.concatenate([bg_sample, target_idx]))
        # Build tiny AnnData for UMAP computation
        tmp = ad.AnnData(obsm={"X_scANVI": X[sub_idx]},
                         obs=obs.iloc[sub_idx].reset_index(drop=False))
        sc.pp.neighbors(tmp, use_rep="X_scANVI", n_neighbors=30, random_state=42)
        sc.tl.umap(tmp, random_state=42)
        umap = tmp.obsm["X_umap"]
        obs_plot = tmp.obs
        is_target = obs_plot["source-chemistry"].eq("PSYCHAD-V3") & \
                    obs_plot["age_years"].astype(float).lt(1) & \
                    obs_plot["age_years"].astype(float).ge(0)
        is_target = is_target.values
    else:
        print(f"  Using cached UMAP from obsm['{umap_key}']...")
        umap = np.array(adata.obsm[umap_key])
        target_mask = _psychad_under1y(obs)
        # Subsample background for plotting speed
        rng = np.random.default_rng(42)
        bg_idx = np.where(~target_mask)[0]
        fg_idx = np.where(target_mask)[0]
        bg_sample = rng.choice(bg_idx, size=min(150_000, len(bg_idx)), replace=False)
        plot_idx = np.concatenate([bg_sample, fg_idx])
        umap = umap[plot_idx]
        obs_plot = obs.iloc[plot_idx]
        is_target = np.zeros(len(plot_idx), dtype=bool)
        is_target[len(bg_sample):] = True

    SC_COLORS = {
        "PSYCHAD-V3": "#E41A1C", "VELMESHEV-V3": "#377EB8",
        "VELMESHEV-V2": "#A6CEE3", "WANG-multiome": "#4DAF4A",
    }
    CLASS_COLORS = {
        "Excitatory": "#E41A1C", "Inhibitory": "#377EB8",
        "Astrocytes": "#4DAF4A", "Oligos": "#984EA3",
        "OPC": "#FF7F00", "Microglia": "#A65628",
        "Endothelial": "#F781BF", "Glia": "#999999",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    s = 0.3

    # Panel 1: all cells by source-chemistry, PsychAD <1y highlighted in gold
    sc_col = obs_plot["source-chemistry"].astype(str).map(SC_COLORS).fillna("#CCCCCC").values
    axes[0].scatter(umap[~is_target, 0], umap[~is_target, 1],
                    c=sc_col[~is_target], s=s, alpha=0.3, rasterized=True)
    axes[0].scatter(umap[is_target, 0], umap[is_target, 1],
                    c="gold", s=8, alpha=0.9, zorder=5, label="PsychAD-V3 <1y")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=v, markersize=8, label=k)
               for k, v in SC_COLORS.items()]
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="gold", markersize=8,
                               label="PsychAD-V3 <1y"))
    axes[0].legend(handles=handles, fontsize=7)
    axes[0].set_title("Coloured by source-chemistry\n(PsychAD-V3 <1y = gold)")
    axes[0].axis("off")

    # Panel 2: PsychAD <1y only, coloured by predicted cell_class
    axes[1].scatter(umap[~is_target, 0], umap[~is_target, 1],
                    c="#DDDDDD", s=s, alpha=0.15, rasterized=True)
    if is_target.any():
        fg_class = obs_plot["cell_class"].iloc[np.where(is_target)[0]].astype(str).values
        fg_c = [CLASS_COLORS.get(c, "#999999") for c in fg_class]
        axes[1].scatter(umap[is_target, 0], umap[is_target, 1],
                        c=fg_c, s=10, alpha=0.9, zorder=5)
    handles2 = [plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=v, markersize=8, label=k)
                for k, v in CLASS_COLORS.items()]
    axes[1].legend(handles=handles2, fontsize=7)
    axes[1].set_title("PsychAD-V3 <1y cells only\ncoloured by predicted cell_class")
    axes[1].axis("off")

    plt.suptitle(f"UMAP — PsychAD-V3 <1y cells highlighted "
                 f"(n={int(is_target.sum()):,})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_png}")


# ── C1: Prediction changes between runs ───────────────────────────────────────
def c1_prediction_changes(obs_a, obs_b, name_a, name_b, out_csv):
    mask_a = _psychad_under1y(obs_a)
    mask_b = _psychad_under1y(obs_b)
    sub_a = obs_a.loc[mask_a, ["cell_type_aligned", "cell_type_aligned_confidence"]].copy()
    sub_b = obs_b.loc[mask_b, ["cell_type_aligned", "cell_type_aligned_confidence"]].copy()

    merged = sub_a.join(sub_b, how="inner", lsuffix="_A", rsuffix="_B")
    n = len(merged)
    if n == 0:
        print("  WARNING: no shared PsychAD <1y cell IDs between runs")
        return None

    n_changed = (merged["cell_type_aligned_A"] != merged["cell_type_aligned_B"]).sum()
    print(f"\n  Cell-level prediction changes (PsychAD <1y, n={n:,} matched):")
    print(f"  Changed prediction: {n_changed:,}/{n:,} = {n_changed/n:.1%}")

    ct = pd.crosstab(merged["cell_type_aligned_A"], merged["cell_type_aligned_B"],
                     rownames=[f"{name_a}"], colnames=[f"{name_b}"])
    ct.to_csv(out_csv)

    changed = merged[merged["cell_type_aligned_A"] != merged["cell_type_aligned_B"]]
    if len(changed):
        to_en   = changed["cell_type_aligned_B"].str.contains("EN|Excitatory", na=False).sum()
        from_en = changed["cell_type_aligned_A"].str.contains("EN|Excitatory", na=False).sum()
        print(f"  Of {len(changed):,} changed cells: {to_en} GAINED EN label (→ no_age), "
              f"{from_en} LOST EN label (→ no_age)")
    return merged


# ── C2: Confidence delta ──────────────────────────────────────────────────────
def c2_confidence_delta(obs_a, obs_b, out_csv):
    mask_a = _psychad_under1y(obs_a)
    mask_b = _psychad_under1y(obs_b)
    sub_a = obs_a.loc[mask_a, ["cell_type_aligned_confidence"]].copy()
    sub_b = obs_b.loc[mask_b, ["cell_type_aligned_confidence"]].copy()
    merged = sub_a.join(sub_b, how="inner", lsuffix="_age", rsuffix="_no_age")
    merged["delta"] = merged["cell_type_aligned_confidence_age"] - \
                      merged["cell_type_aligned_confidence_no_age"]
    merged.to_csv(out_csv)
    d = merged["delta"]
    print(f"\n  Confidence delta (age - no_age) PsychAD <1y: "
          f"mean={d.mean():.4f}  std={d.std():.4f}  max_abs={d.abs().max():.4f}")
    return merged


# ── C4: Latent movement between runs ─────────────────────────────────────────
def c4_latent_movement(X_a, X_b, obs_a, obs_b, out_csv):
    """Per-cell L2 distance in X_scANVI space between the two runs (PsychAD <1y).

    Interpretation:
      - Small values (< 0.5) → age covariate barely shifted latent z → confirms H1-reject is robust
      - Large values (> 2.0) → latent shifted but classifier unchanged → classifier saturation
    """
    if X_a is None or X_b is None:
        print("  WARNING: X_scANVI missing for one run — skipping C4")
        return None

    mask_a = _psychad_under1y(obs_a)
    mask_b = _psychad_under1y(obs_b)
    ids_a = obs_a.index[mask_a]
    ids_b = obs_b.index[mask_b]
    shared = ids_a.intersection(ids_b)
    if len(shared) == 0:
        print("  WARNING: no shared PsychAD <1y cell IDs for C4")
        return None

    pos_a = X_a[obs_a.index.get_indexer(shared)]
    pos_b = X_b[obs_b.index.get_indexer(shared)]
    l2 = np.linalg.norm(pos_a - pos_b, axis=1)

    df = pd.DataFrame({"cell_id": shared, "l2_movement": l2})
    df.to_csv(out_csv, index=False)

    print(f"\n  Latent movement (X_scANVI L2 distance, n={len(shared):,} cells):")
    print(f"  mean={l2.mean():.4f}  P50={np.median(l2):.4f}  "
          f"P90={np.percentile(l2, 90):.4f}  max={l2.max():.4f}")
    print(f"  Baseline: mean distance between random cells ≈ "
          f"{float(np.linalg.norm(X_a[:500].mean(0) - X_b[:500].mean(0))):.4f}")
    return df


# ── Summary markdown ──────────────────────────────────────────────────────────
def write_summary(out_path):
    lines = [
        "# PsychAD relabel diagnostic — summary",
        "",
        "## Composition check results (from composition_check.py)",
        "",
        "| source-chemistry | age_bin | age_run EN% | no_age_run EN% | Δ pp |",
        "|------------------|---------|------------:|---------------:|-----:|",
        "| PSYCHAD-V3 | <1y    |  5.46 |  5.16 | -0.30 |",
        "| PSYCHAD-V3 | 1-5y   | 11.90 | 11.90 |  0.00 |",
        "| PSYCHAD-V3 | 5-18y  | 18.13 | 18.16 | +0.03 |",
        "| PSYCHAD-V3 | 18-30y | 22.68 | 22.70 | +0.02 |",
        "| PSYCHAD-V3 | 30-50y | 31.43 | 31.43 |  0.00 |",
        "",
        "**Key finding:** Removing the `age_years` covariate had essentially no effect.",
        "The EN% gradient monotonically *increases* with age — opposite of biology —",
        "and is reproduced almost identically by two independently-trained models.",
        "→ H1 (age covariate erased developmental signal) is **rejected**.",
        "",
        "## Hypotheses",
        "",
        "| # | Hypothesis | Status | Key output |",
        "|---|-----------|--------|-----------|",
        "| H1 | age covariate erased latent dev signal | REJECTED | S1 test, C4 latent movement |",
        "| H2 | PsychAD-adult labels dominate supervision | see D3/D5 | predicted_vs_raw, confusion |",
        "| H5 | Vel IN removal → IN class is PsychAD-only | see D5 | confusion EN→IN cross |",
        "| H6 | PsychAD 5-18y dev cells labelled adult-IN | see D5 | confusion diagonal |",
        "| H7 | PsychAD subclass is IN-heavy by default | see D3 | raw column marginals |",
        "",
        "## Output files",
        "",
        "**Per-run (`age_run/` and `no_age_run/`):**",
        "- `psychad_predictions_by_age_bin.csv` — D1: top predicted labels per PsychAD age bin",
        "- `psychad_confidence_summary.csv` — D2: confidence quartiles",
        "- `psychad_under1y_predicted_vs_raw_crosstab.csv` — D3: scANVI pred vs raw subclass",
        "- `marker_means.csv` — D4: EN/IN marker expression per predicted class",
        "- `scanvi_confusion_on_labelled.csv` — D5: classifier accuracy on training labels",
        "- `latent_neighbor_composition.csv` — D6: are PsychAD <1y neighbours PsychAD adults?",
        "- `psychad_under1y_confidence_hist.png` — D7: confidence distributions",
        "- `umap_psychad_under1y.png` — D9: UMAP with PsychAD <1y highlighted",
        "",
        "**Cross-run (`comparison/`):**",
        "- `psychad_under1y_prediction_changes.csv` — C1: which cells changed prediction?",
        "- `psychad_under1y_confidence_delta.csv` — C2: confidence deltas",
        "- `latent_movement_distribution.csv` — C4: L2 shift in X_scANVI between runs",
        "",
        "## Next step",
        "",
        "Once `errors.log` is clean and D3/D4/D5/D6 are reviewed, choose v4 approach:",
        "- **S2**: drop ALL PsychAD labels (Wang+Vel as sole reference) — addresses H2/H5/H6",
        "- **S2b**: keep only PsychAD non-neuronal labels (Micro/Endo/Astro/Oligo/OPC)",
        "- **S3**: scArches/scPoli reference-query architecture",
    ]
    out_path.write_text("\n".join(lines))
    print(f"\n  Written: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    adatas = {}
    obs_all = {}
    X_latents = {}

    for name, run_dir in RUNS.items():
        h5ad = Path(run_dir) / "scvi_output/integrated.h5ad"
        print("\n" + "=" * 70)
        print(f"RUN: {name}")
        print(f"  {h5ad}")
        print("=" * 70)

        if not h5ad.exists():
            print(f"  ERROR: file not found — skipping")
            continue

        out = OUTPUT_DIR / name
        out.mkdir(parents=True, exist_ok=True)

        adata = ad.read_h5ad(str(h5ad), backed="r")
        obs = adata.obs.copy()
        adatas[name] = adata
        obs_all[name] = obs

        print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        print(f"  obs columns: {sorted(obs.columns.tolist())}")
        print(f"  obsm keys:   {list(adata.obsm.keys())}")
        print(f"  layers:      {list(adata.layers.keys())}")

        print("\n  --- D1: predictions by age bin ---")
        _safe(f"{name}.D1", d1_predictions_by_age_bin,
              obs, out / "psychad_predictions_by_age_bin.csv")

        print("\n  --- D2: confidence summary ---")
        _safe(f"{name}.D2", d2_confidence_summary,
              obs, out / "psychad_confidence_summary.csv")

        print("\n  --- D3: predicted vs raw crosstab ---")
        _safe(f"{name}.D3", d3_predicted_vs_raw,
              obs, out / "psychad_under1y_predicted_vs_raw_crosstab.csv")

        print("\n  --- D4: marker gene expression ---")
        _safe(f"{name}.D4", d4_marker_means,
              adata, obs, out / "marker_means.csv")

        print("\n  --- D5: classifier confusion on labelled cells ---")
        _safe(f"{name}.D5", d5_scanvi_confusion,
              obs, out / "scanvi_confusion_on_labelled.csv")

        print("\n  --- D6: latent neighbour composition ---")
        result = _safe(f"{name}.D6", d6_latent_neighbors,
                       adata, obs, out / "latent_neighbor_composition.csv")
        if result is not None:
            _, X_lat = result
            if X_lat is not None:
                X_latents[name] = X_lat

        print("\n  --- D7: confidence histograms ---")
        _safe(f"{name}.D7", d7_confidence_hist,
              obs, out / "psychad_under1y_confidence_hist.png")

        print("\n  --- D9: UMAP ---")
        _safe(f"{name}.D9", d9_umap,
              adata, obs, out / "umap_psychad_under1y.png")

        print(f"\n  {name} complete.")

    # Cross-run comparisons
    if len(obs_all) == 2:
        names = list(obs_all.keys())
        obs_a, obs_b = obs_all[names[0]], obs_all[names[1]]

        print("\n" + "=" * 70)
        print("CROSS-RUN COMPARISON")
        print("=" * 70)
        cmp = OUTPUT_DIR / "comparison"
        cmp.mkdir(exist_ok=True)

        print("\n  --- C1: prediction changes ---")
        _safe("C1", c1_prediction_changes, obs_a, obs_b, names[0], names[1],
              cmp / "psychad_under1y_prediction_changes.csv")

        print("\n  --- C2: confidence delta ---")
        _safe("C2", c2_confidence_delta, obs_a, obs_b,
              cmp / "psychad_under1y_confidence_delta.csv")

        if len(X_latents) == 2:
            print("\n  --- C4: latent movement ---")
            Xa = X_latents[names[0]]
            Xb = X_latents[names[1]]
            _safe("C4", c4_latent_movement, Xa, Xb, obs_a, obs_b,
                  cmp / "latent_movement_distribution.csv")

    write_summary(OUTPUT_DIR / "summary.md")

    if _errors:
        err_path = OUTPUT_DIR / "errors.log"
        err_path.write_text("\n\n".join(_errors))
        print(f"\n  {len(_errors)} error(s) written to {err_path}")
    else:
        (OUTPUT_DIR / "errors.log").write_text("No errors.\n")
        print("\n  No errors.")

    print("\n" + "=" * 70)
    print("Done.  Results in:", OUTPUT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
