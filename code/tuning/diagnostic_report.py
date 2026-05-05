"""Standalone diagnostic report for scVI input data characterisation.

Loads the integrated.h5ad from the most recent pipeline run (backed mode where
possible) and produces structured CSVs and figures describing:
  1. Data availability (layers, dtypes, integer count check)
  2. Chemistry × age confound structure + anchor donors
  3. Donor age distribution and confound severity
  4. Cell type label inventory (all obs fields + cell_type_aligned detail)
  5. Excitatory lineage proxy (using cell_type_aligned as rough reference)
  6. Marker gene expression per age bin (sanity check)
  7. Confound summary text report

Usage:
    python -m tuning.diagnostic_report --input <h5ad_path> [--output <dir>]

Default input: integrated.h5ad from VelWangPsychAD_100k_source-chemistry run.
Output:        <output_dir>/<YYYY-MM-DD>_diagnostic/
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INPUT = (
    "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated"
    "/VelWangPsychAD_100k_source-chemistry/scvi_output/integrated.h5ad"
)
DEFAULT_OUTPUT = (
    "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/diagnostics"
)

# These edges match the round4/round5 tuning config.
AGE_BIN_EDGES = [-1.0, -0.5, -0.38, -0.27, -0.15, -0.05, 0.0, 1.0, 5.0, 12.0, 20.0, 40.0, 90.0]
AGE_BIN_LABELS = [
    "GW14-20", "GW20-26", "GW26-33", "GW33-38", "GW38-40", "GW40-Birth",
    "Infant(0-1y)", "Child(1-5y)", "Juvenile(5-12y)", "Adolescent(12-20y)",
    "Adult(20-40y)", "OldAdult(40+y)",
]

BATCH_KEY      = "source-chemistry"
AGE_KEY        = "age_years"
DONOR_KEY      = "individual"
CELL_CLASS_KEY = "cell_class"
CELL_TYPE_ALIGNED_KEY = "cell_type_aligned"

MARKER_GENES = ["EOMES", "DCX", "SATB2", "PAX6", "SOX2", "NEUROD2", "TBR1"]

# Keywords for identifying excitatory/progenitor lineage cells in cell_type_aligned
EXCITATORY_KEYWORDS = [
    "excitatory", "glutamatergic", "radial glia", " rg", "intermediate progenitor",
    r"\bIP\b", "vrg", "org", "newborn neuron", r"\bEN\b", r"\bglun",
]
PROGENITOR_KEYWORDS = [
    "radial glia", " rg", "intermediate progenitor", r"\bIP\b", "vrg", "org",
    "progenitor", "stem",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_age_bins(age_series: pd.Series) -> pd.Categorical:
    return pd.cut(
        age_series,
        bins=AGE_BIN_EDGES,
        labels=AGE_BIN_LABELS,
        right=False,
        include_lowest=True,
    )


def _sample_array(arr, n: int = 1000, seed: int = 0) -> np.ndarray:
    """Draw up to n random values from an array-like for dtype/range inspection."""
    rng = np.random.default_rng(seed)
    if sp.issparse(arr):
        arr = arr[:n].toarray() if arr.shape[0] >= n else arr.toarray()
    else:
        arr = np.asarray(arr)
    flat = arr.ravel()
    if flat.size > n:
        flat = flat[rng.choice(flat.size, n, replace=False)]
    return flat


def _is_integer_counts(sample: np.ndarray) -> bool:
    sample = sample[np.isfinite(sample)]
    if len(sample) == 0:
        return False
    return bool(np.all(sample >= 0) and np.all(sample == np.floor(sample)))


def _label_matches_keywords(label: str, keywords: list[str]) -> bool:
    import re
    label_lower = label.lower()
    return any(re.search(kw.lower(), label_lower) for kw in keywords)


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 120) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Section 1: Data availability
# ---------------------------------------------------------------------------

def section_data_availability(adata: ad.AnnData, out_dir: Path) -> dict[str, Any]:
    print("\n[1] Data availability")
    rows = []

    def _check_store(label: str, arr) -> dict:
        try:
            sample = _sample_array(arr, n=2000)
            vmin = float(np.nanmin(sample)) if len(sample) else float("nan")
            vmax = float(np.nanmax(sample)) if len(sample) else float("nan")
            is_int = _is_integer_counts(sample)
            shape = arr.shape if hasattr(arr, "shape") else "?"
            dtype = str(arr.dtype) if hasattr(arr, "dtype") else "?"
        except Exception as e:
            return {"store": label, "shape": "?", "dtype": "error", "min": float("nan"),
                    "max": float("nan"), "integer_counts": False, "note": str(e)}
        return {
            "store":          label,
            "shape":          str(shape),
            "dtype":          dtype,
            "min":            round(vmin, 3),
            "max":            round(vmax, 3),
            "integer_counts": bool(is_int),
            "note":           "RAW COUNTS ✓" if is_int else "",
        }

    # adata.X
    rows.append(_check_store("X", adata.X))

    # layers
    for layer_name in adata.layers.keys():
        rows.append(_check_store(f"layers['{layer_name}']", adata.layers[layer_name]))

    # adata.raw
    if adata.raw is not None:
        rows.append(_check_store("raw.X", adata.raw.X))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "data_availability.csv", index=False)
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    for r in rows:
        flag = " ← RAW COUNTS" if r["integer_counts"] else ""
        print(f"  {r['store']:30s}  dtype={r['dtype']:12s}  "
              f"range=[{r['min']}, {r['max']}]{flag}")

    # Per-sample (individual) cell counts
    if DONOR_KEY in adata.obs.columns:
        donor_counts = adata.obs[DONOR_KEY].value_counts()
        donor_counts.to_frame("n_cells").head(50).to_csv(out_dir / "per_donor_cell_counts.csv")
        print(f"  Unique donors: {donor_counts.shape[0]:,} (top counts: {donor_counts.head(3).tolist()})")

    # Check for spliced/unspliced
    for ln in ["spliced", "unspliced", "ambiguous"]:
        present = ln in adata.layers
        print(f"  Layer '{ln}': {'present' if present else 'absent'}")

    raw_count_store = next(
        (r["store"] for r in rows if r["integer_counts"]), None
    )
    print(f"  → Raw count store: {raw_count_store or 'NOT FOUND (check manually)'}")
    return {"raw_count_store": raw_count_store, "n_obs": adata.n_obs, "n_vars": adata.n_vars}


# ---------------------------------------------------------------------------
# Section 2: Chemistry × age confound + anchor donors
# ---------------------------------------------------------------------------

def section_chemistry_age(adata: ad.AnnData, out_dir: Path) -> dict[str, Any]:
    print("\n[2] Chemistry × age confound")
    obs = adata.obs.copy()

    if BATCH_KEY not in obs.columns:
        print(f"  WARNING: '{BATCH_KEY}' not in obs. Available: {list(obs.columns[:10])}")
        return {}
    if AGE_KEY not in obs.columns:
        print(f"  WARNING: '{AGE_KEY}' not in obs.")
        return {}

    obs["_age_bin"] = _make_age_bins(obs[AGE_KEY])

    # Cross-tab: chemistry × age bin (cell counts)
    crosstab = pd.crosstab(obs[BATCH_KEY], obs["_age_bin"])
    crosstab.to_csv(out_dir / "chemistry_age_crosstab.csv")
    print(f"  Chemistry × age bin (n_cells):")
    print(crosstab.to_string())

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, max(3, len(crosstab) * 0.7 + 1)))
    im = ax.imshow(crosstab.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="n cells")
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels(list(crosstab.columns), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels(list(crosstab.index), fontsize=8)
    ax.set_title("Chemistry × Age Bin (cell counts)\nEmpty = chemistry absent from that age bin")
    for i in range(crosstab.shape[0]):
        for j in range(crosstab.shape[1]):
            v = crosstab.values[i, j]
            if v > 0:
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=6,
                        color="white" if v > crosstab.values.max() * 0.6 else "black")
    _save_fig(fig, out_dir / "chemistry_age_crosstab.png")

    # Per-bin: how many distinct chemistries?
    n_chem_per_bin = (crosstab > 0).sum(axis=0)
    single_chem_bins = n_chem_per_bin[n_chem_per_bin == 1].index.tolist()
    if single_chem_bins:
        print(f"  WARNING: age bins with only 1 chemistry (cannot batch-correct): {single_chem_bins}")
    zero_chem_bins = n_chem_per_bin[n_chem_per_bin == 0].index.tolist()
    if zero_chem_bins:
        print(f"  Empty age bins (no cells): {zero_chem_bins}")

    # Chemistry × donor cross-tab (n_donors per chemistry)
    if DONOR_KEY in obs.columns:
        chem_donor = obs.groupby(BATCH_KEY, observed=True)[DONOR_KEY].nunique()
        chem_donor.to_frame("n_donors").to_csv(out_dir / "chemistry_donor_counts.csv")
        print(f"  Donors per chemistry:\n{chem_donor.to_string()}")

    # Anchor donors: donors appearing in ≥2 chemistries at the same age bin
    if DONOR_KEY in obs.columns:
        donor_chem_bin = (
            obs[[DONOR_KEY, BATCH_KEY, "_age_bin"]]
            .drop_duplicates()
            .dropna()
        )
        anchor_rows = []
        for (donor, age_bin), grp in donor_chem_bin.groupby([DONOR_KEY, "_age_bin"], observed=True):
            chems = grp[BATCH_KEY].unique().tolist()
            if len(chems) >= 2:
                anchor_rows.append({"donor": donor, "age_bin": age_bin, "chemistries": json.dumps(chems)})
        anchor_df = pd.DataFrame(anchor_rows)
        anchor_df.to_csv(out_dir / "anchor_donors.csv", index=False)
        n_anchors = len(anchor_df)
        print(f"  Anchor donors (≥2 chemistries in same age bin): {n_anchors}")
        if n_anchors == 0:
            print("  WARNING: No anchor donors — within-bin batch correction cannot be empirically validated.")
        else:
            print(anchor_df.to_string())
    else:
        print(f"  '{DONOR_KEY}' not in obs — cannot identify anchor donors")
        n_anchors = 0
        anchor_df = pd.DataFrame()

    return {"n_anchors": n_anchors, "single_chem_bins": single_chem_bins}


# ---------------------------------------------------------------------------
# Section 3: Donor age structure
# ---------------------------------------------------------------------------

def section_donor_age(adata: ad.AnnData, out_dir: Path) -> dict[str, Any]:
    print("\n[3] Donor age structure")
    obs = adata.obs.copy()

    if AGE_KEY not in obs.columns:
        print(f"  WARNING: '{AGE_KEY}' not in obs.")
        return {}

    obs["_age_bin"] = _make_age_bins(obs[AGE_KEY])

    # Summary stats per age bin
    age_stats = obs.groupby("_age_bin", observed=False)[AGE_KEY].agg(
        ["count", "min", "max", "mean", "median"]
    )
    if DONOR_KEY in obs.columns:
        age_stats["n_donors"] = obs.groupby("_age_bin", observed=False)[DONOR_KEY].nunique()
    age_stats.to_csv(out_dir / "donor_age_structure.csv")
    print(age_stats.to_string())

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ax0, ax1 = axes
    obs[AGE_KEY].hist(bins=60, ax=ax0, color="steelblue", edgecolor="none", alpha=0.8)
    ax0.set_xlabel("age_years")
    ax0.set_ylabel("n cells")
    ax0.set_title("Cell distribution by age (raw)")
    ax0.axvline(0, color="red", linestyle="--", linewidth=0.8, label="Birth")
    ax0.legend(fontsize=8)

    np.log1p(np.maximum(obs[AGE_KEY], -0.999)).hist(bins=60, ax=ax1, color="darkorange",
                                                      edgecolor="none", alpha=0.8)
    ax1.set_xlabel("log1p(age_years)")
    ax1.set_title("Cell distribution by log age")
    _save_fig(fig, out_dir / "donor_age_histogram.png")

    # Confound check: is each donor at exactly one age?
    confound_result = {}
    if DONOR_KEY in obs.columns:
        donor_age_counts = obs.groupby(DONOR_KEY, observed=True)[AGE_KEY].nunique()
        n_multi_age = (donor_age_counts > 1).sum()
        confound_result = {
            "n_donors_total": int(donor_age_counts.shape[0]),
            "n_donors_at_multiple_ages": int(n_multi_age),
            "perfectly_confounded": bool(n_multi_age == 0),
        }
        print(f"  Donors at multiple ages: {n_multi_age} / {donor_age_counts.shape[0]}")
        if n_multi_age == 0:
            print("  → Donor ID and age are PERFECTLY CONFOUNDED (each donor = one age value).")
        else:
            print(f"  → {n_multi_age} donors span multiple ages (partially separable).")
        donor_age_counts.to_frame("n_distinct_ages").to_csv(out_dir / "donor_age_confound.csv")

    return confound_result


# ---------------------------------------------------------------------------
# Section 4: Cell type label inventory
# ---------------------------------------------------------------------------

def section_cell_type_labels(adata: ad.AnnData, out_dir: Path) -> dict[str, Any]:
    print("\n[4] Cell type label inventory")
    obs = adata.obs.copy()

    # Scan obs for candidate cell type columns
    candidate_rows = []
    for col in obs.columns:
        if obs[col].dtype in (object, "category") or hasattr(obs[col], "cat"):
            n_unique = int(obs[col].nunique(dropna=True))
            n_na = int(obs[col].isna().sum())
            frac_na = round(n_na / len(obs), 3)
            if n_unique < 500:
                candidate_rows.append({
                    "column":    col,
                    "n_unique":  n_unique,
                    "n_na":      n_na,
                    "frac_na":   frac_na,
                    "dtype":     str(obs[col].dtype),
                    "top5_values": json.dumps(obs[col].value_counts().head(5).index.tolist()),
                })
    pd.DataFrame(candidate_rows).to_csv(out_dir / "cell_type_labels.csv", index=False)
    print(f"  Candidate cell-type columns ({len(candidate_rows)}):")
    for r in candidate_rows:
        print(f"    {r['column']:30s}  n_unique={r['n_unique']:3d}  frac_na={r['frac_na']:.3f}")

    # Detailed cell_type_aligned analysis
    result = {}
    if CELL_TYPE_ALIGNED_KEY in obs.columns:
        print(f"\n  *** cell_type_aligned detail ***")
        print("  NOTE: cell_type_aligned was produced by a SUBOPTIMAL model (age_log_pc")
        print("  included as covariate). Treat as rough reference only — not ground truth.")
        obs["_age_bin"] = _make_age_bins(obs[AGE_KEY]) if AGE_KEY in obs.columns else "unknown"

        vc = obs[CELL_TYPE_ALIGNED_KEY].value_counts(dropna=False)
        vc.to_frame("n_cells").to_csv(out_dir / "cell_type_aligned_counts.csv")
        print(f"  cell_type_aligned value counts (top 20):\n{vc.head(20).to_string()}")

        n_na = int(obs[CELL_TYPE_ALIGNED_KEY].isna().sum())
        frac_na = n_na / len(obs)
        print(f"  NaN / unlabelled: {n_na:,} ({frac_na:.1%})")

        if BATCH_KEY in obs.columns:
            ct_chem = pd.crosstab(obs[CELL_TYPE_ALIGNED_KEY], obs[BATCH_KEY],
                                  dropna=False, margins=False)
            ct_chem.to_csv(out_dir / "cell_type_aligned_x_chemistry.csv")

        if "_age_bin" in obs.columns:
            ct_age = pd.crosstab(obs[CELL_TYPE_ALIGNED_KEY], obs["_age_bin"],
                                 dropna=False, margins=False)
            ct_age.to_csv(out_dir / "cell_type_aligned_x_age.csv")

        result["n_unique_cell_type_aligned"] = int(obs[CELL_TYPE_ALIGNED_KEY].nunique(dropna=True))
        result["frac_na_cell_type_aligned"] = round(frac_na, 4)

    return result


# ---------------------------------------------------------------------------
# Section 5: Excitatory lineage proxy
# ---------------------------------------------------------------------------

def section_excitatory_lineage(adata: ad.AnnData, out_dir: Path) -> dict[str, Any]:
    print("\n[5] Excitatory lineage proxy (cell_type_aligned as rough reference)")
    obs = adata.obs.copy()

    if CELL_TYPE_ALIGNED_KEY not in obs.columns:
        print(f"  '{CELL_TYPE_ALIGNED_KEY}' not in obs — skipping")
        return {}

    obs["_age_bin"] = _make_age_bins(obs[AGE_KEY]) if AGE_KEY in obs.columns else "unknown"
    labels = obs[CELL_TYPE_ALIGNED_KEY].astype(object).fillna("").astype(str)
    all_unique = sorted(labels.unique().tolist())
    print(f"  All unique cell_type_aligned values ({len(all_unique)}):")
    for lbl in all_unique:
        print(f"    {lbl}")

    is_excitatory = labels.apply(
        lambda l: _label_matches_keywords(l, EXCITATORY_KEYWORDS)
    )
    is_progenitor = labels.apply(
        lambda l: _label_matches_keywords(l, PROGENITOR_KEYWORDS)
    )
    is_excit_mature = is_excitatory & ~is_progenitor

    obs["_lineage"] = "other"
    obs.loc[is_excitatory, "_lineage"] = "excitatory"
    obs.loc[is_progenitor, "_lineage"] = "progenitor"
    obs.loc[is_excit_mature, "_lineage"] = "excitatory_mature"

    print(f"\n  Keyword matches (excitatory): {is_excitatory.sum():,} cells")
    print(f"  Keyword matches (progenitor): {is_progenitor.sum():,} cells")

    # Matched labels for review
    matched_excit = sorted(set(labels[is_excitatory].unique()))
    matched_prog  = sorted(set(labels[is_progenitor].unique()))
    print(f"  Matched excitatory labels: {matched_excit}")
    print(f"  Matched progenitor labels: {matched_prog}")

    if "_age_bin" in obs.columns:
        lineage_by_age = pd.crosstab(obs["_age_bin"], obs["_lineage"])
        lineage_by_age.to_csv(out_dir / "excitatory_lineage_proxy.csv")
        print(f"\n  Lineage by age bin:\n{lineage_by_age.to_string()}")

        # Fraction progenitor-like per age bin
        total_per_bin = lineage_by_age.sum(axis=1)
        prog_col = "progenitor" if "progenitor" in lineage_by_age.columns else None
        if prog_col:
            frac_prog = lineage_by_age[prog_col] / total_per_bin.replace(0, np.nan)
            print("\n  Fraction progenitor-like per age bin:")
            for bin_lbl, frac in frac_prog.items():
                flag = "  ← HIGH (lineage metric may be unreliable)" if frac > 0.5 else ""
                print(f"    {str(bin_lbl):20s}: {frac:.1%}{flag}")

    return {"n_excitatory": int(is_excitatory.sum()), "n_progenitor": int(is_progenitor.sum())}


# ---------------------------------------------------------------------------
# Section 6: Marker gene expression per age bin
# ---------------------------------------------------------------------------

def section_marker_expression(
    adata: ad.AnnData,
    out_dir: Path,
    raw_count_store: str | None,
) -> None:
    print("\n[6] Marker gene expression per age bin")
    obs = adata.obs.copy()

    if AGE_KEY not in obs.columns:
        print("  Age key not found — skipping")
        return

    obs["_age_bin"] = _make_age_bins(obs[AGE_KEY])

    # Identify which markers are present
    present_markers = [g for g in MARKER_GENES if g in adata.var_names]
    absent_markers  = [g for g in MARKER_GENES if g not in adata.var_names]
    print(f"  Present markers: {present_markers}")
    if absent_markers:
        print(f"  Absent (not in HVG set): {absent_markers}")

    if not present_markers:
        print("  No marker genes in var_names — skipping expression section")
        return

    # Extract expression for the marker genes
    gene_idx = [list(adata.var_names).index(g) for g in present_markers]

    try:
        # Try to read from the identified raw counts layer first
        layer_name = None
        if raw_count_store and raw_count_store.startswith("layers['"):
            layer_name = raw_count_store[len("layers['"):-2]

        if layer_name and layer_name in adata.layers:
            X_markers = adata.layers[layer_name][:, gene_idx]
        else:
            # Fallback to adata.X
            X_markers = adata.X[:, gene_idx]

        if sp.issparse(X_markers):
            X_markers = X_markers.toarray()
        X_markers = np.asarray(X_markers, dtype=np.float32)
    except Exception as e:
        print(f"  Could not load marker expression: {e}")
        return

    marker_df = pd.DataFrame(X_markers, index=obs.index, columns=present_markers)
    marker_df["_age_bin"] = obs["_age_bin"].values

    mean_by_age = marker_df.groupby("_age_bin", observed=False)[present_markers].mean()
    mean_by_age.to_csv(out_dir / "marker_expression.csv")
    print(f"\n  Mean expression per age bin:\n{mean_by_age.to_string()}")

    # Bar chart: one panel per gene
    n_markers = len(present_markers)
    ncols = min(4, n_markers)
    nrows = math.ceil(n_markers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False)
    for k, gene in enumerate(present_markers):
        ax = axes[k // ncols][k % ncols]
        vals = mean_by_age[gene].values
        ax.bar(range(len(vals)), vals, color="steelblue", alpha=0.85)
        ax.set_xticks(range(len(mean_by_age.index)))
        ax.set_xticklabels(list(mean_by_age.index), rotation=60, ha="right", fontsize=6)
        ax.set_title(gene, fontsize=10)
        ax.set_ylabel("mean counts")
    for k in range(n_markers, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    fig.suptitle(
        "Marker Gene Expression by Age Bin (raw counts)\n"
        "NOTE: computed from suboptimal run (age as covariate) — for sanity check only",
        fontsize=9,
    )
    fig.tight_layout()
    _save_fig(fig, out_dir / "marker_expression.png")


# ---------------------------------------------------------------------------
# Section 7: Confound summary report
# ---------------------------------------------------------------------------

def section_confound_summary(
    out_dir: Path,
    section2: dict,
    section3: dict,
    section4: dict,
) -> None:
    print("\n[7] Confound summary")
    lines = [
        "=" * 70,
        "CONFOUND SUMMARY REPORT",
        f"Generated: {date.today().isoformat()}",
        "=" * 70,
        "",
        "CHEMISTRY × AGE CONFOUND",
        "-" * 40,
    ]

    n_anchors = section2.get("n_anchors", "N/A")
    single_bins = section2.get("single_chem_bins", [])
    lines.append(f"Anchor donors (>=2 chemistries, same age bin): {n_anchors}")
    if n_anchors == 0:
        lines.append(
            "CRITICAL: No anchor donors found. Within-bin batch correction cannot be\n"
            "empirically validated. Any batch mixing score reflects global confound\n"
            "structure, not true batch correction efficacy."
        )
    elif isinstance(n_anchors, int) and n_anchors < 5:
        lines.append(
            f"WARNING: Only {n_anchors} anchor donors. Validation power is limited."
        )
    else:
        lines.append(f"OK: {n_anchors} anchor donors provide cross-chemistry validation.")

    if single_bins:
        lines.append(
            f"\nAge bins with only 1 chemistry (cannot correct): {single_bins}"
        )
        lines.append(
            "Conditioned chemistry mixing score will be undefined for these bins."
        )

    lines += [
        "",
        "DONOR–AGE CONFOUND",
        "-" * 40,
    ]
    perfectly_confounded = section3.get("perfectly_confounded", None)
    n_multi = section3.get("n_donors_at_multiple_ages", "N/A")
    n_total = section3.get("n_donors_total", "N/A")
    lines.append(f"Donors at multiple ages: {n_multi} / {n_total}")
    if perfectly_confounded is True:
        lines.append(
            "PERFECTLY CONFOUNDED: Every donor appears at exactly one age value.\n"
            "Donor ID cannot be used as an additional covariate beyond age binning."
        )
    elif perfectly_confounded is False:
        lines.append(
            f"{n_multi} donors span multiple age values — donor and age are partially\n"
            "separable. If chemistry is not confounded with donor, donor can potentially\n"
            "be added as categorical_covariate_keys."
        )

    lines += [
        "",
        "CELL TYPE LABEL STATUS",
        "-" * 40,
        "cell_type_aligned was produced by a SUBOPTIMAL model (age_log_pc included",
        "as a covariate). Labels should be treated as rough reference only.",
        f"NaN fraction: {section4.get('frac_na_cell_type_aligned', 'N/A')}",
        f"Unique labels: {section4.get('n_unique_cell_type_aligned', 'N/A')}",
        "",
        "METRIC IMPLICATIONS",
        "-" * 40,
    ]

    if n_anchors == 0:
        lines.append(
            "Conditioned chemistry mixing: SCORE ONLY REFLECTS OVERALL INTEGRATION\n"
            "No anchor donors means the metric cannot verify within-bin correction."
        )
    else:
        lines.append("Conditioned chemistry mixing: anchor donors allow empirical validation.")

    frac_na = section4.get("frac_na_cell_type_aligned", 1.0)
    if isinstance(frac_na, float) and frac_na > 0.3:
        lines.append(
            f"Age preservation (per cell type): {frac_na:.0%} of cells have no cell_type_aligned\n"
            "label. Conditioned mixing metric will cover fewer strata — check that\n"
            f"the cell types that DO have labels represent the full age range."
        )

    lines += ["", "=" * 70]

    report = "\n".join(lines)
    (out_dir / "confound_summary.txt").write_text(report)
    print(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic report for scVI input data characterisation"
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to integrated.h5ad")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Root output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(args.output) / f"{date.today().isoformat()}_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    print(f"\nLoading AnnData (backed mode): {input_path}")
    print(f"  File size: {input_path.stat().st_size / 1e9:.1f} GB")

    # Load in backed mode — obs and var are in RAM, arrays are lazy.
    adata = sc.read_h5ad(str(input_path), backed="r")
    print(f"  Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print(f"  layers: {list(adata.layers.keys())}")
    print(f"  obsm: {list(adata.obsm.keys())}")

    sec1 = section_data_availability(adata, out_dir)
    sec2 = section_chemistry_age(adata, out_dir)
    sec3 = section_donor_age(adata, out_dir)
    sec4 = section_cell_type_labels(adata, out_dir)
    _    = section_excitatory_lineage(adata, out_dir)
    section_marker_expression(adata, out_dir, raw_count_store=sec1.get("raw_count_store"))
    section_confound_summary(out_dir, sec2, sec3, sec4)

    # Close backed file handle
    if hasattr(adata, "file") and adata.file is not None:
        try:
            adata.file.close()
        except Exception:
            pass

    print(f"\nDiagnostic report complete: {out_dir}")
    print("NEXT STEPS: Review confound_summary.txt and chemistry_age_crosstab.png")
    print("before proceeding with metric design and hyperparameter tuning.")


if __name__ == "__main__":
    main()
