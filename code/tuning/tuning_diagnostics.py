"""Visualise scVI hyperparameter tuning results.

Usage (from project root):
    micromamba run -n scvi env PYTHONPATH=code python -m tuning.tuning_diagnostics \
        --input_dir <scvi_tuning_output_dir> \
        --output_dir <plot_output_dir>

Produces a multi-page PDF and individual PNGs covering:
  - Trial objective ranking with parameter annotations
  - Per-age-bin batch mixing heatmap across all trials
  - Per-parameter effect plots (n_latent, n_hidden, n_layers, gene_likelihood, batch_size)
  - Prenatal vs postnatal mixing comparison
  - scVI vs scANVI objective comparison for top-k trials
  - Per-batch global mixing scores (identifies which batches are poorly integrated)
  - UMAP grid: top-5 vs bottom-5 trials coloured by batch (2 rows × 5 cols)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# Age bins in chronological order — includes both old coarse and new finer prenatal formats
# so the same diagnostics script works for both run generations.
AGE_BIN_ORDER = [
    # Very early prenatal (typically empty; old or new format lower bound)
    "[-1.0,-0.75)", "[-1.0,-0.5)",
    # Early prenatal old coarse format
    "[-0.75,-0.5)",
    # New finer prenatal format (GW14-GW40 in ~5-7wk windows)
    "[-0.5,-0.38)", "[-0.38,-0.27)", "[-0.27,-0.15)", "[-0.15,-0.05)", "[-0.05,0.0)",
    # Old coarse prenatal format (backward compat)
    "[-0.5,-0.25)", "[-0.25,0.0)",
    # Postnatal (unchanged across formats)
    "[0.0,1.0)", "[1.0,5.0)", "[5.0,12.0)", "[12.0,20.0)", "[20.0,40.0)", "[40.0,90.0)",
]

AGE_BIN_LABELS: dict[str, str] = {
    "[-1.0,-0.75)":  "Prenatal\n<GW14",
    "[-1.0,-0.5)":   "Prenatal\n<GW14",
    "[-0.75,-0.5)":  "Prenatal\nGW14–21",
    "[-0.5,-0.38)":  "Prenatal\nGW14–20",
    "[-0.38,-0.27)": "Prenatal\nGW20–26",
    "[-0.27,-0.15)": "Prenatal\nGW26–33",
    "[-0.15,-0.05)": "Prenatal\nGW33–37",
    "[-0.05,0.0)":   "Prenatal\nGW37+",
    "[-0.5,-0.25)":  "Prenatal\nGW14–27",
    "[-0.25,0.0)":   "Prenatal\nGW27–40",
    "[0.0,1.0)":     "Infant\n0–1y",
    "[1.0,5.0)":     "Toddler\n1–5y",
    "[5.0,12.0)":    "Childhood\n5–12y",
    "[12.0,20.0)":   "Adolescence\n12–20y",
    "[20.0,40.0)":   "Young adult\n20–40y",
    "[40.0,90.0)":   "Older adult\n40–90y",
}

# All bins with negative left edge are prenatal; computed dynamically from data
def _is_prenatal_bin(label: str) -> bool:
    try:
        return float(label.lstrip("[").split(",")[0]) < 0
    except Exception:
        return False

PARAM_COLS = ["n_latent", "n_hidden", "n_layers", "gene_likelihood", "batch_size"]

PALETTE = {1: "#4878CF", 2: "#D65F5F", 3: "#6AAB6A", 4: "#D68A5F", 5: "#9B59B6"}  # n_layers → colour

N_UMAP_CELLS = 10_000  # per-trial subsample for UMAP speed (~10–20 s/panel on CPU)

FIGURE_WIDTH = 14.0  # inches — enforced across all output figures so they align in the PDF


def _load_results(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    trials_path = input_dir / "trial_results.csv"
    if not trials_path.exists():
        raise FileNotFoundError(f"trial_results.csv not found in {input_dir}")
    df = pd.read_csv(trials_path)
    df = df[df["status"] == "ok"].copy()

    # Parse age bin scores from JSON column
    all_age_keys: list[str] = []
    age_score_rows: list[dict] = []
    for _, row in df.iterrows():
        scores = json.loads(row["age_bin_scores_json"])
        age_score_rows.append(scores)
        for key in scores:
            if key not in all_age_keys and not key.endswith("_mode"):
                all_age_keys.append(key)

    # Order by AGE_BIN_ORDER, append unknowns at the end
    ordered = [k for k in AGE_BIN_ORDER if k in all_age_keys]
    ordered += [k for k in all_age_keys if k not in ordered]

    for key in ordered:
        df[key] = [s.get(key, float("nan")) for s in age_score_rows]

    # Parse per-batch scores (added in newer runs; absent in older ones)
    if "batch_scores_json" in df.columns:
        all_batch_score_rows: list[dict] = []
        all_batch_keys: set[str] = set()
        for _, row in df.iterrows():
            if pd.notna(row.get("batch_scores_json")):
                scores = json.loads(row["batch_scores_json"])
            else:
                scores = {}
            all_batch_score_rows.append(scores)
            all_batch_keys.update(scores.keys())

        for key in sorted(all_batch_keys):
            df[f"batch__{key}"] = [s.get(key, float("nan")) for s in all_batch_score_rows]

    df = df.sort_values("objective", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    scanvi_path = input_dir / "scanvi_comparison.csv"
    scanvi = pd.read_csv(scanvi_path) if scanvi_path.exists() else None

    return df, scanvi


def _param_label(row: pd.Series) -> str:
    return (
        f"L={int(row['n_latent'])} H={int(row['n_hidden'])} "
        f"lay={int(row['n_layers'])} {row['gene_likelihood']} bs={int(row['batch_size'])}"
    )


# ---------------------------------------------------------------------------
# Figure 1: Trial ranking
# ---------------------------------------------------------------------------

def fig_trial_ranking(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, max(5, len(df) * 0.55 + 1.5)),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Trial Objective Ranking", fontsize=13, fontweight="bold")

    # Left: horizontal bars coloured by n_layers
    ax = axes[0]
    n_layers_vals = df["n_layers"].astype(int)
    colors = [PALETTE.get(v, "#888888") for v in n_layers_vals]
    y = np.arange(len(df))
    ax.barh(y, df["objective"], color=colors, edgecolor="white", height=0.7)

    # Overlay the batch-mixing-PCA component as a lighter inner bar (80% scaled)
    bm_col = "batch_mixing_pca_score" if "batch_mixing_pca_score" in df.columns else None
    if bm_col:
        ax.barh(y, df[bm_col] * 0.8, color=[c + "88" for c in colors],
                edgecolor="none", height=0.3, label="_bm_pca component")

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["objective"] + 0.002, i, _param_label(row),
                va="center", fontsize=7.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([f"T{int(r['trial'])}" for _, r in df.iterrows()], fontsize=8)
    ax.set_xlabel("Objective score")
    present_layers = sorted(df["n_layers"].astype(int).unique())
    ax.set_title(f"Objective (coloured by n_layers; layers present: {present_layers})")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, df["objective"].max() * 1.35)

    legend_patches = [
        mpatches.Patch(color=PALETTE.get(lay, "#888888"), label=f"n_layers={lay}")
        for lay in present_layers
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    # Right: batch_mixing_pca_score (y) vs decoder_score_norm (x)
    ax2 = axes[1]
    # Use whichever column is present; None means no data available for that axis.
    bm_col  = "batch_mixing_pca_score" if "batch_mixing_pca_score" in df.columns else None
    dec_col = "decoder_score_norm"     if "decoder_score_norm"     in df.columns else None

    x_vals = (
        df[dec_col].values if dec_col and dec_col in df.columns
        else np.full(len(df), float("nan"))
    )
    y_vals = (
        df[bm_col].values if bm_col and bm_col in df.columns
        else np.full(len(df), float("nan"))
    )

    sc = ax2.scatter(
        x_vals, y_vals,
        c=df["objective"], cmap="RdYlGn",
        vmin=df["objective"].min(), vmax=df["objective"].max(),
        s=80, edgecolors="gray", linewidths=0.5, zorder=3,
    )
    for i, (_, row) in enumerate(df.iterrows()):
        ax2.annotate(
            f"T{int(row['trial'])}",
            (x_vals[i], y_vals[i]),
            fontsize=6.5, ha="left", va="bottom",
        )
    fig.colorbar(sc, ax=ax2, label="Objective")
    ax2.set_xlabel("Decoder score norm (50%)")
    ax2.set_ylabel("Batch mixing PCA score (50%)")
    ax2.set_title("Score components\n(50:50 cross-trial normalized)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Age-bin batch mixing heatmap
# ---------------------------------------------------------------------------

def fig_age_bin_heatmap(df: pd.DataFrame) -> plt.Figure:
    # Use _pct_of_max columns only: normalises each bin by its structural ceiling
    # (bins with fewer present batches get lower max entropy; pct_of_max makes bins
    # directly comparable).  Showing only this metric keeps the heatmap compact.
    ordered_pct = [f"{b}_pct_of_max" for b in AGE_BIN_ORDER
                   if f"{b}_pct_of_max" in df.columns and df[f"{b}_pct_of_max"].notna().any()]
    extra_pct = [c for c in df.columns
                 if c.endswith("_pct_of_max")
                 and c not in ordered_pct
                 and df[c].notna().any()]
    pct_cols = ordered_pct + extra_pct
    if not pct_cols:
        # Fallback: no _pct_of_max columns (older run); show raw scores instead
        present_bins = [b for b in AGE_BIN_ORDER if b in df.columns and df[b].notna().any()]
        extra = [c for c in df.columns
                 if c.startswith("[") and c not in present_bins
                 and not any(c.endswith(s) for s in ("_pct_of_max", "_max_score", "_n_batches"))
                 and df[c].notna().any()]
        present_bins = present_bins + extra
        pct_cols = present_bins
        vmin, vmax, cbar_label, fmt = 0, 0.55, "Mixing score", ".2f"
    else:
        present_bins = [c[:-len("_pct_of_max")] for c in pct_cols]
        vmin, vmax, cbar_label, fmt = 0, 100, "% of max possible", ".0f"

    mat = df[pct_cols].values  # shape: (n_trials, n_bins)
    row_labels = [f"T{int(r['trial'])}  {_param_label(r)}" for _, r in df.iterrows()]
    col_labels = [AGE_BIN_LABELS.get(b, b) for b in present_bins]
    prenatal_idxs = [k for k, b in enumerate(present_bins) if _is_prenatal_bin(b)]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, max(5, len(df) * 0.55 + 2)))
    fig.suptitle("Age-Bin Batch Mixing: % of max possible score (higher = better)",
                 fontsize=12, fontweight="bold")

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.6)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title("Rows sorted best→worst by overall objective. Grey = bin absent (too few cells).")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=6.5, color="black" if 20 < v < 80 else "white")

    for idx in prenatal_idxs:
        ax.axvline(idx - 0.5, color="steelblue", linewidth=1.5, alpha=0.7)
        ax.axvline(idx + 0.5, color="steelblue", linewidth=1.5, alpha=0.7)
    if prenatal_idxs:
        ax.text(np.mean(prenatal_idxs), -0.8, "◄ PRENATAL (2× weight) ►",
                ha="center", va="top", fontsize=8, color="steelblue",
                transform=ax.get_xaxis_transform())

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Parameter effect plots
# ---------------------------------------------------------------------------

def fig_parameter_effects(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH, 8))
    fig.suptitle(
        "Hyperparameter Effect on Objective Score\n"
        "(NOTE: effects are confounded — control for n_layers when interpreting n_hidden / n_latent)",
        fontsize=12, fontweight="bold",
    )
    axes = axes.flatten()

    def _jitter(vals, scale=0.08):
        return vals + np.random.default_rng(0).uniform(-scale, scale, len(vals))

    # Use whichever batch mixing column is present (new runs: batch_mixing_pca_score,
    # old runs: age_batch_score for backward compatibility).
    bm_col = (
        "batch_mixing_pca_score" if "batch_mixing_pca_score" in df.columns
        else "age_batch_score" if "age_batch_score" in df.columns
        else None
    )
    metrics = ["objective"] + ([bm_col] if bm_col else [])
    cmap = plt.cm.Set2

    for ax_idx, param in enumerate(PARAM_COLS):
        ax = axes[ax_idx]
        unique_vals = sorted(df[param].unique(), key=lambda x: (str(type(x)), x))

        for m_idx, metric in enumerate(metrics):
            offset = (m_idx - 0.5) * 0.25
            x_pos = np.array([unique_vals.index(v) for v in df[param]], dtype=float) + offset

            # Color by n_layers to expose confounding
            point_colors = [PALETTE.get(int(r["n_layers"]), "#888888") for _, r in df.iterrows()]
            ax.scatter(_jitter(x_pos, 0.06), df[metric], s=55,
                       color=point_colors if m_idx == 0 else [c + "66" for c in point_colors],
                       alpha=0.9, label=metric if ax_idx == 0 else "_")

            # Mean line (all trials)
            for k, val in enumerate(unique_vals):
                subset = df[df[param] == val][metric]
                ax.plot([k + offset - 0.1, k + offset + 0.1],
                        [subset.mean()] * 2, color=cmap(m_idx), linewidth=2.5)

        ax.set_xticks(range(len(unique_vals)))
        ax.set_xticklabels([str(v) for v in unique_vals], fontsize=9)
        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"Effect of {param}")

    # Combined legend: metric colour + n_layers dot colour
    bm_label = bm_col if bm_col else "batch mixing (no column)"
    metric_handles = [
        mpatches.Patch(color=cmap(0), label="objective (filled)"),
    ]
    if bm_col:
        metric_handles.append(mpatches.Patch(color=cmap(1), label=f"{bm_label} (filled)"))
    layer_handles = [
        mpatches.Patch(color=c, label=f"n_layers={lay}")
        for lay, c in sorted(PALETTE.items()) if lay in df["n_layers"].astype(int).values
    ]
    axes[0].legend(handles=metric_handles + layer_handles, fontsize=7, ncol=2)
    axes[-1].axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Prenatal vs postnatal mixing detail
# ---------------------------------------------------------------------------

def fig_prenatal_focus(df: pd.DataFrame) -> plt.Figure:
    prenatal_bins = [b for b in AGE_BIN_ORDER
                     if b in df.columns and df[b].notna().any() and _is_prenatal_bin(b)]
    postnatal_bins = [b for b in AGE_BIN_ORDER
                      if b in df.columns and df[b].notna().any() and not _is_prenatal_bin(b)]

    # Avoid adult bins in "early postnatal" line profile
    early_postnatal = [b for b in postnatal_bins if not any(x in b for x in ["20.0", "40.0"])]

    df2 = df.copy()
    df2["prenatal_mean"] = df[prenatal_bins].mean(axis=1) if prenatal_bins else float("nan")
    df2["postnatal_mean"] = df[early_postnatal].mean(axis=1) if early_postnatal else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 5))
    fig.suptitle("Prenatal vs Postnatal Batch Mixing", fontsize=13, fontweight="bold")

    # Left: scatter prenatal vs postnatal mean, coloured by n_latent
    ax = axes[0]
    latent_vals = sorted(df2["n_latent"].unique())
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(latent_vals), max(latent_vals))
    sc = ax.scatter(df2["postnatal_mean"], df2["prenatal_mean"],
                    c=df2["n_latent"], cmap=cmap, norm=norm, s=90,
                    edgecolors="gray", linewidths=0.5, zorder=3)
    for _, row in df2.iterrows():
        ax.annotate(f"T{int(row['trial'])}", (row["postnatal_mean"], row["prenatal_mean"]),
                    fontsize=7, ha="left", va="bottom")
    fig.colorbar(sc, ax=ax, label="n_latent")
    ax.set_xlabel("Postnatal mean mixing (0–20y bins)")
    ax.set_ylabel("Prenatal mean mixing")
    ax.set_title("Prenatal vs postnatal mixing, coloured by n_latent")
    ax.grid(alpha=0.3)
    all_vals = pd.concat([df2["prenatal_mean"].dropna(), df2["postnatal_mean"].dropna()])
    lo = all_vals.min() - 0.01
    hi = all_vals.max() + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="parity line")
    ax.legend(fontsize=8)

    # Right: bin profiles for top-5 vs bottom-5 trials
    ax2 = axes[1]
    display_bins = prenatal_bins + early_postnatal[:4]
    n_show = min(5, len(df2))
    top_trials = df2.head(n_show)
    bot_trials = df2.tail(n_show)

    for i, (_, row) in enumerate(top_trials.iterrows()):
        scores = [row.get(b, float("nan")) for b in display_bins]
        ax2.plot(range(len(scores)), scores, "o-", color=plt.cm.Greens(0.5 + i * 0.1),
                 label=f"T{int(row['trial'])} (top)", linewidth=1.5, markersize=5)
    for i, (_, row) in enumerate(bot_trials.iterrows()):
        scores = [row.get(b, float("nan")) for b in display_bins]
        ax2.plot(range(len(scores)), scores, "s--", color=plt.cm.Reds(0.4 + i * 0.1),
                 label=f"T{int(row['trial'])} (bot)", linewidth=1.2, markersize=4)

    xlabels = [AGE_BIN_LABELS.get(b, b) for b in display_bins]
    ax2.set_xticks(range(len(xlabels)))
    ax2.set_xticklabels(xlabels, fontsize=7.5, rotation=20, ha="right")
    ax2.axvline(len(prenatal_bins) - 0.5, color="steelblue", linestyle=":", alpha=0.7,
                label="birth")
    ax2.set_ylabel("Batch mixing score")
    ax2.set_title("Top-5 vs bottom-5 trials across prenatal + early postnatal bins")
    ax2.legend(fontsize=6.5, ncol=2)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: scVI vs scANVI comparison
# ---------------------------------------------------------------------------

def fig_scanvi_comparison(df: pd.DataFrame, scanvi: pd.DataFrame) -> plt.Figure:
    merged = scanvi.merge(df[["trial"] + PARAM_COLS], on="trial", how="left")
    merged = merged.sort_values("scvi_objective", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 5))
    fig.suptitle("scVI vs scANVI: Objective and Component Scores", fontsize=13, fontweight="bold")

    ax = axes[0]
    x = np.arange(len(merged))
    w = 0.35
    bars1 = ax.bar(x - w / 2, merged["scvi_objective"], w, label="scVI", color="#4878CF", alpha=0.85)
    bars2 = ax.bar(x + w / 2, merged["scanvi_objective"], w, label="scANVI", color="#D65F5F", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{int(r['trial'])}\n{r['gene_likelihood']} L={int(r['n_latent'])}"
                        for _, r in merged.iterrows()], fontsize=8)
    ax.set_ylabel("Objective score (50:50 normalized)")
    ax.set_title("scVI → scANVI: objective comparison\n(same scVI cross-trial normalization scale)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for b1, b2 in zip(bars1, bars2):
        delta = b2.get_height() - b1.get_height()
        sign = "+" if delta >= 0 else ""
        ax.text(b2.get_x() + b2.get_width() / 2, b2.get_height() + 0.001,
                f"{sign}{delta:.3f}", ha="center", va="bottom", fontsize=7.5, color="#D65F5F")

    ax2 = axes[1]
    # Show batch mixing PCA scores side by side (scVI vs scANVI)
    # Use whichever column name is present for backward compatibility.
    scvi_bm_col = "scvi_batch_mixing_pca_score"
    scanvi_bm_col = "scanvi_batch_mixing_pca_score"
    has_bm = scvi_bm_col in merged.columns and scanvi_bm_col in merged.columns

    if has_bm:
        ax2.bar(x - w / 2, merged[scvi_bm_col],  w, label="scVI bm_pca",   color="#4878CF88", alpha=0.85)
        ax2.bar(x + w / 2, merged[scanvi_bm_col], w, label="scANVI bm_pca", color="#D65F5F88", alpha=0.85)
        ax2.set_ylabel("Batch mixing PCA score (expression space)")
        ax2.set_title("Batch mixing in expression PCA space\n(higher = better integrated)")
        ax2.set_ylim(0, 1.0)
    else:
        # Fallback for old runs that recorded recon_error instead
        scvi_re_col  = "scvi_recon_error"
        scanvi_re_col = "scanvi_recon_error"
        if scvi_re_col in merged.columns and scanvi_re_col in merged.columns:
            ax2.bar(x - w / 2, merged[scvi_re_col],  w, label="scVI recon",   color="#4878CF88", alpha=0.85)
            ax2.bar(x + w / 2, merged[scanvi_re_col], w, label="scANVI recon", color="#D65F5F88", alpha=0.85)
            ax2.set_ylabel("Reconstruction error (lower = better)")
            ax2.set_title("Decoder reconstruction error\n(lower = better generative model)")
        else:
            ax2.set_title("No component breakdown available")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"T{int(r['trial'])}" for _, r in merged.iterrows()], fontsize=8)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Per-batch global mixing scores
# ---------------------------------------------------------------------------

def fig_batch_mixing(df: pd.DataFrame) -> plt.Figure | None:
    """Heatmap showing how well each individual batch mixes with others.

    Uses global k-NN (not age-bin-restricted) so a batch that is globally
    isolated in expression PCA space — not just in one age bin — gets a low score.
    Wang batches are highlighted with a blue border.
    """
    batch_cols = sorted(c for c in df.columns if c.startswith("batch__"))
    if not batch_cols:
        return None

    batch_names = [c[len("batch__"):] for c in batch_cols]
    mat = df[batch_cols].values.T  # (n_batches, n_trials)

    # Sort batches ascending by mean score (worst-mixing at top)
    mean_scores = np.nanmean(mat, axis=1)
    order = np.argsort(mean_scores)
    mat = mat[order]
    sorted_names = [batch_names[i] for i in order]
    sorted_means = mean_scores[order]

    trial_labels = [f"T{int(r['trial'])}" for _, r in df.iterrows()]

    fig_h = max(3, len(sorted_names) * 0.65 + 2)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, fig_h))
    fig.suptitle(
        "Per-Batch Global Mixing Score  (k-NN entropy, higher = better integrated)\n"
        "Global k-NN on expression PCA — identifies batches isolated across the whole space, not just within one age bin.",
        fontsize=11, fontweight="bold",
    )

    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.55,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Mixing score", shrink=0.6)

    ax.set_xticks(range(len(trial_labels)))
    ax.set_xticklabels(trial_labels, fontsize=9)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Trial (sorted best→worst by scVI objective)")
    ax.set_title("Rows sorted worst→best mixing (worst at top). Blue border = Wang batch.")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.15 < v < 0.45 else "white")

    # Highlight Wang batches
    for i, name in enumerate(sorted_names):
        if "wang" in name.lower():
            rect = mpatches.FancyBboxPatch(
                (-0.5, i - 0.5), len(trial_labels), 1,
                boxstyle="square,pad=0", fill=False,
                edgecolor="steelblue", linewidth=2.5,
            )
            ax.add_patch(rect)

    # Right axis: mean score per batch
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(sorted_names)))
    ax2.set_yticklabels([f"μ={s:.3f}" for s in sorted_means], fontsize=8)
    ax2.set_ylabel("Mean mixing score across trials", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 7: UMAP grid — top-5 vs bottom-5 trials coloured by batch
# ---------------------------------------------------------------------------

def fig_umap_grid(
    df: pd.DataFrame,
    input_dir: Path,
    batch_key: str,
    n_top: int = 5,
    n_bot: int = 5,
    n_cells: int = N_UMAP_CELLS,
    seed: int = 77,
) -> plt.Figure | None:
    """2-row × 5-col UMAP grid: top-n (row 1) and bottom-n (row 2) trials coloured by batch.

    Requires trial_XX_latent.npy and obs_tuning.csv produced by tune_scvi_batch.py.
    A fixed random subsample of n_cells is used across all panels so batch colour
    proportions are directly comparable; only the UMAP layout differs between panels.
    Returns None when latent files are absent (e.g. older run results).
    """
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError:
        return None

    obs_path = input_dir / "obs_tuning.csv"
    if not obs_path.exists():
        return None

    obs_full = pd.read_csv(obs_path, index_col=0)
    if batch_key not in obs_full.columns:
        return None

    def _latent_path(t: int) -> Path:
        return input_dir / f"trial_{t:02d}_latent.npy"

    n_top = min(n_top, len(df))
    n_bot = min(n_bot, len(df) - n_top)
    top_rows = df.head(n_top)
    bot_rows = df.tail(n_bot)

    top_mask = [_latent_path(int(r["trial"])).exists() for _, r in top_rows.iterrows()]
    bot_mask = [_latent_path(int(r["trial"])).exists() for _, r in bot_rows.iterrows()]
    top_rows = top_rows[top_mask].reset_index(drop=True)
    bot_rows = bot_rows[bot_mask].reset_index(drop=True)

    if len(top_rows) == 0 and len(bot_rows) == 0:
        return None

    # Fixed subsample index shared across all panels
    rng = np.random.default_rng(seed)
    n_full = len(obs_full)
    n_sub = min(n_cells, n_full)
    sub_idx = np.sort(rng.choice(n_full, n_sub, replace=False))
    batch_sub = obs_full[batch_key].astype(str).iloc[sub_idx].values

    batches = sorted(obs_full[batch_key].astype(str).unique())
    tab_cmap = plt.cm.get_cmap("tab10", len(batches))
    batch_colors = {b: tab_cmap(i) for i, b in enumerate(batches)}

    n_cols = 5
    fig, axes = plt.subplots(2, n_cols, figsize=(FIGURE_WIDTH, 6.5))
    fig.suptitle(
        f"UMAP coloured by batch — top {len(top_rows)} trials (row 1) vs bottom {len(bot_rows)} trials (row 2)\n"
        f"Fixed {n_sub:,}-cell subsample per panel; each UMAP is independent — compare clustering, not coordinates",
        fontsize=10, fontweight="bold",
    )

    def _plot_one(ax: plt.Axes, trial_row: pd.Series) -> None:
        t = int(trial_row["trial"])
        X = np.load(str(_latent_path(t)))[sub_idx].astype(np.float32)

        adata_tmp = ad.AnnData(X=X)
        sc.pp.neighbors(adata_tmp, use_rep="X", n_neighbors=15, random_state=seed)
        sc.tl.umap(adata_tmp, min_dist=0.3, random_state=seed)
        coords = adata_tmp.obsm["X_umap"]

        for batch in batches:
            mask = batch_sub == batch
            if mask.any():
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[batch_colors[batch]], s=1.0, alpha=0.4, rasterized=True,
                )

        ax.set_title(
            f"T{t}  obj={trial_row['objective']:.3f}\n"
            f"lay={int(trial_row['n_layers'])} H={int(trial_row['n_hidden'])} "
            f"L={int(trial_row['n_latent'])} {trial_row['gene_likelihood']}",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for col, (_, row) in enumerate(top_rows.iterrows()):
        _plot_one(axes[0, col], row)
    for col in range(len(top_rows), n_cols):
        axes[0, col].axis("off")
    axes[0, 0].set_ylabel("Top trials", fontsize=9, labelpad=4)

    for col, (_, row) in enumerate(bot_rows.iterrows()):
        _plot_one(axes[1, col], row)
    for col in range(len(bot_rows), n_cols):
        axes[1, col].axis("off")
    axes[1, 0].set_ylabel("Bottom trials", fontsize=9, labelpad=4)

    legend_handles = [mpatches.Patch(color=batch_colors[b], label=b) for b in batches]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(batches),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Programmatic entry point (called from tune_scvi_batch.py after tuning)
# ---------------------------------------------------------------------------

def run_diagnostics(
    input_dir: Path,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    """Generate all diagnostic plots from tuning output.

    Called automatically at the end of run_tuning(), and also available as
    a standalone command via  python -m tuning.tuning_diagnostics.
    """

    def _log(msg: str) -> None:
        if logger:
            logger.info(msg)
        else:
            print(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Loading results from {input_dir} ...")

    try:
        df, scanvi = _load_results(input_dir)
    except FileNotFoundError as exc:
        _log(f"  Skipping plots: {exc}")
        return

    _log(f"  {len(df)} successful trials loaded")

    # Load key names for UMAP figure (written by tune_scvi_batch.py)
    meta_path = input_dir / "tuning_metadata.json"
    batch_key_meta: str | None = None
    if meta_path.exists():
        import json as _json
        with open(meta_path) as _mf:
            _meta = _json.load(_mf)
        batch_key_meta = _meta.get("batch_key")

    def _build_figures() -> dict[str, plt.Figure]:
        d, s = _load_results(input_dir)
        figs: dict[str, plt.Figure] = {
            "1_trial_ranking":     fig_trial_ranking(d),
            "2_age_bin_heatmap":   fig_age_bin_heatmap(d),
            "3_parameter_effects": fig_parameter_effects(d),
            "4_prenatal_focus":    fig_prenatal_focus(d),
        }
        if s is not None:
            _log(f"  scANVI comparison for {len(s)} trials loaded")
            figs["5_scanvi_comparison"] = fig_scanvi_comparison(d, s)
        batch_fig = fig_batch_mixing(d)
        if batch_fig is not None:
            figs["6_batch_mixing"] = batch_fig
        if batch_key_meta:
            _log(f"  Building UMAP grid (top-5 vs bottom-5, batch_key='{batch_key_meta}') ...")
            umap_fig = fig_umap_grid(d, input_dir, batch_key_meta)
            if umap_fig is not None:
                figs["7_umap_grid"] = umap_fig
            else:
                _log("  UMAP grid skipped (latent files not found; run produced by newer code only)")
        return figs

    # Write individual PNGs
    figures = _build_figures()
    for name, fig in figures.items():
        png_path = output_dir / f"{name}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        _log(f"  Wrote {png_path}")
        plt.close(fig)

    # Write combined PDF (regenerate figures to avoid closed-figure issues)
    pdf_path = output_dir / "tuning_diagnostics.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in _build_figures().values():
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    _log(f"  Wrote {pdf_path}")
    _log("Done.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise scVI tuning results")
    parser.add_argument("--input_dir",  required=True, help="Path to scvi_tuning output directory")
    parser.add_argument("--output_dir", default=None,  help="Where to write plots (default: input_dir/plots)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    run_diagnostics(input_dir, output_dir)


if __name__ == "__main__":
    main()
