"""
Soft composition validation for the scANVI v3 pipeline.

Reads pseudobulk_output/by_cell_class.h5ad, computes per-donor broad-class
proportions by (source-chemistry × age-bin), writes:
  - composition_by_age_bin.csv
  - validation_summary.json
  - fig_composition_vs_age.png
  - fig_under_1y_composition_bar.png
  - errors.log (populated only if something crashes internally)

Always exits 0 — failures write to errors.log and emit a logging.warning
in the caller's pipeline.log. The validation checks are soft (warn-only).

Usage (called from run_pipeline.py::step_pseudobulk or directly):
    python -m pipeline.composition_check \\
        --input pseudobulk_output/by_cell_class.h5ad \\
        --output_dir pseudobulk_output/composition_check
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Broad class → EN/IN labels (must match cell_class values in integrated.h5ad)
# ---------------------------------------------------------------------------
EN_CLASS = "Excitatory"
IN_CLASS = "Inhibitory"

# Validation thresholds (from the plan)
EN_GAP_MAX_PP = 15.0          # |Vel-V3 EN% - PsychAD-V3 EN%| < 15 pp
EN_MIN_PCT = 30.0             # both means > 30%
EN_IN_RATIO_MIN = 2.0         # EN% / IN% > 2.0


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Composition validation check")
    p.add_argument("--input", required=True,
                   help="Path to by_cell_class.h5ad (pseudobulk output)")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write outputs")
    return p.parse_args(argv)


def _safe_load_h5ad(path: str):
    """Load h5ad; return (adata, error_msg). error_msg is None on success."""
    try:
        import anndata as ad
        return ad.read_h5ad(path), None
    except Exception:
        return None, traceback.format_exc()


def _compute_proportions(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Given obs with columns [individual, source-chemistry, cell_class, n_cells],
    compute per-donor × source-chemistry proportions of each broad class.

    Returns a DataFrame with columns:
      individual, source_chemistry, age_years, EN_pct, IN_pct, total_cells
    """
    required = {"individual", "source-chemistry", "cell_class", "n_cells", "age_years"}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"obs missing required columns: {missing}")

    # Total cells per donor × source-chemistry
    total = (
        obs.groupby(["individual", "source-chemistry", "age_years"], observed=True)["n_cells"]
        .sum()
        .reset_index()
        .rename(columns={"n_cells": "total_cells"})
    )

    # EN and IN counts
    en_counts = (
        obs[obs["cell_class"] == EN_CLASS]
        .groupby(["individual", "source-chemistry", "age_years"], observed=True)["n_cells"]
        .sum()
        .reset_index()
        .rename(columns={"n_cells": "en_cells"})
    )
    in_counts = (
        obs[obs["cell_class"] == IN_CLASS]
        .groupby(["individual", "source-chemistry", "age_years"], observed=True)["n_cells"]
        .sum()
        .reset_index()
        .rename(columns={"n_cells": "in_cells"})
    )

    df = total.merge(en_counts, on=["individual", "source-chemistry", "age_years"], how="left")
    df = df.merge(in_counts, on=["individual", "source-chemistry", "age_years"], how="left")
    df["en_cells"] = df["en_cells"].fillna(0)
    df["in_cells"] = df["in_cells"].fillna(0)
    df["EN_pct"] = 100.0 * df["en_cells"] / df["total_cells"].replace(0, np.nan)
    df["IN_pct"] = 100.0 * df["in_cells"] / df["total_cells"].replace(0, np.nan)
    df = df.rename(columns={"source-chemistry": "source_chemistry"})
    return df


def _age_bin(age_years: pd.Series) -> pd.Series:
    """Assign each donor to a broad age bin."""
    bins = [-np.inf, 0, 1, 5, 18, 30, 50, np.inf]
    labels = ["prenatal", "<1y", "1-5y", "5-18y", "18-30y", "30-50y", "50+y"]
    return pd.cut(age_years, bins=bins, labels=labels, right=False)


def _composition_by_age_bin(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["age_bin"] = _age_bin(df2["age_years"])
    grouped = (
        df2.groupby(["source_chemistry", "age_bin"], observed=True)
        .agg(
            n_donors=("individual", "count"),
            mean_EN_pct=("EN_pct", "mean"),
            mean_IN_pct=("IN_pct", "mean"),
        )
        .reset_index()
    )
    return grouped


def _run_validation_checks(df: pd.DataFrame) -> dict:
    """
    Run the four validation checks from the plan.
    Returns a dict of {check_name: {passed, value, target, notes}}.
    """
    results = {}

    # Identify Velmeshev-V3 and PsychAD-V3 source-chemistry combos
    vel_key = "VELMESHEV-V3"
    psychad_key = "PSYCHAD-V3"

    df_under1y = df[df["age_years"] < 1.0].copy()

    vel_under1y = df_under1y[df_under1y["source_chemistry"].str.contains("VELMESHEV", case=False, na=False)]
    psychad_under1y = df_under1y[df_under1y["source_chemistry"].str.contains("PSYCHAD", case=False, na=False)]

    vel_en_mean = float(vel_under1y["EN_pct"].mean()) if len(vel_under1y) > 0 else float("nan")
    psychad_en_mean = float(psychad_under1y["EN_pct"].mean()) if len(psychad_under1y) > 0 else float("nan")
    vel_in_mean = float(vel_under1y["IN_pct"].mean()) if len(vel_under1y) > 0 else float("nan")
    psychad_in_mean = float(psychad_under1y["IN_pct"].mean()) if len(psychad_under1y) > 0 else float("nan")

    # 1. EN% gap < 15 pp
    gap = abs(vel_en_mean - psychad_en_mean) if not (np.isnan(vel_en_mean) or np.isnan(psychad_en_mean)) else float("nan")
    results["vel_v3_psychad_v3_en_pct_under_1y_gap"] = {
        "passed": bool(gap < EN_GAP_MAX_PP) if not np.isnan(gap) else False,
        "value": round(gap, 2) if not np.isnan(gap) else None,
        "target": f"< {EN_GAP_MAX_PP} pp",
        "notes": (
            f"Vel-V3 <1y mean EN%={vel_en_mean:.1f}%, "
            f"PsychAD-V3 <1y mean EN%={psychad_en_mean:.1f}%, "
            f"gap={gap:.1f} pp"
            if not np.isnan(gap) else
            f"Insufficient data: Vel n={len(vel_under1y)}, PsychAD n={len(psychad_under1y)}"
        ),
    }

    # 2. Both means > 30%
    both_above = (vel_en_mean > EN_MIN_PCT) and (psychad_en_mean > EN_MIN_PCT)
    results["vel_v3_psychad_v3_en_pct_under_1y_both_above_30"] = {
        "passed": bool(both_above) if not (np.isnan(vel_en_mean) or np.isnan(psychad_en_mean)) else False,
        "value": {
            "vel_en_pct": round(vel_en_mean, 2) if not np.isnan(vel_en_mean) else None,
            "psychad_en_pct": round(psychad_en_mean, 2) if not np.isnan(psychad_en_mean) else None,
        },
        "target": f"both > {EN_MIN_PCT}%",
        "notes": (
            f"Vel-V3 <1y EN%={vel_en_mean:.1f}%, PsychAD-V3 <1y EN%={psychad_en_mean:.1f}%"
            if not (np.isnan(vel_en_mean) or np.isnan(psychad_en_mean)) else
            "Insufficient data"
        ),
    }

    # 3. PsychAD-V3 EN/IN ratio > 2.0
    psychad_ratio = (psychad_en_mean / psychad_in_mean) if (
        not np.isnan(psychad_en_mean) and not np.isnan(psychad_in_mean) and psychad_in_mean > 0
    ) else float("nan")
    results["psychad_v3_under_1y_en_in_ratio"] = {
        "passed": bool(psychad_ratio > EN_IN_RATIO_MIN) if not np.isnan(psychad_ratio) else False,
        "value": round(psychad_ratio, 2) if not np.isnan(psychad_ratio) else None,
        "target": f"> {EN_IN_RATIO_MIN}",
        "notes": (
            f"PsychAD-V3 <1y EN%={psychad_en_mean:.1f}%, IN%={psychad_in_mean:.1f}%, ratio={psychad_ratio:.2f}"
            if not np.isnan(psychad_ratio) else
            f"Insufficient data: PsychAD <1y n={len(psychad_under1y)}, mean_IN%={psychad_in_mean:.1f}%"
        ),
    }

    # 4. Vel-V3 EN/IN ratio > 2.0
    vel_ratio = (vel_en_mean / vel_in_mean) if (
        not np.isnan(vel_en_mean) and not np.isnan(vel_in_mean) and vel_in_mean > 0
    ) else float("nan")
    results["vel_v3_under_1y_en_in_ratio"] = {
        "passed": bool(vel_ratio > EN_IN_RATIO_MIN) if not np.isnan(vel_ratio) else False,
        "value": round(vel_ratio, 2) if not np.isnan(vel_ratio) else None,
        "target": f"> {EN_IN_RATIO_MIN}",
        "notes": (
            f"Vel-V3 <1y EN%={vel_en_mean:.1f}%, IN%={vel_in_mean:.1f}%, ratio={vel_ratio:.2f}"
            if not np.isnan(vel_ratio) else
            f"Insufficient data: Vel <1y n={len(vel_under1y)}, mean_IN%={vel_in_mean:.1f}%"
        ),
    }

    return results


def _plot_composition_vs_age(df: pd.DataFrame, output_dir: Path):
    """Donor-level scatter of EN%/IN% vs age, faceted by source-chemistry."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        source_chemistries = sorted(df["source_chemistry"].unique())
        n_sources = len(source_chemistries)
        if n_sources == 0:
            return

        fig, axes = plt.subplots(
            n_sources, 2, figsize=(10, 3 * n_sources), squeeze=False
        )
        for i, sc in enumerate(source_chemistries):
            sub = df[df["source_chemistry"] == sc]
            ax_en = axes[i, 0]
            ax_in = axes[i, 1]
            ax_en.scatter(sub["age_years"], sub["EN_pct"], s=20, alpha=0.6, color="steelblue")
            ax_en.axvline(1, color="red", linestyle="--", linewidth=0.8, label="1 y")
            ax_en.set_title(f"{sc} — EN%")
            ax_en.set_xlabel("Age (years)")
            ax_en.set_ylabel("EN %")
            ax_en.set_ylim(0, 100)
            ax_in.scatter(sub["age_years"], sub["IN_pct"], s=20, alpha=0.6, color="darkorange")
            ax_in.axvline(1, color="red", linestyle="--", linewidth=0.8, label="1 y")
            ax_in.set_title(f"{sc} — IN%")
            ax_in.set_xlabel("Age (years)")
            ax_in.set_ylabel("IN %")
            ax_in.set_ylim(0, 100)

        fig.tight_layout()
        fig.savefig(str(output_dir / "fig_composition_vs_age.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        logger.warning(f"fig_composition_vs_age.png failed: {traceback.format_exc()}")


def _plot_under1y_bar(df: pd.DataFrame, output_dir: Path):
    """Grouped bar of mean broad-class% per source-chemistry at <1 y."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sub = df[df["age_years"] < 1.0].copy()
        if sub.empty:
            logger.warning("No <1y donors found; skipping fig_under_1y_composition_bar.png")
            return

        means = (
            sub.groupby("source_chemistry", observed=True)
            .agg(mean_EN=("EN_pct", "mean"), mean_IN=("IN_pct", "mean"))
            .reset_index()
        )

        x = np.arange(len(means))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(means) * 1.5), 5))
        ax.bar(x - width / 2, means["mean_EN"], width, label="EN%", color="steelblue")
        ax.bar(x + width / 2, means["mean_IN"], width, label="IN%", color="darkorange")
        ax.axhline(30, color="gray", linestyle="--", linewidth=0.8, label="30% threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(means["source_chemistry"], rotation=30, ha="right")
        ax.set_ylabel("Mean % of cells")
        ax.set_title("Broad class composition at <1 y by source-chemistry")
        ax.legend()
        ax.set_ylim(0, 100)
        fig.tight_layout()
        fig.savefig(
            str(output_dir / "fig_under_1y_composition_bar.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)
    except Exception:
        logger.warning(f"fig_under_1y_composition_bar.png failed: {traceback.format_exc()}")


def main(argv=None):
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors_log = output_dir / "errors.log"
    errors_log.write_text("")  # clear / create

    # Configure logging to stderr so run_pipeline.py sees it
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        stream=sys.stderr,
    )

    try:
        _main_body(args, output_dir, errors_log)
    except Exception:
        tb = traceback.format_exc()
        errors_log.write_text(tb)
        logger.warning(f"composition_check encountered an unhandled error — see {errors_log}")
        logger.warning(tb)
        # Always exit 0 (soft validation)
        sys.exit(0)


def _main_body(args, output_dir: Path, errors_log: Path):
    # --- Load ---
    if not Path(args.input).exists():
        msg = f"Input not found: {args.input}"
        errors_log.write_text(msg + "\n")
        logger.warning(f"composition_check: {msg}")
        return

    adata, err = _safe_load_h5ad(args.input)
    if adata is None:
        errors_log.write_text(err)
        logger.warning(f"composition_check: failed to load {args.input}\n{err}")
        return

    # --- Build obs DataFrame ---
    # by_cell_class.h5ad has obs keyed by (individual, cell_class) pair
    obs = adata.obs.copy()

    # Expect n_cells column (pseudobulk aggregation count)
    if "n_cells" not in obs.columns:
        # Fall back to summing the counts layer
        if "counts" in adata.layers:
            obs["n_cells"] = np.asarray(adata.layers["counts"].sum(axis=1)).flatten()
        else:
            msg = "obs has no 'n_cells' column and counts layer missing — cannot compute proportions"
            errors_log.write_text(msg + "\n")
            logger.warning(f"composition_check: {msg}")
            return

    # source-chemistry may be stored as a hyphen-separated column
    if "source-chemistry" not in obs.columns and "source" in obs.columns and "chemistry" in obs.columns:
        obs["source-chemistry"] = obs["source"].astype(str) + "-" + obs["chemistry"].astype(str)

    # age_years: might be stored as mean per donor-cellclass combo
    if "age_years" not in obs.columns:
        msg = "'age_years' not found in obs — skipping composition check"
        errors_log.write_text(msg + "\n")
        logger.warning(f"composition_check: {msg}")
        return

    # cell_class column
    if "cell_class" not in obs.columns:
        msg = "'cell_class' not found in obs — skipping composition check"
        errors_log.write_text(msg + "\n")
        logger.warning(f"composition_check: {msg}")
        return

    if "individual" not in obs.columns:
        msg = "'individual' not found in obs — skipping composition check"
        errors_log.write_text(msg + "\n")
        logger.warning(f"composition_check: {msg}")
        return

    # --- Compute proportions ---
    df = _compute_proportions(obs)

    # --- Composition by age bin ---
    age_bin_df = _composition_by_age_bin(df)
    age_bin_df.to_csv(str(output_dir / "composition_by_age_bin.csv"), index=False)
    logger.info(f"composition_check: wrote composition_by_age_bin.csv ({len(age_bin_df)} rows)")

    # --- Validation checks ---
    results = _run_validation_checks(df)
    with open(str(output_dir / "validation_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"composition_check: wrote validation_summary.json")

    # --- Summary log line (matches format in plan) ---
    n_passed = sum(v["passed"] for v in results.values())
    n_total = len(results)
    details = []
    for k, v in results.items():
        status = "PASS" if v["passed"] else f"FAIL"
        val = f"{v['value']}" if v["value"] is not None else "N/A"
        details.append(f"{k}: {val} {status}")
    summary_line = f"Composition check: {n_passed}/{n_total} PASSED ({', '.join(details)})"
    logger.info(summary_line)

    # Emit individual warnings for failed checks
    for k, v in results.items():
        if not v["passed"]:
            logger.warning(f"  FAIL {k}: {v['notes']} (target: {v['target']})")

    # --- Plots ---
    _plot_composition_vs_age(df, output_dir)
    _plot_under1y_bar(df, output_dir)

    logger.info(f"composition_check: outputs written to {output_dir}")


if __name__ == "__main__":
    main()
