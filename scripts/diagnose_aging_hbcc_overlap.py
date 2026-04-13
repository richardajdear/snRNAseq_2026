"""
Diagnose the degree of donor and cell-barcode overlap between the full
AGING (Aging_Cohort.h5ad) and HBCC (HBCC_Cohort.h5ad) source files.

Key question: is the near-zero cell-barcode overlap between AGING and HBCC
a property of the raw data (different physical cells captured), or an
artifact of the per-dataset random downsampling in the pipeline?

Reads both h5ads in backed mode (obs only — expression matrix never loaded).

Outputs:
    scripts/outputs/diagnose_aging_hbcc_overlap.txt

Usage:
    PYTHONPATH=code python scripts/diagnose_aging_hbcc_overlap.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = "/home/rajd2/rds/hpc-work/snRNAseq_2026"
sys.path.insert(0, os.path.join(REPO_ROOT, "code"))

from environment import get_environment
env = get_environment()
rds_dir = env["rds_dir"]

AGING_PATH = os.path.join(rds_dir, "Cam_PsychAD/RNAseq/Aging_Cohort.h5ad")
HBCC_PATH  = os.path.join(rds_dir, "Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad")
OUT_DIR    = os.path.join(REPO_ROOT, "scripts/outputs")
OUT_TXT    = os.path.join(OUT_DIR, "diagnose_aging_hbcc_overlap.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def pct(n, total):
    return f"{100 * n / total:.1f}%" if total > 0 else "N/A"


def read_obs_backed(path, label):
    """Open h5ad in backed mode, extract obs DataFrame, then close. No matrix loaded."""
    print(f"  Reading {label} obs from {path} ...", flush=True)
    ad = sc.read_h5ad(path, backed="r")
    obs = ad.obs.copy()           # lightweight: just the obs DataFrame
    ad.file.close()
    print(f"    {label}: {len(obs):,} cells, {len(obs.columns)} obs columns", flush=True)
    return obs


def donor_col(obs, label):
    """Return the donor/individual column name and values."""
    for col in ("individualID", "individual", "donor_id"):
        if col in obs.columns:
            return col, obs[col].astype(str)
    raise ValueError(f"{label}: no donor column found. Available: {list(obs.columns)}")


def print_and_write(lines, fh):
    text = "\n".join(lines)
    print(text)
    fh.write(text + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading obs from full source h5ads (backed mode) ...", flush=True)
    aging_obs = read_obs_backed(AGING_PATH, "AGING")
    hbcc_obs  = read_obs_backed(HBCC_PATH,  "HBCC")

    aging_dcol, aging_donors_s = donor_col(aging_obs, "AGING")
    hbcc_dcol,  hbcc_donors_s  = donor_col(hbcc_obs,  "HBCC")

    aging_barcodes = set(aging_obs.index.astype(str))
    hbcc_barcodes  = set(hbcc_obs.index.astype(str))

    aging_donors = set(aging_donors_s.unique())
    hbcc_donors  = set(hbcc_donors_s.unique())

    shared_donors   = aging_donors & hbcc_donors
    shared_barcodes = aging_barcodes & hbcc_barcodes

    # ── Cells belonging to shared donors ─────────────────────────────────────

    aging_cells_shared_donor = aging_obs[aging_donors_s.isin(shared_donors)]
    hbcc_cells_shared_donor  = hbcc_obs[hbcc_donors_s.isin(shared_donors)]

    # ── Per-donor cell counts for shared donors ───────────────────────────────

    aging_n_per_shared = aging_donors_s[aging_donors_s.isin(shared_donors)].value_counts()
    hbcc_n_per_shared  = hbcc_donors_s[hbcc_donors_s.isin(shared_donors)].value_counts()

    # ── Are shared barcodes (if any) on the same donor? ──────────────────────

    shared_bc_info = []
    if shared_barcodes:
        for bc in sorted(shared_barcodes):
            a_donor = aging_donors_s.get(bc, "—") if bc in aging_obs.index else "—"
            h_donor = hbcc_donors_s.get(bc, "—")  if bc in hbcc_obs.index  else "—"
            shared_bc_info.append((bc, a_donor, h_donor, a_donor == h_donor))

    # ── Age distribution of shared donors ────────────────────────────────────

    age_col_aging = next((c for c in ("development_stage", "age", "age_years") if c in aging_obs.columns), None)
    age_col_hbcc  = next((c for c in ("development_stage", "age", "age_years") if c in hbcc_obs.columns), None)

    shared_donor_ages = {}
    if age_col_aging:
        # One row per donor (take first cell's value — all cells from same donor share age)
        shared_mask_a = aging_donors_s.isin(shared_donors)
        age_series = aging_obs.loc[shared_mask_a, age_col_aging]
        shared_donor_ages = (
            pd.Series(aging_donors_s[shared_mask_a].values, index=age_series.index)
            .rename("donor")
            .to_frame()
            .assign(age=age_series.values)
            .groupby("donor")["age"]
            .first()
        )

    # ── Sex of shared donors ──────────────────────────────────────────────────

    sex_col = next((c for c in ("sex", "gender") if c in aging_obs.columns), None)
    if sex_col:
        shared_mask_a = aging_donors_s.isin(shared_donors)
        sex_series = aging_obs.loc[shared_mask_a, sex_col]
        shared_donor_sex = (
            pd.Series(aging_donors_s[shared_mask_a].values, index=sex_series.index)
            .rename("donor")
            .to_frame()
            .assign(sex=sex_series.values)
            .groupby("donor")["sex"]
            .first()
        )
    else:
        shared_donor_sex = pd.Series(dtype=str)

    # ── Disease/diagnosis column ──────────────────────────────────────────────

    dx_col = next((c for c in ("disease", "diagnosis", "Dx", "dx") if c in aging_obs.columns), None)

    # ── Format report ─────────────────────────────────────────────────────────

    lines = []
    sep  = "=" * 72
    sep2 = "-" * 72

    lines += [
        sep,
        "  OVERLAP DIAGNOSIS: AGING vs HBCC (full source h5ads)",
        sep,
        "",
        "  This checks whether cell-barcode overlap is a property of the raw",
        "  source data or an artifact of downsampling.",
        "",
    ]

    # 1. Basic counts
    lines += [
        "── 1. CELL AND DONOR COUNTS ─────────────────────────────────────────",
        f"  AGING  total cells:  {len(aging_obs):>10,}",
        f"  HBCC   total cells:  {len(hbcc_obs):>10,}",
        f"  AGING  unique donors ({aging_dcol}):  {len(aging_donors):,}",
        f"  HBCC   unique donors ({hbcc_dcol}):   {len(hbcc_donors):,}",
        "",
    ]

    # 2. Donor overlap
    lines += [
        "── 2. DONOR OVERLAP ─────────────────────────────────────────────────",
        f"  Shared donors (AGING ∩ HBCC):  {len(shared_donors):,}",
        f"    as % of AGING donors:        {pct(len(shared_donors), len(aging_donors))}",
        f"    as % of HBCC donors:         {pct(len(shared_donors), len(hbcc_donors))}",
        f"  AGING-only donors:             {len(aging_donors - hbcc_donors):,}",
        f"  HBCC-only donors:              {len(hbcc_donors - aging_donors):,}",
        "",
        f"  Cells in AGING from shared donors:  {len(aging_cells_shared_donor):,}  "
        f"({pct(len(aging_cells_shared_donor), len(aging_obs))} of AGING cells)",
        f"  Cells in HBCC  from shared donors:  {len(hbcc_cells_shared_donor):,}  "
        f"({pct(len(hbcc_cells_shared_donor), len(hbcc_obs))} of HBCC cells)",
        "",
    ]

    # 3. Cell barcode overlap (the key question)
    lines += [
        "── 3. CELL BARCODE OVERLAP (raw source data) ────────────────────────",
        f"  THIS IS THE KEY QUESTION: are shared donors sequenced as the same",
        f"  physical cells, or as different cells in independent experiments?",
        "",
        f"  Shared cell barcodes (AGING ∩ HBCC):  {len(shared_barcodes):,}",
        f"    as % of AGING barcodes:              {pct(len(shared_barcodes), len(aging_barcodes))}",
        f"    as % of HBCC barcodes:               {pct(len(shared_barcodes), len(hbcc_barcodes))}",
        "",
    ]

    if len(shared_barcodes) == 0:
        lines += [
            "  CONCLUSION: Zero shared cell barcodes in the raw source files.",
            "  The donor overlap is NOT due to downsampling. AGING and HBCC are",
            "  independent sequencing experiments on the same donors: the same",
            "  brain tissue was profiled twice, yielding different single cells",
            "  each time. The ~27 coincidental barcode matches previously seen",
            "  were an artifact of the downsampled per_dataset h5ads.",
            "",
        ]
    elif len(shared_barcodes) <= 50:
        lines += [
            f"  Very few shared barcodes ({len(shared_barcodes)}). Likely coincidental",
            "  barcode collisions (10x barcodes are short and can repeat across runs).",
            "  These are not the same physical cells.",
            "",
            "  Shared barcode details (barcode, AGING donor, HBCC donor, same donor?):",
        ]
        for bc, ad_, hd, same in shared_bc_info[:20]:
            lines.append(f"    {bc:30s}  AGING:{ad_:15s}  HBCC:{hd:15s}  same={'YES' if same else 'no'}")
        if len(shared_bc_info) > 20:
            lines.append(f"    ... ({len(shared_bc_info) - 20} more not shown)")
        lines.append("")
    else:
        lines += [
            f"  WARNING: {len(shared_barcodes):,} shared barcodes found.",
            "  This is unexpectedly high. These may be truly identical cells",
            "  (same physical capture) and would need careful de-duplication.",
            "",
        ]

    # 4. Per-donor cell counts for shared donors
    if len(shared_donors) > 0:
        merged = pd.DataFrame({
            "AGING_cells": aging_n_per_shared,
            "HBCC_cells":  hbcc_n_per_shared,
        }).fillna(0).astype(int)
        merged["total"] = merged["AGING_cells"] + merged["HBCC_cells"]

        lines += [
            "── 4. PER-DONOR CELL COUNTS (shared donors, full data) ──────────────",
            f"  N shared donors: {len(merged)}",
            f"  AGING cells per shared donor — mean: {merged['AGING_cells'].mean():.0f}, "
            f"median: {merged['AGING_cells'].median():.0f}, "
            f"range: {merged['AGING_cells'].min()}–{merged['AGING_cells'].max()}",
            f"  HBCC  cells per shared donor — mean: {merged['HBCC_cells'].mean():.0f}, "
            f"median: {merged['HBCC_cells'].median():.0f}, "
            f"range: {merged['HBCC_cells'].min()}–{merged['HBCC_cells'].max()}",
            "",
            "  Top 15 shared donors by total cell count:",
            f"  {'Donor':20s}  {'AGING':>10s}  {'HBCC':>10s}  {'Total':>10s}",
            f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}",
        ]
        for donor, row in merged.sort_values("total", ascending=False).head(15).iterrows():
            lines.append(
                f"  {str(donor):20s}  {row['AGING_cells']:>10,}  {row['HBCC_cells']:>10,}  {row['total']:>10,}"
            )
        lines.append("")

    # 5. Age distribution of shared donors
    if len(shared_donor_ages) > 0:
        lines += [
            "── 5. AGE DISTRIBUTION OF SHARED DONORS ────────────────────────────",
            f"  (ages read from AGING obs['{age_col_aging}'])",
        ]
        try:
            # Try to parse numeric age if the column is development_stage strings
            import re
            def _parse_age(s):
                m = re.search(r"([\d.]+)\s*year", str(s))
                return float(m.group(1)) if m else np.nan
            numeric_ages = shared_donor_ages.map(_parse_age).dropna()
            if len(numeric_ages) > 0:
                lines += [
                    f"  N with parseable age: {len(numeric_ages)} / {len(shared_donor_ages)}",
                    f"  Mean age:   {numeric_ages.mean():.1f} years",
                    f"  Median age: {numeric_ages.median():.1f} years",
                    f"  Range:      {numeric_ages.min():.1f}–{numeric_ages.max():.1f} years",
                ]
                bins = [0, 20, 40, 60, 80, 200]
                labels = ["<20", "20-40", "40-60", "60-80", "80+"]
                binned = pd.cut(numeric_ages, bins=bins, labels=labels)
                counts = binned.value_counts().sort_index()
                lines.append("  Age bin distribution:")
                for lbl, cnt in counts.items():
                    lines.append(f"    {lbl:10s}: {cnt:3d} donors")
            else:
                # Raw string values
                counts = shared_donor_ages.value_counts().head(10)
                lines += [f"  Raw values (top 10): {dict(counts)}"]
        except Exception as e:
            lines.append(f"  (could not parse ages: {e})")
        lines.append("")

    # 6. Sex distribution of shared donors
    if len(shared_donor_sex) > 0:
        counts = shared_donor_sex.value_counts()
        lines += [
            "── 6. SEX DISTRIBUTION OF SHARED DONORS ────────────────────────────",
        ]
        for sex, cnt in counts.items():
            lines.append(f"  {sex}: {cnt} donors")
        lines.append("")

    # 7. Available obs columns (for reference)
    lines += [
        "── 7. AVAILABLE OBS COLUMNS ─────────────────────────────────────────",
        f"  AGING: {sorted(aging_obs.columns.tolist())}",
        f"  HBCC:  {sorted(hbcc_obs.columns.tolist())}",
        "",
        sep,
        "  END OF REPORT",
        sep,
    ]

    # ── Write output ──────────────────────────────────────────────────────────

    with open(OUT_TXT, "w") as fh:
        for line in lines:
            fh.write(line + "\n")

    for line in lines:
        print(line)

    print(f"\nReport written to: {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
