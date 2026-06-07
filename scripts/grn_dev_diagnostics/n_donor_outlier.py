#!/usr/bin/env python3
"""
N — investigate PsychAD pediatric outlier donor(s).

User recalls an earlier candidate outlier (~Donor_1400 / Donor_1004) with
abnormally high immature-neuron %. This script:
  1. For every PsychAD-V3 donor in the 1-25 y window, tabulates marker-
     annotation composition (% ExN_mature / ExN_immature / ExN_weak / InN),
     median UMI, n_cells, age, mean per-cell C3+ score, layer composition.
  2. Flags donors that are >2 SD outliers on (a) ExN_immature %, (b) per-
     cell-CPM C3+ score, (c) cell count, (d) median UMI.
  3. Reads f3_psychad_devwindow_donors.csv to confirm clinical
     metadata (already established normals, all HBCC).
  4. Recomputes the fuzzy d at the main 3-12 k UMI window with each
     candidate outlier removed, one at a time.

Login-safe — uses the per-cell parquet cache built by M.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (OUT_DIR, fuzzy_d_from_donor_scores,
                   FUZZY_BOUNDARIES, AGE_LO, AGE_HI)

PER_CELL_CACHE = OUT_DIR / "m_per_cell_cache.parquet"
F3_DONORS      = OUT_DIR / "f3_psychad_devwindow_donors.csv"
MAIN_LO, MAIN_HI = 3000, 12000
MIN_CELLS = 5


def main():
    per_cell = pd.read_parquet(PER_CELL_CACHE)
    psy = per_cell[per_cell["group"] == "PsychAD-V3"].copy()
    print(f"PsychAD-V3 cells: {len(psy):,}; donors: {psy['individual'].nunique()}")

    # ---------- per-donor table ----------
    rows = []
    for don, sub in psy.groupby("individual"):
        mc = sub["marker_annotation"].value_counts(normalize=True)
        lc = sub["layer"].value_counts(normalize=True)
        rows.append({
            "individual": don,
            "age_years": float(sub["age_years"].iloc[0]),
            "n_cells": int(len(sub)),
            "median_umi": float(sub["total_umi"].median()),
            "mean_umi":   float(sub["total_umi"].mean()),
            "pct_exn_mature":   float(mc.get("ExN_mature",   0.0)),
            "pct_exn_immature": float(mc.get("ExN_immature", 0.0)),
            "pct_exn_weak":     float(mc.get("ExN_weak",     0.0)),
            "pct_upper":  float(lc.get("upper",     0.0)),
            "pct_L5_ET":  float(lc.get("L5_ET",     0.0)),
            "pct_L6_CT":  float(lc.get("L6_CT",     0.0)),
            "pct_L6_IT":  float(lc.get("L6_IT",     0.0)),
            "pct_ambig":  float(lc.get("ambiguous", 0.0)),
            "c3_per_cell_mean": float(sub["per_cell_c3"].mean()),
        })
    don_df = pd.DataFrame(rows)
    # add depth-matched donor score too
    psy_dm = psy[(psy["total_umi"] >= MAIN_LO) & (psy["total_umi"] < MAIN_HI)]
    dm = (psy_dm.groupby("individual")["per_cell_c3"]
                  .agg(["mean", "size"]).rename(
                      columns={"mean": "c3_dm_main", "size": "n_cells_dm"}))
    don_df = don_df.merge(dm, left_on="individual", right_index=True, how="left")
    don_df["stage"] = np.where(don_df["age_years"] < 9, "child",
                       np.where(don_df["age_years"] < 25, "adol", "adult"))
    don_df = don_df.sort_values("age_years").reset_index(drop=True)
    don_df.to_csv(OUT_DIR / "n_psychad_per_donor.csv", index=False)
    print(f"Wrote per-donor table: {OUT_DIR/'n_psychad_per_donor.csv'} "
          f"({len(don_df)} donors)")

    # ---------- outlier flagging ----------
    children = don_df[don_df["stage"] == "child"]
    print(f"\nChild donors (age<9 y): {len(children)}")

    def z(col):
        mu = don_df[col].mean(); sd = don_df[col].std()
        return (don_df[col] - mu) / sd

    don_df["z_immature"] = z("pct_exn_immature")
    don_df["z_c3_full"]  = z("c3_per_cell_mean")
    don_df["z_c3_dm"]    = z("c3_dm_main")
    don_df["z_ncells"]   = z("n_cells")
    don_df["z_umi"]      = z("median_umi")

    print("\nTop 10 donors by ExN_immature % (overall pool):")
    cols = ["individual", "age_years", "stage", "n_cells",
            "pct_exn_immature", "pct_exn_mature", "pct_exn_weak",
            "c3_per_cell_mean", "c3_dm_main", "z_immature", "z_c3_dm"]
    print(don_df.sort_values("pct_exn_immature", ascending=False)
                  [cols].head(10).to_string(index=False))

    print("\nTop 5 child donors by ExN_immature %:")
    print(don_df[don_df["stage"] == "child"]
            .sort_values("pct_exn_immature", ascending=False)[cols]
            .head(5).to_string(index=False))

    print("\nChild donors ordered by C3+ score (lowest first — anti-drop-driving):")
    print(don_df[don_df["stage"] == "child"]
            .sort_values("c3_dm_main")[cols]
            .to_string(index=False))

    # ---------- leave-one-out fuzzy d at main 3-12 k window ----------
    def fuzzy_d_for_subset(don_df_subset):
        df = don_df_subset.dropna(subset=["c3_dm_main"])
        df = df[df["n_cells_dm"] >= MIN_CELLS]
        df = df[(df["age_years"] >= AGE_LO) & (df["age_years"] < AGE_HI)]
        if len(df) < 4:
            return np.nan
        r = fuzzy_d_from_donor_scores(df["age_years"].values,
                                        df["c3_dm_main"].values)
        return r["mean_d"]

    full_d = fuzzy_d_for_subset(don_df)
    print(f"\nFuzzy d at 3-12 k (all PsychAD-V3 donors): {full_d:+.3f}")

    # Build leave-one-out table for every child donor; also for all donors
    # so we can rank by impact.
    loo_rows = []
    for don in don_df["individual"]:
        sub = don_df[don_df["individual"] != don]
        d = fuzzy_d_for_subset(sub)
        meta = don_df[don_df["individual"] == don].iloc[0]
        loo_rows.append({
            "removed_donor": don,
            "age": float(meta["age_years"]),
            "stage": meta["stage"],
            "n_cells": int(meta["n_cells"]),
            "pct_immature": float(meta["pct_exn_immature"]),
            "c3_dm": float(meta["c3_dm_main"])
                       if not pd.isna(meta["c3_dm_main"]) else np.nan,
            "fuzzy_d_after_removal": d,
            "delta_d_vs_full": d - full_d,
        })
    loo = pd.DataFrame(loo_rows).sort_values("delta_d_vs_full",
                                              ascending=False)
    loo.to_csv(OUT_DIR / "n_leave_one_out_d.csv", index=False)
    print(f"\nWrote LOO table: {OUT_DIR/'n_leave_one_out_d.csv'}")
    print("\nTop 10 donors whose removal MOST increases fuzzy d "
          "(would help PsychAD agree with Vel):")
    print(loo.head(10).to_string(index=False))
    print("\nTop 5 child donors whose removal helps:")
    print(loo[loo["stage"] == "child"].head(5).to_string(index=False))

    # ---------- clinical metadata cross-reference ----------
    if F3_DONORS.exists():
        f3 = pd.read_csv(F3_DONORS)
        merged = loo.merge(f3[["individual", "source", "region"]],
                              left_on="removed_donor", right_on="individual",
                              how="left")
        print("\nChild-donor LOO with F3 metadata (source/region):")
        cols2 = ["removed_donor", "age", "n_cells", "pct_immature",
                 "c3_dm", "fuzzy_d_after_removal", "delta_d_vs_full",
                 "source", "region"]
        print(merged[merged["stage"] == "child"][cols2].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
