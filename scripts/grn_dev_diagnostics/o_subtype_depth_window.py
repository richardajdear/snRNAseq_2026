#!/usr/bin/env python3
"""
O — ExN subtype × depth window fuzzy d.

Re-investigate the within-ExN subtype split (ExN_mature / immature /
weak) AND the continuous maturity-score quantile split (PsychAD-V3
only, from H), now *combined* with the 3-12 k UMI depth window from
the depth-matched analysis. Earlier E1 / H were done without the
depth window.

Login-safe — uses the m_per_cell_cache built by M.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (OUT_DIR, fuzzy_d_from_donor_scores,
                   FUZZY_BOUNDARIES, AGE_LO, AGE_HI)

PER_CELL_CACHE = OUT_DIR / "m_per_cell_cache.parquet"
H_MATURITY     = OUT_DIR / "h_continuous_maturity_per_cell.parquet"
GROUPS         = ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]
WINDOWS        = [("none", None, None), ("3k-12k", 3000, 12000)]
SUBTYPES       = ["ExN_mature", "ExN_immature", "ExN_weak", "ALL_ExN"]
MIN_CELLS      = 5
EXCLUDE_DONORS = {"Donor_1400"}


def fuzzy_d_for_donors(donor_df, age_col="age_years", score_col="score"):
    df = donor_df.dropna(subset=[score_col, age_col])
    df = df[(df[age_col] >= AGE_LO) & (df[age_col] < AGE_HI)]
    if len(df) < 4:
        return np.nan, 0, []
    r = fuzzy_d_from_donor_scores(df[age_col].values, df[score_col].values)
    return r["mean_d"], len(df), r["per_boundary"]


def main():
    per_cell = pd.read_parquet(PER_CELL_CACHE)
    per_cell = per_cell[~per_cell["individual"].isin(EXCLUDE_DONORS)]

    # =====================================================================
    # 1. ExN subtype × window × group
    # =====================================================================
    print("="*72)
    print("PART 1 — ExN subtype × UMI window × group  (fuzzy d)")
    print("="*72)
    rows = []
    for g in GROUPS:
        for w_label, lo, hi in WINDOWS:
            sub = per_cell[per_cell["group"] == g]
            if lo is not None: sub = sub[sub["total_umi"] >= lo]
            if hi is not None: sub = sub[sub["total_umi"] < hi]
            for sty in SUBTYPES:
                if sty == "ALL_ExN":
                    s2 = sub
                else:
                    s2 = sub[sub["marker_annotation"] == sty]
                if len(s2) == 0:
                    rows.append({"group": g, "window": w_label,
                                  "subtype": sty,
                                  "n_donors": 0, "n_cells": 0,
                                  "fuzzy_d": np.nan})
                    continue
                don = (s2.groupby("individual", observed=True)
                          .agg(score=("per_cell_c3", "mean"),
                               n_cells=("per_cell_c3", "size"),
                               age_years=("age_years", "first"))
                          .reset_index())
                don = don[don["n_cells"] >= MIN_CELLS]
                d, n, _ = fuzzy_d_for_donors(don)
                rows.append({"group": g, "window": w_label,
                              "subtype": sty,
                              "n_donors": n,
                              "n_cells": int(len(s2)),
                              "median_cells_per_donor":
                                  float(don["n_cells"].median()) if len(don) else np.nan,
                              "fuzzy_d": d})
    df1 = pd.DataFrame(rows)
    df1.to_csv(OUT_DIR / "o1_subtype_window_d.csv", index=False)

    # pretty pivot per group
    for g in GROUPS:
        sub = df1[df1["group"] == g]
        pv = sub.pivot_table(index="subtype", columns="window",
                              values="fuzzy_d").round(3)
        nv = sub.pivot_table(index="subtype", columns="window",
                              values="n_donors").astype("Int64")
        print(f"\n--- {g} ---")
        print("fuzzy d:")
        print(pv.reindex(SUBTYPES).to_string())
        print("n donors:")
        print(nv.reindex(SUBTYPES).to_string())

    # =====================================================================
    # 2. PsychAD-V3 maturity-score quantile × window
    # =====================================================================
    print("\n" + "="*72)
    print("PART 2 — PsychAD-V3 maturity quantile × UMI window  (fuzzy d)")
    print("="*72)
    if not H_MATURITY.exists():
        print("H maturity parquet missing — skipping.")
        return
    mat = pd.read_parquet(H_MATURITY)
    print(f"loaded H maturity table: {len(mat):,} cells")
    print("columns:", mat.columns.tolist())
    # join maturity_score onto per_cell cache via obs_name if possible —
    # cache doesn't carry obs_name though, so we re-aggregate from H + per-cell scores
    # using the cell index. Need to recompute per_cell_c3 on H... actually the H
    # parquet doesn't carry per_cell_c3.
    #
    # Strategy: merge per_cell cache (which has per_cell_c3, total_umi, age, individual,
    # group) with H maturity_score using obs_name. The cache doesn't have obs_name —
    # so we save cache rows in obs_name order? It does NOT preserve it.
    #
    # Workaround: H has individual via obs_name prefix. Each obs_name starts
    # "Donor_NNN-...".  Cache has individual but not the cell-level key.
    # Without a join key we can't directly attach H scores to cache rows.
    #
    # Alternative: recompute maturity_score for each cache row using cache's
    # missing marker UMIs. But cache doesn't carry RBFOX3/DCX UMIs either.
    #
    # So restrict PART 2 to within-H analysis: use H's per-cell entries and
    # re-derive a C3+ score via... no, H doesn't have C3+ scores.
    #
    # Best lightweight path: use H to *bin* cells by maturity quantile, then
    # for each bin compute a donor-level C3+ score from the cache restricted to
    # cells with matching obs_name. This requires the cache to have obs_name.
    # It doesn't.
    #
    # Pragmatic resolution: skip PART 2 with a note. Rebuilding the cache to
    # add obs_name+maturity is a separate task; user said "investigate", and
    # PART 1 is the more interpretable analysis.
    print("\nPART 2 needs a cell-level join key between per_cell_cache and H.")
    print("Cache currently lacks obs_name. To do this properly we'd extend the")
    print("cache or recompute C3+ on the H cells. Deferring; PART 1 is the")
    print("primary requested analysis.\n")


if __name__ == "__main__":
    main()
