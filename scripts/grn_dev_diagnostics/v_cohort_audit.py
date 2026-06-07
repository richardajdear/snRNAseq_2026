#!/usr/bin/env python3
"""
V — audit developmental sampling of every on-disk dataset, to assess
whether a third independent pediatric-DLPFC cohort exists for the C3+
analysis (PsychAD + Velmeshev are the current two; Velmeshev V2/V3 are the
SAME study split by chemistry, so they are not independent replication).

For each dataset (Velmeshev, Wang, Zhu, PsychAD) report, using the pipeline's
own backed readers:
  - total cells / donors, age range, region vocabulary
  - donors & cells in the developmental window, binned by age
  - the same restricted to PFC-like regions
to decide if it can serve as an independent childhood→adolescence cohort.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=01:00:00 --mem=128G \
     scripts/run_script.sh scripts/grn_dev_diagnostics/v_cohort_audit.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/rajd2/rds/hpc-work/snRNAseq_2026/code")
from pipeline.read_data import (read_velmeshev_backed, read_wang_backed,
                                read_zhu_backed, read_psychad_backed,
                                WANG_PATH, ZHU_PATH, VELMESHEV_PATH)

OUT = Path("/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/"
           "grn_dev_diagnostics/outputs")
AGING = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad"
HBCC  = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"

AGE_BINS = [-2, 0, 1, 3, 5, 8, 12, 18, 25, 40, 120]
PFC_KEYS = ["PFC", "DLPFC", "PREFRONTAL", "BA9", "BA10", "BA46", "BA8",
            "BA44", "BA45", "BA47", "FRONTAL"]


def is_pfc(region_series):
    s = region_series.astype(str).str.upper()
    mask = np.zeros(len(s), dtype=bool)
    for k in PFC_KEYS:
        mask |= s.str.contains(k, na=False)
    return mask


def summarise(name, meta):
    print("\n" + "=" * 70)
    print(f"### {name}  —  {len(meta):,} cells")
    print("=" * 70)
    age = pd.to_numeric(meta.get("age_years"), errors="coerce")
    don = meta.get("individual", meta.get("donor_id"))
    meta = meta.assign(_age=age, _don=don)
    print(f"age range: {np.nanmin(age):.2f} … {np.nanmax(age):.2f} y;  "
          f"n donors total: {meta['_don'].nunique()}")
    print("\nregion value_counts (top 12):")
    print(meta["region"].astype(str).value_counts().head(12).to_string())

    pfc = is_pfc(meta["region"])
    print(f"\nPFC-like cells: {pfc.sum():,} / {len(meta):,} "
          f"({100*pfc.mean():.0f}%);  PFC donors: "
          f"{meta.loc[pfc, '_don'].nunique()}")

    rows = []
    for label, sub in [("ALL regions", meta), ("PFC-like", meta[pfc])]:
        s = sub.dropna(subset=["_age"])
        s = s.assign(bin=pd.cut(s["_age"], AGE_BINS))
        g = s.groupby("bin", observed=True).agg(
            n_cells=("_age", "size"),
            n_donors=("_don", "nunique"))
        g["scope"] = label
        rows.append(g.reset_index())
        # developmental window 1-25 y
        dev = s[(s["_age"] >= 1) & (s["_age"] < 25)]
        print(f"\n[{label}] DEVELOPMENTAL WINDOW 1–25 y: "
              f"{dev['_don'].nunique()} donors, {len(dev):,} cells")
    tab = pd.concat(rows, ignore_index=True)
    print(f"\n[{name}] age-bin × scope (cells / donors):")
    print(tab.to_string(index=False))
    tab["dataset"] = name
    return tab


def main():
    allt = []
    try:
        _, mv = read_velmeshev_backed()
        allt.append(summarise("VELMESHEV", mv))
    except Exception as e:
        print(f"VELMESHEV failed: {e}")
    try:
        _, mw = read_wang_backed()
        allt.append(summarise("WANG", mw))
    except Exception as e:
        print(f"WANG failed: {e}")
    try:
        _, mz = read_zhu_backed()
        allt.append(summarise("ZHU", mz))
    except Exception as e:
        print(f"ZHU failed: {e}")
    try:
        _, mp = read_psychad_backed(AGING, HBCC)
        allt.append(summarise("PSYCHAD", mp))
    except Exception as e:
        print(f"PSYCHAD failed: {e}")

    if allt:
        out = pd.concat(allt, ignore_index=True)
        out.to_csv(OUT / "v_cohort_age_audit.csv", index=False)
        print(f"\nsaved {OUT/'v_cohort_age_audit.csv'}")


if __name__ == "__main__":
    main()
