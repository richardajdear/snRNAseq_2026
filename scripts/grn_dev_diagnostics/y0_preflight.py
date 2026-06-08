#!/usr/bin/env python3
"""
Y0 — pre-flight for the ExN-only joint V3 embedding plan.

Answers the plan's critical question before we write the Stage-1 config:
  - chemistry x age-bin cell counts for Velmeshev and PsychAD (PFC only), so we
    know whether there are enough PRENATAL V3 Velmeshev cells to anchor the
    immature end of the maturation axis (prenatal is Velmeshev-only; may be
    V2-heavy). If prenatal is V2-only we default to V3-only and anchor with
    youngest-postnatal + DCX+ cells instead.
  - confirms the ExN/InN/glia marker genes used for cluster annotation are
    present in each source's var.

Uses the pipeline's OWN backed readers so the counts match exactly what
downsample.py will see (same age_years, chemistry, region, cell_class columns).

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=00:40:00 --mem=120G \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y0_preflight.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from pipeline import read_data

OUT = Path("scripts/grn_dev_diagnostics/outputs")
PSYCHAD_AGING = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad"
PSYCHAD_HBCC = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"

AGE_BINS = [(-np.inf, 0), (0, 1), (1, 10), (10, 18), (18, 30), (30, np.inf)]
AGE_LABELS = ["prenatal", "0-1y", "1-10y", "10-18y", "18-30y", "30y+"]

EXN_MARKERS = ["SLC17A7", "SATB2", "RBFOX3", "NEUROD2", "NEUROD6", "TBR1"]
INN_MARKERS = ["GAD1", "GAD2", "SLC32A1", "LHX6", "ADARB2", "DLX1"]
GLIA_MARKERS = ["AQP4", "GFAP", "MBP", "PLP1", "PDGFRA", "CX3CR1", "P2RY12"]
ALL_MARKERS = EXN_MARKERS + INN_MARKERS + GLIA_MARKERS


def age_bin(a):
    for (lo, hi), lab in zip(AGE_BINS, AGE_LABELS):
        if (a >= lo) and (a < hi):
            return lab
    return "NA"


def tab_source(name, meta_df):
    m = meta_df.copy()
    if "region" in m.columns:
        m = m[m["region"] == "prefrontal cortex"]
    m = m[m["age_years"].notna()]
    m["age_bin"] = m["age_years"].map(age_bin)
    m["chemistry"] = m.get("chemistry", "unknown").astype(str)
    ct = (m.groupby(["chemistry", "age_bin"]).size()
          .rename("n_cells").reset_index())
    ct["source"] = name
    # donor counts too
    dc = (m.groupby(["chemistry", "age_bin"])["individual"].nunique()
          .rename("n_donors").reset_index())
    ct = ct.merge(dc, on=["chemistry", "age_bin"], how="left")
    return ct[["source", "chemistry", "age_bin", "n_donors", "n_cells"]]


def check_markers(name, adata_backed):
    var = adata_backed.var
    names = set(adata_backed.var_names.astype(str))
    fn = set(var["feature_name"].astype(str)) if "feature_name" in var.columns else set()
    present = {}
    for g in ALL_MARKERS:
        present[g] = (g in names) or (g in fn)
    missing = [g for g, ok in present.items() if not ok]
    print(f"\n[{name}] marker presence: {sum(present.values())}/{len(ALL_MARKERS)} found"
          + (f"; MISSING: {missing}" if missing else "; all present"))
    return present


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = []

    print("=" * 70, "\nVELMESHEV\n", "=" * 70)
    v_backed, v_meta = read_data.read_velmeshev_backed(cell_type_field="Cell_Type")
    tabs.append(tab_source("VELMESHEV", v_meta))
    check_markers("VELMESHEV", v_backed)
    del v_backed

    print("=" * 70, "\nPSYCHAD\n", "=" * 70)
    aging_b, hbcc_b, hbcc_unique, p_meta = read_data.read_psychad_backed(
        PSYCHAD_AGING, PSYCHAD_HBCC, cell_type_field="subclass")
    tabs.append(tab_source("PSYCHAD", p_meta))
    check_markers("PSYCHAD", aging_b)
    del aging_b, hbcc_b

    tab = pd.concat(tabs, ignore_index=True)
    tab = tab.sort_values(["source", "chemistry", "age_bin"]).reset_index(drop=True)
    tab.to_csv(OUT / "y0_chem_by_age.csv", index=False)

    print("\n" + "=" * 70)
    print("PFC chemistry x age-bin cell/donor counts")
    print("=" * 70)
    # pivot for readability
    piv = tab.pivot_table(index=["source", "chemistry"], columns="age_bin",
                          values="n_cells", aggfunc="sum", fill_value=0)
    piv = piv.reindex(columns=[c for c in AGE_LABELS if c in piv.columns])
    print(piv.to_string())

    # the key question: prenatal V3 Velmeshev availability
    vel = tab[tab["source"] == "VELMESHEV"]
    pre = vel[vel["age_bin"] == "prenatal"]
    print("\n--- PRENATAL Velmeshev by chemistry ---")
    if len(pre):
        print(pre[["chemistry", "n_donors", "n_cells"]].to_string(index=False))
        v3pre = pre[pre["chemistry"].str.contains("V3", na=False)]["n_cells"].sum()
        v2pre = pre[pre["chemistry"].str.contains("V2", na=False)]["n_cells"].sum()
        print(f"\nPrenatal V3 cells = {v3pre:,}; prenatal V2 cells = {v2pre:,}")
        print("DECISION: " + ("enough prenatal V3 -> include prenatal V3"
                              if v3pre > 2000 else
                              "prenatal is V2-heavy/sparse -> default V3-only, "
                              "anchor immature end with youngest-postnatal + DCX+ cells"))
    else:
        print("No prenatal Velmeshev PFC cells found.")
    print(f"\nsaved {OUT / 'y0_chem_by_age.csv'}")


if __name__ == "__main__":
    main()
