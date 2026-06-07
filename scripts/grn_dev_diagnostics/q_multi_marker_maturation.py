#!/usr/bin/env python3
"""
Q — multi-marker maturation analysis and binary-vs-continuous reconciliation.

Part A (reconciliation):
    Why does binary `ExN_immature` give PsychAD-V3 fuzzy d = +0.45 while
    continuous-maturity-Q0 (lowest quintile of RBFOX3-DCX score) gives only
    +0.09 — when Q0 should *contain* the binary-immature pool?
    Hypothesis: the continuous score uses CP10k normalisation, which
    inflates shallow cells' normalised marker values. So Q0 ends up
    capturing some deep DCX-rich RBFOX3-low cells that the binary
    classifier (raw-count thresholds) correctly calls "mature". Those
    deep cells carry the anti-drop signal and dilute Q0's d.
    Tests:
      Q-R1. Crosstab of binary marker_annotation × continuous maturity quintile.
      Q-R2. Median UMI per (binary × quintile) cell.
      Q-R3. Fuzzy d in cells that are BOTH binary-immature AND continuous Q0.
      Q-R4. Alternative raw-count "binary-style" continuous: maturity = (RBFOX3>=1).astype(int) - (DCX>=1).astype(int).
            Show that binning by THIS score reproduces the binary ExN_immature
            d much more closely.

Part B (multi-marker module):
    Test each of these maturation markers individually AND as a module:
        DCX        (immature)
        RBFOX3     (mature, pan-neuronal)
        RBFOX1     (mature, pan-neuronal)
        NEUROD2    (cortical neuronal differentiation)
        BCL11B     (mature deep-layer)
        SATB2      (mature upper-layer)
        MEF2C      (mature, cortical)
        NEFL       (mature, neurofilament)
        NEFM       (mature, neurofilament)
    For each marker:
      Q-M1. binary expressing vs not (raw count >=1): fuzzy d per group in each group.
      Q-M2. continuous quintile of log1p_cp10k expression: fuzzy d per quintile per group.
    For module:
      Q-MD1. module_mature_5 = mean log1p_cp10k of NEUROD2/BCL11B/SATB2/MEF2C/NEFL/NEFM (6 markers — pure mature, no DCX)
      Q-MD2. module_maturation_diff = module_mature_5 − log1p_cp10k(DCX)
      For each: binary above-median and continuous quintile → fuzzy d per group.

Run via:
    cd /home/rajd2/rds/hpc-work/snRNAseq_2026
    sbatch --time=01:00:00 --mem=200G \\
       scripts/run_script.sh scripts/grn_dev_diagnostics/q_multi_marker_maturation.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (OUT_DIR, fuzzy_d_from_donor_scores, build_c3plus_table,
                   FUZZY_BOUNDARIES, AGE_LO, AGE_HI)

GROUPS = ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]
GROUP_TO_DATASET = {
    "PsychAD-V3":   ("PsychAD",   "V3"),
    "Velmeshev-V2": ("Velmeshev", "V2"),
    "Velmeshev-V3": ("Velmeshev", "V3"),
}
INPUTS = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/scvi_output/integrated.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad",
}
MANUAL = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
}

# canonical Ensembl IDs (verified in earlier scripts where possible)
MARKERS = {
    "DCX":     "ENSG00000077279",
    "RBFOX3":  "ENSG00000167281",
    "RBFOX1":  "ENSG00000078328",
    "NEUROD2": "ENSG00000171532",
    "BCL11B":  "ENSG00000127152",
    "SATB2":   "ENSG00000119042",
    "MEF2C":   "ENSG00000081189",
    "NEFL":    "ENSG00000277956",
    "NEFM":    "ENSG00000104722",
}
# pure-mature module (excludes DCX and the pan-neuronal markers used
# directly by the binary classifier, so as not to circularly recapitulate it)
MATURE_MODULE = ["NEUROD2", "BCL11B", "SATB2", "MEF2C", "NEFL", "NEFM"]

CACHE_V3 = OUT_DIR / "m_per_cell_cache_v3.parquet"
EXCLUDE_DONORS = {"Donor_1400"}
MIN_CELLS = 5
COLORS = {"PsychAD-V3": "#C0392B", "Velmeshev-V2": "#27AE60",
           "Velmeshev-V3": "#2980B9"}


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------

def build_cache():
    if CACHE_V3.exists():
        print(f"Using cached: {CACHE_V3}", flush=True)
        return pd.read_parquet(CACHE_V3)

    weights = build_c3plus_table().set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes", flush=True)

    frames = []
    for g in GROUPS:
        dataset, chem = GROUP_TO_DATASET[g]
        print(f"\n=== {g} ({dataset}, {chem})", flush=True)
        a = ad.read_h5ad(INPUTS[dataset], backed="r")
        obs = a.obs.copy()
        if "chemistry" not in obs.columns and "source-chemistry" in obs.columns:
            obs["chemistry"] = obs["source-chemistry"].astype(str).str.extract(
                r"(V2|V3)")[0].fillna("unknown")
        obs["chemistry"] = obs["chemistry"].astype(str)
        ma = pd.read_parquet(MANUAL[dataset])
        obs = obs.join(ma, how="left")
        age = pd.to_numeric(obs["age_years"], errors="coerce")
        mask = ((age >= AGE_LO) & (age < AGE_HI)
                & obs["marker_annotation"].isin(
                    ["ExN_mature", "ExN_immature", "ExN_weak"])
                & (obs["chemistry"].values == chem))
        obs_idx = np.where(mask.values)[0]
        print(f"  ExN cells: {len(obs_idx):,}", flush=True)
        counts = sp.csr_matrix(a.layers["counts"])[obs_idx, :]
        obs = obs.iloc[obs_idx].copy().reset_index(drop=True)
        obs["age_years"] = age.values[obs_idx]
        obs["individual"] = obs.get("individual", obs.get("donor_id"))
        obs["total_umi"] = np.asarray(counts.sum(axis=1)).ravel()
        var_names = a.var_names.values
        var_pos = {v: i for i, v in enumerate(var_names)}
        total = obs["total_umi"].values.astype(np.float64)

        # marker raw counts and log1p_cp10k
        present_markers = []
        for mname, ens in MARKERS.items():
            if ens in var_pos:
                cnts = np.asarray(counts[:, var_pos[ens]].todense()).ravel()
                obs[f"raw_{mname}"] = cnts.astype(np.int32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cp = np.where(total > 0, cnts / total * 1e4, 0.0)
                obs[f"log1p_cp10k_{mname}"] = np.log1p(cp).astype(np.float32)
                present_markers.append(mname)
            else:
                obs[f"raw_{mname}"] = 0
                obs[f"log1p_cp10k_{mname}"] = 0.0
        print(f"  markers found in var: {present_markers}", flush=True)

        # per-cell C3+
        present = [g_ for g_ in weights.index if g_ in var_pos]
        grn_cols = np.array([var_pos[g_] for g_ in present], dtype=np.int64)
        grn_w = np.array([weights[g_] for g_ in present], dtype=np.float64)
        raw_dot = np.asarray(counts[:, grn_cols] @ grn_w).ravel()
        obs["per_cell_c3"] = np.where(total > 0, raw_dot / total * 1e6, 0.0)

        keep = ["individual", "age_years", "chemistry", "total_umi",
                 "marker_annotation", "per_cell_c3"]
        for mname in MARKERS:
            keep += [f"raw_{mname}", f"log1p_cp10k_{mname}"]
        out = obs[keep].copy(); out["group"] = g
        frames.append(out)
        print(f"  done.", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(CACHE_V3)
    print(f"\nSaved v3 cache: {CACHE_V3} ({len(df):,} rows)", flush=True)
    return df


# ---------------------------------------------------------------------------
# Fuzzy d helper
# ---------------------------------------------------------------------------

def fuzzy_d_from_per_cell(per_cell, min_cells=MIN_CELLS):
    don = (per_cell.groupby("individual", observed=True)
              .agg(score=("per_cell_c3", "mean"),
                   n_cells=("per_cell_c3", "size"),
                   age_years=("age_years", "first"))
              .reset_index())
    don = don[don["n_cells"] >= min_cells]
    don = don.dropna(subset=["score", "age_years"])
    don = don[(don["age_years"] >= AGE_LO) & (don["age_years"] < AGE_HI)]
    if len(don) < 4:
        return np.nan, 0
    r = fuzzy_d_from_donor_scores(don["age_years"].values, don["score"].values)
    return r["mean_d"], len(don)


# ---------------------------------------------------------------------------
# Part A: reconciliation
# ---------------------------------------------------------------------------

def reconciliation(df):
    print("\n" + "="*72)
    print("PART A — binary-vs-continuous-maturity reconciliation")
    print("="*72)
    rows_xt, rows_d, rows_umi = [], [], []
    for g in GROUPS:
        sub = df[df["group"] == g].copy()
        # continuous maturity (existing definition)
        sub["maturity_score_cp10k"] = (sub["log1p_cp10k_RBFOX3"]
                                          - sub["log1p_cp10k_DCX"])
        sub["mat_q_cp10k"] = pd.qcut(sub["maturity_score_cp10k"], 5,
                                       labels=False, duplicates="drop")
        # raw-count-based maturity (alternative)
        sub["raw_immature_flag"] = ((sub["raw_DCX"] >= 1)
                                     & (sub["raw_RBFOX3"] < 1)).astype(int)
        sub["raw_mature_flag"]   = (sub["raw_RBFOX3"] >= 1).astype(int)
        sub["raw_maturity"]      = (sub["raw_mature_flag"]
                                     - sub["raw_immature_flag"])
        # Q-R1: crosstab marker_annotation × mat_q
        ct = pd.crosstab(sub["marker_annotation"], sub["mat_q_cp10k"],
                          margins=True)
        print(f"\n--- {g}: crosstab of binary marker_annotation × continuous-Q ---")
        print(ct.to_string())
        ct_long = ct.reset_index().melt(id_vars="marker_annotation",
                                          var_name="mat_q_cp10k", value_name="n")
        ct_long["group"] = g; rows_xt.append(ct_long)

        # Q-R2: per (binary × quintile) median UMI
        for sty in ["ExN_mature", "ExN_immature", "ExN_weak"]:
            for q in sorted(sub["mat_q_cp10k"].dropna().unique()):
                s2 = sub[(sub["marker_annotation"] == sty)
                          & (sub["mat_q_cp10k"] == q)]
                if len(s2) < 5: continue
                rows_umi.append({"group": g, "subtype": sty,
                                  "mat_q_cp10k": int(q),
                                  "n_cells": int(len(s2)),
                                  "median_umi": float(s2["total_umi"].median()),
                                  "median_dcx_raw": float(s2["raw_DCX"].median()),
                                  "median_rbfox3_raw":
                                      float(s2["raw_RBFOX3"].median())})

        # Q-R3: fuzzy d in cells that are binary-immature AND Q0
        for label, mask in [
            ("binary_ExN_immature", sub["marker_annotation"] == "ExN_immature"),
            ("continuous_Q0", sub["mat_q_cp10k"] == 0),
            ("binary_immature_AND_Q0",
                (sub["marker_annotation"] == "ExN_immature")
                & (sub["mat_q_cp10k"] == 0)),
            ("binary_mature_AND_Q0",
                (sub["marker_annotation"] == "ExN_mature")
                & (sub["mat_q_cp10k"] == 0)),
            ("raw_immature_flag",
                sub["raw_immature_flag"] == 1),
            ("raw_maturity_minus1_class",
                sub["raw_maturity"] == -1),
            ("raw_maturity_zero_class",
                sub["raw_maturity"] == 0),
            ("raw_maturity_plus1_class",
                sub["raw_maturity"] == 1),
        ]:
            s2 = sub[mask]
            d, n = fuzzy_d_from_per_cell(s2)
            rows_d.append({"group": g, "definition": label,
                            "n_cells": int(len(s2)), "n_donors": n,
                            "fuzzy_d": d})

    pd.concat(rows_xt, ignore_index=True).to_csv(
        OUT_DIR / "qA_binary_x_cont_crosstab.csv", index=False)
    pd.DataFrame(rows_umi).to_csv(
        OUT_DIR / "qA_binary_x_cont_median_umi.csv", index=False)
    qA_d = pd.DataFrame(rows_d)
    qA_d.to_csv(OUT_DIR / "qA_reconciliation_d.csv", index=False)
    print("\n--- Q-R3 fuzzy d across definitions ---")
    print(qA_d.pivot_table(index="definition", columns="group", values="fuzzy_d")
            .round(3).to_string())
    print("\n--- Q-R3 n_donors across definitions ---")
    print(qA_d.pivot_table(index="definition", columns="group", values="n_donors")
            .astype("Int64").to_string())


# ---------------------------------------------------------------------------
# Part B: individual markers + module
# ---------------------------------------------------------------------------

def individual_markers(df):
    print("\n" + "="*72)
    print("PART B — per-marker fuzzy d (binary + continuous quintile)")
    print("="*72)
    rows_binary, rows_quint = [], []
    for g in GROUPS:
        sub = df[df["group"] == g].copy()
        for mname in MARKERS:
            raw_col = f"raw_{mname}"
            cp_col  = f"log1p_cp10k_{mname}"
            # binary
            for label, mask in [
                (f"{mname}_expressing(raw>=1)", sub[raw_col] >= 1),
                (f"{mname}_not_expressing(raw==0)", sub[raw_col] == 0),
            ]:
                d, n = fuzzy_d_from_per_cell(sub[mask])
                rows_binary.append({"group": g, "marker": mname,
                                     "split": label,
                                     "n_cells": int(mask.sum()),
                                     "n_donors": n, "fuzzy_d": d})
            # continuous quintile of log1p_cp10k
            if sub[cp_col].std() == 0: continue
            try:
                sub["_q"] = pd.qcut(sub[cp_col], 5, labels=False, duplicates="drop")
            except Exception:
                continue
            for q in sorted(sub["_q"].dropna().unique()):
                s2 = sub[sub["_q"] == q]
                d, n = fuzzy_d_from_per_cell(s2)
                rows_quint.append({"group": g, "marker": mname, "q": int(q),
                                    "n_cells": int(len(s2)), "n_donors": n,
                                    "fuzzy_d": d,
                                    "median_expr": float(s2[cp_col].median()),
                                    "median_umi":  float(s2["total_umi"].median())})
    qb_bin = pd.DataFrame(rows_binary); qb_bin.to_csv(
        OUT_DIR / "qB_per_marker_binary_d.csv", index=False)
    qb_q = pd.DataFrame(rows_quint); qb_q.to_csv(
        OUT_DIR / "qB_per_marker_quintile_d.csv", index=False)

    # Visualisation: binary expressing-vs-not bar plot per marker per group
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
    for i, mname in enumerate(MARKERS):
        ax = axes[i // 3, i % 3]
        for j, g in enumerate(GROUPS):
            s = qb_bin[(qb_bin["group"] == g) & (qb_bin["marker"] == mname)]
            vals = []
            for lab in [f"{mname}_expressing(raw>=1)", f"{mname}_not_expressing(raw==0)"]:
                r = s[s["split"] == lab]
                vals.append(float(r["fuzzy_d"].iloc[0]) if len(r) else np.nan)
            x = np.array([0, 1]) + j*0.27 - 0.27
            ax.bar(x, vals, 0.25, color=COLORS[g],
                    label=g if i == 0 else None)
            for k_, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(x[k_], v + (0.04 if v >= 0 else -0.06),
                             f"{v:+.2f}", ha="center",
                             fontsize=7, fontweight="bold", color=COLORS[g])
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{mname}+\n(raw≥1)", f"{mname}-\n(raw=0)"], fontsize=8)
        ax.set_title(mname, fontsize=10)
        if i == 0: ax.legend(loc="best", frameon=False, fontsize=7)
    fig.suptitle("Q-B per-marker binary fuzzy d (expressing vs not expressing)",
                  fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qB_per_marker_binary_d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved qB_per_marker_binary_d.png")

    # Quintile bar plot per marker per group (low expr → high expr)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)
    for i, mname in enumerate(MARKERS):
        ax = axes[i // 3, i % 3]
        for j, g in enumerate(GROUPS):
            s = qb_q[(qb_q["group"] == g) & (qb_q["marker"] == mname)]
            qs = sorted(s["q"].unique())
            vals = [float(s[s["q"] == q]["fuzzy_d"].iloc[0]) for q in qs]
            x = np.array(qs) + j*0.27 - 0.27
            ax.bar(x, vals, 0.25, color=COLORS[g], label=g if i == 0 else None)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel(f"{mname} log1p_cp10k quintile (low→high)", fontsize=8)
        ax.set_title(mname, fontsize=10)
        if i == 0: ax.legend(loc="best", frameon=False, fontsize=7)
    fig.suptitle("Q-B per-marker continuous-quintile fuzzy d "
                  "(quintile 0 = lowest expression, quintile 4 = highest)",
                  fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qB_per_marker_quintile_d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved qB_per_marker_quintile_d.png")


def module_score(df):
    print("\n" + "="*72)
    print("PART C — multi-marker mature module fuzzy d")
    print("="*72)
    # build module per group from the same cells
    rows = []
    for g in GROUPS:
        sub = df[df["group"] == g].copy()
        cp_cols = [f"log1p_cp10k_{m}" for m in MATURE_MODULE]
        sub["mature_module"] = sub[cp_cols].mean(axis=1)
        sub["mature_module_minus_dcx"] = (sub["mature_module"]
                                            - sub["log1p_cp10k_DCX"])

        for score_name in ["mature_module", "mature_module_minus_dcx"]:
            # binary above-median split
            med = sub[score_name].median()
            for label, mask in [(f"{score_name}_above_median",
                                  sub[score_name] >= med),
                                 (f"{score_name}_below_median",
                                  sub[score_name] < med)]:
                d, n = fuzzy_d_from_per_cell(sub[mask])
                rows.append({"group": g, "score": score_name,
                              "split": label,
                              "n_cells": int(mask.sum()), "n_donors": n,
                              "fuzzy_d": d})
            # continuous quintile
            try:
                qcol = pd.qcut(sub[score_name], 5, labels=False, duplicates="drop")
                for q in sorted(qcol.dropna().unique()):
                    mask = (qcol == q)
                    d, n = fuzzy_d_from_per_cell(sub[mask])
                    rows.append({"group": g, "score": score_name,
                                  "split": f"q{int(q)}",
                                  "n_cells": int(mask.sum()), "n_donors": n,
                                  "fuzzy_d": d,
                                  "median_expr":
                                      float(sub[mask][score_name].median())})
            except Exception:
                pass
    qC = pd.DataFrame(rows)
    qC.to_csv(OUT_DIR / "qC_module_d.csv", index=False)

    # Plot: quintile fuzzy d per module per group
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, score in zip(axes, ["mature_module", "mature_module_minus_dcx"]):
        x = np.arange(5)
        for j, g in enumerate(GROUPS):
            s = qC[(qC["group"] == g) & (qC["score"] == score)
                    & (qC["split"].str.startswith("q"))]
            s = s.sort_values("split")
            vals = [float(s[s["split"] == f"q{q}"]["fuzzy_d"].iloc[0])
                    if len(s[s["split"] == f"q{q}"]) else np.nan
                    for q in range(5)]
            ax.bar(x + j*0.27 - 0.27, vals, 0.25, color=COLORS[g], label=g)
            for k_, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(x[k_] + j*0.27 - 0.27,
                             v + (0.04 if v >= 0 else -0.06),
                             f"{v:+.2f}", ha="center", fontsize=7,
                             fontweight="bold", color=COLORS[g])
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(["q0\nlow", "q1", "q2", "q3", "q4\nhigh"])
        ax.set_xlabel(f"{score} quintile")
        ax.set_title(score, fontsize=10)
        ax.legend(loc="best", frameon=False, fontsize=8)
    axes[0].set_ylabel("Cohen's d (fuzzy boundary)")
    fig.suptitle("Q-C multi-marker mature module fuzzy d by quintile",
                  fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qC_module_quintile_d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved qC_module_quintile_d.png")

    print("\n--- Q-C above/below median ---")
    print(qC[qC["split"].str.contains("median")][
        ["group", "score", "split", "n_cells", "n_donors", "fuzzy_d"]
    ].to_string(index=False))
    print("\n--- Q-C quintile pivot ---")
    qpivot = (qC[qC["split"].str.startswith("q")]
                .pivot_table(index=["score", "split"],
                              columns="group", values="fuzzy_d").round(3))
    print(qpivot.to_string())


def main():
    df = build_cache()
    df = df[~df["individual"].isin(EXCLUDE_DONORS)].reset_index(drop=True)
    reconciliation(df)
    individual_markers(df)
    module_score(df)
    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
