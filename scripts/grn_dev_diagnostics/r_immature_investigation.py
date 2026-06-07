#!/usr/bin/env python3
"""
R — Is the C3+ child→adolescent drop a MATURITY effect (not a depth effect)?

Motivating finding (from Q): with NO depth filter at all,
    PsychAD-V3 mature_module quintile-0 (least-mature cells) → fuzzy d = +0.52,
which EXCEEDS the binary DCX+RBFOX3- "ExN_immature" d = +0.45. The
"below-median module = +0.19" earlier was just dilution from a coarse
50% split (median mixes q0 +0.52 with q1 -0.14 and q2 -0.33).

So the +0.45 is NOT a lucky hit: a principled multi-marker maturity index
recovers AND improves it, without any depth window. This script nails that
down and tests the user's central hypothesis: *the depth filter that
"reveals" the C3+ trend in PsychAD-V3 is actually a maturity confound* —
shallow == immature, and the FANS shallow-dropout removes the very
(immature, high-C3+) cells that carry the childhood peak.

Predictions if the hypothesis is TRUE:
  (P-a) A direct maturity index recovers d≈+0.5 in PsychAD-V3 with NO
        depth filter (already seen: module q0 = +0.52). Cross-cohort, the
        immature-bin d's should agree better than the all-ExN aggregate.
  (P-b) Within a FIXED depth stratum, the maturity gradient persists
        (least-mature cells still drop) — i.e. maturity carries signal
        *beyond* depth. (depth × module 2D)
  (P-c) Conversely, within a fixed maturity bin, depth should add little —
        i.e. once you condition on maturity, the depth window is no longer
        needed. (same 2D, read the other way)
  (P-d) The effect is not driven by one layer (layer × module 2D) nor a
        few donors (leave-one-out).

Parts:
  R1. Resolve marker Ensembl IDs from var.feature_name (FIXES NEFL etc.),
      rebuild v4 per-cell cache adding: corrected mature markers, a layer
      label (argmax layer-TF module), and a stable cell_key for joins.
  R2. Maturity-index cascade (NO depth filter): for several maturity
      definitions compute the full quantile→fuzzy-d profile and the
      "extreme immature bin" d. Definitions:
        - mature_module (mean log1p_cp10k of corrected mature markers)
        - detection module (count of mature markers with raw>=1)  [detection-based]
        - n_immature_markers detected (DCX/SOX11/...) [detection-based]
        - binary DCX+RBFOX3-  (reference, +0.45)
      Question: which most cleanly/strongly recovers the childhood peak?
  R3. Detection-based vs CP10k-normalised module head-to-head (the user's
      explicit question). Same marker set, scored two ways.
  R4. depth × module 2D fuzzy-d heatmap (P-b / P-c).
  R5. layer × module 2D fuzzy-d heatmap (P-d).
  R6. Per-donor decomposition + leave-one-out robustness of module-q0.
  R7. Cross-cohort concordance summary table: immature-bin d for all three
      groups, NO depth filter, several definitions.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=01:30:00 --mem=240G \
     scripts/run_script.sh scripts/grn_dev_diagnostics/r_immature_investigation.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from _lib import (OUT_DIR, fuzzy_d_from_donor_scores, build_c3plus_table,
                   AGE_LO, AGE_HI)

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

# Resolve these BY SYMBOL from var.feature_name (robust; auto-fixes NEFL).
# Mature / differentiation markers (the module). Pan-neuronal RBFOX3/RBFOX1
# kept separate (used by binary classifier). DCX/immature kept separate.
MATURE_MARKERS   = ["NEUROD2", "BCL11B", "SATB2", "MEF2C", "NEFL", "NEFM",
                    "NEFH", "SYT1", "SNAP25", "MAP2", "RBFOX3", "RBFOX1"]
IMMATURE_MARKERS = ["DCX", "SOX11", "SOX4", "NEUROD1", "TBR1", "EOMES",
                    "DPYSL3", "STMN2"]
# the "pure mature module" used for the headline index (no DCX, no
# pan-neuronal RBFOX so it is not circular with the binary classifier)
MODULE_MATURE    = ["NEUROD2", "BCL11B", "SATB2", "MEF2C", "NEFL", "NEFM",
                    "NEFH", "SYT1", "SNAP25", "MAP2"]
ALL_SYMBOLS = sorted(set(MATURE_MARKERS + IMMATURE_MARKERS))

# layer-TF modules (copied from K) for layer assignment
LAYER_MODULES = {
    "upper": ["SATB2", "CUX2", "CUX1", "RORB"],
    "L5_ET": ["FEZF2", "BCL11B", "POU3F1"],
    "L6_CT": ["TBR1", "FOXP2", "TLE4", "NXPH4", "SYT6"],
    "L6_IT": ["SULF1", "OPRK1"],
}
LAYER_SYMBOLS = sorted({s for v in LAYER_MODULES.values() for s in v})

CACHE_V4 = OUT_DIR / "r_per_cell_cache_v4.parquet"
EXCLUDE_DONORS = {"Donor_1400"}
MIN_CELLS = 5
N_Q = 5
COLORS = {"PsychAD-V3": "#C0392B", "Velmeshev-V2": "#27AE60",
          "Velmeshev-V3": "#2980B9"}


# ---------------------------------------------------------------------------
# R1 — cache build with symbol-resolved markers + layer + cell_key
# ---------------------------------------------------------------------------

# Known symbol→Ensembl IDs (from q/k scripts) as a fallback when a dataset
# lacks a feature_name column. The integrated h5ads use Ensembl var_names.
KNOWN_IDS = {
    "DCX": "ENSG00000077279", "RBFOX3": "ENSG00000167281",
    "RBFOX1": "ENSG00000078328", "NEUROD2": "ENSG00000171532",
    "BCL11B": "ENSG00000127152", "SATB2": "ENSG00000119042",
    "MEF2C": "ENSG00000081189", "NEFM": "ENSG00000104722",
    "CUX2": "ENSG00000111249", "CUX1": "ENSG00000257923",
    "RORB": "ENSG00000198963", "FEZF2": "ENSG00000153266",
    "POU3F1": "ENSG00000185650", "TBR1": "ENSG00000136535",
    "FOXP2": "ENSG00000128573", "TLE4": "ENSG00000106829",
    "NXPH4": "ENSG00000182379", "SYT6": "ENSG00000147642",
    "SULF1": "ENSG00000137573", "OPRK1": "ENSG00000082556",
}


def build_symbol_map(wanted):
    """Build a global {symbol: ensembl_id} map. Prefer Velmeshev's
    feature_name column (authoritative symbol→ID), fall back to KNOWN_IDS.
    The resulting Ensembl IDs are then matched against each dataset's
    var_names (which are Ensembl in both integrated h5ads)."""
    mapping = dict(KNOWN_IDS)
    a = ad.read_h5ad(INPUTS["Velmeshev"], backed="r")
    var = a.var
    name_col = next((c for c in ["feature_name", "gene_name", "gene_symbol",
                                 "symbol", "Gene"] if c in var.columns), None)
    if name_col is not None:
        fn = var[name_col].astype(str)
        for sym in wanted:
            hit = var.index[fn.values == sym]
            if len(hit):
                mapping[sym] = hit[0]
    for sym in wanted:
        mapping.setdefault(sym, None)
    unresolved = [s for s in wanted if mapping.get(s) is None]
    print(f"  symbol map: {len(wanted)-len(unresolved)}/{len(wanted)} resolved; "
          f"unresolved={unresolved}", flush=True)
    return mapping


def build_cache():
    if CACHE_V4.exists():
        print(f"Using cached: {CACHE_V4}", flush=True)
        return pd.read_parquet(CACHE_V4)

    weights = build_c3plus_table().set_index("ensembl_id")["weight"]
    print(f"C3+ weights: {len(weights)} genes", flush=True)

    print("\nBuilding global symbol→Ensembl map...", flush=True)
    sym2ens = build_symbol_map(ALL_SYMBOLS + LAYER_SYMBOLS)
    pd.DataFrame([{"symbol": s, "ensembl": sym2ens.get(s)}
                 for s in ALL_SYMBOLS + LAYER_SYMBOLS]).to_csv(
                     OUT_DIR / "r1_marker_id_resolution.csv", index=False)

    frames = []
    for g in GROUPS:
        dataset, chem = GROUP_TO_DATASET[g]
        print(f"\n=== {g} ({dataset}, {chem})", flush=True)
        a = ad.read_h5ad(INPUTS[dataset], backed="r")
        var = a.var
        present_here = [s for s in ALL_SYMBOLS
                        if sym2ens.get(s) in set(var.index)]
        print(f"  expression markers present in this var: "
              f"{len(present_here)}/{len(ALL_SYMBOLS)}", flush=True)

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
        cell_keys = a.obs_names.values[obs_idx]
        obs = obs.iloc[obs_idx].copy().reset_index(drop=True)
        obs["cell_key"] = cell_keys
        obs["age_years"] = age.values[obs_idx]
        obs["individual"] = obs.get("individual", obs.get("donor_id"))
        total = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
        obs["total_umi"] = total
        var_names = var.index.values
        var_pos = {v: i for i, v in enumerate(var_names)}

        def col_raw(ens):
            return np.asarray(counts[:, var_pos[ens]].todense()).ravel()

        # marker raw + log1p_cp10k
        for sym in ALL_SYMBOLS:
            ens = sym2ens.get(sym)
            if ens is not None and ens in var_pos:
                cnts = col_raw(ens).astype(np.int32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    cp = np.where(total > 0, cnts / total * 1e4, 0.0)
                obs[f"raw_{sym}"] = cnts
                obs[f"cp_{sym}"] = np.log1p(cp).astype(np.float32)
            else:
                obs[f"raw_{sym}"] = 0
                obs[f"cp_{sym}"] = 0.0

        # layer assignment (argmax of layer-TF module log1p_cp10k mean)
        lscore = {}
        for lname, syms in LAYER_MODULES.items():
            cols = [var_pos[sym2ens[s]] for s in syms
                    if sym2ens.get(s) in var_pos]
            if not cols:
                lscore[lname] = np.zeros(len(obs)); continue
            sub = np.asarray(counts[:, cols].todense())
            with np.errstate(divide="ignore", invalid="ignore"):
                cp = np.where(total[:, None] > 0,
                              sub / total[:, None] * 1e4, 0.0)
            lscore[lname] = np.log1p(cp).mean(axis=1)
        ldf = pd.DataFrame(lscore)
        lmax = ldf.max(axis=1)
        layer = ldf.idxmax(axis=1)
        layer[lmax == 0] = "ambiguous"
        obs["layer"] = layer.values

        # per-cell C3+ (per-cell CPM)
        present = [g_ for g_ in weights.index if g_ in var_pos]
        grn_cols = np.array([var_pos[g_] for g_ in present], dtype=np.int64)
        grn_w = np.array([weights[g_] for g_ in present], dtype=np.float64)
        raw_dot = np.asarray(counts[:, grn_cols] @ grn_w).ravel()
        obs["per_cell_c3"] = np.where(total > 0, raw_dot / total * 1e6, 0.0)

        keep = ["cell_key", "individual", "age_years", "chemistry",
                "total_umi", "marker_annotation", "layer", "per_cell_c3"]
        for sym in ALL_SYMBOLS:
            keep += [f"raw_{sym}", f"cp_{sym}"]
        out = obs[keep].copy(); out["group"] = g
        frames.append(out)
        print(f"  layer counts: {obs['layer'].value_counts().to_dict()}",
              flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(CACHE_V4)
    print(f"\nSaved v4 cache: {CACHE_V4} ({len(df):,} rows)", flush=True)
    return df


# ---------------------------------------------------------------------------
# fuzzy-d helper on an arbitrary cell subset
# ---------------------------------------------------------------------------

def fuzzy_d(cells, min_cells=MIN_CELLS):
    don = (cells.groupby("individual", observed=True)
                .agg(score=("per_cell_c3", "mean"),
                     n_cells=("per_cell_c3", "size"),
                     age_years=("age_years", "first"))
                .reset_index())
    don = don[don["n_cells"] >= min_cells]
    don = don.dropna(subset=["score", "age_years"])
    don = don[(don["age_years"] >= AGE_LO) & (don["age_years"] < AGE_HI)]
    if len(don) < 4:
        return np.nan, len(don)
    r = fuzzy_d_from_donor_scores(don["age_years"].values, don["score"].values)
    return r["mean_d"], len(don)


def add_indices(sub):
    """Add the maturity indices to a per-group frame."""
    mm = [f"cp_{m}" for m in MODULE_MATURE if f"cp_{m}" in sub.columns]
    sub["mature_module"] = sub[mm].mean(axis=1)
    # detection-based mature module: count of mature markers with raw>=1
    det_cols = [f"raw_{m}" for m in MODULE_MATURE if f"raw_{m}" in sub.columns]
    sub["mature_detect_n"] = (sub[det_cols] >= 1).sum(axis=1)
    # immature detection count
    imm_cols = [f"raw_{m}" for m in IMMATURE_MARKERS
                if f"raw_{m}" in sub.columns]
    sub["immature_detect_n"] = (sub[imm_cols] >= 1).sum(axis=1)
    # net detection maturity = mature_detect - immature_detect
    sub["net_detect_maturity"] = sub["mature_detect_n"] - sub["immature_detect_n"]
    return sub


# ---------------------------------------------------------------------------
# R2 — maturity-index cascade (NO depth filter)
# ---------------------------------------------------------------------------

def quantile_profile(sub, score_col, nq=N_Q):
    """Return list of (q, n_cells, n_donors, d) low->high."""
    try:
        q = pd.qcut(sub[score_col], nq, labels=False, duplicates="drop")
    except Exception:
        return []
    out = []
    for qi in sorted(pd.unique(q.dropna())):
        cells = sub[q == qi]
        d, n = fuzzy_d(cells)
        out.append((int(qi), int(len(cells)), n, d,
                    float(sub.loc[q == qi, score_col].median())))
    return out


def r2_cascade(df):
    print("\n" + "=" * 72)
    print("R2 — maturity-index cascade (NO depth filter)")
    print("=" * 72)
    rows = []
    for g in GROUPS:
        sub = add_indices(df[df["group"] == g].copy())
        # binary reference
        d_bin, n_bin = fuzzy_d(sub[sub["marker_annotation"] == "ExN_immature"])
        rows.append({"group": g, "index": "binary_ExN_immature",
                     "bin": "immature", "n_cells":
                     int((sub["marker_annotation"] == "ExN_immature").sum()),
                     "n_donors": n_bin, "fuzzy_d": d_bin})
        # continuous indices: report q0 (least mature) AND full profile
        for idx_name, col, immature_is_low in [
            ("mature_module_cp10k", "mature_module", True),
            ("mature_detect_n",     "mature_detect_n", True),
            ("net_detect_maturity", "net_detect_maturity", True),
        ]:
            prof = quantile_profile(sub, col)
            for qi, ncell, ndon, d, med in prof:
                # least-mature bin == lowest q for these (immature_is_low True)
                rows.append({"group": g, "index": idx_name,
                             "bin": f"q{qi}", "n_cells": ncell,
                             "n_donors": ndon, "fuzzy_d": d,
                             "median_score": med})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "r2_maturity_cascade.csv", index=False)
    print(out.to_string(index=False))

    # plot: quantile profiles, one panel per index, mark binary ref line
    indices = ["mature_module_cp10k", "mature_detect_n", "net_detect_maturity"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, idx_name in zip(axes, indices):
        for g in GROUPS:
            s = out[(out["group"] == g) & (out["index"] == idx_name)
                    & (out["bin"].str.startswith("q"))].copy()
            s["qi"] = s["bin"].str[1:].astype(int)
            s = s.sort_values("qi")
            ax.plot(s["qi"], s["fuzzy_d"], "o-", color=COLORS[g], label=g)
            # binary ref as dashed
            b = out[(out["group"] == g)
                    & (out["index"] == "binary_ExN_immature")]
            if len(b):
                ax.axhline(b["fuzzy_d"].iloc[0], color=COLORS[g],
                           ls=":", lw=1, alpha=.6)
        ax.axhline(0, color="k", lw=.5)
        ax.set_title(idx_name)
        ax.set_xlabel("quantile (0 = least mature)")
    axes[0].set_ylabel("fuzzy Cohen's d (child→adol)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("R2 — least-mature cells carry the C3+ childhood peak "
                 "(dotted = binary DCX+RBFOX3- reference)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "r2_maturity_cascade.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("saved r2_maturity_cascade.png")


# ---------------------------------------------------------------------------
# R3 — detection-based vs CP10k module head-to-head
# ---------------------------------------------------------------------------

def r3_detection_vs_cp10k(df):
    print("\n" + "=" * 72)
    print("R3 — detection-based vs CP10k mature module (least-mature bin d)")
    print("=" * 72)
    rows = []
    for g in GROUPS:
        sub = add_indices(df[df["group"] == g].copy())
        # CP10k module, lowest quintile
        prof_cp = quantile_profile(sub, "mature_module")
        if prof_cp:
            qi, nc, nd, d, med = prof_cp[0]
            rows.append({"group": g, "method": "cp10k_module_q0",
                         "n_cells": nc, "n_donors": nd, "fuzzy_d": d})
        # detection module: cells detecting 0 mature markers (most immature)
        for k in range(0, 4):
            cells = sub[sub["mature_detect_n"] == k]
            d, n = fuzzy_d(cells)
            rows.append({"group": g, "method": f"detect_n=={k}",
                         "n_cells": int(len(cells)), "n_donors": n,
                         "fuzzy_d": d})
        # detection module <= median split (least-mature half)
        med = sub["mature_detect_n"].median()
        cells = sub[sub["mature_detect_n"] <= med]
        d, n = fuzzy_d(cells)
        rows.append({"group": g, "method": f"detect_<=median({med:.0f})",
                     "n_cells": int(len(cells)), "n_donors": n, "fuzzy_d": d})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "r3_detection_vs_cp10k.csv", index=False)
    print(out.to_string(index=False))


# ---------------------------------------------------------------------------
# R4 — depth × module 2D fuzzy-d heatmap
# ---------------------------------------------------------------------------

def two_d_heatmap(df, xcol, xlabel, fname, title, nq=4):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    allrows = []
    for ax, g in zip(axes, GROUPS):
        sub = add_indices(df[df["group"] == g].copy())
        # bin by module (maturity) and by xcol within this group
        try:
            sub["mat_bin"] = pd.qcut(sub["mature_module"], nq, labels=False,
                                     duplicates="drop")
        except Exception:
            continue
        if xcol == "total_umi":
            sub["x_bin"] = pd.qcut(sub[xcol], nq, labels=False,
                                   duplicates="drop")
            xcats = list(range(nq))
            xticklabels = [f"Q{i}" for i in xcats]
        else:  # categorical (layer)
            xcats = [c for c in ["upper", "L5_ET", "L6_IT", "L6_CT",
                                 "ambiguous"] if c in sub[xcol].unique()]
            sub["x_bin"] = sub[xcol]
            xticklabels = xcats
        mat_cats = sorted(pd.unique(sub["mat_bin"].dropna()))
        M = np.full((len(mat_cats), len(xcats)), np.nan)
        N = np.zeros_like(M)
        for i, mb in enumerate(mat_cats):
            for j, xb in enumerate(xcats):
                cells = sub[(sub["mat_bin"] == mb) & (sub["x_bin"] == xb)]
                d, n = fuzzy_d(cells)
                M[i, j] = d
                N[i, j] = len(cells)
                allrows.append({"group": g, "mat_bin": int(mb),
                                "x_bin": xb, "n_cells": int(len(cells)),
                                "n_donors": n, "fuzzy_d": d})
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto",
                       origin="lower")
        ax.set_xticks(range(len(xcats)))
        ax.set_xticklabels(xticklabels, rotation=30, fontsize=8)
        ax.set_yticks(range(len(mat_cats)))
        ax.set_yticklabels([f"matQ{int(m)}" for m in mat_cats], fontsize=8)
        ax.set_xlabel(xlabel)
        if g == GROUPS[0]:
            ax.set_ylabel("maturity module bin (0=least mature)")
        ax.set_title(g)
        for i in range(len(mat_cats)):
            for j in range(len(xcats)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:+.2f}\nn{int(N[i,j])}",
                            ha="center", va="center", fontsize=6,
                            color="k")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(allrows).to_csv(
        OUT_DIR / fname.replace(".png", ".csv"), index=False)
    print(f"saved {fname}")
    return pd.DataFrame(allrows)


def r4_depth_x_module(df):
    print("\n" + "=" * 72)
    print("R4 — depth × maturity-module 2D fuzzy d")
    print("=" * 72)
    out = two_d_heatmap(
        df, "total_umi", "depth quartile (Q0 shallow → Q3 deep)",
        "r4_depth_x_module.png",
        "R4 — fuzzy d by depth × maturity. If maturity (rows) drives the "
        "drop independent of depth (cols), each row is ~constant in colour.")
    print(out.to_string(index=False))


def r5_layer_x_module(df):
    print("\n" + "=" * 72)
    print("R5 — layer × maturity-module 2D fuzzy d")
    print("=" * 72)
    out = two_d_heatmap(
        df, "layer", "cortical layer (argmax TF module)",
        "r5_layer_x_module.png",
        "R5 — fuzzy d by layer × maturity. Tests whether the immature-cell "
        "drop is layer-specific.")
    print(out.to_string(index=False))


# ---------------------------------------------------------------------------
# R6 — per-donor decomposition + leave-one-out of module-q0
# ---------------------------------------------------------------------------

def r6_donor_robustness(df):
    print("\n" + "=" * 72)
    print("R6 — module-q0 per-donor decomposition + leave-one-out (PsychAD-V3)")
    print("=" * 72)
    g = "PsychAD-V3"
    sub = add_indices(df[df["group"] == g].copy())
    sub["mat_bin"] = pd.qcut(sub["mature_module"], N_Q, labels=False,
                             duplicates="drop")
    q0 = sub[sub["mat_bin"] == 0].copy()
    don = (q0.groupby("individual", observed=True)
              .agg(score=("per_cell_c3", "mean"),
                   n_cells=("per_cell_c3", "size"),
                   age_years=("age_years", "first"))
              .reset_index())
    don = don[(don["n_cells"] >= MIN_CELLS)
              & (don["age_years"] >= AGE_LO)
              & (don["age_years"] < AGE_HI)].dropna(subset=["score"])
    don = don.sort_values("age_years")
    don.to_csv(OUT_DIR / "r6_module_q0_per_donor.csv", index=False)
    print(f"module-q0 donors in window: {len(don)}")
    print(don.to_string(index=False))
    if len(don) < 5:
        print("Too few module-q0 donors for leave-one-out; skipping.")
        return

    # leave-one-out fuzzy d
    base = fuzzy_d_from_donor_scores(don["age_years"].values,
                                     don["score"].values)["mean_d"]
    loo = []
    for ind in don["individual"]:
        d2 = don[don["individual"] != ind]
        r = fuzzy_d_from_donor_scores(d2["age_years"].values,
                                      d2["score"].values)["mean_d"]
        loo.append({"dropped": ind, "d_without": r,
                    "delta": r - base})
    loo = pd.DataFrame(loo).sort_values("d_without")
    loo.to_csv(OUT_DIR / "r6_module_q0_leave_one_out.csv", index=False)
    print(f"\nbase module-q0 fuzzy d = {base:+.3f}")
    print("leave-one-out range: "
          f"{loo['d_without'].min():+.3f} … {loo['d_without'].max():+.3f}")
    print(loo.to_string(index=False))


# ---------------------------------------------------------------------------
# R7 — cross-cohort concordance summary (NO depth filter)
# ---------------------------------------------------------------------------

def r7_concordance(df):
    print("\n" + "=" * 72)
    print("R7 — cross-cohort immature-bin d, NO depth filter")
    print("=" * 72)
    rows = []
    for g in GROUPS:
        sub = add_indices(df[df["group"] == g].copy())
        defs = {}
        # all ExN (the FINAL_REPORT all-cell baseline, no depth filter)
        defs["all_ExN_no_depth"] = sub
        defs["binary_ExN_immature"] = sub[
            sub["marker_annotation"] == "ExN_immature"]
        try:
            q = pd.qcut(sub["mature_module"], N_Q, labels=False,
                        duplicates="drop")
            defs["module_q0"] = sub[q == 0]
        except Exception:
            pass
        defs["detect_n==0"] = sub[sub["mature_detect_n"] == 0]
        for name, cells in defs.items():
            d, n = fuzzy_d(cells)
            rows.append({"group": g, "definition": name,
                         "n_cells": int(len(cells)), "n_donors": n,
                         "fuzzy_d": d})
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index="definition", columns="group",
                          values="fuzzy_d")
    out.to_csv(OUT_DIR / "r7_concordance.csv", index=False)
    print(piv.round(3).to_string())
    print("\nInterpretation: if module_q0 / detect_n==0 bring PsychAD-V3 and")
    print("Vel-V3 closer than all_ExN_no_depth does, maturity (not depth)")
    print("is the operative variable.")


def main():
    df = build_cache()
    df = df[~df["individual"].isin(EXCLUDE_DONORS)].reset_index(drop=True)
    print(f"\nTotal cells (Donor_1400 excluded): {len(df):,}")
    r2_cascade(df)
    r3_detection_vs_cp10k(df)
    r4_depth_x_module(df)
    r5_layer_x_module(df)
    r6_donor_robustness(df)
    r7_concordance(df)
    print(f"\nAll R outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
