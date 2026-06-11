#!/usr/bin/env python3
"""
X — Why is Velmeshev-V2 technically confounded for the child→adolescent
contrast? A clean, self-contained demonstration.

Claim: V2's large child→adol C3+ "drop" is partly a NON-BIOLOGICAL,
depth-driven, transcriptome-wide shift, not a specific synaptic-maturation
signal. The diagnostic logic:

  A real developmental signal is SPECIFIC  — it moves a coherent gene
     programme (e.g. synaptic genes) and leaves the rest of the transcriptome
     roughly unchanged, so the genome-wide mean child→adol effect ≈ 0.
  A technical depth artefact is NON-SPECIFIC and EXPRESSION-DEPENDENT — if
     childhood and adolescent donors differ in sequencing depth, then after
     per-cell CPM + log1p the per-gene means shift systematically across the
     WHOLE transcriptome, most strongly for low-abundance genes (whose
     log1p-CPM is dominated by the Poisson/zero-inflation floor and is highly
     depth-sensitive). C3+ is a large, low-to-moderate-expression synaptic
     gene set — exactly the genes this artefact hits hardest.

Three panels:
  A. per-donor MEAN per-cell depth vs age (3 cohorts): is there a child-vs-adol
     depth imbalance, and how shallow is V2?
  B. distribution of per-gene child→adol Cohen's d over ALL genes (3 cohorts):
     a clean cohort is centred at 0; V2's whole transcriptome is shifted
     (the "background" shift) — the signature of a global technical effect.
  C. per-gene child→adol d binned by gene expression decile (3 cohorts):
     in V2 the shift is concentrated in low-expression genes (depth-sensitive)
     = mechanism. V3/PsychAD are flat.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=00:40:00 --mem=120G \
     scripts/run_script.sh scripts/grn_dev_diagnostics/x_v2_confound.py
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
from _lib import OUT_DIR, build_c3plus_table, AGE_LO, AGE_HI

GROUPS = ["PsychAD-V3", "Velmeshev-V2", "Velmeshev-V3"]
GROUP_CHEM = {"PsychAD-V3": "V3", "Velmeshev-V2": "V2", "Velmeshev-V3": "V3"}
DATASET = {"PsychAD-V3": "PsychAD", "Velmeshev-V2": "Velmeshev",
           "Velmeshev-V3": "Velmeshev"}
PSEUDOBULK = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
}
EXCLUDE_DONORS = {"Donor_1400"}
BOUND = 10
COLORS = {"PsychAD-V3": "#C0392B", "Velmeshev-V2": "#27AE60",
          "Velmeshev-V3": "#2980B9"}


def cohens_d_cols(X, is_child):
    c, a = X[is_child], X[~is_child]
    n1, n2 = c.shape[0], a.shape[0]
    if n1 < 2 or n2 < 2:
        return np.full(X.shape[1], np.nan)
    m1, m2 = c.mean(0), a.mean(0)
    v1, v2 = c.var(0, ddof=1), a.var(0, ddof=1)
    sp_ = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sp_ > 0, (m1 - m2) / sp_, np.nan)


def find_ncells(obs):
    for c in ["n_cells", "n_cells_pseudobulk", "ncells", "n_obs", "size",
              "cell_count", "counts_n_cells"]:
        if c in obs.columns:
            return pd.to_numeric(obs[c], errors="coerce").values
    return None


def main():
    c3_ens = set(build_c3plus_table()["ensembl_id"])
    depth_rows, summ_rows = [], []
    per_gene_d, per_gene_expr = {}, {}

    for g in GROUPS:
        ds, chem = DATASET[g], GROUP_CHEM[g]
        a = ad.read_h5ad(PSEUDOBULK[ds])
        obs = a.obs.copy()
        if "chemistry" not in obs and "source-chemistry" in obs:
            obs["chemistry"] = obs["source-chemistry"].astype(str).str.extract(r"(V2|V3)")[0]
        if ds == "Velmeshev" and "chemistry" in obs:
            keep = obs["chemistry"].astype(str).str.contains(chem, na=False).values
        else:
            keep = np.ones(len(obs), bool)
        age = pd.to_numeric(obs["age_years"], errors="coerce")
        ind = obs.index.astype(str)
        keep = keep & (age >= AGE_LO).values & (age < AGE_HI).values \
            & ~ind.isin(EXCLUDE_DONORS) & age.notna().values
        ki = np.where(keep)[0]

        counts = sp.csr_matrix(a.layers["counts"] if "counts" in a.layers else a.X)[ki, :]
        tot = np.asarray(counts.sum(1)).ravel()           # per-donor total UMI
        ncells = find_ncells(obs)
        ncells = ncells[ki] if ncells is not None else None
        # per-cell mean depth (preferred) else per-donor total
        if ncells is not None and np.all(np.isfinite(ncells)) and np.all(ncells > 0):
            depth = tot / ncells
            depth_label = "mean per-cell UMI"
        else:
            depth = tot
            depth_label = "per-donor total UMI"
        agem = age.values[ki]
        is_child = agem < BOUND

        # per-donor depth vs age
        for d_, ag_, ic in zip(depth, agem, is_child):
            depth_rows.append({"group": g, "age": ag_, "depth": d_,
                               "stage": "child" if ic else "adol"})
        # child vs adol depth test
        dc, da = depth[is_child], depth[~is_child]
        mwu_p = stats.mannwhitneyu(dc, da).pvalue if (len(dc) >= 2 and len(da) >= 2) else np.nan

        # per-cell CPM (each donor pseudobulk treated as a sample) -> log1p
        cpm = counts.multiply(1.0 / np.where(tot > 0, tot, 1)[:, None]).tocsr() * 1e6
        X = np.log1p(np.asarray(cpm.todense()))
        d = cohens_d_cols(X, is_child)                    # + = higher in child
        expr = X.mean(0)                                  # mean log1p-CPM per gene
        gd = pd.Series(d, index=a.var_names.values)
        ge = pd.Series(expr, index=a.var_names.values)
        per_gene_d[g] = gd
        per_gene_expr[g] = ge

        is_c3 = a.var_names.isin(c3_ens)
        fin = np.isfinite(d)
        bg = d[~is_c3 & fin]
        # C3+ stats: mean d and expression-matched background
        c3_d_vals = d[is_c3 & fin]
        c3_e_vals = expr[is_c3 & fin]
        if len(c3_e_vals) > 0:
            lo, hi = np.nanpercentile(c3_e_vals, [25, 75])
            in_range = (~is_c3) & fin & (expr >= lo) & (expr <= hi)
            c3_expr_matched_bg = float(np.nanmean(d[in_range])) if in_range.sum() > 0 else np.nan
        else:
            c3_expr_matched_bg = np.nan
        summ_rows.append({
            "group": g, "depth_label": depth_label,
            "n_child_donors": int(is_child.sum()),
            "n_adol_donors": int((~is_child).sum()),
            "child_median_depth": float(np.median(dc)) if len(dc) else np.nan,
            "adol_median_depth": float(np.median(da)) if len(da) else np.nan,
            "child/adol_depth_ratio": float(np.median(dc)/np.median(da)) if (len(dc) and len(da)) else np.nan,
            "depth_child_vs_adol_MWU_p": float(mwu_p),
            "background_mean_age_d": float(np.nanmean(bg)),
            "background_median_age_d": float(np.nanmedian(bg)),
            "frac_genes_abs_d_gt_0.5": float(np.mean(np.abs(d[fin]) > 0.5)),
            "spearman_age_d_vs_expr": float(stats.spearmanr(
                expr[fin], d[fin]).correlation),
            "c3_mean_age_d": float(np.nanmean(c3_d_vals)) if len(c3_d_vals) > 0 else np.nan,
            "c3_expr_matched_bg_d": c3_expr_matched_bg,
        })
        print(f"{g}: child_depth={np.median(dc):.0f} adol_depth={np.median(da):.0f} "
              f"({depth_label}); bg_mean_d={np.nanmean(bg):+.3f}; "
              f"rho(d,expr)={summ_rows[-1]['spearman_age_d_vs_expr']:+.3f}")

    summ = pd.DataFrame(summ_rows)
    summ.to_csv(OUT_DIR / "x_v2_confound_summary.csv", index=False)
    pd.DataFrame(depth_rows).to_csv(OUT_DIR / "x_v2_donor_depth.csv", index=False)

    # Per-gene CSV for V2 (used to verify floor-inflation mechanism for C3+)
    v2g = "Velmeshev-V2"
    v2_gene_df = pd.DataFrame({
        "gene": per_gene_d[v2g].index,
        "d_V2": per_gene_d[v2g].values,
        "mean_expr_V2": per_gene_expr[v2g].values,
        "is_c3": per_gene_d[v2g].index.isin(c3_ens),
    })
    v2_gene_df.to_csv(OUT_DIR / "x_v2_per_gene_d_expr.csv", index=False)

    # -------------------- figure --------------------
    dd = pd.DataFrame(depth_rows)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.4))

    # Panel A: per-donor depth vs age
    axA = axes[0]
    axA.axvspan(AGE_LO, BOUND, color="gold", alpha=0.10)
    for g in GROUPS:
        s = dd[dd["group"] == g]
        axA.scatter(s["age"], s["depth"], s=42, color=COLORS[g], alpha=0.8,
                    edgecolors="k", linewidths=0.3, label=g)
    axA.axvline(BOUND, color="grey", ls="--", lw=0.8)
    axA.set_yscale("log")
    axA.set_xlabel("donor age (y)")
    axA.set_ylabel("mean per-cell UMI (log scale)")
    axA.set_title("A. Per-donor sequencing depth vs age\n"
                  "V2 is ~3–4× shallower; check child/adol balance", fontsize=10)
    axA.legend(fontsize=8, loc="upper right")

    # Panel B: distribution of per-gene child->adol d over ALL genes
    axB = axes[1]
    xs = np.linspace(-1.5, 1.5, 400)
    for g in GROUPS:
        d = per_gene_d[g].values
        d = d[np.isfinite(d)]
        kde = stats.gaussian_kde(d)
        axB.plot(xs, kde(xs), color=COLORS[g], lw=2, label=g)
        m = np.nanmean(d)
        axB.axvline(m, color=COLORS[g], ls="--", lw=1.2)
    axB.axvline(0, color="k", lw=0.8)
    axB.set_xlabel("per-gene child→adol Cohen's d  (+ = higher in childhood)")
    axB.set_ylabel("density (all genes)")
    axB.set_title("B. Transcriptome-wide shift = technical, not biological\n"
                  "clean cohorts centre at 0; V2's WHOLE transcriptome shifts",
                  fontsize=10)
    axB.legend(fontsize=8)

    # Panel C: per-gene d by expression decile
    axC = axes[2]
    for g in GROUPS:
        df = pd.DataFrame({"d": per_gene_d[g], "e": per_gene_expr[g]}).dropna()
        df = df[df["e"] > 0]
        df["eq"] = pd.qcut(df["e"], 10, labels=False, duplicates="drop")
        bm = df.groupby("eq").agg(e=("e", "median"), d=("d", "mean"))
        axC.plot(bm["e"], bm["d"], "o-", color=COLORS[g], lw=1.8, ms=5, label=g)
    axC.axhline(0, color="k", lw=0.8)
    axC.set_xlabel("gene expression decile (median log1p-CPM →)")
    axC.set_ylabel("mean child→adol Cohen's d in decile")
    axC.set_title("C. The shift is expression-dependent in V2\n"
                  "low-expression (depth-sensitive) genes carry it; flat in V3/PsychAD",
                  fontsize=10)
    axC.legend(fontsize=8)

    # Panel D: V2 per-gene d vs expression, C3+ highlighted
    # Shows that C3+ genes (d≈−0.06) are near-neutral while expression-matched
    # background is strongly negative (d≈−0.32): C3+ is resistant to depth suppression,
    # not inflated by it — the positive module score runs counter to the depth artefact.
    axD = axes[3]
    v2g = "Velmeshev-V2"
    dfD = pd.DataFrame({"d": per_gene_d[v2g], "e": per_gene_expr[v2g]}).dropna()
    dfD = dfD[dfD["e"] > 0]
    is_c3_D = dfD.index.isin(c3_ens)
    axD.scatter(dfD.loc[~is_c3_D, "e"], dfD.loc[~is_c3_D, "d"],
                s=2, c="grey", alpha=0.12, rasterized=True, label="_nolegend_")
    axD.scatter(dfD.loc[is_c3_D, "e"], dfD.loc[is_c3_D, "d"],
                s=14, c="#E74C3C", alpha=0.75, zorder=5,
                label=f"C3+ genes (n={is_c3_D.sum()})")
    # Background decile curve (non-C3+ only)
    dfD_bg = dfD[~is_c3_D].copy()
    dfD_bg["eq"] = pd.qcut(dfD_bg["e"], 10, labels=False, duplicates="drop")
    bmD = dfD_bg.groupby("eq").agg(e=("e", "median"), d=("d", "mean"))
    axD.plot(bmD["e"], bmD["d"], "ko-", lw=2, ms=5, zorder=6,
             label="background (decile mean, non-C3+)")
    axD.axhline(0, color="k", lw=0.8)
    # Vertical band for C3+ IQR expression range
    c3_e = dfD.loc[is_c3_D, "e"]
    axD.axvspan(c3_e.quantile(0.25), c3_e.quantile(0.75), alpha=0.08,
                color="#E74C3C", label="C3+ expr range (p25–p75)")
    axD.set_xlabel("mean log1p-CPM (Velmeshev-V2)")
    axD.set_ylabel("child→adol Cohen's d  (+ = higher in child)")
    axD.set_title("D. V2: C3+ genes resistant to depth suppression\n"
                  "C3+ (red) near d=0; expression-matched background ≈ −0.32",
                  fontsize=10)
    axD.legend(fontsize=7, loc="upper left")

    fig.suptitle("Why Velmeshev-V2 is technically confounded for the child→adolescent contrast",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "x_v2_confound.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n--- summary ---")
    print(summ.round(3).to_string(index=False))
    print(f"\nsaved x_v2_confound.png, x_v2_confound_summary.csv, x_v2_donor_depth.csv,")
    print(f"      x_v2_per_gene_d_expr.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()
