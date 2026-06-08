#!/usr/bin/env python3
"""
W — Is there a data-driven childhood→adolescent maturation axis, and does
the externally-derived C3+ weighting align with it?

Context: C3+ is the positive tail of the 3rd spatial component of the Allen
Human Brain Atlas (Dear et al. 2024 Nat Neurosci) — derived from ADULT bulk
microarray, INDEPENDENT of these snRNA-seq data. So testing whether C3+
aligns with a data-driven developmental axis is genuine cross-validation,
not circularity. The current 9-gene maturity module captures only early
neuronal differentiation (flat with age 1–25 y); the genes for LATER
childhood/adolescent circuit maturation are largely unknown — the hypothesis
is that C3+ captures them.

Two questions:
  PART 1 — do childhood vs adolescent ExN cells occupy different regions of
           the batch-corrected scVI embedding? (grouped-CV classifier AUC +
           donor-latent–age correlation). If yes, a late-maturation axis
           exists in the data.
  PART 2 — do the genes that change from childhood to adolescence align with
           the C3+ weighting? Per-gene child-vs-adol effect (donor
           pseudobulk) vs C3+ weight: weighted mean, rank correlation,
           enrichment of C3+ among age-changing genes, vs the 9-gene module
           and a genome-wide background. This tests whether C3+ represents
           childhood/adolescent synaptic maturation beyond birth differentiation.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=01:30:00 --mem=200G \
     scripts/run_script.sh scripts/grn_dev_diagnostics/w_age_axis.py
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
INTEGRATED = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/scvi_output/integrated.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad",
}
MANUAL = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/manual_annotations.parquet",
}
PSEUDOBULK = {
    "PsychAD":   "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/PsychAD_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
    "Velmeshev": "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/Vel_prepost_noage_tuning5/pseudobulk_output/ExN_manual_by_donor.h5ad",
}
EXCLUDE_DONORS = {"Donor_1400"}
BOUND = 10
MODULE_MATURE_ENS = {  # the 9-gene early-differentiation module (ensembl)
    "NEUROD2": "ENSG00000171532", "BCL11B": "ENSG00000127152",
    "SATB2": "ENSG00000119042", "MEF2C": "ENSG00000081189",
    "NEFM": "ENSG00000104722", "NEFH": "ENSG00000100285",
    "SYT1": "ENSG00000067715", "SNAP25": "ENSG00000132639",
    "MAP2": "ENSG00000078018",
}
COLORS = {"PsychAD-V3": "#C0392B", "Velmeshev-V2": "#27AE60",
          "Velmeshev-V3": "#2980B9"}


# ---------------------------------------------------------------------------
# PART 1 — child/adol separability in the scVI embedding
# ---------------------------------------------------------------------------

def part1():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    print("\n" + "=" * 70)
    print("PART 1 — child vs adolescent separability in scVI latent")
    print("=" * 70)
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, g in zip(axes, GROUPS):
        ds, chem = DATASET[g], GROUP_CHEM[g]
        a = ad.read_h5ad(INTEGRATED[ds], backed="r")
        obs = a.obs.copy()
        if "chemistry" not in obs and "source-chemistry" in obs:
            obs["chemistry"] = obs["source-chemistry"].astype(str).str.extract(r"(V2|V3)")[0]
        obs["chemistry"] = obs["chemistry"].astype(str)
        ma = pd.read_parquet(MANUAL[ds]); obs = obs.join(ma, how="left")
        age = pd.to_numeric(obs["age_years"], errors="coerce")
        ind = obs.get("individual", obs.get("donor_id")).astype(str)
        mask = ((age >= AGE_LO) & (age < AGE_HI)
                & obs["marker_annotation"].isin(["ExN_mature", "ExN_immature", "ExN_weak"])
                & (obs["chemistry"].values == chem)
                & ~ind.isin(EXCLUDE_DONORS)).values
        # latent
        key = "X_scVI" if "X_scVI" in a.obsm else ("X_scvi" if "X_scvi" in a.obsm else None)
        Z = np.asarray(a.obsm[key])[mask]
        agem = age.values[mask]; indm = ind.values[mask]
        y = (agem < BOUND).astype(int)   # 1 = child
        n_don = len(np.unique(indm)); n_child_don = len(np.unique(indm[y == 1]))
        print(f"\n{g}: {mask.sum():,} ExN cells, {n_don} donors "
              f"({n_child_don} child)")
        # grouped CV AUC (hold out whole donors)
        aucs = []
        if n_child_don >= 2 and (n_don - n_child_don) >= 2:
            nsplit = min(5, n_child_don, n_don - n_child_don)
            gkf = GroupKFold(n_splits=nsplit)
            for tr, te in gkf.split(Z, y, groups=indm):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(Z[tr], y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(Z[te])[:, 1]))
        auc = float(np.mean(aucs)) if aucs else np.nan
        # donor-mean latent vs age (best single dim)
        dmean = pd.DataFrame(Z, index=indm).groupby(level=0).mean()
        dage = pd.Series(agem, index=indm).groupby(level=0).first().reindex(dmean.index)
        rhos = [abs(stats.spearmanr(dmean[c], dage)[0]) for c in dmean.columns]
        best_rho = float(np.nanmax(rhos))
        print(f"  grouped-CV AUC(child vs adol) = {auc:.3f}  "
              f"(folds={len(aucs)});  max|ρ(latent_dim, age)| = {best_rho:.2f}")
        rows.append({"group": g, "n_cells": int(mask.sum()), "n_donors": n_don,
                     "n_child_donors": n_child_don, "cv_auc": auc,
                     "max_abs_rho_latent_age": best_rho})
        # 2D donor-latent (PCA of donor means) coloured by age
        from numpy.linalg import svd
        M = dmean.values - dmean.values.mean(0)
        U, S, Vt = svd(M, full_matrices=False)
        pc = U[:, :2] * S[:2]
        scv = ax.scatter(pc[:, 0], pc[:, 1], c=dage.values, cmap="viridis",
                         s=40, edgecolors="k", linewidths=0.3)
        ax.set_title(f"{g}\nAUC={auc:.2f}, max|ρ|={best_rho:.2f}", fontsize=10)
        ax.set_xlabel("donor-mean latent PC1"); ax.set_ylabel("PC2")
        fig.colorbar(scv, ax=ax, label="age (y)", fraction=0.046)
    fig.suptitle("PART 1 — childhood vs adolescent ExN in the scVI embedding "
                 "(donor-mean latent; AUC from cell-level grouped-CV classifier)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "w1_latent_separability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT_DIR / "w1_latent_separability.csv", index=False)
    print("\nsaved w1_latent_separability.{png,csv}")


# ---------------------------------------------------------------------------
# PART 2 — per-gene child→adol effect vs C3+ weight
# ---------------------------------------------------------------------------

def cohens_d_per_gene(X, is_child):
    c = X[is_child]; a = X[~is_child]
    n1, n2 = c.shape[0], a.shape[0]
    if n1 < 2 or n2 < 2:
        return np.full(X.shape[1], np.nan)
    m1, m2 = c.mean(0), a.mean(0)
    v1, v2 = c.var(0, ddof=1), a.var(0, ddof=1)
    sp_ = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(sp_ > 0, (m1 - m2) / sp_, np.nan)
    return d


def part2():
    print("\n" + "=" * 70)
    print("PART 2 — per-gene child→adol effect vs C3+ weight")
    print("=" * 70)
    c3 = build_c3plus_table().set_index("ensembl_id")["weight"]
    print(f"C3+ genes: {len(c3)}")

    per_gene = {}
    summ_rows = []
    for g in GROUPS:
        ds, chem = DATASET[g], GROUP_CHEM[g]
        a = ad.read_h5ad(PSEUDOBULK[ds])
        obs = a.obs.copy()
        if "chemistry" not in obs and "source-chemistry" in obs:
            obs["chemistry"] = obs["source-chemistry"].astype(str).str.extract(r"(V2|V3)")[0]
        # PsychAD PB is all V3; Velmeshev split by chemistry
        if "chemistry" in obs:
            keep = obs["chemistry"].astype(str).str.contains(chem, na=False) if ds == "Velmeshev" else np.ones(len(obs), bool)
        else:
            keep = np.ones(len(obs), bool)
        age = pd.to_numeric(obs["age_years"], errors="coerce")
        ind = obs.index.astype(str)
        keep = keep & (age >= AGE_LO).values & (age < AGE_HI).values & ~ind.isin(EXCLUDE_DONORS)
        counts = a.layers["counts"] if "counts" in a.layers else a.X
        counts = sp.csr_matrix(counts)[np.where(keep)[0], :]
        tot = np.asarray(counts.sum(1)).ravel()
        cpm = counts.multiply(1.0 / np.where(tot > 0, tot, 1)[:, None]).tocsr() * 1e6
        X = np.log1p(np.asarray(cpm.todense()))
        agem = age.values[keep]
        is_child = agem < BOUND
        d = cohens_d_per_gene(X, is_child)   # + = higher in childhood
        gd = pd.Series(d, index=a.var_names.values)
        per_gene[g] = gd
        # merge with C3+
        df = pd.DataFrame({"age_d": gd}).join(c3.rename("c3_weight"), how="left")
        in_c3 = df["c3_weight"].notna()
        bg = df.loc[~in_c3, "age_d"].dropna()
        c3d = df.loc[in_c3].dropna(subset=["age_d"])
        wmean = np.average(c3d["age_d"], weights=c3d["c3_weight"])
        rho_w = stats.spearmanr(c3d["c3_weight"], c3d["age_d"])[0]
        # enrichment of C3+ in top-300 child-elevated genes
        allg = df.dropna(subset=["age_d"])
        topN = 300
        top = allg["age_d"].nlargest(topN).index
        k = in_c3.reindex(top).fillna(False).sum()
        K = int(in_c3.sum()); N = int(len(allg)); n = topN
        hp = stats.hypergeom.sf(k-1, N, K, n)
        # module genes age_d
        mod_d = gd.reindex(list(MODULE_MATURE_ENS.values())).dropna()
        summ_rows.append({
            "group": g, "n_child_donors": int(is_child.sum()),
            "n_adol_donors": int((~is_child).sum()),
            "C3+_mean_age_d": float(c3d["age_d"].mean()),
            "C3+_weighted_mean_age_d": float(wmean),
            "background_mean_age_d": float(bg.mean()),
            "spearman_weight_vs_age_d": float(rho_w),
            "top300_C3+_count": int(k), "top300_expected": round(n*K/N, 1),
            "hypergeom_p": float(hp),
            "module9_mean_age_d": float(mod_d.mean())})
        print(f"\n{g}: child={is_child.sum()} adol={(~is_child).sum()} donors")
        print(f"  C3+ mean age_d = {c3d['age_d'].mean():+.3f} "
              f"(weighted {wmean:+.3f}) vs background {bg.mean():+.3f}")
        print(f"  Spearman(C3+ weight, age_d) = {rho_w:+.3f}")
        print(f"  top-300 child-elevated: {k} C3+ (expected {n*K/N:.0f}), "
              f"hypergeom p={hp:.1e}")
        print(f"  9-gene module mean age_d = {mod_d.mean():+.3f} "
              f"(early-diff markers, expected ~0)")

    summ = pd.DataFrame(summ_rows)
    summ.to_csv(OUT_DIR / "w2_age_vs_c3_summary.csv", index=False)
    # per-gene table merged across cohorts
    pg = pd.DataFrame(per_gene)
    pg = pg.join(c3.rename("c3_weight"))
    pg.to_csv(OUT_DIR / "w2_per_gene_age_d.csv")

    # scatter: C3+ weight vs age_d per cohort
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, g in zip(axes, GROUPS):
        df = pd.DataFrame({"age_d": per_gene[g]}).join(c3.rename("w"), how="inner").dropna()
        ax.scatter(df["w"], df["age_d"], s=5, alpha=0.25, color=COLORS[g])
        rho = stats.spearmanr(df["w"], df["age_d"])[0]
        # binned mean
        try:
            df["wq"] = pd.qcut(df["w"], 10, labels=False, duplicates="drop")
            bm = df.groupby("wq").agg(w=("w", "median"), d=("age_d", "mean"))
            ax.plot(bm["w"], bm["d"], "o-", color="k", lw=1.5, ms=4)
        except Exception:
            pass
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{g}\nρ(C3+ weight, child→adol d) = {rho:+.2f}", fontsize=10)
        ax.set_xlabel("C3+ weight (AHBA, external)")
        if g == GROUPS[0]:
            ax.set_ylabel("per-gene child→adol Cohen's d\n(+ = higher in childhood)")
    fig.suptitle("PART 2 — do high-C3+-weight genes decline from childhood to "
                 "adolescence? (C3+ weights are external: AHBA / Dear 2024)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "w2_age_vs_c3_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n--- PART 2 summary ---")
    print(summ.round(3).to_string(index=False))
    print("saved w2_age_vs_c3_{summary.csv,scatter.png}, w2_per_gene_age_d.csv")


def main():
    try:
        part1()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"PART 1 failed: {e}")
    try:
        part2()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"PART 2 failed: {e}")
    print(f"\nAll W outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
