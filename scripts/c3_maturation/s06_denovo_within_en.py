#!/usr/bin/env python3
"""Step 2 — de novo within-EN developmental program, and its alignment to C3.

Instead of asking whether the C3-weighted *aggregate* score moves within EN
(weak; Step 1), ask the per-gene question: WHICH genes change with age within
mature EN, do they REPLICATE across cohorts, and are they the C3 / synaptic genes?

For each cohort (PsychAD-V3, Herring-V3, U01-V2), on mature-EN postnatal
pseudobulks (excitatory_by_celltype):
  - per-gene within-EN age slope = slope of log1p(CPM) on age after residualising
    BOTH on subtype identity + sequencing depth (so it is a within-subtype,
    depth-controlled developmental slope).
Then:
  (Q1) cross-cohort replication: correlate the per-gene slope vectors between cohorts.
  (Q2) alignment to C3: correlate per-gene slope with signed C3 weight (vs C1/C2 control).
  (Q3) biology: do canonical synaptic / maturation genes carry the signal?

Inline-safe (small EN pseudobulks).
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L
from s04_within_en_cohorts import (VEL_MATURE, PSY_MATURE, AGE_LO_POST, AGE_LO_PSY, AGE_HI)

# Canonical synaptic / maturation genes (incl. the prior NRXN1/NLGN1/GRIK... drop set)
SYNAPTIC = ["NRXN1", "NRXN3", "NLGN1", "NLGN4X", "GRIK1", "GRIK2", "GRM7", "GRM1",
            "KIRREL3", "NTM", "MEF2C", "DLGAP1", "DLGAP2", "DLG2", "DLG4", "SHANK2",
            "SHANK3", "HOMER1", "SYN1", "SYN2", "SYT1", "SNAP25", "GRIN1", "GRIN2A",
            "GRIN2B", "GRIA1", "GRIA2", "GABRA1", "GABRB2", "CACNA1C", "ANK3",
            "RBFOX1", "CNTNAP2", "NRG1", "ERBB4"]
SWITCH = {"GRIN2B": "NMDA-immature", "GRIN2A": "NMDA-mature",
          "GABRA2": "GABA-immature", "GABRA1": "GABA-mature",
          "SLC12A2": "Cl-immature(NKCC1)", "SLC12A5": "Cl-mature(KCC2)"}


def sym2ens_map():
    ref = ad.read_h5ad(L.PB["Vel_ExN_by_donor"])
    col = "gene_symbol" if "gene_symbol" in ref.var.columns else "feature_name"
    return (ref.var.reset_index().rename(columns={"index": "ens"})
            .dropna(subset=[col]).drop_duplicates(col).set_index(col)["ens"])


def per_gene_slopes(path, mature, age_lo, subsource=None):
    a = ad.read_h5ad(path)
    obs = a.obs
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    sub = obs["cell_type_aligned"].astype(str).values
    nc = obs["n_cells"].values
    keep = np.isin(sub, mature) & (age >= age_lo) & (age < AGE_HI) & (nc >= 20)
    if subsource is not None:
        keep &= (obs["dataset"].astype(str).values == subsource)
    a = a[keep].copy()
    cpm = L.cpm_matrix(a)
    M = np.log1p(cpm)                       # samples x genes
    age = pd.to_numeric(a.obs["age_years"], errors="coerce").values
    log10_total = np.log10(np.asarray(a.layers["counts"].sum(1)).ravel().clip(1))
    subt = a.obs["cell_type_aligned"].astype(str).values
    # design Z = [1, subtype dummies, depth]
    sts = sorted(np.unique(subt))[1:]
    Z = [np.ones(len(age)), log10_total]
    for s in sts:
        Z.append((subt == s).astype(float))
    Z = np.column_stack(Z)
    # residualise age and M on Z
    Zi = np.linalg.pinv(Z.T @ Z) @ Z.T
    a_res = age - Z @ (Zi @ age)
    M_res = M - Z @ (Zi @ M)
    denom = float(a_res @ a_res)
    slopes = (M_res.T @ a_res) / denom        # gene -> slope
    mean_expr = pd.Series(M.mean(0), index=a.var_names.values, name="mean_expr")
    return pd.Series(slopes, index=a.var_names.values, name="slope"), mean_expr, len(age)


def main():
    vel = L.B / "Vel_prepost_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad"
    psy = L.B / "PsychAD_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad"
    cohorts, mexpr = {}, {}
    cohorts["PsychAD-V3"], mexpr["PsychAD-V3"], n1 = per_gene_slopes(psy, PSY_MATURE, AGE_LO_PSY)
    cohorts["Herring-V3"], mexpr["Herring-V3"], n2 = per_gene_slopes(vel, VEL_MATURE, AGE_LO_POST, subsource="Herring")
    cohorts["U01-V2"], mexpr["U01-V2"], n3 = per_gene_slopes(vel, VEL_MATURE, AGE_LO_POST, subsource="U01")
    for k, n in zip(cohorts, [n1, n2, n3]):
        print(f"  {k}: {len(cohorts[k])} genes, {n} samples")

    slopes = pd.DataFrame(cohorts)

    # ---- Q1: cross-cohort replication of the per-gene developmental program ----
    print("\n(Q1) cross-cohort Spearman correlation of per-gene within-EN age slopes:")
    cohort_names = list(cohorts)
    rep = pd.DataFrame(index=cohort_names, columns=cohort_names, dtype=float)
    for i in cohort_names:
        for j in cohort_names:
            if i == j:
                rep.loc[i, j] = 1.0; continue
            d = slopes[[i, j]].dropna()
            rep.loc[i, j] = stats.spearmanr(d[i].values, d[j].values).statistic
    print(rep.round(3).to_string())

    # ---- Q2: alignment to C3 (and C1/C2 control) ----
    ahba = L.ahba_weights_ensembl(("C1", "C2", "C3"))
    print("\n(Q2) Spearman(per-gene within-EN age slope, AHBA weight):")
    q2 = []
    for coh in cohort_names:
        row = {"cohort": coh}
        for comp in ["C1", "C2", "C3"]:
            d = pd.concat([slopes[coh].rename("slope"), ahba[comp].rename("w")], axis=1).dropna()
            row[comp] = stats.spearmanr(d["slope"].values, d["w"].values).statistic
            dp = d[d["w"] > 0]
            row[f"{comp}+only"] = stats.spearmanr(dp["slope"].values, dp["w"].values).statistic
        q2.append(row)
    q2 = pd.DataFrame(q2)
    print(q2.round(3).to_string(index=False))
    q2.to_csv(L.OUT_DIR / "s06_slope_vs_ahba.csv", index=False)

    # expression-level confound check: is the C1 alignment just mean-expression?
    print("\n(Q2b) expression-confound check (PsychAD-V3):")
    def psp(x, y, z):  # partial spearman of x,y given z
        from s01a_within_celltype_trajectory import partial_spearman as ps
        return ps(x, y, z)
    coh = "PsychAD-V3"
    me = mexpr[coh]
    for comp in ["C1", "C3"]:
        d = pd.concat([slopes[coh].rename("slope"), ahba[comp].rename("w"),
                       me.rename("expr")], axis=1).dropna()
        r_raw = stats.spearmanr(d["slope"].values, d["w"].values).statistic
        r_partial = psp(d["slope"].values, d["w"].values, d["expr"].values)
        r_w_expr = stats.spearmanr(d["w"].values, d["expr"].values).statistic
        r_slope_expr = stats.spearmanr(d["slope"].values, d["expr"].values).statistic
        print(f"  {comp}: slope~w raw={r_raw:+.3f}  partial|expr={r_partial:+.3f}  "
              f"(w~expr={r_w_expr:+.3f}, slope~expr={r_slope_expr:+.3f})")

    # ---- Q3: synaptic / maturation genes ----
    s2e = sym2ens_map()
    syn_ens = {s2e[g]: g for g in SYNAPTIC if g in s2e.index}
    print(f"\n(Q3) synaptic gene within-EN age slopes (sign: - = declines with age):")
    syn_tab = slopes.loc[slopes.index.intersection(list(syn_ens))].rename(index=syn_ens)
    syn_tab = syn_tab.reindex([syn_ens[e] for e in syn_ens if e in slopes.index])
    print(syn_tab.round(2).to_string())
    syn_tab.to_csv(L.OUT_DIR / "s06_synaptic_gene_slopes.csv")
    # synaptic set vs background: median slope
    print("\n  median within-EN slope: synaptic set vs all genes:")
    for coh in cohort_names:
        med_syn = syn_tab[coh].median()
        med_all = slopes[coh].median()
        u = stats.mannwhitneyu(syn_tab[coh].dropna(),
                               slopes[coh].dropna(), alternative="two-sided")
        print(f"    {coh}: synaptic median={med_syn:+.3f}  all={med_all:+.3f}  MWU p={u.pvalue:.2g}")

    # switch genes
    sw_ens = {s2e[g]: g for g in SWITCH if g in s2e.index}
    sw_tab = slopes.loc[slopes.index.intersection(list(sw_ens))].rename(index=sw_ens)
    print("\n  maturation-switch genes (immature->mature expected):")
    print(sw_tab.assign(role=[SWITCH[g] for g in sw_tab.index]).round(2).to_string())

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # (a) cross-cohort replication scatter
    d = slopes[["PsychAD-V3", "Herring-V3"]].dropna()
    axes[0].scatter(d["PsychAD-V3"], d["Herring-V3"], s=4, alpha=.2)
    axes[0].axhline(0, c="grey", lw=.5); axes[0].axvline(0, c="grey", lw=.5)
    axes[0].set_title(f"Q1 cross-cohort slopes\nrho={rep.loc['PsychAD-V3','Herring-V3']:.2f}")
    axes[0].set_xlabel("PsychAD-V3 within-EN age slope"); axes[0].set_ylabel("Herring-V3")
    # (b) slope vs C3 weight (PsychAD)
    d = pd.concat([slopes["PsychAD-V3"], ahba["C3"]], axis=1).dropna()
    axes[1].scatter(d["C3"], d["PsychAD-V3"], s=4, alpha=.2)
    axes[1].axhline(0, c="grey", lw=.5); axes[1].axvline(0, c="grey", lw=.5)
    axes[1].set_title(f"Q2 PsychAD-V3 slope vs C3 weight\nrho={q2[q2.cohort=='PsychAD-V3']['C3'].iloc[0]:.2f}")
    axes[1].set_xlabel("AHBA C3 weight"); axes[1].set_ylabel("within-EN age slope")
    # (c) synaptic gene slopes across cohorts
    st = syn_tab.dropna()
    y = np.arange(len(st))
    for k, c in zip(cohort_names, ["C0", "C2", "C1"]):
        axes[2].scatter(st[k], y, s=20, label=k, color=c, alpha=.8)
    axes[2].axvline(0, c="grey", lw=.8); axes[2].set_yticks(y); axes[2].set_yticklabels(st.index, fontsize=6)
    axes[2].set_title("Q3 synaptic gene within-EN slopes"); axes[2].set_xlabel("age slope"); axes[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(L.OUT_DIR / "s06_denovo_within_en.png", dpi=140, bbox_inches="tight")
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
