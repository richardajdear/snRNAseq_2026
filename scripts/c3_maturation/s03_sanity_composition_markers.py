#!/usr/bin/env python3
"""C3-maturation — DATA SANITY CHECKS (appendix).

Addresses the misclassification concern: PsychAD native labels came from an
aging/dementia reference, so young-donor EN subtypes may be unreliable.

(A) Composition vs age, per cohort, using PsychAD's *marker-based* annotation
    (ExN_*/InN/glia/Unknown — independent of the aging reference) and
    Velmeshev's cell_class_original. Do young donors have plausible EN fractions?
(B) Marker expression INSIDE the labeled-EN subtype pseudobulks
    (excitatory_by_celltype), by age. If young "mature EN" pseudobulks express
    high IN/glial markers, the labels are contaminated and the within-EN
    trajectory is suspect.

Cohorts treated separately: PsychAD-V3, Herring-V3 (Velmeshev), U01-V2 (Velmeshev),
Ramos (Velmeshev, mostly prenatal).

Inline-safe: composition from backed obs; markers from the (<500MB) EN objects.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import _lib_c3 as L

MARKERS = {  # Ensembl -> (symbol, lineage)
    "ENSG00000104888": ("SLC17A7", "EN"), "ENSG00000119042": ("SATB2", "EN"),
    "ENSG00000167281": ("RBFOX3", "EN"),
    "ENSG00000128683": ("GAD1", "IN"), "ENSG00000136750": ("GAD2", "IN"),
    "ENSG00000171885": ("AQP4", "Astro"), "ENSG00000131095": ("GFAP", "Astro"),
    "ENSG00000123560": ("PLP1", "Oligo"), "ENSG00000168314": ("MOBP", "Oligo"),
    "ENSG00000134853": ("PDGFRA", "OPC"), "ENSG00000182578": ("CSF1R", "Micro"),
    "ENSG00000077279": ("DCX", "Immature"),
}
EXN = {"ExN_mature", "ExN_immature", "ExN_weak"}


def psychad_cohort(obs):
    return pd.Series("PsychAD-V3", index=obs.index)


def vel_cohort(obs):
    ds = obs["dataset"].astype(str); ch = obs["chemistry"].astype(str)
    return ds.where(ds != "Herring", "Herring-V3").where(ds != "U01", "U01-V2") \
             .map(lambda x: {"Herring": "Herring-V3", "U01": "U01-V2",
                             "Ramos": "Ramos-V3"}.get(x, x)) \
             .where(~ds.isin(["Herring", "U01", "Ramos"]),
                    ds.map({"Herring": "Herring-V3", "U01": "U01-V2", "Ramos": "Ramos-V3"}))


# ---------------- (A) composition vs age ----------------

def composition_table():
    rows = []
    # PsychAD: marker_annotation
    pa = ad.read_h5ad(L.B / "PsychAD_noage_tuning5/pseudobulk_output/by_cell_class_manual.h5ad",
                      backed="r").obs.copy()
    pa["broad"] = np.where(pa["marker_annotation"].astype(str).isin(EXN), "EN",
                  np.where(pa["marker_annotation"].astype(str) == "InN", "IN",
                  np.where(pa["marker_annotation"].astype(str) == "Unknown", "Unknown", "Glia")))
    pa["cohort"] = "PsychAD-V3"
    # Velmeshev: cell_class_original
    ve = ad.read_h5ad(L.B / "Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class.h5ad",
                      backed="r").obs.copy()
    cc = ve["cell_class_original"].astype(str)
    ve["broad"] = np.where(cc == "Excitatory", "EN",
                  np.where(cc == "Inhibitory", "IN", "Glia"))
    ve["cohort"] = vel_cohort(ve)

    out = {}
    for nm, df in [("PsychAD", pa), ("Velmeshev", ve)]:
        g = (df.groupby(["cohort", "individual", "age_years", "broad"], observed=True)["n_cells"]
             .sum().reset_index())
        tot = g.groupby(["cohort", "individual", "age_years"], observed=True)["n_cells"].transform("sum")
        g["frac"] = g["n_cells"] / tot
        out[nm] = g
    return out


# ---------------- (B) marker expression in labeled-EN pseudobulks ----------------

def en_marker_table():
    rows = []
    for nm, path, stcol, cohfn in [
        ("PsychAD-V3", L.B / "PsychAD_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad",
         "cell_type_aligned", lambda o: pd.Series("PsychAD-V3", index=o.index)),
        ("Velmeshev", L.B / "Vel_prepost_noage_tuning5/pseudobulk_output/excitatory_by_celltype.h5ad",
         "cell_type_aligned", vel_cohort),
    ]:
        a = ad.read_h5ad(path)
        cpm = L.cpm_matrix(a)
        vn = list(a.var_names)
        sub = pd.DataFrame(index=a.obs_names)
        for ens, (sym, lin) in MARKERS.items():
            if ens in vn:
                sub[f"{sym}|{lin}"] = np.log1p(cpm[:, vn.index(ens)])
        sub["subtype"] = a.obs[stcol].astype(str).values
        sub["age_years"] = pd.to_numeric(a.obs["age_years"], errors="coerce").values
        sub["cohort"] = cohfn(a.obs).values if nm == "Velmeshev" else nm
        sub["n_cells"] = a.obs["n_cells"].values
        rows.append(sub)
    return pd.concat(rows)


def main():
    comp = composition_table()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (nm, g) in zip(axes, comp.items()):
        en = g[g["broad"] == "EN"]
        for coh, s in en.groupby("cohort"):
            ax.scatter(s["age_years"], s["frac"], s=28, alpha=.7, label=f"{coh} (n={len(s)})")
        ax.axhline(0.4, ls=":", c="grey"); ax.set_ylim(0, 1)
        ax.set_title(f"{nm}: EN fraction vs age (marker-based)")
        ax.set_xlabel("age (years)"); ax.set_ylabel("EN fraction of (EN+IN+Glia)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(L.OUT_DIR / "s03A_EN_fraction_vs_age.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    for nm, g in comp.items():
        g.to_csv(L.OUT_DIR / f"s03A_composition_{nm}.csv", index=False)
        # young-vs-old EN fraction summary
        en = g[g["broad"] == "EN"].copy()
        en["agebin"] = pd.cut(en["age_years"], [-1, 2, 5, 10, 20, 100],
                              labels=["<2", "2-5", "5-10", "10-20", "20+"])
        print(f"\n{nm}: EN fraction by age bin (marker-based)")
        print(en.groupby(["cohort", "agebin"], observed=True)["frac"]
              .agg(["median", "min", "max", "size"]).round(3).to_string())

    # (B) markers in labeled-EN pseudobulks
    mk = en_marker_table()
    mk = mk[mk["n_cells"] >= 20]
    mk["agebin"] = pd.cut(mk["age_years"], [-1, 5, 10, 20, 100], labels=["<5", "5-10", "10-20", "20+"])
    en_cols = [c for c in mk.columns if "|" in str(c) and c.endswith("|EN")]
    cont_cols = [c for c in mk.columns if "|" in str(c)
                 and c.split("|")[1] in {"IN", "Astro", "Oligo", "OPC", "Micro"}]
    mk["EN_marker_mean"] = mk[en_cols].mean(1)
    mk["contam_marker_mean"] = mk[cont_cols].mean(1)
    mk["EN_minus_contam"] = mk["EN_marker_mean"] - mk["contam_marker_mean"]
    print("\n(B) labeled-EN pseudobulk marker purity (log1p CPM), by cohort x agebin:")
    print(mk.groupby(["cohort", "agebin"], observed=True)[
        ["EN_marker_mean", "contam_marker_mean", "EN_minus_contam"]].median().round(2).to_string())
    mk.to_csv(L.OUT_DIR / "s03B_en_subtype_markers.csv")

    # figure: EN vs contaminant markers by age, per cohort
    cohs = [c for c in mk["cohort"].unique() if isinstance(c, str)]
    fig, axes = plt.subplots(1, len(cohs), figsize=(5 * len(cohs), 4.5), squeeze=False)
    for ax, coh in zip(axes[0], cohs):
        s = mk[mk["cohort"] == coh]
        ax.scatter(s["age_years"], s["EN_marker_mean"], s=18, alpha=.6, c="C0", label="EN markers")
        ax.scatter(s["age_years"], s["contam_marker_mean"], s=18, alpha=.6, c="C3", label="IN+glia markers")
        ax.set_title(f"{coh}: labeled-EN purity"); ax.set_xlabel("age"); ax.set_ylabel("log1p CPM")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(L.OUT_DIR / "s03B_en_marker_purity.png", dpi=140, bbox_inches="tight")
    print(f"\nOutputs -> {L.OUT_DIR}")


if __name__ == "__main__":
    main()
