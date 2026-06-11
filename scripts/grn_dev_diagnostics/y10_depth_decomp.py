#!/usr/bin/env python3
"""
Y10 — what IS the +0.85 per-cell rho(module, C3+) in postnatal (1-25y) cells?

Decompose the correlation into three additive sources and test the user's
question (depth artefact vs real biology):

  (A) TECHNICAL / depth: do both scores ride on log total counts & n_genes?
      -> r(c3,depth), r(module,depth); partial r(c3,module | depth, n_genes)
  (B) WITHIN-DONOR cell-state: after centering each score within its donor,
      does the c3<->module coupling survive (and survive depth control)?
  (C) BETWEEN-DONOR / AGE: donor-pseudobulk r(c3,module), and does either
      track AGE across donors? (this is the developmental component)

Also: per-module-gene r(c3, gene) to see if 1-2 genes carry it.

Run on the GOOD embedding (the one we trust most so far).
  sbatch --time=01:00:00 --mem=200G --cpus-per-task=8 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y10_depth_decomp.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, scipy.stats as stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR, build_c3plus_table

GOOD = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_V3only_tuning5/scvi_output/integrated.h5ad"
MODULE_ENS = ["ENSG00000171532", "ENSG00000127152", "ENSG00000119042",
              "ENSG00000081189", "ENSG00000104722", "ENSG00000100285",
              "ENSG00000067715", "ENSG00000132639", "ENSG00000078018"]
COHORTS = {"PsychAD-V3": "PSYCHAD-V3", "Velmeshev-V3": "VELMESHEV-V3"}


def rankz(x):
    r = stats.rankdata(x); return (r - r.mean()) / (r.std() + 1e-12)


def partial(x, y, ctrls):
    """Partial correlation of x,y controlling for columns of ctrls (rank-based)."""
    X = np.column_stack([rankz(c) for c in ctrls]); X = np.column_stack([np.ones(len(x)), X])
    rx = rankz(x) - X @ np.linalg.lstsq(X, rankz(x), rcond=None)[0]
    ry = rankz(y) - X @ np.linalg.lstsq(X, rankz(y), rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def sr(x, y):
    return float(stats.spearmanr(x, y).correlation)


def main():
    a = ad.read_h5ad(GOOD, backed="r")
    obs = a.obs
    bk = "source-chemistry" if "source-chemistry" in obs else "source"
    batch = obs[bk].astype(str).values
    age = pd.to_numeric(obs["age_years"], errors="coerce").values
    don = obs["individual"].astype(str).values
    counts = sp.csr_matrix(a.X[:])
    tot = np.asarray(counts.sum(1)).ravel(); inv = 1.0 / np.where(tot > 0, tot, 1)
    ngene = np.asarray((counts > 0).sum(1)).ravel()
    var = pd.Index(a.var_names.astype(str))
    c3w = build_c3plus_table().set_index("ensembl_id")["weight"]
    hit = var.intersection(c3w.index)
    c3 = np.asarray(counts[:, [var.get_loc(g) for g in hit]].multiply(inv[:, None])
                    .dot(c3w.reindex(hit).values)).ravel() * 1e6
    midx = [var.get_loc(g) for g in MODULE_ENS if g in var]
    gene_cp = np.asarray(np.log1p(counts[:, midx].multiply(inv[:, None]) * 1e4).todense())
    module = gene_cp.mean(1)
    logtot = np.log10(np.clip(tot, 1, None)); logng = np.log10(np.clip(ngene, 1, None))

    rows = []
    fig, axes = plt.subplots(2, len(COHORTS), figsize=(7 * len(COHORTS), 11))
    for j, (lab, tag) in enumerate(COHORTS.items()):
        m = (batch == tag) & (age >= 1) & (age < 25) & np.isfinite(age)
        c3m, mom, ltm, lnm, agm, dm = c3[m], module[m], logtot[m], logng[m], age[m], don[m]
        raw = sr(c3m, mom)
        r_c3_dep, r_mo_dep = sr(c3m, ltm), sr(mom, ltm)
        r_c3_ng, r_mo_ng = sr(c3m, lnm), sr(mom, lnm)
        p_dep = partial(c3m, mom, [ltm])
        p_depng = partial(c3m, mom, [ltm, lnm])
        # within-donor centering
        df = pd.DataFrame({"c3": c3m, "mo": mom, "lt": ltm, "ln": lnm, "d": dm})
        cen = df.groupby("d").transform(lambda v: v - v.mean())
        r_within = sr(cen["c3"], cen["mo"])
        p_within = partial(cen["c3"].values, cen["mo"].values, [cen["lt"].values, cen["ln"].values])
        # donor pseudobulk
        pb = df.assign(age=agm).groupby("d").mean(numeric_only=True)
        r_pb = sr(pb["c3"], pb["mo"])
        r_c3_age = sr(pb["c3"], pb["age"]); r_mo_age = sr(pb["mo"], pb["age"])
        p_pb_dep = partial(pb["c3"].values, pb["mo"].values, [pb["lt"].values]) if len(pb) > 5 else np.nan
        rows.append(dict(cohort=lab, n=int(m.sum()), n_donor=pb.shape[0],
                         raw_r_c3_module=round(raw, 3),
                         r_c3_depth=round(r_c3_dep, 3), r_module_depth=round(r_mo_dep, 3),
                         r_c3_ngene=round(r_c3_ng, 3), r_module_ngene=round(r_mo_ng, 3),
                         partial_depth=round(p_dep, 3), partial_depth_ngene=round(p_depng, 3),
                         within_donor_r=round(r_within, 3), within_donor_partial_depth=round(p_within, 3),
                         pseudobulk_r=round(r_pb, 3), pseudobulk_partial_depth=round(p_pb_dep, 3),
                         pb_r_c3_age=round(r_c3_age, 3), pb_r_module_age=round(r_mo_age, 3)))
        # per-module-gene
        for gi, g in enumerate([g for g in MODULE_ENS if g in var]):
            rows.append(dict(cohort=lab, n=int(m.sum()), gene=g,
                             raw_r_c3_module=round(sr(c3m, gene_cp[m, gi]), 3)))
        # plots: (top) per-cell c3 vs module colored by depth; (bottom) donor pb
        s = np.random.default_rng(0).choice(m.sum(), min(20000, m.sum()), replace=False)
        sc0 = axes[0, j].scatter(c3m[s], mom[s], c=ltm[s], s=3, cmap="magma", alpha=0.4, linewidths=0)
        axes[0, j].set_title(f"{lab}: per-cell raw r={raw:.2f} | partial(depth+ngene)={p_depng:.2f}", fontsize=10)
        axes[0, j].set_xlabel("C3+ (CPM-wt)"); axes[0, j].set_ylabel("maturity module (logCP10K)")
        plt.colorbar(sc0, ax=axes[0, j], label="log10 total counts", fraction=0.046)
        axes[1, j].scatter(pb["c3"], pb["mo"], c=pb["age"], s=40, cmap="viridis", edgecolors="k", linewidths=0.4)
        axes[1, j].set_title(f"{lab}: donor pseudobulk r={r_pb:.2f} | r(c3,age)={r_c3_age:.2f} r(mod,age)={r_mo_age:.2f}", fontsize=10)
        axes[1, j].set_xlabel("donor-mean C3+"); axes[1, j].set_ylabel("donor-mean module")
    fig.suptitle("Y10: decomposing the per-cell module<->C3+ correlation (good embedding, 1-25y)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "y10_depth_decomp.png", dpi=150, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(OUT_DIR / "y10_depth_decomp.csv", index=False)
    print(pd.DataFrame([r for r in rows if "gene" not in r]).to_string(index=False))
    print("DONE")


if __name__ == "__main__":
    main()
