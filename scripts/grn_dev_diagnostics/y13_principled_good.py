#!/usr/bin/env python3
"""
Y13 — recompute the PRINCIPLED ExN definition on the GOOD baseline embedding
(VelWangPsychAD_200k_prepost_V3only_tuning5) and check it is believable.

The principled definition = gene-threshold ANCHORS (embedding-independent) +
a kNN lineage-vote on X_scVI for ambiguous immature neurons (embedding-dependent).
It was only ever computed on the dev30 (under-mixed) embedding. The user's worry:
on that bad embedding, non-neuronal cells (oligos etc.) may have been voted ExN.

This re-runs it on the good embedding and outputs:
  - confusion of principled_exn vs NATIVE cell_class, per source (the key
    "are oligos/glia being called ExN?" table)
  - 4-panel UMAP: native cell_class | anchors+ambiguous | principled_exn |
    principled-ExN cells recoloured by native cell_class (believability)
  - principled ExN id sets on the good embedding (for downstream rebuilds)

Run:
  sbatch --time=02:00:00 --mem=240G --cpus-per-task=32 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y13_principled_good.py
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scanpy as sc, scipy.sparse as sp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
from _lib import OUT_DIR as OUT

GOOD = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_V3only_tuning5/scvi_output/integrated.h5ad"
EXC_TF = ["NEUROD2", "NEUROD6", "TBR1", "EOMES", "SLC17A7", "SLC17A6", "SATB2", "BCL11B", "FEZF2"]
INH_TF = ["GAD1", "GAD2", "SLC32A1", "DLX1", "DLX2", "DLX5", "LHX6", "ADARB2", "SP8"]
GLIA = ["AQP4", "GFAP", "MBP", "PLP1", "CX3CR1", "P2RY12", "PDGFRA"]
NEUR = ["RBFOX3", "DCX", "RBFOX1"]
ALLG = sorted(set(EXC_TF + INH_TF + GLIA + NEUR))
KNN = 30


def resolve_symbols(var):
    # good-run gene_symbol stores collision-disambiguated names, e.g.
    # 'SAMD11_ENSG00000187634'; strip the trailing _ENSG\d+ to recover 'SAMD11'.
    for c in ["gene_symbol", "feature_name", "gene_name", "symbol"]:
        if c in var.columns:
            clean = np.array([re.sub(r"_ENSG\d+$", "", s) for s in var[c].astype(str).values])
            return clean, c
    return var.index.astype(str).values, "var_names"


def main():
    a = sc.read_h5ad(GOOD, backed="r")
    sym, symcol = resolve_symbols(a.var)
    print(f"using symbol column: {symcol}")
    pos = {}
    for i, s in enumerate(sym):
        pos.setdefault(s, i)
    have = [g for g in ALLG if g in pos]
    print("markers present:", {g: (g in pos) for g in ALLG})
    n = a.n_obs
    Z = np.asarray(a.obsm["X_scVI"][:])
    src = a.obs["source"].astype(str).values
    native = a.obs["cell_class"].astype(str).values
    age = pd.to_numeric(a.obs["age_years"], errors="coerce").values
    sub = a[:, [pos[g] for g in have]].to_memory()
    Xs = sp.csr_matrix(sub.X)
    raw = {g: np.asarray(Xs[:, i].todense()).ravel() for i, g in enumerate(have)}
    print("library size (chunked) ...")
    ls = np.zeros(n); step = 100000
    for s0 in range(0, n, step):
        ls[s0:s0 + step] = np.asarray(a.X[s0:s0 + step].sum(1)).ravel()
    inv = 1.0 / np.where(ls > 0, ls, 1)

    def score(genes):
        gs = [g for g in genes if g in raw]
        return np.vstack([np.log1p(raw[g] * inv * 1e4) for g in gs]).T.mean(1)
    exc = score(EXC_TF); inh = score(INH_TF)
    gad_max = np.maximum.reduce([raw.get(g, np.zeros(n)) for g in ["GAD1", "GAD2", "SLC32A1"]])
    slc17 = np.maximum.reduce([raw.get(g, np.zeros(n)) for g in ["SLC17A7", "SLC17A6"]])
    glia_det = np.maximum.reduce([raw.get(g, np.zeros(n)) for g in GLIA]) >= 1
    is_neuron = (raw.get("RBFOX3", np.zeros(n)) >= 1) | (raw.get("DCX", np.zeros(n)) >= 1)
    conf_inn = is_neuron & (gad_max >= 10)
    conf_exn = is_neuron & (gad_max < 10) & (slc17 >= 1)
    glia = (~is_neuron) & glia_det
    ambiguous = is_neuron & (~conf_inn) & (~conf_exn)
    print(f"anchors: conf_ExN={conf_exn.sum():,} conf_InN={conf_inn.sum():,} "
          f"glia={glia.sum():,} ambiguous={ambiguous.sum():,}")

    print("neighbors on good X_scVI ...")
    g = ad.AnnData(X=sp.csr_matrix((n, 1))); g.obsm["X_scVI"] = Z
    sc.pp.neighbors(g, use_rep="X_scVI", n_neighbors=KNN)
    G = (g.obsp["distances"] > 0).astype(np.float32)
    ie = G.dot(conf_exn.astype(np.float32)); ii = G.dot(conf_inn.astype(np.float32))
    denom = ie + ii
    exn_nb_frac = np.divide(ie, denom, out=np.full(n, np.nan), where=denom > 0)
    vote_exn = exn_nb_frac >= 0.5
    principled_exn = conf_exn | (ambiguous & vote_exn)
    grp = np.select([conf_exn, conf_inn, glia, ambiguous],
                    ["conf_ExN", "conf_InN", "glia", "ambiguous"], default="other")
    print(f"principled ExN total: {principled_exn.sum():,}")

    # ---- believability: confusion of principled_exn vs native cell_class ----
    df = pd.DataFrame({"source": src, "native": native, "pexn": principled_exn,
                       "age": age, "grp": grp})
    rows = []
    for s in ["PSYCHAD", "VELMESHEV", "WANG"]:
        d = df[df.source == s]
        ct = pd.crosstab(d["native"], d["pexn"])
        ct.to_csv(OUT / f"y13_confusion_{s}.csv")
        print(f"\n=== {s}: native cell_class x principled_exn ===\n{ct.to_string()}")
        pe = d[d.pexn]
        comp = pe["native"].value_counts(normalize=True).round(3)
        print(f"  principled-ExN composition by native label: {dict(comp)}")
        # of native non-Excitatory, how many got called ExN?
        nonexc = d[d.native != "Excitatory"]
        leak = nonexc["pexn"].mean()
        nat_exc = d[d.native == "Excitatory"]
        recall = nat_exc["pexn"].mean()
        rows.append(dict(source=s, n=len(d), principled_exn=int(d.pexn.sum()),
                         native_exc=int((d.native == "Excitatory").sum()),
                         recall_of_native_exc=round(recall, 3),
                         leak_from_non_exc=round(leak, 3),
                         frac_pexn_that_is_native_exc=round((pe.native == "Excitatory").mean(), 3)))
    pd.DataFrame(rows).to_csv(OUT / "y13_believability.csv", index=False)
    print("\nbelievability summary:\n", pd.DataFrame(rows).to_string(index=False))

    # write principled ids on the good embedding
    for s in ["VELMESHEV", "PSYCHAD", "WANG"]:
        ids = a.obs_names[(src == s) & principled_exn]
        pd.DataFrame(index=pd.Index(ids, name="cell_id")).to_parquet(
            OUT / f"y13_principledexn_good_{s}.parquet")
        print(f"  wrote y13_principledexn_good_{s}.parquet: {len(ids):,}")

    # ---- UMAP (subsample) ----
    rng = np.random.default_rng(0)
    si = rng.choice(n, min(120000, n), replace=False)
    sa = ad.AnnData(X=sp.csr_matrix((len(si), 1))); sa.obsm["X_scVI"] = Z[si]
    sc.pp.neighbors(sa, use_rep="X_scVI", n_neighbors=15); sc.tl.umap(sa)
    U = sa.obsm["X_umap"]
    fig, ax = plt.subplots(1, 4, figsize=(26, 6.5))

    def cat(axx, lab, title, palette=None):
        s_ = pd.Series(lab); u = sorted(s_.unique())
        cm = palette or {c: plt.cm.tab20(i % 20) for i, c in enumerate(u)}
        for c in u:
            m = (s_ == c).values
            axx.scatter(U[m, 0], U[m, 1], s=2, color=cm[c], alpha=0.5,
                        label=f"{c} ({m.sum()})", linewidths=0)
        axx.legend(markerscale=4, fontsize=6); axx.set_title(title, fontsize=10)
        axx.set_xticks([]); axx.set_yticks([])

    cat(ax[0], native[si], "native cell_class")
    grpcol = {"conf_ExN": "#C0392B", "conf_InN": "#2471A3", "glia": "#BDC3C7",
              "ambiguous": "#F39C12", "other": "#EAECEE"}
    cat(ax[1], grp[si], "anchors + ambiguous", grpcol)
    cat(ax[2], np.where(principled_exn[si], "principled_ExN", "not")[:],
        "principled ExN assignment", {"principled_ExN": "#C0392B", "not": "#D5DBDB"})
    # panel 4: only the principled-ExN cells, coloured by their native label
    pe_mask = principled_exn[si]
    sub_native = native[si][pe_mask]
    u = sorted(pd.unique(sub_native))
    cm = {c: plt.cm.tab20(i % 20) for i, c in enumerate(u)}
    for c in u:
        m = sub_native == c
        ax[3].scatter(U[pe_mask][m, 0], U[pe_mask][m, 1], s=2, color=cm[c],
                      alpha=0.6, label=f"{c} ({m.sum()})", linewidths=0)
    ax[3].legend(markerscale=4, fontsize=6)
    ax[3].set_title("principled-ExN cells, coloured by NATIVE label\n(non-Excitatory here = leakage)", fontsize=10)
    ax[3].set_xticks([]); ax[3].set_yticks([])
    fig.suptitle("Y13 — principled ExN on the GOOD embedding: is the re-assignment believable?",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "y13_principled_good_umap.png", dpi=150, bbox_inches="tight")
    print("\nsaved y13_principled_good_umap.png + confusion/believability csvs + id parquets")
    print("DONE")


if __name__ == "__main__":
    main()
