#!/usr/bin/env python3
"""
Y3 — two design checks before rebuilding the ExN embedding.

CONCERN 1 (ExN definition we believe in): our cluster-based ExN call reproduced
the NATIVE labels, which Appendix A.1 rejected for young PsychAD (~5-6% ExN in
infants; 4-11x ExN-marker deficit). The project's believed definition is the
permissive MARKER RULE (annotation_by_markers: GAD>=10->InN; else RBFOX3>=1 /
DCX>=1 / RBFOX1>=1 -> ExN). Compare cluster-ExN vs marker-rule-ExN by
source x age, and inspect the young-PsychAD cells the cluster call drops:
do they actually carry positive ExN markers (i.e. real ExN we're losing)?

CONCERN 2 (is Velmeshev-V3 == Herring?): tabulate Velmeshev source
dataset x chemistry, and whether V3 cells carry obs['dataset']=='Herring' /
'Herring_' barcode prefix.

Run:
  cd /home/rajd2/rds/hpc-work/snRNAseq_2026
  sbatch --time=00:40:00 --mem=200G --cpus-per-task=8 \
     scripts/run_script.sh scripts/grn_dev_diagnostics/y3_diagnose.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

sys.path.insert(0, "code")
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import read_data

OUT = Path("scripts/grn_dev_diagnostics/outputs")
STAGE1 = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelPsychAD_V3_allcell_dev30/scvi_output/integrated.h5ad"

MARK = ["GAD1", "GAD2", "SLC32A1", "RBFOX3", "DCX", "RBFOX1",
        "AQP4", "GFAP", "MBP", "PLP1", "CX3CR1", "P2RY12", "PDGFRA",
        "SLC17A7", "SATB2", "NEUROD2", "NEUROD6"]
AGE_BINS = [(-1, 0), (0, 1), (1, 10), (10, 18), (18, 31)]
AGE_LABS = ["prenatal", "0-1", "1-10", "10-18", "18-30"]


def agebin(a):
    for (lo, hi), lab in zip(AGE_BINS, AGE_LABS):
        if lo <= a < hi:
            return lab
    return "NA"


def col(adata, sym, gene_pos):
    """Dense 1D raw-count vector for a gene symbol (zeros if absent)."""
    if sym not in gene_pos:
        return np.zeros(adata.n_obs)
    return np.asarray(adata.X[:, gene_pos[sym]].todense()).ravel()


def concern1():
    print("=" * 64 + "\nCONCERN 1 — cluster-ExN vs marker-rule-ExN\n" + "=" * 64)
    a = ad.read_h5ad(STAGE1, backed="r")
    sym = a.var["gene_symbol"].astype(str)
    gene_pos = {}
    for pos, s in enumerate(sym.values):
        gene_pos.setdefault(s, pos)
    present = {m: (m in gene_pos) for m in MARK}
    print("marker in HVG:", {m: present[m] for m in MARK})

    # pull marker count columns into memory
    cols_needed = [gene_pos[m] for m in MARK if m in gene_pos]
    names_needed = [m for m in MARK if m in gene_pos]
    sub = a[:, cols_needed].to_memory()
    C = {m: np.asarray(sp.csr_matrix(sub.X)[:, i].todense()).ravel()
         for i, m in enumerate(names_needed)}
    g = lambda m: C.get(m, np.zeros(a.n_obs))

    inn = np.maximum.reduce([g("GAD1"), g("GAD2"), g("SLC32A1")]) >= 10
    mature = (~inn) & (g("RBFOX3") >= 1)
    immature = (~inn) & (~(g("RBFOX3") >= 1)) & (g("DCX") >= 1)
    glia_det = np.maximum.reduce([g(x) for x in ["AQP4", "GFAP", "MBP", "PLP1", "CX3CR1", "P2RY12", "PDGFRA"]]) >= 1
    glia = (~inn) & (~mature) & (~immature) & glia_det
    weak = (~inn) & (~mature) & (~immature) & (~glia) & (g("RBFOX1") >= 1)
    marker_exn = mature | immature | weak

    obs = a.obs[["source", "age_years", "cell_class"]].copy()
    obs["agebin"] = obs["age_years"].map(agebin)
    obs["marker_exn"] = marker_exn
    obs["marker_label"] = np.select(
        [inn, mature, immature, glia, weak],
        ["InN", "ExN_mature", "ExN_immature", "glia", "ExN_weak"], default="Unknown")
    exn_ids = set(pd.read_parquet(OUT / "y1_exn_cellids_ALL.parquet").index.astype(str))
    obs["cluster_exn"] = obs.index.astype(str).isin(exn_ids)
    obs["native_exn"] = obs["cell_class"] == "Excitatory"

    print("\n--- ExN fraction by source x agebin: native | cluster | marker-rule ---")
    tab = obs.groupby(["source", "agebin"]).agg(
        n=("marker_exn", "size"),
        native=("native_exn", "mean"),
        cluster=("cluster_exn", "mean"),
        marker=("marker_exn", "mean")).round(3)
    print(tab.to_string())
    tab.to_csv(OUT / "y3_exn_fraction_3defs.csv")

    print("\n--- marker_label breakdown for PsychAD by agebin (%) ---")
    pa = obs[obs.source == "PSYCHAD"]
    br = (pa.groupby("agebin")["marker_label"].value_counts(normalize=True)
          .unstack().fillna(0) * 100).round(1)
    print(br.to_string())

    # the cells the cluster call DROPS but the marker rule keeps (PsychAD <10y)
    young = obs[(obs.source == "PSYCHAD") & (obs.agebin.isin(["0-1", "1-10"]))]
    gained = young[young.marker_exn & ~young.cluster_exn]
    print(f"\nPsychAD <10y: cluster-ExN={young.cluster_exn.sum():,}  "
          f"marker-ExN={young.marker_exn.sum():,}  "
          f"marker-only (gained)={len(gained):,}")
    print("  gained cells marker_label:", gained["marker_label"].value_counts().to_dict())
    print("  gained cells native cell_class:", gained["cell_class"].value_counts().to_dict())
    # do the gained cells carry positive ExN markers?
    gi = obs.index.isin(gained.index)
    for m in ["RBFOX3", "DCX", "RBFOX1", "SLC17A7", "SATB2", "GAD1"]:
        v = g(m)[gi]
        print(f"  gained: {m} detect-rate(>=1)={100*np.mean(v>=1):.0f}%  mean={v.mean():.2f}")

    # write the marker-rule ExN id set (union of mature/immature/weak), per source
    obs["is_marker_exn"] = marker_exn
    for s in ["VELMESHEV", "PSYCHAD"]:
        ids = obs.index[(obs.source == s) & marker_exn]
        pd.DataFrame(index=pd.Index(ids, name="cell_id")).to_parquet(
            OUT / f"y3_markerexn_cellids_{s}.parquet")
        print(f"  wrote y3_markerexn_cellids_{s}.parquet: {len(ids):,} cells")
    # union with cluster-ExN too (recall-maximising option)
    union = obs.index[(marker_exn) | (obs["cluster_exn"])]
    for s in ["VELMESHEV", "PSYCHAD"]:
        ids = obs.index[((obs.source == s)) & ((marker_exn) | (obs["cluster_exn"].values))]
        pd.DataFrame(index=pd.Index(ids, name="cell_id")).to_parquet(
            OUT / f"y3_unionexn_cellids_{s}.parquet")


def concern2():
    print("\n" + "=" * 64 + "\nCONCERN 2 — is Velmeshev-V3 == Herring?\n" + "=" * 64)
    _, vmeta = read_data.read_velmeshev_backed(cell_type_field="Cell_Type")
    vmeta = vmeta.copy()
    vmeta["chemistry"] = vmeta["chemistry"].astype(str)
    vmeta["dataset"] = vmeta["dataset"].astype(str)
    print("\n--- dataset x chemistry (cell counts) ---")
    ct = pd.crosstab(vmeta["dataset"], vmeta["chemistry"])
    print(ct.to_string())
    ct.to_csv(OUT / "y3_vel_dataset_by_chem.csv")
    print("\n--- within each chemistry, dataset composition (%) ---")
    print((pd.crosstab(vmeta["chemistry"], vmeta["dataset"], normalize="index") * 100).round(1).to_string())
    # barcode prefix check
    v3 = vmeta[vmeta["chemistry"].str.contains("V3", na=False)]
    herr_prefix = v3.index.astype(str).str.contains("Herring").mean()
    print(f"\nV3 cells: {len(v3):,}; fraction with 'Herring' in barcode = {100*herr_prefix:.1f}%")
    print(f"V3 dataset value_counts: {v3['dataset'].value_counts().to_dict()}")
    v2 = vmeta[vmeta["chemistry"].str.contains("V2", na=False)]
    print(f"V2 dataset value_counts: {v2['dataset'].value_counts().to_dict()}")


def main():
    concern1()
    concern2()
    print("\nDONE")


if __name__ == "__main__":
    main()
