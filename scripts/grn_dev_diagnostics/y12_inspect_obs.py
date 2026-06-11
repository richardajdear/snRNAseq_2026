#!/usr/bin/env python3
"""Y12 — inspect good-run obs via h5py (no X load): columns + native labels per source."""
import h5py, numpy as np, pandas as pd
GOOD = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_V3only_tuning5/scvi_output/integrated.h5ad"


def col(o, name):
    """Read an obs column (categorical or plain) -> np array of strings/values."""
    node = o[name]
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        cats = node["categories"][:]
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
        codes = node["codes"][:]
        out = np.where(codes >= 0, cats[np.clip(codes, 0, len(cats) - 1)], "NA")
        return out
    arr = node[:]
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
    return arr


with h5py.File(GOOD, "r") as f:
    o = f["obs"]
    cols = [k for k in o.keys() if k != "__categories"]
    print("obs columns:", cols)
    src = col(o, "source").astype(str)
    print("\nsources:", dict(pd.Series(src).value_counts()))
    for bk in ["source-chemistry", "source_chemistry"]:
        if bk in o:
            print(f"{bk}:", dict(pd.Series(col(o, bk).astype(str)).value_counts())); break
    cand = [c for c in cols if any(k in c.lower() for k in
            ["class", "type", "subclass", "label", "annot", "lineage", "celltype", "cluster"])]
    print("\ncandidate label columns:", cand)
    for c in cand:
        vals = col(o, c).astype(str)
        print(f"\n=== {c} ===")
        for s in np.unique(src):
            vc = pd.Series(vals[src == s]).value_counts()
            print(f"  [{s}] {dict(list(vc.items())[:30])}")
print("DONE")
