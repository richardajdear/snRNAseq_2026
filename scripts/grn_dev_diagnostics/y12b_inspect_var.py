#!/usr/bin/env python3
"""Inspect good-run var via h5py: columns + whether marker symbols are present anywhere."""
import h5py, numpy as np
GOOD = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc/Cam_snRNAseq/integrated/VelWangPsychAD_200k_prepost_V3only_tuning5/scvi_output/integrated.h5ad"
MARK = ["NEUROD2", "GAD1", "SLC17A7", "RBFOX3", "AQP4", "MBP"]


def col(g, name):
    node = g[name]
    if isinstance(node, h5py.Group) and "categories" in node:
        cats = [c.decode() if isinstance(c, bytes) else c for c in node["categories"][:]]
        codes = node["codes"][:]; cats = np.array(cats)
        return np.where(codes >= 0, cats[np.clip(codes, 0, len(cats) - 1)], "NA").astype(str)
    arr = node[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr]) if arr.dtype.kind in ("S", "O") else arr.astype(str)


with h5py.File(GOOD, "r") as f:
    v = f["var"]
    cols = list(v.keys())
    print("var columns:", cols)
    print("var index (_index) examples:", col(v, "_index")[:5].tolist())
    for c in cols:
        try:
            vals = col(v, c)
            ex = vals[:5].tolist()
            hits = sum(int(m in set(vals.tolist())) for m in MARK)
            print(f"  {c}: dtype-ok n={len(vals)} examples={ex} | marker-hits={hits}/{len(MARK)}")
        except Exception as e:
            print(f"  {c}: ERR {e}")
print("DONE")
