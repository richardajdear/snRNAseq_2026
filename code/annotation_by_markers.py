#!/usr/bin/env python3
"""
Marker-based cell type annotation for snRNA-seq h5ad files.

Works with HBCC_Cohort.h5ad and velmeshev.h5ad (auto-detects differences).

Public API
----------
annotate(h5ad_path, age_max=None, inn_threshold=INN_THRESHOLD) -> pd.Series
    Returns cell-type labels indexed by obs_names.

classify_cell(counts: dict, inn_threshold=INN_THRESHOLD) -> str
    Pure function; assign a label given {gene_name: raw_count}.

summarise(labels, dev_stages, donor_ids) -> pd.DataFrame
    Per-donor breakdown table.

CLASSIFICATION LOGIC
--------------------
Step 1  — InN  max(GAD1, GAD2, SLC32A1) >= inn_threshold (default 10).
          Elevated threshold reduces ambient-RNA false positives.
Step 2  — ExN  RBFOX3 or DCX >= 1 (neuron-exclusive, override any glial signal).
          DCX-only → ExN_immature; RBFOX3 → ExN_mature.
Step 3  — Glia  (only when no neuronal marker detected)
          Astro > Oligo > Micro > OPC.
Step 4  — ExN_weak  RBFOX1 >= 1, no glia markers.
Step 5  — Unknown.

DATASET DIFFERENCES (handled by _H5Dataset)
--------------------------------------------
  HBCC:      raw counts in X;     gene names in var/gene_name
  Velmeshev: raw counts in raw/X; gene names in var/feature_name
  Age:       HBCC  "N-month-old stage" / "N-year-old stage"
             Velm  "N-month-old human stage" / "N-year-old human stage"
                   / "newborn human stage"
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INN_THRESHOLD = 10

MARKERS = {
    "NeuronStrong": ["RBFOX3", "DCX"],
    "NeuronWeak":   ["RBFOX1"],
    "InN":          ["GAD1", "GAD2", "SLC32A1"],
    "Astro":        ["AQP4", "GFAP"],
    "Oligo":        ["MBP", "PLP1"],
    "OPC":          ["PDGFRA"],
    "Micro":        ["CX3CR1", "P2RY12"],
}
ALL_MARKER_GENES = [g for genes in MARKERS.values() for g in genes]

CLASS_ORDER = [
    "ExN_mature", "ExN_immature", "ExN_weak",
    "InN",
    "Astro", "Oligo", "OPC", "Micro",
    "Unknown",
]

_AGE_PATTERNS = [
    (re.compile(r"^(\d+)-month-old stage$"),          lambda m: int(m.group(1)) / 12.0),
    (re.compile(r"^(\d+)-year-old stage$"),            lambda m: float(m.group(1))),
    (re.compile(r"^(\d+)-month-old human stage$"),     lambda m: int(m.group(1)) / 12.0),
    (re.compile(r"^(\d+)-year-old human stage$"),      lambda m: float(m.group(1))),
    (re.compile(r"^newborn human stage$"),              lambda _: 0.0),
    (re.compile(r"fetal|embryo|trimester", re.I),      lambda _: None),
]


def parse_age_years(stage: str) -> Optional[float]:
    """Return age in years, 0.0 for newborn, None for fetal/unknown."""
    for pat, fn in _AGE_PATTERNS:
        m = pat.match(stage)
        if m:
            return fn(m)
    return None


# ---------------------------------------------------------------------------
# H5AD adapter — no anndata/scanpy dependency
# ---------------------------------------------------------------------------

class _H5Dataset:
    """Thin wrapper that hides HBCC vs Velmeshev structural differences."""

    def __init__(self, h5_file: h5py.File):
        self._f = h5_file
        self._detect()

    def _detect(self):
        f = self._f
        # Velmeshev stores raw integer counts under raw/X
        if "raw" in f and "X" in f["raw"]:
            self._x_grp = f["raw"]["X"]
            self._var_grp = f["raw"]["var"] if "var" in f["raw"] else f["var"]
        else:
            self._x_grp = f["X"]
            self._var_grp = f["var"]

        # Gene name column varies by dataset
        for col in ("gene_name", "feature_name", "_index"):
            if col in self._var_grp:
                self._gene_col = col
                break
        else:
            raise KeyError("Cannot find gene name column in var group")

    def _read_col(self, group, key):
        item = group[key]
        if isinstance(item, h5py.Group):
            codes = item["codes"][:]
            cats = [c.decode() if isinstance(c, bytes) else c
                    for c in item["categories"][:]]
            return [cats[c] for c in codes]
        data = item[:]
        if data.dtype.kind == "S":
            return [d.decode() for d in data]
        # variable-length strings (h5py vlen dtype, kind=="O") also come back as bytes
        if data.dtype.kind == "O":
            return [d.decode() if isinstance(d, bytes) else str(d) for d in data]
        return list(data)

    def obs_names(self):
        # anndata stores obs_names under the key named by obs.attrs['_index']
        # (e.g. 'barcodekey' for PsychAD, '_index' for Velmeshev)
        key = self._f["obs"].attrs.get("_index", "_index")
        return self._read_col(self._f["obs"], key)

    def read_obs_col(self, key):
        return self._read_col(self._f["obs"], key)

    def gene_names(self):
        return self._read_col(self._var_grp, self._gene_col)

    @property
    def indptr(self):
        return self._x_grp["indptr"][:]

    @property
    def indices_dset(self):
        return self._x_grp["indices"]

    @property
    def data_dset(self):
        return self._x_grp["data"]

    def read_cell_block(self, sorted_cell_idx: np.ndarray,
                        target_col_idx: np.ndarray) -> np.ndarray:
        """
        Read raw counts for target genes from a contiguous cell block.

        sorted_cell_idx : 1-D int array of global cell row indices, must be sorted.
        target_col_idx  : 1-D int array of gene column indices.
        Returns float32 array of shape (len(sorted_cell_idx), len(target_col_idx)).
        """
        ip = self.indptr
        ci0 = int(sorted_cell_idx[0])
        ci1 = int(sorted_cell_idx[-1]) + 1
        ds = int(ip[ci0])
        de = int(ip[ci1])

        n = len(sorted_cell_idx)
        counts = np.zeros((n, len(target_col_idx)), dtype=np.float32)
        if ds >= de:
            return counts

        blk_col = self.indices_dset[ds:de]
        blk_val = self.data_dset[ds:de]
        lptr = ip[ci0:ci1 + 1] - ds

        g2l = {int(c): li for li, c in enumerate(sorted_cell_idx)}
        for global_ci in sorted_cell_idx:
            li = g2l[int(global_ci)]
            s = int(lptr[global_ci - ci0])
            e = int(lptr[global_ci - ci0 + 1])
            if s >= e:
                continue
            row_cols = blk_col[s:e]
            row_vals = blk_val[s:e]
            for j, tc in enumerate(target_col_idx):
                hit = row_cols == tc
                if hit.any():
                    counts[li, j] = row_vals[hit].sum()

        return counts


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_cell(counts: dict, inn_threshold: int = INN_THRESHOLD) -> str:
    """
    Assign a cell-type label given {gene_name: raw_count}.

    Returns one of: ExN_mature, ExN_immature, ExN_weak, InN,
                    Astro, Oligo, OPC, Micro, Unknown.
    """
    def c(g):
        return float(counts.get(g, 0))

    rbfox3 = c("RBFOX3") >= 1
    dcx    = c("DCX")    >= 1
    rbfox1 = c("RBFOX1") >= 1

    is_inn   = max(c("GAD1"), c("GAD2"), c("SLC32A1")) >= inn_threshold
    is_astro = c("AQP4") >= 1 or c("GFAP")    >= 1
    is_oligo = c("MBP")  >= 1 or c("PLP1")    >= 1
    is_micro = c("CX3CR1") >= 1 or c("P2RY12") >= 1
    is_opc   = c("PDGFRA") >= 1

    if is_inn:
        return "InN"

    # RBFOX3 and DCX are neuron-exclusive; any co-expressed glial signal is ambient
    if rbfox3 or dcx:
        if dcx and not rbfox3:
            return "ExN_immature"
        return "ExN_mature"

    if is_astro:
        return "Astro"
    if is_oligo:
        return "Oligo"
    if is_micro:
        return "Micro"
    if is_opc:
        return "OPC"

    # RBFOX1 is also expressed in oligos → lower confidence, only if no glia
    if rbfox1:
        return "ExN_weak"

    return "Unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate(
    h5ad_path: str,
    age_max: Optional[float] = None,
    inn_threshold: int = INN_THRESHOLD,
    chunk_size: int = 5000,
) -> pd.Series:
    """
    Annotate all cells in an h5ad file using marker-based classification.

    Parameters
    ----------
    h5ad_path     : path to .h5ad file (HBCC or Velmeshev format).
    age_max       : if given, only cells from donors younger than this (years)
                    are classified; the rest receive label "skipped".
    inn_threshold : minimum GAD1/GAD2/SLC32A1 count to call a cell InN.
    chunk_size    : number of cells to process per HDF5 read.

    Returns
    -------
    pd.Series of string labels indexed by obs_names.
    """
    with h5py.File(h5ad_path, "r") as f:
        ds = _H5Dataset(f)

        obs_names  = ds.obs_names()
        dev_stages = ds.read_obs_col("development_stage")
        gene_names = ds.gene_names()

        gene_idx = {g: i for i, g in enumerate(gene_names)}
        found    = [g for g in ALL_MARKER_GENES if g in gene_idx]
        missing  = [g for g in ALL_MARKER_GENES if g not in gene_idx]
        if missing:
            print(f"WARNING: genes not in dataset: {missing}", file=sys.stderr)

        target_cols = np.array([gene_idx[g] for g in found], dtype=np.int64)
        gene_to_j   = {g: j for j, g in enumerate(found)}

        ages = np.array([
            (parse_age_years(s) if parse_age_years(s) is not None else 999.0)
            for s in dev_stages
        ])

        if age_max is not None:
            active_mask = ages < age_max
        else:
            active_mask = np.ones(len(ages), dtype=bool)

        active_idx = np.where(active_mask)[0]
        labels = np.full(len(obs_names), "skipped", dtype=object)

        if len(active_idx) == 0:
            return pd.Series(labels, index=obs_names, name="marker_annotation")

        sorted_active = np.sort(active_idx)
        n = len(sorted_active)

        for start in range(0, n, chunk_size):
            chunk = sorted_active[start:start + chunk_size]
            counts_mat = ds.read_cell_block(chunk, target_cols)
            for li, global_i in enumerate(chunk):
                cell_counts = {g: float(counts_mat[li, gene_to_j[g]]) for g in found}
                labels[global_i] = classify_cell(cell_counts, inn_threshold)

    return pd.Series(labels, index=obs_names, name="marker_annotation")


def summarise(
    labels: pd.Series,
    dev_stages: list,
    donor_ids: list,
) -> pd.DataFrame:
    """
    Per-donor composition table.

    Parameters
    ----------
    labels      : pd.Series of cell-type labels (as returned by annotate).
    dev_stages  : list of development_stage strings, same length as labels.
    donor_ids   : list of donor_id strings, same length as labels.

    Returns
    -------
    pd.DataFrame with one row per donor, sorted by age.
    """
    df = pd.DataFrame({
        "donor": donor_ids,
        "stage": dev_stages,
        "label": labels.values,
        "age_y": [
            (parse_age_years(s) if parse_age_years(s) is not None else 999.0)
            for s in dev_stages
        ],
    })
    df = df[df["label"] != "skipped"]

    rows = []
    for donor, grp in df.groupby("donor"):
        n     = len(grp)
        stage = grp["stage"].iloc[0]
        age_y = grp["age_y"].iloc[0]
        cts   = grp["label"].value_counts()

        row = {"donor": donor, "stage": stage, "age_y": age_y, "n_cells": n}
        for cls in CLASS_ORDER:
            row[f"n_{cls}"]   = int(cts.get(cls, 0))
            row[f"pct_{cls}"] = round(100.0 * cts.get(cls, 0) / n, 1)

        exn_n = sum(cts.get(c, 0) for c in ["ExN_mature", "ExN_immature", "ExN_weak"])
        row["n_ExN_all"]   = int(exn_n)
        row["pct_ExN_all"] = round(100.0 * exn_n / n, 1)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("age_y")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Marker-based cell type annotation for snRNA-seq h5ad files."
    )
    parser.add_argument(
        "--input", action="append", dest="inputs", metavar="PATH",
        help="Path to .h5ad file (repeat for multiple files; first wins on duplicate barcodes)",
    )
    parser.add_argument(
        "--config",
        help="Pipeline config YAML: source h5ad paths are read from the sources section",
    )
    parser.add_argument(
        "--age-max", type=float, default=None,
        help="Only annotate cells younger than this age in years (default: all cells)",
    )
    parser.add_argument(
        "--inn-threshold", type=int, default=INN_THRESHOLD,
        help=f"Min GAD1/GAD2/SLC32A1 count to call a cell InN (default: {INN_THRESHOLD})",
    )
    parser.add_argument(
        "--no-age-filter", action="store_true",
        help="Annotate all cells regardless of age (overrides --age-max)",
    )
    parser.add_argument(
        "--save", metavar="PATH",
        help="Save annotations as a parquet file at this path (index=obs_names, col=marker_annotation)",
    )
    args = parser.parse_args()

    age_max = None if args.no_age_filter else args.age_max

    # Build list of source h5ad paths
    h5ad_paths: list = list(args.inputs or [])
    if args.config:
        import yaml
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        for src in cfg.get("sources", []):
            if "paths" in src:
                # PsychAD: aging first → preferred in duplicate barcodes
                h5ad_paths.append(str(src["paths"]["aging"]))
                h5ad_paths.append(str(src["paths"]["hbcc"]))
            elif "path" in src:
                h5ad_paths.append(str(src["path"]))

    if not h5ad_paths:
        parser.error("Provide at least one --input path or a --config with sources")

    print(f"Source files:  {h5ad_paths}", flush=True)
    print(f"Age max:       {age_max if age_max is not None else 'all'}", flush=True)
    print(f"InN threshold: {args.inn_threshold}", flush=True)
    print("", flush=True)

    all_labels: list = []
    for h5ad_path in h5ad_paths:
        print(f"Annotating {h5ad_path} ...", flush=True)
        labels = annotate(h5ad_path, age_max=age_max, inn_threshold=args.inn_threshold)
        print(f"  {len(labels):,} cells", flush=True)
        all_labels.append(labels)

    if len(all_labels) == 1:
        labels = all_labels[0]
        h5ad_path = h5ad_paths[0]
    else:
        combined = pd.concat(all_labels)
        labels = combined[~combined.index.duplicated(keep="first")]
        h5ad_path = h5ad_paths[0]
        print(f"Combined: {len(labels):,} unique cells (deduped, first file wins)", flush=True)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        labels.to_frame(name="marker_annotation").to_parquet(args.save)
        print(f"Saved annotations → {args.save}", flush=True)

    # Read metadata for summary (from first source file)
    with h5py.File(h5ad_path, "r") as f:
        ds = _H5Dataset(f)
        dev_stages = ds.read_obs_col("development_stage")
        donor_ids  = ds.read_obs_col("donor_id")
        obs_names_src = ds.obs_names()

    # Build a dev_stage/donor_id lookup keyed by obs_name
    name_to_stage  = dict(zip(obs_names_src, dev_stages))
    name_to_donor  = dict(zip(obs_names_src, donor_ids))
    aligned_stages = [name_to_stage.get(n, "unknown") for n in labels.index]
    aligned_donors = [name_to_donor.get(n, "unknown")  for n in labels.index]

    summary = summarise(labels, aligned_stages, aligned_donors)

    sep = "-" * 110
    print(f"\n{'='*110}")
    print(f"MARKER-BASED CLASSIFICATION  —  {Path(h5ad_path).name}")
    print(f"InN threshold: >={args.inn_threshold} counts  |  all other markers: >=1 count")
    print(f"{'='*110}\n")

    pct_cols = (
        ["donor", "stage", "n_cells", "pct_ExN_all", "pct_InN"]
        + [f"pct_{c}" for c in ["Astro", "Oligo", "OPC", "Micro", "Unknown"]]
    )
    print("-- Summary (ExN combined) --")
    print(summary[pct_cols].to_string(index=False, float_format="%.1f"))

    det_cols = (
        ["donor", "stage", "n_cells"]
        + [f"pct_{c}" for c in CLASS_ORDER]
    )
    print(f"\n{sep}")
    print("-- Detailed (ExN split by maturity) --")
    print(summary[det_cols].to_string(index=False, float_format="%.1f"))

    active = labels[labels != "skipped"]
    total  = len(active)
    print(f"\n{sep}")
    print("-- Overall totals --")
    print(f"  Total annotated cells: {total:,}")
    for cls in CLASS_ORDER:
        nc  = (active == cls).sum()
        pct = 100.0 * nc / total if total else 0.0
        print(f"  {cls:15s}: {nc:6,}  ({pct:.1f}%)")
    exn_all = active.isin(["ExN_mature", "ExN_immature", "ExN_weak"]).sum()
    print(f"\n  {'ExN (all)':15s}: {exn_all:6,}  ({100.0*exn_all/total:.1f}%)")


if __name__ == "__main__":
    main()
