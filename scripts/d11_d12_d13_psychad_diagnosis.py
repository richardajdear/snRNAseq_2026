"""
D11/D12/D13 — Biological characterisation of PsychAD <1y EN gap.

Loads raw counts from per-dataset BASE FILES (not integrated.h5ad):
  Wang, Velmeshev : .raw.X  (full-transcriptome raw integer counts)
  PsychAD         : .X      (IS raw; no .raw slot)

Obs metadata (age_years, source-chemistry, individual, cell_type_aligned)
comes from integrated.h5ad, which is read in backed mode and closed after
extracting the obs table.

D11 — Per-donor EN/IN marker ratio for PsychAD <1y.
      Is the low EN/IN consistent across donors, or driven by 1–2 outliers?

D12 — Fraction of cells with detectable EN/IN marker expression (count >= 1, >= 3).
      Distinguishes genuine sparse EN composition from per-cell weak expression.

D13 — Ambient RNA contamination check.
      Do non-neuronal cells (cell_type_aligned != Excitatory/Inhibitory) in
      PsychAD <1y show elevated GAD1/GAD2 vs Wang/Vel non-neuronal?

Output: scripts/relabel_comparison/d11_d12_d13/
Run with:
  sbatch --mem=200G --time=00:45:00 scripts/run_script.sh scripts/d11_d12_d13_psychad_diagnosis.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import scanpy as sc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 1

# ─── Paths ───────────────────────────────────────────────────────────────────
RDS        = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
INTEGRATED = f"{RDS}/Cam_snRNAseq/integrated/VelWangPsychAD_semisup3_tuning5/scvi_output/integrated.h5ad"
WANG_PATH  = f"{RDS}/Cam_snRNAseq/wang/wang.h5ad"
VEL_PATH   = f"{RDS}/Cam_snRNAseq/velmeshev/velmeshev.h5ad"
HBCC_PATH  = f"{RDS}/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"
AGING_PATH = f"{RDS}/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad"

OUT = Path("/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/relabel_comparison/d11_d12_d13")
OUT.mkdir(parents=True, exist_ok=True)

# ─── Markers (ENSG IDs matching var_names in all per-dataset files) ──────────
EN_MARKERS = {
    "SLC17A7": "ENSG00000104888",
    "NEUROD2":  "ENSG00000171532",
    "SATB2":    "ENSG00000119042",
    "RBFOX3":   "ENSG00000128266",
    "TBR1":     "ENSG00000136535",
}
IN_MARKERS = {
    "GAD1":    "ENSG00000128683",
    "GAD2":    "ENSG00000136750",
    "LHX6":    "ENSG00000106688",
    "SLC32A1": "ENSG00000101438",
}
ALL_MARKERS = {**EN_MARKERS, **IN_MARKERS}

# cell_type_aligned uses EN_*/IN_* prefixes (e.g. EN_L2_3, IN_PV, IN_SST)
NEURONAL_PREFIXES = ("EN_", "IN_")

def is_neuronal(cta_val):
    return any(cta_val.startswith(p) for p in NEURONAL_PREFIXES)

print("=" * 70)
print("D11/D12/D13 — PsychAD <1y biological characterisation (raw base files)")
print(f"  Output: {OUT}")
print("=" * 70)


# ─── Helper: extract raw integer counts from a loaded AnnData ────────────────
def extract_raw(a, label):
    """Return an AnnData whose .X is raw integer counts.

    Wang/Vel (CellxGene format): .raw.X has full-transcriptome raw counts with
    its own var. PsychAD: .X IS raw counts (no .raw slot).
    """
    if a.raw is not None:
        Xr = a.raw.X
        var_raw = a.raw.var.copy()
        new = ad.AnnData(X=Xr, obs=a.obs.copy(), var=var_raw)
        if sp.issparse(new.X):
            sample = new.X.data[:2000] if new.X.data.size > 2000 else new.X.data
        else:
            sample = new.X.flatten()[:2000]
        is_int = bool(np.allclose(sample, np.round(sample)))
        print(f"  {label}: .raw.X  n_vars={new.n_vars:,}  integer={is_int}  max={sample.max():.1f}")
        if not is_int:
            print(f"  WARNING {label}: .raw.X is NOT integer — check source file!")
        return new
    else:
        X = a.X
        if sp.issparse(X):
            sample = X.data[:2000] if X.data.size > 2000 else X.data
        else:
            sample = X.flatten()[:2000]
        is_int = bool(np.allclose(sample, np.round(sample)))
        print(f"  {label}: .X (no .raw)  n_vars={a.n_vars:,}  integer={is_int}  max={sample.max():.1f}")
        if not is_int:
            print(f"  WARNING {label}: .X is NOT integer — check source file!")
        return a


def load_raw_subset(label, path, target_barcodes):
    """Load cells matching target_barcodes from a backed h5ad, return raw AnnData."""
    print(f"\n  [{label}] {path}")
    a = ad.read_h5ad(path, backed="r")
    print(f"    full: {a.n_obs:,} × {a.n_vars:,}  .raw={'YES' if a.raw is not None else 'NO'}")
    mask = a.obs.index.isin(target_barcodes)
    idx  = np.where(mask)[0]
    print(f"    matched: {len(idx):,}")
    if len(idx) == 0:
        a.file.close()
        return None
    sub = a[idx].to_memory()
    a.file.close()
    return extract_raw(sub, label)


def get_marker_indices(var_names_arr, marker_dict):
    """Return dict of sym -> column index for markers present in var_names."""
    var_pos = {g: i for i, g in enumerate(var_names_arr)}
    return {sym: var_pos[ensg] for sym, ensg in marker_dict.items() if ensg in var_pos}


def mean_cpm(X, lib, col_indices):
    """Mean CPM across cells for given column indices. lib shape (n_cells,)."""
    if not col_indices:
        return 0.0
    block = X[:, list(col_indices.values())]
    if sp.issparse(block):
        block = block.toarray()
    return float((block / lib[:, None] * 1e4).mean())


def frac_nonzero(X, col_indices, threshold=1):
    """Fraction of cells with count >= threshold for each marker."""
    out = {}
    for sym, idx in col_indices.items():
        col = X[:, idx]
        if sp.issparse(col):
            col = np.asarray(col.todense()).flatten()
        out[sym] = float((col >= threshold).mean())
    return out


# ─── 1. Load integrated obs ──────────────────────────────────────────────────
print("\n[1] Reading integrated h5ad obs (metadata only)...")
ai  = ad.read_h5ad(INTEGRATED, backed="r")
obs = ai.obs.copy()
ai.file.close()
print(f"  {len(obs):,} cells  columns: {list(obs.columns)[:10]}...")

age  = obs["age_years"].values.astype(float)
src  = obs["source-chemistry"].astype(str).values
names = obs.index.values

individual_col = None
for c in ("individual", "individualID", "donor_id"):
    if c in obs.columns:
        individual_col = c
        break
individual = obs[individual_col].astype(str).values if individual_col else names

cta_col = "cell_type_aligned" if "cell_type_aligned" in obs.columns else None
cell_type_aligned = obs[cta_col].astype(str).values if cta_col else np.full(len(obs), "Unknown")

# Group definitions (aligned with plan)
gm = {
    "PSYCHAD_under1y": (src == "PSYCHAD-V3")    & (age >= 0) & (age < 1),
    "PSYCHAD_1_5y":    (src == "PSYCHAD-V3")    & (age >= 1) & (age < 5),
    "WANG_under1y":    (src == "WANG-multiome") & (age >= 0) & (age < 1),
    "VEL_V3_under1y":  (src == "VELMESHEV-V3")  & (age >= 0) & (age < 1),
}
for g, m in gm.items():
    print(f"  {g}: {m.sum():,}")

# Barcode sets for loading
bc_psy_all = set(names[(src == "PSYCHAD-V3") & (age >= 0) & (age < 5)])
bc_wang    = set(names[gm["WANG_under1y"]])
bc_vel     = set(names[gm["VEL_V3_under1y"]])

# Per-barcode lookups needed for D11 and D13
bc_to_individual = dict(zip(names[gm["PSYCHAD_under1y"]],
                            individual[gm["PSYCHAD_under1y"]]))
bc_to_cta        = dict(zip(names, cell_type_aligned))
bc_to_age        = dict(zip(names, age))   # age_years from integrated obs
# Age group label per PsychAD barcode
bc_to_agegroup = {}
for g in ("PSYCHAD_under1y", "PSYCHAD_1_5y"):
    for bc in names[gm[g]]:
        bc_to_agegroup[bc] = g

print(f"\n  PsychAD <1y+1-5y barcodes to load: {len(bc_psy_all):,}")
print(f"  Wang <1y barcodes to load: {len(bc_wang):,}")
print(f"  Vel-V3 <1y barcodes to load: {len(bc_vel):,}")


# ─── 2. Load raw counts from base files ──────────────────────────────────────
print("\n[2] Loading raw counts from base files...")

wang = load_raw_subset("WANG", WANG_PATH, bc_wang)
vel  = load_raw_subset("VEL",  VEL_PATH,  bc_vel)

# Load from HBCC only — HBCC is the sole source of young donors (ages 0.2-85y).
# MSSM/Aging is adults-only (20-97y). Both files share 899k cells with identical
# counts, but HBCC is the correct primary for any young-cell analysis, and
# HBCC-unique young cells would be missed if loading from Aging.
psy = load_raw_subset("PSY_HBCC", HBCC_PATH, bc_psy_all)
if psy is None:
    raise RuntimeError("No PsychAD cells matched in HBCC_Cohort.h5ad")
print(f"\n  PsychAD (HBCC): {psy.n_obs:,} × {psy.n_vars:,}")

# Extract HBCC-native class and development_stage for D13b and D11 age annotation.
# 'class' uses Mathys 2023 adult reference labels: Astro/EN/Endo/IN/Immune/Mural/OPC/Oligo
# These reliably identify non-neuronal cells regardless of age composition issues.
bc_to_hbcc_class = {}
bc_to_devstage   = {}
if 'class' in psy.obs.columns:
    bc_to_hbcc_class = dict(zip(psy.obs.index, psy.obs['class'].astype(str)))
    print(f"  HBCC class categories: {sorted(set(bc_to_hbcc_class.values()))}")
else:
    print("  WARNING: 'class' column not found in HBCC obs — D13b will fall back to cell_type_aligned")
if 'development_stage' in psy.obs.columns:
    bc_to_devstage = dict(zip(psy.obs.index, psy.obs['development_stage'].astype(str)))

# Check marker presence in each dataset
print("\n  Marker presence check:")
for lbl, a in [("PsychAD", psy), ("Wang", wang), ("Vel", vel)]:
    vn = set(a.var_names)
    en_found = [s for s, e in EN_MARKERS.items() if e in vn]
    in_found = [s for s, e in IN_MARKERS.items() if e in vn]
    print(f"  {lbl}: EN markers {en_found}, IN markers {in_found}")


# ─── Helper: get the X matrix and lib sizes for a subset of cells ────────────
def get_X(a, barcodes=None):
    """Return (X_csr, obs_names) for cells matching barcodes (or all if None)."""
    if barcodes is not None:
        mask = a.obs.index.isin(barcodes)
        obs_sel = a.obs.index[mask]
        X = a.X[mask, :]
    else:
        obs_sel = a.obs.index
        X = a.X
    if sp.issparse(X) and not sp.isspmatrix_csr(X):
        X = X.tocsr()
    elif not sp.issparse(X):
        X = sp.csr_matrix(X)
    return X, obs_sel


# ─── 3. D11 — PsychAD <1y per-donor EN/IN ────────────────────────────────────
print("\n" + "=" * 70)
print("[D11] Per-donor EN/IN ratio — PsychAD <1y")
print("=" * 70)

psy_u1_barcodes = set(names[gm["PSYCHAD_under1y"]])
X_psy_u1, obs_psy_u1 = get_X(psy, psy_u1_barcodes)
vn_psy = psy.var_names.values

en_idx = get_marker_indices(vn_psy, EN_MARKERS)
in_idx = get_marker_indices(vn_psy, IN_MARKERS)
print(f"  EN marker indices: {list(en_idx.keys())}")
print(f"  IN marker indices: {list(in_idx.keys())}")

# Build per-donor records
donors = [bc_to_individual.get(bc, "Unknown") for bc in obs_psy_u1]
donor_arr = np.array(donors)
unique_donors = sorted(set(donor_arr))
print(f"  PsychAD <1y: {X_psy_u1.shape[0]:,} cells, {len(unique_donors)} donors")

lib_u1 = np.asarray(X_psy_u1.sum(axis=1)).flatten().astype(float)
lib_u1[lib_u1 == 0] = 1.0
n_genes_u1 = np.asarray((X_psy_u1 > 0).sum(axis=1)).flatten()

d11_rows = []
for donor in unique_donors:
    dmask = donor_arr == donor
    Xd = X_psy_u1[dmask, :]
    libd = lib_u1[dmask]
    n = int(dmask.sum())
    en_mean  = mean_cpm(Xd, libd, en_idx)
    in_mean  = mean_cpm(Xd, libd, in_idx)
    en_in    = en_mean / (in_mean + 1e-9)
    total_umi_mean  = float(libd.mean())
    n_genes_mean    = float(n_genes_u1[dmask].mean())
    # All cells from one donor share the same age — take first barcode
    donor_bc0    = obs_psy_u1[dmask][0]
    age_val      = float(bc_to_age.get(donor_bc0, np.nan))
    age_months   = round(age_val * 12) if not np.isnan(age_val) else np.nan
    devstage_val = bc_to_devstage.get(donor_bc0, "Unknown")
    d11_rows.append({
        "individual":       donor,
        "age_years":        age_val,
        "age_months":       age_months,
        "development_stage": devstage_val,
        "n_cells":          n,
        "total_UMI_mean":   total_umi_mean,
        "n_genes_mean":     n_genes_mean,
        "EN_mean_cpm":      en_mean,
        "IN_mean_cpm":      in_mean,
        "EN_over_IN":       en_in,
    })
    # Per-marker CPM detail
    for sym, cidx in en_idx.items():
        col = Xd[:, cidx]
        if sp.issparse(col): col = np.asarray(col.todense()).flatten()
        d11_rows[-1][f"EN_{sym}_cpm"] = float((col / libd * 1e4).mean())
    for sym, cidx in in_idx.items():
        col = Xd[:, cidx]
        if sp.issparse(col): col = np.asarray(col.todense()).flatten()
        d11_rows[-1][f"IN_{sym}_cpm"] = float((col / libd * 1e4).mean())

    # Per-donor frac_nonzero for key EN-specific (SATB2, SLC17A7) and
    # IN-specific (GAD1, SLC32A1) markers at thresholds >=1 and >=3.
    # SATB2/SLC17A7 are EN-specific in DLPFC → frac(>=1) is a lower-bound EN%
    # estimate that is library-size-independent.
    # SATB2_retention = frac3/frac1 diagnoses whether signal is genuine
    # (high retention ≈ Wang/Vel) vs mostly 1-count noise (low retention).
    for sym, cidx in {**en_idx, **in_idx}.items():
        col = Xd[:, cidx]
        if sp.issparse(col): col = np.asarray(col.todense()).flatten()
        d11_rows[-1][f"{sym}_frac1"] = float((col >= 1).mean())
        d11_rows[-1][f"{sym}_frac3"] = float((col >= 3).mean())
    satb2_f1 = d11_rows[-1].get("SATB2_frac1", 1e-9)
    satb2_f3 = d11_rows[-1].get("SATB2_frac3", 0.0)
    d11_rows[-1]["SATB2_retention"] = satb2_f3 / (satb2_f1 + 1e-9)

d11 = pd.DataFrame(d11_rows).sort_values("age_years", ascending=True)
d11.to_csv(OUT / "d11_per_donor.csv", index=False)

print(f"\n  Per-donor EN% estimates (sorted by age):")
frac_cols = ["individual", "age_months", "development_stage", "n_cells", "total_UMI_mean",
             "SATB2_frac1", "SATB2_frac3", "SATB2_retention",
             "SLC17A7_frac1", "SLC17A7_frac3",
             "GAD1_frac1", "GAD1_frac3", "SLC32A1_frac1"]
print(d11[[c for c in frac_cols if c in d11.columns]].to_string(index=False))
print(f"\n  SATB2_frac1 (EN% lower bound) across donors: "
      f"mean={d11['SATB2_frac1'].mean():.3f}  "
      f"median={d11['SATB2_frac1'].median():.3f}  "
      f"range=[{d11['SATB2_frac1'].min():.3f}, {d11['SATB2_frac1'].max():.3f}]")
print(f"  SATB2_retention (frac3/frac1): "
      f"mean={d11['SATB2_retention'].mean():.3f}  "
      f"median={d11['SATB2_retention'].median():.3f}  "
      f"(Wang cohort-level retention={0.368/0.479:.3f}, Vel={0.466/0.624:.3f})")
print(f"\n  EN/IN CPM ratio across donors: mean={d11['EN_over_IN'].mean():.3f}  "
      f"median={d11['EN_over_IN'].median():.3f}  "
      f"range=[{d11['EN_over_IN'].min():.3f}, {d11['EN_over_IN'].max():.3f}]")


# ─── 4. D12 — Fraction of cells with detectable marker expression ─────────────
print("\n" + "=" * 70)
print("[D12] Fraction of cells with detectable EN/IN marker expression")
print("=" * 70)

# Build group data: {group_name: (X_csr, var_names)}
groups_data = {}

# PsychAD groups (from psy, which has PSYCHAD <1y + 1-5y)
for g in ("PSYCHAD_under1y", "PSYCHAD_1_5y"):
    bc_g = set(names[gm[g]])
    Xg, _ = get_X(psy, bc_g)
    groups_data[g] = (Xg, vn_psy)

def _to_csr(X):
    if sp.isspmatrix_csr(X):
        return X
    if sp.issparse(X):
        return X.tocsr()
    return sp.csr_matrix(X)

groups_data["WANG_under1y"]   = (_to_csr(wang.X), wang.var_names.values)
groups_data["VEL_V3_under1y"] = (_to_csr(vel.X),  vel.var_names.values)

d12_rows = []
for grp, (X, vn) in groups_data.items():
    en_i = get_marker_indices(vn, EN_MARKERS)
    in_i = get_marker_indices(vn, IN_MARKERS)
    n = X.shape[0]
    row = {"group": grp, "n_cells": n}

    frac1_en = frac_nonzero(X, en_i, threshold=1)
    frac3_en = frac_nonzero(X, en_i, threshold=3)
    frac1_in = frac_nonzero(X, in_i, threshold=1)
    frac3_in = frac_nonzero(X, in_i, threshold=3)

    # Aggregate: mean fraction across EN markers, mean across IN markers
    row["EN_frac_nonzero_1"] = float(np.mean(list(frac1_en.values()))) if frac1_en else np.nan
    row["EN_frac_nonzero_3"] = float(np.mean(list(frac3_en.values()))) if frac3_en else np.nan
    row["IN_frac_nonzero_1"] = float(np.mean(list(frac1_in.values()))) if frac1_in else np.nan
    row["IN_frac_nonzero_3"] = float(np.mean(list(frac3_in.values()))) if frac3_in else np.nan
    ratio1 = row["EN_frac_nonzero_1"] / (row["IN_frac_nonzero_1"] + 1e-9)
    ratio3 = row["EN_frac_nonzero_3"] / (row["IN_frac_nonzero_3"] + 1e-9)
    row["EN_over_IN_frac1"] = ratio1
    row["EN_over_IN_frac3"] = ratio3

    # Per-marker columns
    for sym, f in frac1_en.items():
        row[f"EN_{sym}_frac1"] = f
    for sym, f in frac3_en.items():
        row[f"EN_{sym}_frac3"] = f
    for sym, f in frac1_in.items():
        row[f"IN_{sym}_frac1"] = f
    for sym, f in frac3_in.items():
        row[f"IN_{sym}_frac3"] = f

    d12_rows.append(row)
    print(f"  {grp:25s} n={n:>7,}  EN_frac(>=1)={row['EN_frac_nonzero_1']:.3f}  "
          f"IN_frac(>=1)={row['IN_frac_nonzero_1']:.3f}  EN/IN_frac={ratio1:.3f}")

d12 = pd.DataFrame(d12_rows)
d12.to_csv(OUT / "d12_frac_nonzero.csv", index=False)

# Print per-marker detail for PSYCHAD_under1y vs others
print("\n  Per-marker frac_nonzero (threshold >= 1):")
cols_to_show = [f"EN_{s}_frac1" for s in EN_MARKERS if any(f"EN_{s}_frac1" in r for r in d12_rows)] + \
               [f"IN_{s}_frac1" for s in IN_MARKERS if any(f"IN_{s}_frac1" in r for r in d12_rows)]
show_cols = ["group"] + [c for c in cols_to_show if c in d12.columns]
print(d12[show_cols].to_string(index=False))


# ─── 5. D13 — Ambient RNA contamination check ─────────────────────────────────
print("\n" + "=" * 70)
print("[D13] Ambient RNA check — IN marker signal in non-neuronal cells")
print("=" * 70)

if cta_col is None:
    print("  WARNING: 'cell_type_aligned' not in integrated obs; skipping D13.")
else:
    d13_rows = []

    # For each group, filter to non-neuronal cells using cell_type_aligned
    for grp, (X_full, vn) in groups_data.items():
        # Get obs barcodes for this group
        if grp.startswith("PSYCHAD"):
            age_grp = "PSYCHAD_under1y" if "under1y" in grp else "PSYCHAD_1_5y"
            bc_list = list(names[gm[age_grp]])
            adata_for_grp = psy
        elif grp == "WANG_under1y":
            bc_list = list(names[gm["WANG_under1y"]])
            adata_for_grp = wang
        else:  # VEL_V3_under1y
            bc_list = list(names[gm["VEL_V3_under1y"]])
            adata_for_grp = vel

        # Build non-neuronal mask using cell_type_aligned from integrated obs
        # Values use EN_*/IN_* prefixes (e.g. EN_L2_3, IN_PV) — use is_neuronal()
        nonneuronal_bcs = [bc for bc in bc_list
                           if not is_neuronal(bc_to_cta.get(bc, "Unknown"))]
        neuronal_bcs    = [bc for bc in bc_list
                           if is_neuronal(bc_to_cta.get(bc, "Unknown"))]
        print(f"\n  {grp}: {len(bc_list):,} total  |  "
              f"non-neuronal (cell_type_aligned not in EN/IN): {len(nonneuronal_bcs):,}  |  "
              f"neuronal: {len(neuronal_bcs):,}")

        if len(nonneuronal_bcs) == 0:
            print(f"  (no non-neuronal cells found for {grp} — check cell_type_aligned values)")
            continue

        # Load non-neuronal subset
        bc_nn_set = set(nonneuronal_bcs)
        mask_nn   = adata_for_grp.obs.index.isin(bc_nn_set)
        Xnn = adata_for_grp.X[mask_nn, :]
        if sp.issparse(Xnn) and not sp.isspmatrix_csr(Xnn):
            Xnn = Xnn.tocsr()
        elif not sp.issparse(Xnn):
            Xnn = sp.csr_matrix(Xnn)

        lib_nn = np.asarray(Xnn.sum(axis=1)).flatten().astype(float)
        lib_nn[lib_nn == 0] = 1.0

        en_i = get_marker_indices(vn, EN_MARKERS)
        in_i = get_marker_indices(vn, IN_MARKERS)

        en_mean_nn = mean_cpm(Xnn, lib_nn, en_i)
        in_mean_nn = mean_cpm(Xnn, lib_nn, in_i)

        row = {
            "group":              grp,
            "n_nonneuronal":      int(mask_nn.sum()),
            "n_neuronal":         len(neuronal_bcs),
            "EN_mean_cpm":        en_mean_nn,
            "IN_mean_cpm":        in_mean_nn,
            "IN_over_EN_nonneuronal": in_mean_nn / (en_mean_nn + 1e-9),
        }
        # Per-marker
        for sym, cidx in {**en_i, **in_i}.items():
            col = Xnn[:, cidx]
            if sp.issparse(col): col = np.asarray(col.todense()).flatten()
            row[f"{sym}_mean_cpm"] = float((col / lib_nn * 1e4).mean())
            row[f"{sym}_frac_nonzero"] = float((col > 0).mean())

        d13_rows.append(row)
        print(f"  EN_mean={en_mean_nn:.3f}  IN_mean={in_mean_nn:.3f}  "
              f"IN/EN={in_mean_nn/(en_mean_nn+1e-9):.2f}")

    d13 = pd.DataFrame(d13_rows)
    d13.to_csv(OUT / "d13_ambient_check.csv", index=False)

    print("\n  Summary table (non-neuronal cells only):")
    key_cols = ["group", "n_nonneuronal"] + \
               [f"{s}_mean_cpm" for s in ("GAD1", "GAD2", "SLC32A1", "SLC17A7")
                if any(f"{s}_mean_cpm" in r for r in d13_rows)] + \
               ["IN_mean_cpm", "EN_mean_cpm", "IN_over_EN_nonneuronal"]
    show = [c for c in key_cols if c in d13.columns]
    print(d13[show].to_string(index=False))

    print("\n  Interpretation guide:")
    print("  If PsychAD <1y non-neuronal cells show elevated GAD1/GAD2 vs Wang/Vel")
    print("  non-neuronal → ambient RNA contamination inflating IN signal.")
    print("  If similar across sources → IN signal is genuine cell-intrinsic.")


# ─── 5b. D13b — Ambient RNA check using source-native cell type labels ────────
print("\n" + "=" * 70)
print("[D13b] Ambient RNA check — source-native cell type labels")
print("  PsychAD: HBCC 'class' (EN/IN=neuronal; Astro/Oligo/OPC/Mural/Endo/Immune=non-neuronal)")
print("  Wang/Vel: cell_type_aligned (reliable — scANVI supervised on their own labels)")
print("=" * 70)

HBCC_NEURONAL = {"EN", "IN"}
d13b_rows = []

for grp, (X_full, vn) in groups_data.items():
    if grp.startswith("PSYCHAD"):
        age_grp = "PSYCHAD_under1y" if "under1y" in grp else "PSYCHAD_1_5y"
        bc_list = list(names[gm[age_grp]])
        adata_for_grp = psy
        if bc_to_hbcc_class:
            nonneuronal_bcs = [bc for bc in bc_list
                               if bc_to_hbcc_class.get(bc, "Unknown") not in HBCC_NEURONAL]
            neuronal_bcs    = [bc for bc in bc_list
                               if bc_to_hbcc_class.get(bc, "Unknown") in HBCC_NEURONAL]
            label_source = "HBCC_class"
        else:
            nonneuronal_bcs = [bc for bc in bc_list
                               if not is_neuronal(bc_to_cta.get(bc, "Unknown"))]
            neuronal_bcs    = [bc for bc in bc_list
                               if is_neuronal(bc_to_cta.get(bc, "Unknown"))]
            label_source = "cell_type_aligned_fallback"
    elif grp == "WANG_under1y":
        bc_list = list(names[gm["WANG_under1y"]])
        adata_for_grp = wang
        nonneuronal_bcs = [bc for bc in bc_list
                           if not is_neuronal(bc_to_cta.get(bc, "Unknown"))]
        neuronal_bcs    = [bc for bc in bc_list
                           if is_neuronal(bc_to_cta.get(bc, "Unknown"))]
        label_source = "cell_type_aligned"
    else:  # VEL_V3_under1y
        bc_list = list(names[gm["VEL_V3_under1y"]])
        adata_for_grp = vel
        nonneuronal_bcs = [bc for bc in bc_list
                           if not is_neuronal(bc_to_cta.get(bc, "Unknown"))]
        neuronal_bcs    = [bc for bc in bc_list
                           if is_neuronal(bc_to_cta.get(bc, "Unknown"))]
        label_source = "cell_type_aligned"

    print(f"\n  {grp} [{label_source}]: {len(bc_list):,} total  |  "
          f"non-neuronal: {len(nonneuronal_bcs):,}  |  neuronal: {len(neuronal_bcs):,}")

    if len(nonneuronal_bcs) == 0:
        print("  (no non-neuronal cells found — check label source)")
        continue

    bc_nn_set = set(nonneuronal_bcs)
    mask_nn   = adata_for_grp.obs.index.isin(bc_nn_set)
    Xnn = adata_for_grp.X[mask_nn, :]
    if sp.issparse(Xnn) and not sp.isspmatrix_csr(Xnn):
        Xnn = Xnn.tocsr()
    elif not sp.issparse(Xnn):
        Xnn = sp.csr_matrix(Xnn)

    lib_nn = np.asarray(Xnn.sum(axis=1)).flatten().astype(float)
    lib_nn[lib_nn == 0] = 1.0

    en_i = get_marker_indices(vn, EN_MARKERS)
    in_i = get_marker_indices(vn, IN_MARKERS)

    en_mean_nn = mean_cpm(Xnn, lib_nn, en_i)
    in_mean_nn = mean_cpm(Xnn, lib_nn, in_i)

    row = {
        "group":                  grp,
        "label_source":           label_source,
        "n_nonneuronal":          int(mask_nn.sum()),
        "n_neuronal":             len(neuronal_bcs),
        "EN_mean_cpm":            en_mean_nn,
        "IN_mean_cpm":            in_mean_nn,
        "IN_over_EN_nonneuronal": in_mean_nn / (en_mean_nn + 1e-9),
    }
    for sym, cidx in {**en_i, **in_i}.items():
        col = Xnn[:, cidx]
        if sp.issparse(col): col = np.asarray(col.todense()).flatten()
        row[f"{sym}_mean_cpm"] = float((col / lib_nn * 1e4).mean())

    d13b_rows.append(row)
    print(f"  EN_mean={en_mean_nn:.3f}  IN_mean={in_mean_nn:.3f}  "
          f"IN/EN={in_mean_nn/(en_mean_nn+1e-9):.2f}")

d13b = pd.DataFrame(d13b_rows)
d13b.to_csv(OUT / "d13b_ambient_native_labels.csv", index=False)

print("\n  Summary table (non-neuronal, source-native labels):")
key_cols_b = ["group", "label_source", "n_nonneuronal"] + \
             [f"{s}_mean_cpm" for s in ("GAD1", "GAD2", "SLC32A1", "SLC17A7")
              if any(f"{s}_mean_cpm" in r for r in d13b_rows)] + \
             ["IN_mean_cpm", "EN_mean_cpm", "IN_over_EN_nonneuronal"]
show_b = [c for c in key_cols_b if c in d13b.columns]
print(d13b[show_b].to_string(index=False))

print("\n  Interpretation:")
print("  PsychAD non-neuronal here = HBCC class NOT in {EN, IN} — not dependent on scANVI.")
print("  If GAD1/GAD2 in PsychAD non-neuronal >> Wang/Vel → ambient RNA (FANS should prevent this).")
print("  If similar across sources → PsychAD IN signal is genuine, not contamination.")


# ─── 6. Summary markdown ──────────────────────────────────────────────────────
print("\n[6] Writing summary.md...")

lines = [
    "# D11/D12/D13 — PsychAD <1y biological characterisation",
    "",
    f"Raw counts from base files: Wang/Vel via `.raw.X`, PsychAD via `.X`.",
    "",
    "## D11 — Per-donor marker analysis (PsychAD <1y, sorted by age)",
    "",
    "| individual | age_mo | devstage | n_cells | UMI/cell | SATB2_frac1 | SATB2_frac3 | SATB2_ret | SLC17A7_frac1 | GAD1_frac1 | GAD1_frac3 | EN/IN_cpm |",
    "|-----------|-------:|---------|--------:|---------:|------------:|------------:|----------:|--------------:|-----------:|-----------:|----------:|",
]
for _, r in d11.iterrows():
    lines.append(
        f"| {r['individual']} | {r.get('age_months', '')} | {r.get('development_stage', '')} | "
        f"{r['n_cells']:,} | {r['total_UMI_mean']:,.0f} | "
        f"{r.get('SATB2_frac1', 0):.3f} | {r.get('SATB2_frac3', 0):.3f} | "
        f"{r.get('SATB2_retention', 0):.2f} | {r.get('SLC17A7_frac1', 0):.3f} | "
        f"{r.get('GAD1_frac1', 0):.3f} | {r.get('GAD1_frac3', 0):.3f} | "
        f"{r['EN_over_IN']:.3f} |"
    )

lines += [
    "",
    f"SATB2_frac1 range: [{d11['SATB2_frac1'].min():.3f}, {d11['SATB2_frac1'].max():.3f}]  "
    f"SATB2_retention (frac3/frac1): mean={d11['SATB2_retention'].mean():.2f}  "
    f"(Wang cohort={0.368/0.479:.2f}, Vel={0.466/0.624:.2f})",
    "",
    "## D12 — Fraction of cells with detectable marker expression",
    "",
    "| group | n_cells | EN_frac(>=1) | IN_frac(>=1) | EN_frac(>=3) | IN_frac(>=3) | EN/IN_frac1 | EN/IN_frac3 |",
    "|-------|--------:|-------------:|-------------:|-------------:|-------------:|------------:|------------:|",
]
for _, r in d12.iterrows():
    lines.append(
        f"| {r['group']} | {r['n_cells']:,} | {r.get('EN_frac_nonzero_1', 0):.3f} | "
        f"{r.get('IN_frac_nonzero_1', 0):.3f} | {r.get('EN_frac_nonzero_3', 0):.3f} | "
        f"{r.get('IN_frac_nonzero_3', 0):.3f} | {r.get('EN_over_IN_frac1', 0):.3f} | "
        f"{r.get('EN_over_IN_frac3', 0):.3f} |"
    )

if 'd13' in dir() and len(d13) > 0:
    lines += [
        "",
        "## D13 — Ambient RNA check (non-neuronal via scANVI cell_type_aligned)",
        "",
        "| group | n_nonneuronal | GAD1_cpm | GAD2_cpm | SLC32A1_cpm | SLC17A7_cpm | IN/EN ratio |",
        "|-------|-------------:|---------:|---------:|------------:|------------:|------------:|",
    ]
    for _, r in d13.iterrows():
        lines.append(
            f"| {r['group']} | {r['n_nonneuronal']:,} | "
            f"{r.get('GAD1_mean_cpm', 0):.3f} | {r.get('GAD2_mean_cpm', 0):.3f} | "
            f"{r.get('SLC32A1_mean_cpm', 0):.3f} | {r.get('SLC17A7_mean_cpm', 0):.3f} | "
            f"{r.get('IN_over_EN_nonneuronal', 0):.2f} |"
        )

if 'd13b' in dir() and len(d13b) > 0:
    lines += [
        "",
        "## D13b — Ambient RNA check (source-native labels: HBCC class / cell_type_aligned)",
        "",
        "| group | label_source | n_nonneuronal | GAD1_cpm | GAD2_cpm | SLC32A1_cpm | SLC17A7_cpm | IN/EN ratio |",
        "|-------|-------------|-------------:|---------:|---------:|------------:|------------:|------------:|",
    ]
    for _, r in d13b.iterrows():
        lines.append(
            f"| {r['group']} | {r.get('label_source', '')} | {r['n_nonneuronal']:,} | "
            f"{r.get('GAD1_mean_cpm', 0):.3f} | {r.get('GAD2_mean_cpm', 0):.3f} | "
            f"{r.get('SLC32A1_mean_cpm', 0):.3f} | {r.get('SLC17A7_mean_cpm', 0):.3f} | "
            f"{r.get('IN_over_EN_nonneuronal', 0):.2f} |"
        )

with open(OUT / "summary.md", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nWritten:")
print(f"  {OUT / 'd11_per_donor.csv'}")
print(f"  {OUT / 'd12_frac_nonzero.csv'}")
print(f"  {OUT / 'd13_ambient_check.csv'}")
print(f"  {OUT / 'd13b_ambient_native_labels.csv'}")
print(f"  {OUT / 'summary.md'}")
print("\nDone.")
