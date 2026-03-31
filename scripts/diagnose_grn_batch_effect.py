"""
Diagnose source-level differences in ahbaC3 C3+ GRN scores after scVI correction.
Main focus: why do VELMESHEV (and WANG) show higher C3+ in childhood (ages 1-10)?

Checks:
  1. Overall and Excitatory-only means by source
  2. Regression C3+ ~ age + source (Excitatory)
  3. Regression C3+ ~ age + cell_subclass + source (Excitatory)
  4. Cell subclass composition per source (with fixed source-routing)
  5. QC metrics (library size, gene detection) per source
  6. Top C3+ genes by between-source variance
  7. Genetic ancestry per source (from PsychAD donor metadata)
  -- Childhood-focused analysis (ages 1-10) --
  8. N donors, pseudobulk means per source x age bin in childhood
  9. QC metrics for childhood cells specifically
  10. Regression in childhood window: does source effect persist after age control?
  11. Top genes driving VEL > AGING/HBCC in childhood
  12. Contamination check: re-run excluding VEL cells whose mapped subclass is Inhibitory

Outputs:
  diagnose_grn_batch_effect.png      -- main 2x3 panel (all ages)
  diagnose_grn_batch_effect_pb.png   -- pseudobulk + subclass panels
  diagnose_grn_childhood.png         -- childhood-focused 2x3 panel
  diagnose_grn_batch_effect.txt      -- text summary
"""

import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from scipy import stats
from pathlib import Path

CODE_DIR = "/home/rajd2/rds/hpc-work/snRNAseq_2026/code"
sys.path.insert(0, CODE_DIR)

RDS = "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
DATA_FILE = f"{RDS}/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld/scvi_output/integrated.h5ad"
GRN_FILE  = "/home/rajd2/rds/hpc-work/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv"
AGING_PATH = f"{RDS}/Cam_PsychAD/RNAseq/Aging_Cohort.h5ad"
HBCC_PATH  = f"{RDS}/Cam_PsychAD/RNAseq/HBCC_Cohort.h5ad"

OUT_DIR  = "/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs"
OUT_FIG1 = f"{OUT_DIR}/diagnose_grn_batch_effect.png"
OUT_FIG2 = f"{OUT_DIR}/diagnose_grn_batch_effect_pb.png"
OUT_FIG3 = f"{OUT_DIR}/diagnose_grn_childhood.png"
OUT_TXT  = f"{OUT_DIR}/diagnose_grn_batch_effect.txt"

RANDOM_SEED = 42
AGE_BINS   = [-1, 0, 1, 3, 6, 10, 15, 20, 30, 40, 100]
AGE_LABELS = ["prenatal", "0-1", "1-3", "3-6", "6-10",
              "10-15", "15-20", "20-30", "30-40", "40+"]
CHILDHOOD  = (1.0, 10.0)   # age window of interest

SET1 = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
        "#FF7F00", "#A65628", "#F781BF", "#999999"]

EXCIT_SUBCLASSES = [
    "EN_L2_3_IT", "EN_L4_IT", "EN_L5_IT", "EN_L5_ET",
    "EN_L5_6_NP", "EN_L6_IT", "EN_L6_CT", "EN_L6B",
    "EN_Immature", "Excitatory",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def to_dense(mat):
    return mat.toarray() if issparse(mat) else np.asarray(mat)

def cpm(counts):
    s = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(s, 1) * 1e6

def ols_summary(y, X_df):
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(len(y)), X_df.values.astype(float)])
    cols = ["intercept"] + list(X_df.columns)
    b, _, _, _ = lstsq(X, y, rcond=None)
    resid = y - X @ b
    n, p = X.shape
    s2 = (resid ** 2).sum() / max(n - p, 1)
    XtXinv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(s2 * XtXinv.diagonal())
    t  = b / np.where(se > 0, se, np.nan)
    pv = 2 * stats.t.sf(np.abs(t), df=n - p)
    r2 = 1 - (resid ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return pd.DataFrame({"coef": b, "se": se, "t": t, "p": pv}, index=cols), r2


# ── import subclass mapping ───────────────────────────────────────────────────

try:
    from read_data import (map_cellxgene_subclass, map_velmeshev_subclass,
                           collapse_en_subclass)
    SUBCLASS_OK = True
    print("Imported subclass mapping from read_data.py", flush=True)
except Exception as e:
    SUBCLASS_OK = False
    print(f"Warning: subclass import failed ({e})", flush=True)


# ── load genetic ancestry from PsychAD source files ──────────────────────────

print("Loading donor genetic ancestry from PsychAD sources…", flush=True)
donor_ancestry = {}
for path in [AGING_PATH, HBCC_PATH]:
    try:
        a = sc.read_h5ad(path, backed="r")
        if "genetic_ancestry" in a.obs.columns and "donor_id" in a.obs.columns:
            mapping = (a.obs[["donor_id", "genetic_ancestry"]]
                       .drop_duplicates("donor_id")
                       .set_index("donor_id")["genetic_ancestry"])
            donor_ancestry.update(mapping.to_dict())
        print(f"  {path.split('/')[-1]}: {len(mapping)} donors", flush=True)
    except Exception as e:
        print(f"  Could not read {path}: {e}", flush=True)
donor_ancestry_series = pd.Series(donor_ancestry, name="genetic_ancestry")


# ── load main data ────────────────────────────────────────────────────────────

print("\nLoading integrated h5ad…", flush=True)
adata = sc.read_h5ad(DATA_FILE)
print(f"Shape: {adata.shape}  layers: {list(adata.layers.keys())}", flush=True)


# ── derive cell_subclass (source-routed) ──────────────────────────────────────

if SUBCLASS_OK and "cell_type" in adata.obs.columns:
    print("Deriving cell_subclass (source-routed)…", flush=True)
    sub = adata.obs["cell_type"].astype(str).copy()
    vel = adata.obs["source"] == "VELMESHEV"
    # VELMESHEV cell_type contains source-specific labels (L2-3, INT, …)
    sub[vel]  = sub[vel].map(map_velmeshev_subclass).map(collapse_en_subclass)
    # WANG / AGING / HBCC cell_type contains CellxGene ontology terms
    sub[~vel] = sub[~vel].map(map_cellxgene_subclass).map(collapse_en_subclass)
    adata.obs["cell_subclass"] = sub.values
    # Flag VEL cells where subclass is Inhibitory despite cell_class=Excitatory
    inconsistent = (
        (adata.obs["source"] == "VELMESHEV") &
        (adata.obs["cell_class"] == "Excitatory") &
        (adata.obs["cell_subclass"].isin(["Inhibitory", "IN_SST", "IN_PVALB",
                                           "IN_VIP", "IN_LAMP5", "IN_ADARB2",
                                           "IN_Immature"]))
    )
    adata.obs["vel_subclass_inconsistent"] = inconsistent
    n_incon = inconsistent.sum()
    print(f"  VEL excitatory cells with Inhibitory-mapped subclass: {n_incon:,}", flush=True)
else:
    adata.obs["cell_subclass"] = "unknown"
    adata.obs["vel_subclass_inconsistent"] = False


# ── attach genetic ancestry ───────────────────────────────────────────────────

if "donor_id" in adata.obs.columns and len(donor_ancestry_series) > 0:
    adata.obs["genetic_ancestry"] = (
        adata.obs["donor_id"].map(donor_ancestry_series).fillna("unknown")
    )
else:
    adata.obs["genetic_ancestry"] = "unknown"
print("Genetic ancestry counts:\n" +
      adata.obs["genetic_ancestry"].value_counts().to_string(), flush=True)


# ── load GRN and compute scores ───────────────────────────────────────────────

from regulons import get_ahba_GRN
from gene_mapping import map_grn_symbols_to_ensembl

print("\nLoading GRN…", flush=True)
ahba_GRN   = get_ahba_GRN(path_to_ahba_weights=GRN_FILE, use_weights=True)
ahba_GRN   = map_grn_symbols_to_ensembl(ahba_GRN, adata)
grn_pivot  = ahba_GRN.pivot_table(
    index="Network", columns="Gene", values="Importance", fill_value=0)

def aligned_weights(network):
    w = pd.Series(0.0, index=adata.var_names)
    if network in grn_pivot.index:
        for g, v in grn_pivot.loc[network].items():
            if g in w.index and v != 0:
                w[g] = v
    return w.values.astype(np.float32)

w_c3pos = aligned_weights("C3+")
print(f"C3+ genes mapped: {(w_c3pos != 0).sum()}", flush=True)

print("Computing GRN scores…", flush=True)
raw_counts  = to_dense(adata.layers["counts"]).astype(np.float32)
score_raw   = cpm(raw_counts) @ w_c3pos
del raw_counts

scvi_expr   = to_dense(adata.layers["scvi_normalized"]).astype(np.float32)
score_scvi  = cpm(scvi_expr) @ w_c3pos
del scvi_expr


# ── build obs frame ───────────────────────────────────────────────────────────

obs = adata.obs.copy()
obs["score_raw"]  = score_raw
obs["score_scvi"] = score_scvi
obs["age_bin"]    = pd.cut(obs["age_years"], bins=AGE_BINS, labels=AGE_LABELS,
                           right=False)

raw_for_qc = to_dense(adata.layers["counts"])
obs["log_total_counts"]  = np.log1p(raw_for_qc.sum(axis=1))
obs["n_genes_detected"]  = (raw_for_qc > 0).sum(axis=1)
c3pos_idx                = np.where(w_c3pos != 0)[0]
obs["c3_gene_detection"] = (raw_for_qc[:, c3pos_idx] > 0).mean(axis=1)
del raw_for_qc

excit_mask = obs["cell_class"] == "Excitatory"
obs_ex     = obs[excit_mask].copy()
sources    = sorted(obs_ex["source"].unique())
src_col    = {s: SET1[i] for i, s in enumerate(sources)}

individual_col = next((c for c in ["individual", "donor_id", "donor"]
                       if c in obs.columns), None)

# Childhood mask
child_mask = (obs_ex["age_years"] >= CHILDHOOD[0]) & (obs_ex["age_years"] < CHILDHOOD[1])
obs_child  = obs_ex[child_mask].copy()
print(f"\nExcitatory cells: {excit_mask.sum():,} total, "
      f"{child_mask.sum():,} in childhood ({CHILDHOOD[0]}-{CHILDHOOD[1]} yrs)", flush=True)


# ── helper: pseudobulk ────────────────────────────────────────────────────────

def pseudobulk(df, individual_col):
    if individual_col is None:
        return None
    return (df.groupby([individual_col, "source"])
            .agg(score_scvi_mean=("score_scvi", "mean"),
                 score_raw_mean=("score_raw", "mean"),
                 age_years=("age_years", "median"),
                 n_cells=(individual_col, "count"),
                 c3_gene_detection=("c3_gene_detection", "mean"),
                 n_genes_detected=("n_genes_detected", "mean"))
            .reset_index())


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

lines = []
def say(s=""):
    print(s, flush=True)
    lines.append(s)

say("=" * 72)
say("DIAGNOSIS: ahbaC3 C3+ score — source differences (scVI corrected)")
say("=" * 72)

# 1. Overall means
for label, df in [("All cells", obs), ("Excitatory only", obs_ex)]:
    say(f"\n--- {label}: mean C3+ by source ---")
    say(f"  {'Source':12}  {'N':>7}  {'raw':>10}  {'scvi':>10}")
    for src in sources:
        d = df[df["source"] == src]
        say(f"  {src:12}  {len(d):>7,}  "
            f"{d['score_raw'].mean():>10.1f}  {d['score_scvi'].mean():>10.1f}")

# 2. Age-stratified (Excitatory, scVI)
say("\n--- Excitatory: mean scVI C3+ by source x age bin ---")
age_src = (obs_ex.groupby(["age_bin", "source"], observed=True)["score_scvi"]
           .mean().unstack("source"))
say(age_src.round(0).to_string())

# 3. Regression: C3+ ~ age + source (Excitatory)
say("\n--- Regression: scVI C3+ ~ age + source (Excitatory, all ages) ---")
reg_df  = obs_ex[["score_scvi", "age_years", "source"]].dropna()
X_reg   = pd.concat([reg_df[["age_years"]],
                     pd.get_dummies(reg_df["source"], drop_first=True, dtype=float)],
                    axis=1)
coefs, r2 = ols_summary(reg_df["score_scvi"].values, X_reg)
say(f"  R2 = {r2:.4f}")
say(coefs.to_string(float_format=lambda x: f"{x:10.1f}"))

# 4. Regression: C3+ ~ age + subclass + source
say("\n--- Regression: scVI C3+ ~ age + subclass + source (Excitatory) ---")
if "cell_subclass" in obs_ex.columns:
    r2_df   = obs_ex[["score_scvi", "age_years", "cell_subclass", "source"]].dropna()
    X_r2    = pd.concat([r2_df[["age_years"]],
                         pd.get_dummies(r2_df["cell_subclass"], drop_first=True, dtype=float),
                         pd.get_dummies(r2_df["source"],        drop_first=True, dtype=float)],
                        axis=1)
    coefs2, r2_2 = ols_summary(r2_df["score_scvi"].values, X_r2)
    say(f"  R2 = {r2_2:.4f}")
    show = [r for r in coefs2.index
            if r.startswith(tuple(sources)) or r in ("intercept", "age_years")]
    say(coefs2.loc[show].to_string(float_format=lambda x: f"{x:10.1f}"))
else:
    say("  (cell_subclass not available)")

# 5. QC metrics
say("\n--- QC metrics by source (all cells) ---")
say(f"  {'Source':12}  {'log_lib':>8}  {'n_genes':>8}  {'c3_det%':>8}  {'chemistry':>10}")
for src in sources:
    d   = obs[obs["source"] == src]
    ch  = d["chemistry"].value_counts().index[0] if "chemistry" in d.columns else "?"
    say(f"  {src:12}  {d['log_total_counts'].mean():>8.2f}  "
        f"{d['n_genes_detected'].mean():>8.0f}  "
        f"{d['c3_gene_detection'].mean() * 100:>7.1f}%  {ch:>10}")

# 6. Genetic ancestry per source
say("\n--- Genetic ancestry by source (PsychAD; unknown for VEL/WANG) ---")
anc_src = (obs.groupby(["source", "genetic_ancestry"])
           .size().unstack(fill_value=0))
say(anc_src.to_string())

# 7. Cell subclass composition (Excitatory)
if "cell_subclass" in obs_ex.columns:
    say("\n--- Cell subclass composition (Excitatory, %) by source ---")
    comp     = obs_ex.groupby(["source", "cell_subclass"]).size().unstack(fill_value=0)
    comp_pct = (comp.T / comp.sum(axis=1) * 100).T.round(1)
    say(comp_pct.to_string())
    say(f"\n  VEL Excitatory cells with inconsistent subclass: "
        f"{obs_ex.get('vel_subclass_inconsistent', pd.Series(False)).sum():,}")

# 8. Pseudobulk all-age summary
pb_all = pseudobulk(obs_ex, individual_col)
if pb_all is not None:
    say(f"\n--- Pseudobulk (per donor, Excitatory, all ages): {len(pb_all)} donors ---")
    say(f"  {'Source':12}  {'N donors':>9}  {'pb_scvi':>10}  {'pb_raw':>10}")
    for src in sources:
        d = pb_all[pb_all["source"] == src]
        say(f"  {src:12}  {len(d):>9,}  "
            f"{d['score_scvi_mean'].mean():>10.1f}  {d['score_raw_mean'].mean():>10.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CHILDHOOD ANALYSIS  (ages 1–10)
# ═══════════════════════════════════════════════════════════════════════════════

say("\n" + "=" * 72)
say(f"CHILDHOOD ANALYSIS: ages {CHILDHOOD[0]}–{CHILDHOOD[1]} years")
say("=" * 72)

say(f"\n--- Cell counts in childhood (Excitatory) ---")
say(f"  {'Source':12}  {'N cells':>8}  {'N donors':>9}")
pb_child = pseudobulk(obs_child, individual_col)
for src in sources:
    nc = (obs_child["source"] == src).sum()
    nd = len(pb_child[pb_child["source"] == src]) if pb_child is not None else "?"
    say(f"  {src:12}  {nc:>8,}  {nd:>9}")

# Age bin breakdown in childhood
say(f"\n--- Excitatory scVI C3+ by source x age bin (childhood) ---")
child_age = (obs_child.groupby(["age_bin", "source"], observed=True)["score_scvi"]
             .mean().unstack("source"))
say(child_age.round(0).to_string())

# Pseudobulk in childhood
if pb_child is not None:
    say(f"\n--- Pseudobulk (per donor, Excitatory, childhood) ---")
    say(f"  {'Source':12}  {'N donors':>9}  {'mean':>10}  {'sem':>8}")
    for src in sources:
        d = pb_child[pb_child["source"] == src]["score_scvi_mean"]
        if len(d) == 0:
            continue
        say(f"  {src:12}  {len(d):>9,}  {d.mean():>10.1f}  "
            f"{d.sem():>8.1f}  (range {d.min():.0f}–{d.max():.0f})")
    pb_child["age_bin"] = pd.cut(pb_child["age_years"], bins=AGE_BINS,
                                 labels=AGE_LABELS, right=False)
    say(f"\n--- Pseudobulk scVI C3+ by source x age bin (childhood) ---")
    pb_age = (pb_child.groupby(["age_bin", "source"], observed=True)["score_scvi_mean"]
              .agg(["mean", "count"]).unstack("source"))
    say(pb_age["mean"].round(0).to_string())
    say("\n  (N donors per bin)")
    say(pb_age["count"].to_string())

# Regression in childhood
say(f"\n--- Regression: scVI C3+ ~ age + source (Excitatory, childhood) ---")
if len(obs_child) > 100:
    rc_df  = obs_child[["score_scvi", "age_years", "source"]].dropna()
    srcs_c = [s for s in sources if (rc_df["source"] == s).sum() > 0]
    if len(srcs_c) > 1:
        Xc = pd.concat([rc_df[["age_years"]],
                        pd.get_dummies(rc_df["source"], drop_first=True, dtype=float)],
                       axis=1)
        cc, r2c = ols_summary(rc_df["score_scvi"].values, Xc)
        say(f"  R2 = {r2c:.4f}")
        say(cc.to_string(float_format=lambda x: f"{x:10.1f}"))

# QC metrics in childhood
say(f"\n--- QC metrics (childhood Excitatory cells only) ---")
say(f"  {'Source':12}  {'log_lib':>8}  {'n_genes':>8}  {'c3_det%':>8}")
for src in sources:
    d = obs_child[obs_child["source"] == src]
    if len(d) == 0:
        continue
    say(f"  {src:12}  {d['log_total_counts'].mean():>8.2f}  "
        f"{d['n_genes_detected'].mean():>8.0f}  "
        f"{d['c3_gene_detection'].mean() * 100:>7.1f}%")

# Genetic ancestry in childhood
say(f"\n--- Genetic ancestry in childhood (Excitatory donors) ---")
if pb_child is not None and "genetic_ancestry" in obs_child.columns:
    anc_child = (obs_child.groupby(["source", "genetic_ancestry"])
                 .size().unstack(fill_value=0))
    say(anc_child.to_string())

# Contamination check: re-run excluding VEL cells mapped to Inhibitory subclass
say(f"\n--- Contamination check: exclude VEL Excitatory cells with Inhibitory subclass ---")
obs_child_clean = obs_child[~obs_child.get("vel_subclass_inconsistent",
                                            pd.Series(False, index=obs_child.index))]
n_before = len(obs_child)
n_after  = len(obs_child_clean)
say(f"  Childhood excitatory cells: {n_before:,} -> {n_after:,} after exclusion")
say(f"  {'Source':12}  {'N (before)':>12}  {'N (after)':>10}  "
    f"{'mean_scvi (before)':>20}  {'mean_scvi (after)':>18}")
for src in sources:
    db = obs_child[obs_child["source"] == src]
    da = obs_child_clean[obs_child_clean["source"] == src]
    say(f"  {src:12}  {len(db):>12,}  {len(da):>10,}  "
        f"{db['score_scvi'].mean() if len(db) else float('nan'):>20.1f}  "
        f"{da['score_scvi'].mean() if len(da) else float('nan'):>18.1f}")

# Top genes in childhood — which genes are higher in VEL?
say(f"\n--- Top C3+ genes: VEL vs AGING mean scVI expression (childhood Excitatory) ---")
scvi_full   = to_dense(adata.layers["scvi_normalized"])

# child_idx   = np.where(excit_mask & child_mask)[0]
child_mask_full = (
    excit_mask &
    (obs["age_years"] >= CHILDHOOD[0]) &
    (obs["age_years"] < CHILDHOOD[1])
)
child_idx = np.where(child_mask_full)[0]

c3_gene_idx = np.where(w_c3pos != 0)[0]
scvi_child  = scvi_full[child_idx][:, c3_gene_idx]
del scvi_full

obs_child2 = obs_ex[child_mask]   # aligned with scvi_child rows
child_gene_means = {}
for src in sources:
    smask = obs_child2["source"].values == src
    if smask.sum() > 0:
        child_gene_means[src] = scvi_child[smask].mean(axis=0)

c3pos_ensembl = adata.var_names[w_c3pos != 0]
c3pos_weights = w_c3pos[w_c3pos != 0]
if "feature_name" in adata.var.columns:
    symbols = adata.var.loc[c3pos_ensembl, "feature_name"].values
elif "gene_symbols" in adata.var.columns:
    symbols = adata.var.loc[c3pos_ensembl, "gene_symbols"].values
else:
    symbols = c3pos_ensembl

gdf = pd.DataFrame(child_gene_means, index=c3pos_ensembl)
gdf["weight"]  = c3pos_weights
gdf["symbol"]  = symbols

avail_src = list(child_gene_means.keys())
if "VELMESHEV" in avail_src and "AGING" in avail_src:
    gdf["VEL_minus_AGING"] = gdf["VELMESHEV"] - gdf["AGING"]
    gdf["abs_diff_x_w"]    = gdf["VEL_minus_AGING"].abs() * gdf["weight"].abs()
    top20 = gdf.nlargest(20, "abs_diff_x_w")
    say(f"\n  Top 20 genes by |VEL-AGING| x |weight| (childhood Excitatory):")
    say(f"  {'gene':15}  {'weight':>7}  " +
        "  ".join(f"{s:>10}" for s in avail_src) + "  VEL-AGING")
    for _, row in top20.iterrows():
        vals = "  ".join(f"{row[s]:>10.5f}" for s in avail_src if s in row.index)
        say(f"  {str(row['symbol'])[:15]:15}  {row['weight']:>7.4f}  "
            f"{vals}  {row['VEL_minus_AGING']:>+9.5f}")

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines))
print(f"\nText saved to {OUT_TXT}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: main 2x3 (all ages)
# ═══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 1…", flush=True)
rng  = np.random.RandomState(RANDOM_SEED)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.subplots_adjust(hspace=0.35, wspace=0.3, left=0.07, right=0.97,
                    top=0.93, bottom=0.07)

for col, (sc_col, title) in enumerate([
        ("score_raw",  "C3+ — Raw (CPM)"),
        ("score_scvi", "C3+ — scVI corrected"),
]):
    ax = axes[0, col]
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Age (years)", fontsize=8)
    ax.set_ylabel("C3+ score", fontsize=8)
    ax.axvline(0, color="#CCC", lw=0.6, ls="--")
    for src in sources:
        d = obs_ex[obs_ex["source"] == src]
        n = min(3000, len(d))
        ix = rng.choice(len(d), n, replace=False)
        ax.scatter(d["age_years"].values[ix], d[sc_col].values[ix],
                   c=src_col[src], s=1.5, alpha=0.25, rasterized=True,
                   linewidths=0, label=src)
        d2 = d.copy(); d2["_b"] = pd.cut(d2["age_years"], bins=20)
        mn = d2.groupby("_b", observed=True)[sc_col].mean()
        ax.plot([iv.mid for iv in mn.index], mn.values,
                color=src_col[src], lw=1.5)
    ax.legend(fontsize=6, markerscale=4, frameon=False)

ax = axes[0, 2]
ax.set_title("Mean scVI C3+ by age bin (Excitatory)", fontsize=9)
pivot = (obs_ex.groupby(["age_bin", "source"], observed=True)["score_scvi"]
         .mean().unstack("source"))
x = np.arange(len(pivot)); bw = 0.8 / len(sources)
for i, src in enumerate(sources):
    if src in pivot.columns:
        ax.bar(x + i*bw - 0.4 + bw/2, pivot[src].fillna(0),
               width=bw, color=src_col[src], alpha=0.8, label=src)
ax.set_xticks(x)
ax.set_xticklabels(pivot.index.astype(str), rotation=45, ha="right", fontsize=6)
ax.legend(fontsize=6, frameon=False)

ax = axes[1, 0]
if "cell_subclass" in obs_ex.columns:
    ax.set_title("Excit. subclass composition by source", fontsize=9)
    comp     = obs_ex.groupby(["source", "cell_subclass"]).size().unstack(fill_value=0)
    comp_pct = (comp.T / comp.sum(axis=1) * 100).T
    tab20    = list(plt.cm.tab20.colors)
    sub_cols = {s: tab20[i % 20] for i, s in enumerate(comp_pct.columns)}
    bot = np.zeros(len(sources)); x = np.arange(len(sources))
    for sub in comp_pct.columns:
        vals = comp_pct.reindex(sources)[sub].fillna(0).values
        ax.bar(x, vals, bottom=bot, color=sub_cols[sub], alpha=0.85, label=sub)
        bot += vals
    ax.set_xticks(x); ax.set_xticklabels(sources, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% Excitatory", fontsize=8)
    ax.legend(fontsize=5, loc="upper right", frameon=False, ncol=2)
else:
    ax.text(0.5, 0.5, "cell_subclass not available",
            ha="center", va="center", transform=ax.transAxes)

ax = axes[1, 1]
ax.set_title("C3+ gene detection rate by source", fontsize=9)
bp = ax.boxplot([obs[obs["source"] == s]["c3_gene_detection"].dropna().values
                 for s in sources],
                labels=sources, patch_artist=True, flierprops=dict(markersize=1))
for patch, src in zip(bp["boxes"], sources):
    patch.set_facecolor(src_col[src]); patch.set_alpha(0.8)
ax.set_ylabel("Fraction C3+ genes detected", fontsize=8)
ax.tick_params(axis="x", labelrotation=20, labelsize=8)

ax = axes[1, 2]
ax.set_title("Genetic ancestry per source", fontsize=9)
if "genetic_ancestry" in obs.columns:
    anc = obs.groupby(["source", "genetic_ancestry"]).size().unstack(fill_value=0)
    anc_pct = (anc.T / anc.sum(axis=1) * 100).T
    anc_colors = {"European": "#4393C3", "African": "#D6604D",
                  "East Asian": "#74C476", "South Asian": "#FD8D3C",
                  "unknown": "#BBBBBB"}
    bot = np.zeros(len(sources)); x = np.arange(len(sources))
    for anc_label in anc_pct.columns:
        vals = anc_pct.reindex(sources)[anc_label].fillna(0).values
        ax.bar(x, vals, bottom=bot,
               color=anc_colors.get(anc_label, "#BBBBBB"), alpha=0.85, label=anc_label)
        bot += vals
    ax.set_xticks(x); ax.set_xticklabels(sources, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% cells", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
else:
    ax.text(0.5, 0.5, "no ancestry data", ha="center", va="center",
            transform=ax.transAxes)

fig.savefig(OUT_FIG1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure 1 saved to {OUT_FIG1}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: pseudobulk + subclass (2x3)
# ═══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 2…", flush=True)
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
fig2.subplots_adjust(hspace=0.38, wspace=0.32, left=0.07, right=0.97,
                     top=0.93, bottom=0.07)

pb_all = pseudobulk(obs_ex, individual_col)
if pb_all is not None:
    for col, (sc_col, title) in enumerate([
            ("score_raw_mean",  "Pseudobulk C3+ — Raw"),
            ("score_scvi_mean", "Pseudobulk C3+ — scVI"),
    ]):
        ax = axes2[0, col]
        ax.set_title(title + " (per donor, Excitatory)", fontsize=9)
        ax.set_xlabel("Donor median age (years)", fontsize=8)
        ax.set_ylabel("Mean C3+ per donor", fontsize=8)
        ax.axvline(0, color="#CCC", lw=0.6, ls="--")
        for src in sources:
            d = pb_all[pb_all["source"] == src]
            ax.scatter(d["age_years"], d[sc_col], c=src_col[src],
                       s=18, alpha=0.7, linewidths=0, label=src, zorder=3)
            if len(d) >= 4:
                ix = np.argsort(d["age_years"].values)
                xs = d["age_years"].values[ix]; ys = d[sc_col].values[ix]
                wsz = max(3, len(d) // 6)
                ys_sm = np.convolve(ys, np.ones(wsz)/wsz, mode="valid")
                xs_sm = xs[wsz//2: wsz//2 + len(ys_sm)]
                ax.plot(xs_sm, ys_sm, color=src_col[src], lw=1.5, alpha=0.9)
        ax.legend(fontsize=7, markerscale=1.5, frameon=False)
    ax = axes2[0, 2]
    ax.set_title("Pseudobulk scVI C3+ — boxplot (Excitatory)", fontsize=9)
    bdata = [pb_all[pb_all["source"] == s]["score_scvi_mean"].dropna().values
             for s in sources]
    bp2 = ax.boxplot(bdata, labels=sources, patch_artist=True,
                     flierprops=dict(markersize=2))
    for patch, src in zip(bp2["boxes"], sources):
        patch.set_facecolor(src_col[src]); patch.set_alpha(0.8)
    for i, src in enumerate(sources):
        d = pb_all[pb_all["source"] == src]["score_scvi_mean"].dropna()
        jx = rng.normal(i + 1, 0.07, len(d))
        axes2[0, 2].scatter(jx, d, c=src_col[src], s=4, alpha=0.5,
                            zorder=3, linewidths=0)
    ax.set_ylabel("Mean C3+ per donor", fontsize=8)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)

ax = axes2[1, 0]
ax.set_title("C3+ (scVI) vs age — coloured by subclass (Excitatory)", fontsize=9)
ax.set_xlabel("Age (years)", fontsize=8); ax.set_ylabel("C3+ score", fontsize=8)
ax.axvline(0, color="#CCC", lw=0.6, ls="--")
if "cell_subclass" in obs_ex.columns:
    present = [s for s in EXCIT_SUBCLASSES if s in obs_ex["cell_subclass"].values]
    others  = sorted(s for s in obs_ex["cell_subclass"].unique()
                     if s not in EXCIT_SUBCLASSES)
    all_subs = present + others
    sc_colors = {s: (list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors))[i % 40]
                 for i, s in enumerate(all_subs)}
    for sub in all_subs:
        d = obs_ex[obs_ex["cell_subclass"] == sub]
        if len(d) == 0: continue
        ix = rng.choice(len(d), min(1500, len(d)), replace=False)
        ax.scatter(d["age_years"].values[ix], d["score_scvi"].values[ix],
                   c=[sc_colors[sub]], s=1.2, alpha=0.2, rasterized=True,
                   linewidths=0, label=sub)
    ax.legend(fontsize=5, markerscale=4, frameon=False, ncol=2, loc="upper left")
else:
    ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)

ax = axes2[1, 1]
ax.set_title("Mean scVI C3+ per subclass x source", fontsize=9)
if "cell_subclass" in obs_ex.columns:
    ss = (obs_ex.groupby(["cell_subclass", "source"])["score_scvi"]
          .mean().unstack("source"))
    keep = obs_ex.groupby("cell_subclass").size()
    ss   = ss.loc[keep[keep >= 50].index].dropna(how="all")
    x = np.arange(len(ss)); bw = 0.8 / len(sources)
    for i, src in enumerate(sources):
        if src in ss.columns:
            ax.bar(x + i*bw - 0.4 + bw/2, ss[src].fillna(0),
                   width=bw, color=src_col[src], alpha=0.8, label=src)
    ax.set_xticks(x)
    ax.set_xticklabels(ss.index.astype(str), rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("Mean C3+ score", fontsize=8)
    ax.legend(fontsize=6, frameon=False)

ax = axes2[1, 2]
ax.set_title("Excit. subclass fraction by age bin", fontsize=9)
if "cell_subclass" in obs_ex.columns:
    sa  = (obs_ex.groupby(["age_bin", "cell_subclass"], observed=True)
           .size().unstack(fill_value=0))
    sap = (sa.T / sa.sum(axis=1) * 100).T
    top8 = obs_ex["cell_subclass"].value_counts().head(8).index.tolist()
    sap2 = sap[[s for s in top8 if s in sap.columns]].copy()
    sap2["Other"] = 100 - sap2.sum(axis=1)
    bot = np.zeros(len(sap2)); x = np.arange(len(sap2))
    tab20 = list(plt.cm.tab20.colors)
    for j, sub in enumerate(sap2.columns):
        ax.bar(x, sap2[sub].values, bottom=bot, color=tab20[j % 20], alpha=0.85, label=sub)
        bot += sap2[sub].values
    ax.set_xticks(x)
    ax.set_xticklabels(sap2.index.astype(str), rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("% Excitatory", fontsize=8)
    ax.legend(fontsize=5, frameon=False, ncol=2)

fig2.savefig(OUT_FIG2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Figure 2 saved to {OUT_FIG2}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: childhood-focused (2x3)
# ═══════════════════════════════════════════════════════════════════════════════

print("Plotting Figure 3 (childhood)…", flush=True)
fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
fig3.subplots_adjust(hspace=0.38, wspace=0.32, left=0.07, right=0.97,
                     top=0.93, bottom=0.07)

# 3A: scatter cell-level, childhood, colour=source
ax = axes3[0, 0]
ax.set_title(f"C3+ scVI vs age — source (childhood Excitatory)", fontsize=9)
ax.set_xlabel("Age (years)", fontsize=8); ax.set_ylabel("C3+ score", fontsize=8)
for src in sources:
    d = obs_child[obs_child["source"] == src]
    if len(d) == 0: continue
    ix = rng.choice(len(d), min(3000, len(d)), replace=False)
    ax.scatter(d["age_years"].values[ix], d["score_scvi"].values[ix],
               c=src_col[src], s=2, alpha=0.3, rasterized=True, linewidths=0, label=src)
    d2 = d.copy(); d2["_b"] = pd.cut(d2["age_years"], bins=10)
    mn = d2.groupby("_b", observed=True)["score_scvi"].mean()
    ax.plot([iv.mid for iv in mn.index], mn.values, color=src_col[src], lw=2)
ax.legend(fontsize=7, markerscale=3, frameon=False)

# 3B: pseudobulk scatter, childhood, colour=source
ax = axes3[0, 1]
ax.set_title("Pseudobulk scVI C3+ — childhood donors", fontsize=9)
ax.set_xlabel("Donor median age (years)", fontsize=8)
ax.set_ylabel("Mean C3+ per donor", fontsize=8)
if pb_child is not None:
    for src in sources:
        d = pb_child[pb_child["source"] == src]
        if len(d) == 0: continue
        ax.scatter(d["age_years"], d["score_scvi_mean"],
                   c=src_col[src], s=30, alpha=0.8, linewidths=0,
                   label=f"{src} (n={len(d)})", zorder=3)
    ax.legend(fontsize=7, frameon=False)

# 3C: mean C3+ per source x age bin (childhood), with error bars from pseudobulk
ax = axes3[0, 2]
ax.set_title("Pseudobulk C3+ by source x age bin (childhood)", fontsize=9)
ax.set_xlabel("Age bin", fontsize=8); ax.set_ylabel("Mean C3+ per donor", fontsize=8)
if pb_child is not None:
    pb_agg = (pb_child.groupby(["source", "age_bin"], observed=True)["score_scvi_mean"]
              .agg(["mean", "sem", "count"]).reset_index())
    age_bins_c = sorted(pb_child["age_bin"].dropna().unique().tolist(),
                        key=lambda x: AGE_LABELS.index(str(x))
                        if str(x) in AGE_LABELS else 99)
    x = np.arange(len(age_bins_c)); bw = 0.8 / len(sources)
    for i, src in enumerate(sources):
        d = pb_agg[pb_agg["source"] == src].set_index("age_bin")
        vals = [d.loc[b, "mean"] if b in d.index else np.nan for b in age_bins_c]
        errs = [d.loc[b, "sem"]  if b in d.index else 0       for b in age_bins_c]
        ns   = [int(d.loc[b, "count"]) if b in d.index else 0 for b in age_bins_c]
        ax.bar(x + i*bw - 0.4 + bw/2, vals, width=bw,
               yerr=errs, capsize=3,
               color=src_col[src], alpha=0.8, label=src)
        for xi, (v, n) in enumerate(zip(vals, ns)):
            if not np.isnan(v):
                ax.text(xi + i*bw - 0.4 + bw/2, v + 500, str(n),
                        ha="center", va="bottom", fontsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in age_bins_c], rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=6, frameon=False)

# 3D: QC comparison in childhood
ax = axes3[1, 0]
ax.set_title("n_genes per cell — childhood Excitatory", fontsize=9)
ax.set_xlabel("Source", fontsize=8); ax.set_ylabel("Genes detected per cell", fontsize=8)
qc_data = [obs_child[obs_child["source"] == s]["n_genes_detected"].dropna().values
           for s in sources if (obs_child["source"] == s).sum() > 0]
qc_srcs = [s for s in sources if (obs_child["source"] == s).sum() > 0]
bp3 = ax.boxplot(qc_data, labels=qc_srcs, patch_artist=True,
                 flierprops=dict(markersize=1))
for patch, src in zip(bp3["boxes"], qc_srcs):
    patch.set_facecolor(src_col[src]); patch.set_alpha(0.8)
ax.tick_params(axis="x", labelrotation=20, labelsize=8)

# 3E: C3+ gene detection in childhood
ax = axes3[1, 1]
ax.set_title("C3+ gene detection — childhood Excitatory", fontsize=9)
ax.set_xlabel("Source", fontsize=8)
ax.set_ylabel("Fraction C3+ genes detected", fontsize=8)
det_data = [obs_child[obs_child["source"] == s]["c3_gene_detection"].dropna().values
            for s in qc_srcs]
bp4 = ax.boxplot(det_data, labels=qc_srcs, patch_artist=True,
                 flierprops=dict(markersize=1))
for patch, src in zip(bp4["boxes"], qc_srcs):
    patch.set_facecolor(src_col[src]); patch.set_alpha(0.8)
ax.tick_params(axis="x", labelrotation=20, labelsize=8)

# 3F: Top genes, VEL vs AGING, childhood
ax = axes3[1, 2]
ax.set_title("Top C3+ genes: VEL−AGING (childhood Excit.)", fontsize=9)
if "VELMESHEV" in child_gene_means and "AGING" in child_gene_means:
    top15 = gdf.nlargest(15, "abs_diff_x_w")
    y_pos = np.arange(len(top15))
    for i, src in enumerate([s for s in sources if s in child_gene_means]):
        vals = top15[src].values if src in top15.columns else np.zeros(len(top15))
        ax.barh(y_pos - i*0.18 + 0.27, vals, height=0.18,
                color=src_col[src], alpha=0.8, label=src)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15["symbol"].astype(str).values, fontsize=6)
    ax.set_xlabel("Mean scVI expression", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, "insufficient sources", ha="center", va="center",
            transform=ax.transAxes)

fig3.savefig(OUT_FIG3, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"Figure 3 saved to {OUT_FIG3}", flush=True)

print("\nDone.", flush=True)
