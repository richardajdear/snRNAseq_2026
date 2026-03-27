"""
Diagnose the apparent source-level difference in ahbaC3 GRN scores
even after scVI batch correction.

Checks:
  1. Is the effect present in both raw and scVI-corrected expression?
  2. Does it persist after controlling for age and cell subclass in a regression?
  3. Is it driven by differences in cell subclass composition per source?
  4. Is it correlated with QC metrics (library size, n_genes detected)?
  5. Which individual GRN genes are driving the between-source difference?
  6. Is the effect visible in pseudobulk (donor-level) data?
  7. How does it look stratified by cell subclass?

Outputs:
  diagnose_grn_batch_effect.png        — main 2x3 panel
  diagnose_grn_batch_effect_pb.png     — pseudobulk + subclass panels
  diagnose_grn_batch_effect.txt        — text summary
"""

import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.sparse import issparse
from scipy import stats

# -- Paths --------------------------------------------------------------------

CODE_DIR = "/home/rajd2/rds/hpc-work/snRNAseq_2026/code"
sys.path.insert(0, CODE_DIR)

DATA_FILE = (
    "/home/rajd2/rds/rds-cam-psych-transc-Pb9UGUlrwWc"
    "/Cam_snRNAseq/combined/VelWangPsychad_100k_PFC_lessOld"
    "/scvi_output/integrated.h5ad"
)
GRN_FILE = "/home/rajd2/rds/hpc-work/snRNAseq_2026/reference/ahba_dme_hcp_top8kgenes_weights.csv"
OUTPUT_FIG    = "/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/diagnose_grn_batch_effect.png"
OUTPUT_FIG_PB = "/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/diagnose_grn_batch_effect_pb.png"
OUTPUT_TEXT   = "/home/rajd2/rds/hpc-work/snRNAseq_2026/scripts/outputs/diagnose_grn_batch_effect.txt"

RANDOM_SEED = 42
AGE_BINS   = [-1, 0, 1, 3, 6, 10, 15, 20, 30, 40, 100]
AGE_LABELS = ["prenatal", "0-1", "1-3", "3-6", "6-10",
              "10-15", "15-20", "20-30", "30-40", "40+"]

SET1 = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3",
        "#FF7F00", "#A65628", "#F781BF", "#999999"]

# Major excitatory subclasses to highlight in subclass plots
EXCIT_SUBCLASSES = [
    "EN_L2_3_IT", "EN_L4_IT", "EN_L5_IT", "EN_L5_ET",
    "EN_L5_6_NP", "EN_L6_IT", "EN_L6_CT", "EN_L6B",
    "EN_Immature", "Excitatory",
]


# -- Helpers ------------------------------------------------------------------

def to_dense(mat):
    return mat.toarray() if issparse(mat) else np.asarray(mat)

def cpm(counts):
    s = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(s, 1) * 1e6

def grn_score(expr_cpm, weights_aligned):
    return expr_cpm @ weights_aligned

def ols_summary(y, X_df):
    from numpy.linalg import lstsq
    X = X_df.values.astype(float)
    X = np.column_stack([np.ones(len(y)), X])
    cols = ["intercept"] + list(X_df.columns)
    b, _, _, _ = lstsq(X, y, rcond=None)
    y_hat = X @ b
    resid = y - y_hat
    n, p = X.shape
    s2 = (resid**2).sum() / (n - p)
    var_b = s2 * np.linalg.inv(X.T @ X).diagonal()
    se = np.sqrt(var_b)
    t  = b / se
    pv = 2 * stats.t.sf(np.abs(t), df=n - p)
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1 - (resid**2).sum() / ss_tot
    df = pd.DataFrame({"coef": b, "se": se, "t": t, "p": pv}, index=cols)
    return df, r2


# -- Import subclass mapping --------------------------------------------------

try:
    from read_data import map_cellxgene_subclass, collapse_en_subclass
    print("Imported subclass mapping from read_data.py", flush=True)
    _SUBCLASS_IMPORT_OK = True
except Exception as e:
    print(f"Warning: could not import from read_data.py ({e}); subclass unavailable",
          flush=True)
    _SUBCLASS_IMPORT_OK = False


# -- Load data ----------------------------------------------------------------

print("Loading data...", flush=True)
adata = sc.read_h5ad(DATA_FILE)
print(f"Shape: {adata.shape}")
print(f"Layers: {list(adata.layers.keys())}")
print(f"obs cols: {list(adata.obs.columns)}", flush=True)


# -- Derive cell_subclass from cell_type (CellxGene ontology) -----------------

if _SUBCLASS_IMPORT_OK and "cell_type" in adata.obs.columns:
    print("Deriving cell_subclass from cell_type...", flush=True)
    adata.obs["cell_subclass"] = (
        adata.obs["cell_type"]
        .map(map_cellxgene_subclass)
        .map(collapse_en_subclass)
    )
    n_mapped = (adata.obs["cell_subclass"] != adata.obs["cell_type"]).sum()
    print(f"  Mapped {n_mapped:,} / {len(adata.obs):,} cells")
    print(f"  Subclass counts:\n{adata.obs['cell_subclass'].value_counts().head(20)}",
          flush=True)
else:
    print("cell_subclass not derivable; proceeding without it", flush=True)


# -- Load and map GRN ---------------------------------------------------------

from regulons import get_ahba_GRN
from gene_mapping import map_grn_symbols_to_ensembl

print("\nLoading GRN...", flush=True)
ahba_GRN = get_ahba_GRN(path_to_ahba_weights=GRN_FILE, use_weights=True)
ahba_GRN = map_grn_symbols_to_ensembl(ahba_GRN, adata)

grn_pivot = ahba_GRN.pivot_table(
    index="Network", columns="Gene", values="Importance", fill_value=0)

def aligned_weights(network_name):
    w = pd.Series(0.0, index=adata.var_names)
    if network_name in grn_pivot.index:
        for g, v in grn_pivot.loc[network_name].items():
            if g in w.index and v != 0:
                w[g] = v
    return w.values.astype(np.float32)

w_c3pos = aligned_weights("C3+")
w_c3neg = aligned_weights("C3-")
n_c3pos = (w_c3pos != 0).sum()
print(f"C3+ genes mapped: {n_c3pos}", flush=True)


# -- Compute GRN scores -------------------------------------------------------

print("\nComputing raw C3+ scores...", flush=True)
raw_counts = to_dense(adata.layers["counts"]).astype(np.float32)
raw_cpm    = cpm(raw_counts)
score_raw  = grn_score(raw_cpm, w_c3pos)
del raw_counts, raw_cpm

print("Computing scVI-corrected C3+ scores...", flush=True)
scvi_expr  = to_dense(adata.layers["scvi_normalized"]).astype(np.float32)
scvi_cpm   = cpm(scvi_expr)
score_scvi = grn_score(scvi_cpm, w_c3pos)
del scvi_expr, scvi_cpm


# -- Build metadata frame -----------------------------------------------------

obs = adata.obs.copy()
obs["score_raw"]  = score_raw
obs["score_scvi"] = score_scvi
obs["age_bin"] = pd.cut(obs["age_years"], bins=AGE_BINS, labels=AGE_LABELS,
                        right=False)

# QC metrics
raw_counts_qc = to_dense(adata.layers["counts"])
obs["log_total_counts"] = np.log1p(raw_counts_qc.sum(axis=1))
obs["n_genes_detected"]  = (raw_counts_qc > 0).sum(axis=1)
c3pos_idx = np.where(w_c3pos != 0)[0]
obs["c3_gene_detection"] = (raw_counts_qc[:, c3pos_idx] > 0).mean(axis=1)
del raw_counts_qc

# Excitatory-only mask
excit_mask = obs["cell_class"] == "Excitatory"
print(f"\nExcitatory cells: {excit_mask.sum():,} / {len(obs):,}", flush=True)

obs_ex = obs[excit_mask].copy()
sources = sorted(obs_ex["source"].unique())
src_colors = {s: SET1[i] for i, s in enumerate(sources)}

# Detect individual/donor column
individual_col = None
for col in ["individual", "donor_id", "donor", "subject"]:
    if col in obs.columns:
        individual_col = col
        break
print(f"Individual column: {individual_col}", flush=True)


# -- Text output --------------------------------------------------------------

lines = []
def say(s=""):
    print(s, flush=True)
    lines.append(s)

say("=" * 70)
say("DIAGNOSIS: ahbaC3 GRN C3+ score -- source differences")
say("=" * 70)

# 1. Overall mean by source
for label, df in [("All cells", obs), ("Excitatory only", obs_ex)]:
    say(f"\n--- {label}: mean C3+ score by source ---")
    say(f"  {'Source':12s}  {'N':>7}  {'raw_mean':>10}  {'scvi_mean':>10}")
    for src in sources:
        d = df[df["source"] == src]
        say(f"  {src:12s}  {len(d):>7,}  "
            f"{d['score_raw'].mean():>10.4f}  {d['score_scvi'].mean():>10.4f}")

# 2. Age-stratified comparison
say("\n--- Excitatory: mean scVI C3+ score by source x age bin ---")
age_src = (obs_ex.groupby(["age_bin", "source"], observed=True)["score_scvi"]
           .agg(["mean", "count"]).unstack("source"))
say(age_src["mean"].round(4).to_string())

# 3. Regression: C3+ ~ age + source
say("\n--- Regression: C3+ ~ age_years + source (Excitatory, scVI) ---")
reg_df = obs_ex[["score_scvi", "age_years", "source"]].dropna()
dummies = pd.get_dummies(reg_df["source"], drop_first=True, dtype=float)
X_reg = pd.concat([reg_df[["age_years"]], dummies], axis=1)
coefs, r2 = ols_summary(reg_df["score_scvi"].values, X_reg)
say(f"  R2 = {r2:.4f}")
say(coefs.to_string(float_format=lambda x: f"{x:10.4f}"))

# 4. Regression: C3+ ~ age + cell_subclass + source
say("\n--- Regression: C3+ ~ age + cell_subclass + source (Excitatory, scVI) ---")
if "cell_subclass" in obs_ex.columns:
    reg2_df = obs_ex[["score_scvi", "age_years", "cell_subclass", "source"]].dropna()
    sub_dummies = pd.get_dummies(reg2_df["cell_subclass"], drop_first=True, dtype=float)
    src_dummies = pd.get_dummies(reg2_df["source"], drop_first=True, dtype=float)
    X_reg2 = pd.concat([reg2_df[["age_years"]], sub_dummies, src_dummies], axis=1)
    coefs2, r2_2 = ols_summary(reg2_df["score_scvi"].values, X_reg2)
    say(f"  R2 = {r2_2:.4f}")
    show_rows = [r for r in coefs2.index
                 if r.startswith(tuple(sources)) or r in ("intercept", "age_years")]
    say(coefs2.loc[show_rows].to_string(float_format=lambda x: f"{x:10.4f}"))
else:
    say("  (cell_subclass not available)")

# 5. QC metrics by source
say("\n--- QC metrics by source (all cells) ---")
say(f"  {'Source':12s}  {'log_lib_size':>13}  {'n_genes':>8}  "
    f"{'c3_detection':>13}  {'chemistry':>10}")
for src in sources:
    d = obs[obs["source"] == src]
    chem = d["chemistry"].value_counts().index[0] if "chemistry" in d.columns else "?"
    say(f"  {src:12s}  "
        f"{d['log_total_counts'].mean():>13.3f}  "
        f"{d['n_genes_detected'].mean():>8.0f}  "
        f"{d['c3_gene_detection'].mean():>13.4f}  "
        f"{str(chem):>10}")

# 6. Cell subclass composition
if "cell_subclass" in obs_ex.columns:
    say("\n--- Cell subclass composition (Excitatory, %) by source ---")
    comp = (obs_ex.groupby(["source", "cell_subclass"])
            .size().unstack(fill_value=0))
    comp_pct = (comp.T / comp.sum(axis=1) * 100).T.round(1)
    say(comp_pct.to_string())

# 7. Pseudobulk summary
if individual_col is not None:
    say(f"\n--- Pseudobulk (per-donor mean C3+ scVI, Excitatory) by source ---")
    pb = (obs_ex.groupby([individual_col, "source"])
          .agg(score_scvi_mean=("score_scvi", "mean"),
               score_raw_mean=("score_raw", "mean"),
               age_years=("age_years", "median"),
               n_cells=(individual_col, "count"))
          .reset_index())
    say(f"  Total donors: {len(pb)}")
    say(f"  {'Source':12s}  {'N donors':>9}  {'mean_pb_scvi':>13}  {'mean_pb_raw':>12}")
    for src in sources:
        d = pb[pb["source"] == src]
        say(f"  {src:12s}  {len(d):>9,}  "
            f"{d['score_scvi_mean'].mean():>13.4f}  "
            f"{d['score_raw_mean'].mean():>12.4f}")

    # Pseudobulk age-stratified
    say(f"\n--- Pseudobulk C3+ scVI by source x age_bin (Excitatory donors) ---")
    pb["age_bin"] = pd.cut(pb["age_years"], bins=AGE_BINS, labels=AGE_LABELS,
                           right=False)
    pb_age = (pb.groupby(["age_bin", "source"], observed=True)["score_scvi_mean"]
              .agg(["mean", "count"]).unstack("source"))
    say(pb_age["mean"].round(4).to_string())

# 8. Top genes driving between-source differences
say("\n--- Top 20 C3+ genes: mean expression by source (scVI, Excitatory) ---")
c3pos_ensembl = adata.var_names[w_c3pos != 0]
c3pos_weights = w_c3pos[w_c3pos != 0]
scvi_full = to_dense(adata.layers["scvi_normalized"])
ex_idx = np.where(excit_mask)[0]
scvi_ex = scvi_full[ex_idx][:, np.where(w_c3pos != 0)[0]]
del scvi_full

gene_means = {}
for src in sources:
    src_mask = obs_ex["source"].values == src
    gene_means[src] = scvi_ex[src_mask].mean(axis=0)

gene_means_df = pd.DataFrame(gene_means, index=c3pos_ensembl)
gene_means_df["weight"] = c3pos_weights
gene_means_df["source_var"] = gene_means_df[sources].var(axis=1)
gene_means_df["weighted_var"] = (gene_means_df["source_var"]
                                  * np.abs(gene_means_df["weight"]))

if "feature_name" in adata.var.columns:
    gene_means_df["symbol"] = adata.var.loc[c3pos_ensembl, "feature_name"].values
elif "gene_symbols" in adata.var.columns:
    gene_means_df["symbol"] = adata.var.loc[c3pos_ensembl, "gene_symbols"].values
else:
    gene_means_df["symbol"] = gene_means_df.index

top_genes = gene_means_df.nlargest(20, "weighted_var")
say(f"\n  Top 20 genes by (between-source variance x |weight|):")
say(f"  {'gene':15s}  {'weight':>8}  " + "  ".join(f"{s:>12}" for s in sources))
for _, row in top_genes.iterrows():
    say(f"  {str(row['symbol'])[:15]:15s}  {row['weight']:>8.4f}  " +
        "  ".join(f"{row[s]:>12.4f}" for s in sources))

with open(OUTPUT_TEXT, "w") as f:
    f.write("\n".join(lines))
print(f"\nText report saved to {OUTPUT_TEXT}", flush=True)


# -- Figure 1: Main diagnosis (2x3) -------------------------------------------

print("\nPlotting Figure 1 (main)...", flush=True)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.subplots_adjust(hspace=0.35, wspace=0.3, left=0.07, right=0.97,
                    top=0.93, bottom=0.07)

rng = np.random.RandomState(RANDOM_SEED)

# Panels A/B: C3+ vs age by source (raw, scVI)
for col, (score_col, title) in enumerate([
        ("score_raw",  "C3+ score -- Raw (CPM)"),
        ("score_scvi", "C3+ score -- scVI (transform_batch=VELMESHEV)"),
]):
    ax = axes[0, col]
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Age (years)", fontsize=8)
    ax.set_ylabel("C3+ score", fontsize=8)
    ax.axvline(0, color="#CCCCCC", linewidth=0.6, linestyle="--")
    for src in sources:
        d = obs_ex[obs_ex["source"] == src]
        n = min(3000, len(d))
        idx = rng.choice(len(d), n, replace=False)
        ax.scatter(d["age_years"].values[idx], d[score_col].values[idx],
                   c=src_colors[src], s=1.5, alpha=0.25, rasterized=True,
                   linewidths=0, label=src)
        d2 = d.copy()
        d2["age_bin2"] = pd.cut(d2["age_years"], bins=20)
        mn = d2.groupby("age_bin2", observed=True)[score_col].mean()
        cx = [iv.mid for iv in mn.index]
        ax.plot(cx, mn.values, color=src_colors[src], linewidth=1.5)
    ax.legend(fontsize=6, markerscale=4, frameon=False)

# Panel C: Mean scVI score by age bin x source
ax = axes[0, 2]
ax.set_title("Mean C3+ (scVI) by age bin -- Excitatory", fontsize=9)
ax.set_xlabel("Age bin", fontsize=8)
ax.set_ylabel("Mean C3+ score", fontsize=8)
pivot = (obs_ex.groupby(["age_bin", "source"], observed=True)["score_scvi"]
         .mean().unstack("source"))
x = np.arange(len(pivot))
w = 0.8 / len(sources)
for i, src in enumerate(sources):
    if src in pivot.columns:
        ax.bar(x + i * w - 0.4 + w/2, pivot[src].fillna(0),
               width=w, color=src_colors[src], alpha=0.8, label=src)
ax.set_xticks(x)
ax.set_xticklabels(pivot.index.astype(str), rotation=45, ha="right", fontsize=6)
ax.legend(fontsize=6, frameon=False)

# Panel D: Cell subclass composition
ax = axes[1, 0]
if "cell_subclass" in obs_ex.columns:
    ax.set_title("Excit. subclass composition by source", fontsize=9)
    comp = (obs_ex.groupby(["source", "cell_subclass"])
            .size().unstack(fill_value=0))
    comp_pct = (comp.T / comp.sum(axis=1) * 100).T
    subclasses = comp_pct.columns.tolist()
    tab20 = plt.cm.tab20.colors
    sub_colors = {s: tab20[i % 20] for i, s in enumerate(subclasses)}
    bottom = np.zeros(len(sources))
    x = np.arange(len(sources))
    for sub in subclasses:
        if sub in comp_pct.columns:
            vals = comp_pct.reindex(sources)[sub].fillna(0).values
            ax.bar(x, vals, bottom=bottom, color=sub_colors[sub],
                   alpha=0.85, label=sub)
            bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("% Excitatory cells", fontsize=8)
    ax.legend(fontsize=5, loc="upper right", frameon=False, ncol=2)
else:
    ax.text(0.5, 0.5, "cell_subclass not available",
            ha="center", va="center", transform=ax.transAxes)

# Panel E: C3+ gene detection rate boxplot
ax = axes[1, 1]
ax.set_title("C3+ gene detection rate by source", fontsize=9)
data_for_box = [obs[obs["source"] == src]["c3_gene_detection"].dropna().values
                for src in sources]
bp = ax.boxplot(data_for_box, labels=sources, patch_artist=True,
                flierprops=dict(markersize=1))
for patch, src in zip(bp["boxes"], sources):
    patch.set_facecolor(src_colors[src])
    patch.set_alpha(0.8)
ax.set_xlabel("Source", fontsize=8)
ax.set_ylabel("Fraction C3+ genes detected per cell", fontsize=8)
ax.tick_params(axis='x', labelrotation=20, labelsize=8)

# Panel F: Top genes
ax = axes[1, 2]
ax.set_title("Top 15 C3+ genes (between-source var x |weight|)", fontsize=9)
top15 = gene_means_df.nlargest(15, "weighted_var")
y_pos = np.arange(len(top15))
for i, src in enumerate(sources):
    vals = top15[src].values if src in top15.columns else np.zeros(len(top15))
    ax.barh(y_pos - i * 0.18 + 0.27, vals, height=0.18,
            color=src_colors[src], alpha=0.8, label=src)
ax.set_yticks(y_pos)
ax.set_yticklabels(top15["symbol"].astype(str).values, fontsize=6)
ax.set_xlabel("Mean scVI expression (Excitatory)", fontsize=8)
ax.legend(fontsize=6, frameon=False)
ax.invert_yaxis()

from pathlib import Path
Path(OUTPUT_FIG).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure 1 saved to {OUTPUT_FIG}", flush=True)


# -- Figure 2: Pseudobulk + subclass developmental (2x3) ---------------------

print("Plotting Figure 2 (pseudobulk + subclass)...", flush=True)

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
fig2.subplots_adjust(hspace=0.38, wspace=0.32, left=0.07, right=0.97,
                     top=0.93, bottom=0.07)

# Panels 2A/2B: Pseudobulk per-donor C3+ vs age (raw, scVI)
if individual_col is not None:
    pb = (obs_ex.groupby([individual_col, "source"])
          .agg(score_scvi_mean=("score_scvi", "mean"),
               score_raw_mean=("score_raw", "mean"),
               age_years=("age_years", "median"),
               n_cells=(individual_col, "count"))
          .reset_index())

    for col, (score_col, title) in enumerate([
            ("score_raw_mean",  "Pseudobulk C3+ -- Raw (per donor, Excitatory)"),
            ("score_scvi_mean", "Pseudobulk C3+ -- scVI (per donor, Excitatory)"),
    ]):
        ax = axes2[0, col]
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Donor median age (years)", fontsize=8)
        ax.set_ylabel("Mean C3+ score (per donor)", fontsize=8)
        ax.axvline(0, color="#CCCCCC", linewidth=0.6, linestyle="--")
        for src in sources:
            d = pb[pb["source"] == src]
            ax.scatter(d["age_years"], d[score_col],
                       c=src_colors[src], s=20, alpha=0.7,
                       linewidths=0, label=src, zorder=3)
            if len(d) >= 4:
                order = np.argsort(d["age_years"].values)
                xs = d["age_years"].values[order]
                ys = d[score_col].values[order]
                w_size = max(3, len(d) // 6)
                ys_sm = np.convolve(ys, np.ones(w_size) / w_size, mode="valid")
                xs_sm = xs[w_size // 2: w_size // 2 + len(ys_sm)]
                ax.plot(xs_sm, ys_sm, color=src_colors[src], linewidth=1.5, alpha=0.9)
        ax.legend(fontsize=7, markerscale=1.5, frameon=False)

    # Panel 2C: Pseudobulk boxplot per source (scVI)
    ax = axes2[0, 2]
    ax.set_title("Pseudobulk C3+ (scVI) -- Excitatory donors", fontsize=9)
    pb_data = [pb[pb["source"] == src]["score_scvi_mean"].dropna().values
               for src in sources]
    bp2 = ax.boxplot(pb_data, labels=sources, patch_artist=True,
                     flierprops=dict(markersize=2))
    for patch, src in zip(bp2["boxes"], sources):
        patch.set_facecolor(src_colors[src])
        patch.set_alpha(0.8)
    for i, src in enumerate(sources):
        d = pb[pb["source"] == src]["score_scvi_mean"].dropna()
        jx = rng.normal(i + 1, 0.07, len(d))
        ax.scatter(jx, d, c=src_colors[src], s=5, alpha=0.6, zorder=3, linewidths=0)
    ax.set_ylabel("Mean C3+ per donor", fontsize=8)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
else:
    for col in range(3):
        axes2[0, col].text(0.5, 0.5, "individual column not found",
                           ha="center", va="center",
                           transform=axes2[0, col].transAxes)

# Panel 2D: C3+ vs age coloured by cell_subclass (Excitatory)
ax = axes2[1, 0]
ax.set_title("C3+ (scVI) vs age -- coloured by subclass (Excitatory)", fontsize=9)
ax.set_xlabel("Age (years)", fontsize=8)
ax.set_ylabel("C3+ score", fontsize=8)
ax.axvline(0, color="#CCCCCC", linewidth=0.6, linestyle="--")
if "cell_subclass" in obs_ex.columns:
    present_subs = [s for s in EXCIT_SUBCLASSES if s in obs_ex["cell_subclass"].values]
    other_subs = sorted([s for s in obs_ex["cell_subclass"].unique()
                         if s not in EXCIT_SUBCLASSES])
    all_subs = present_subs + other_subs
    tab20_colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    sub_color_map = {s: tab20_colors[i % len(tab20_colors)]
                     for i, s in enumerate(all_subs)}
    n_per_sub = 1500
    for sub in all_subs:
        mask = obs_ex["cell_subclass"] == sub
        d = obs_ex[mask]
        if len(d) == 0:
            continue
        n = min(n_per_sub, len(d))
        idx = rng.choice(len(d), n, replace=False)
        ax.scatter(d["age_years"].values[idx], d["score_scvi"].values[idx],
                   c=[sub_color_map[sub]], s=1.2, alpha=0.2, rasterized=True,
                   linewidths=0, label=sub)
    ax.legend(fontsize=5, markerscale=4, frameon=False, ncol=2, loc="upper left")
else:
    ax.text(0.5, 0.5, "cell_subclass not available",
            ha="center", va="center", transform=ax.transAxes)

# Panel 2E: Mean C3+ per subclass x source (Excitatory)
ax = axes2[1, 1]
ax.set_title("Mean C3+ (scVI) per subclass x source (Excitatory)", fontsize=9)
if "cell_subclass" in obs_ex.columns:
    sub_src = (obs_ex.groupby(["cell_subclass", "source"])["score_scvi"]
               .mean().unstack("source"))
    sub_counts = obs_ex.groupby("cell_subclass").size()
    keep_subs = sub_counts[sub_counts >= 50].index
    sub_src = sub_src.loc[sub_src.index.isin(keep_subs)].dropna(how="all")
    n_subs = len(sub_src)
    x = np.arange(n_subs)
    w = 0.8 / len(sources)
    for i, src in enumerate(sources):
        if src in sub_src.columns:
            ax.bar(x + i * w - 0.4 + w / 2, sub_src[src].fillna(0),
                   width=w, color=src_colors[src], alpha=0.8, label=src)
    ax.set_xticks(x)
    ax.set_xticklabels(sub_src.index.astype(str), rotation=55, ha="right", fontsize=6)
    ax.set_ylabel("Mean C3+ score", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
else:
    ax.text(0.5, 0.5, "cell_subclass not available",
            ha="center", va="center", transform=ax.transAxes)

# Panel 2F: Excit. subclass composition by age bin
ax = axes2[1, 2]
ax.set_title("Excit. subclass fraction by age bin", fontsize=9)
if "cell_subclass" in obs_ex.columns:
    sub_age = (obs_ex.groupby(["age_bin", "cell_subclass"], observed=True)
               .size().unstack(fill_value=0))
    sub_age_pct = (sub_age.T / sub_age.sum(axis=1) * 100).T
    top_subs_total = obs_ex["cell_subclass"].value_counts().head(8).index.tolist()
    sub_age_top = sub_age_pct[[s for s in top_subs_total
                                if s in sub_age_pct.columns]].copy()
    sub_age_top["Other"] = 100 - sub_age_top.sum(axis=1)
    tab20_colors = list(plt.cm.tab20.colors)
    sub_cols = sub_age_top.columns.tolist()
    bottom = np.zeros(len(sub_age_top))
    x = np.arange(len(sub_age_top))
    for j, sub in enumerate(sub_cols):
        ax.bar(x, sub_age_top[sub].values, bottom=bottom,
               color=tab20_colors[j % 20], alpha=0.85, label=sub)
        bottom += sub_age_top[sub].values
    ax.set_xticks(x)
    ax.set_xticklabels(sub_age_top.index.astype(str), rotation=45, ha="right",
                       fontsize=6)
    ax.set_ylabel("% Excitatory cells", fontsize=8)
    ax.legend(fontsize=5, frameon=False, ncol=2, loc="upper right")
else:
    ax.text(0.5, 0.5, "cell_subclass not available",
            ha="center", va="center", transform=ax.transAxes)

fig2.savefig(OUTPUT_FIG_PB, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Figure 2 saved to {OUTPUT_FIG_PB}", flush=True)

print("\nDone.", flush=True)
