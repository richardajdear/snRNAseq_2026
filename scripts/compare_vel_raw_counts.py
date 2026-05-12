"""
Diagnostic: compare the VELMESHEV raw counts layer between two pipeline runs.

Run A (Vel only):         Vel_prepost_noage_tuning5
Run B (VelWangPsychAD):   VelWangPsychAD_200k_prepost_noage_tuning5

Pure h5py implementation — anndata is never used.
Root cause of prior OOM: anndata backed='r' still materialises dense layers
(scanvi_normalized, scvi_normalized ~30 GB each) into RAM on open.
"""
import os, sys, gc
import numpy as np
import pandas as pd
import h5py

REPO_ROOT = "/home/rajd2/rds/hpc-work/snRNAseq_2026"
sys.path.insert(0, os.path.join(REPO_ROOT, 'code'))

from environment import get_environment
env = get_environment()
rds_dir = env['rds_dir']

BASE = rds_dir + "/Cam_snRNAseq/integrated"
PATH_A = BASE + "/Vel_prepost_noage_tuning5/scvi_output/integrated.h5ad"
PATH_B = BASE + "/VelWangPsychAD_200k_prepost_noage_tuning5_allVel/scvi_output/integrated.h5ad"

SAMPLE_N = 10000
N_CHEM   = 10000   # cells per chemistry group for V2/V3 comparison
RNG = np.random.default_rng(42)

# ── Helpers ──────────────────────────────────────────────────────────────────

def rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return -1

def mem(label=''):
    print(f"  [MEM {rss_mb():.0f} MB] {label}", flush=True)

def sep(title=''):
    print(f"\n{'='*70}", flush=True)
    if title:
        print(title, flush=True)

def h5_read_categorical(f, path):
    """Read a categorical obs column from h5py file, return pandas Categorical."""
    grp = f[path]
    codes = grp['codes'][:]
    cats  = grp['categories'][:].astype(str)
    return pd.Categorical.from_codes(codes, cats)

def h5_read_obs(path, columns):
    """
    Read selected obs columns and obs_names from an h5ad file using h5py only.
    Returns a DataFrame. Never opens anndata.
    """
    with h5py.File(path, 'r') as f:
        obs_grp = f['obs']
        # obs_names
        idx = obs_grp['_index'][:].astype(str)
        data = {'obs_names': idx}
        for col in columns:
            if col not in obs_grp:
                continue
            obj = obs_grp[col]
            if isinstance(obj, h5py.Group) and 'codes' in obj:
                data[col] = h5_read_categorical(f, f'obs/{col}')
            else:
                arr = obj[:]
                # decode bytes to str if needed
                if arr.dtype.kind in ('S', 'O'):
                    arr = arr.astype(str)
                data[col] = arr
    df = pd.DataFrame(data).set_index('obs_names')
    return df

def h5_var_names(path):
    with h5py.File(path, 'r') as f:
        return f['var/_index'][:].astype(str)

def h5_n_cells(path):
    with h5py.File(path, 'r') as f:
        return f['obs/_index'].shape[0]

def read_csr_rows(path, group_path, row_indices_sorted, n_cols):
    """
    Read specific rows from a CSR sparse group in h5ad (data/indices/indptr).
    Returns dense float32 array (n_rows, n_cols).
    Reads indptr once (small), then one slice per selected row.
    """
    rows = np.asarray(row_indices_sorted, dtype=np.intp)
    result = np.zeros((len(rows), n_cols), dtype=np.float32)
    with h5py.File(path, 'r') as f:
        grp = f[group_path]
        indptr = grp['indptr'][:]          # n_cells+1 ints, always tiny
        for i, row_idx in enumerate(rows):
            s, e = int(indptr[row_idx]), int(indptr[row_idx + 1])
            if e > s:
                result[i, grp['indices'][s:e]] = grp['data'][s:e]
    return result

# ── 0. Quick file summary ─────────────────────────────────────────────────────

mem("start")
for label, path in [("Run A", PATH_A), ("Run B", PATH_B)]:
    size_mb = os.path.getsize(path) / 1024**2
    n = h5_n_cells(path)
    print(f"\n{label}: {n:,} cells, file {size_mb:.0f} MB  — {path}", flush=True)
mem("after file summary")

# ── 1. Load obs metadata via h5py (no anndata) ────────────────────────────────

sep("1. Loading obs metadata (h5py only — no anndata)")

OBS_COLS = ['source', 'individual', 'donor_id', 'age_years', 'sex',
            'cell_class', 'cell_class_original', 'Disorder', 'sample',
            'chemistry', 'source-chemistry']

print("Reading Run A obs ...", flush=True)
mem("before obs A")
obs_a = h5_read_obs(PATH_A, OBS_COLS)
mem("after obs A")
print(f"  Run A: {obs_a.shape[0]:,} cells, cols: {list(obs_a.columns)}", flush=True)

print("Reading Run B obs ...", flush=True)
mem("before obs B")
obs_b = h5_read_obs(PATH_B, OBS_COLS)
mem("after obs B")
print(f"  Run B: {obs_b.shape[0]:,} cells, cols: {list(obs_b.columns)}", flush=True)

var_a = h5_var_names(PATH_A)
var_b = h5_var_names(PATH_B)
n_genes_a = len(var_a)
n_genes_b = len(var_b)
mem("after var_names")

# ── 2. VELMESHEV cell and donor counts ────────────────────────────────────────

sep("2. VELMESHEV cell and donor counts")

mask_a = np.array(obs_a['source']) == 'VELMESHEV'
mask_b = np.array(obs_b['source']) == 'VELMESHEV'
obs_a_vel = obs_a.loc[mask_a]
obs_b_vel = obs_b.loc[mask_b]

for obs_v, label in [(obs_a_vel, "Run A"), (obs_b_vel, "Run B")]:
    n = len(obs_v)
    for col in ('individual', 'donor_id', 'sample'):
        if col in obs_v.columns:
            donors = obs_v[col].unique()
            print(f"  [{label}] n_cells={n:,}  n_donors ({col})={len(donors)}", flush=True)
            print(f"    donors: {sorted(str(d) for d in donors)}", flush=True)
            break
    else:
        print(f"  [{label}] n_cells={n:,}  (no donor column present)", flush=True)

mem("after VELMESHEV masks")

# ── 3. Obs metadata distributions ────────────────────────────────────────────

sep("3. Obs metadata distributions for VELMESHEV cells")

for col in ('age_years', 'sex', 'cell_class', 'Disorder'):
    for obs_v, label in [(obs_a_vel, "Run A"), (obs_b_vel, "Run B")]:
        if col not in obs_v.columns:
            continue
        vals = obs_v[col]
        if vals.dtype.kind in ('f', 'i'):
            vals = vals.astype(float)
            print(f"  [{label}] {col}: mean={vals.mean():.1f}  "
                  f"min={vals.min():.1f}  max={vals.max():.1f}", flush=True)
        else:
            vc = vals.value_counts()
            print(f"  [{label}] {col}:\n{vc.to_string()}", flush=True)

# ── 4. Gene sets ─────────────────────────────────────────────────────────────

sep("4. Gene sets")

genes_a = set(var_a)
genes_b = set(var_b)
only_a  = genes_a - genes_b
only_b  = genes_b - genes_a
common_genes = sorted(genes_a & genes_b)
print(f"  Run A: {len(genes_a):,} genes", flush=True)
print(f"  Run B: {len(genes_b):,} genes", flush=True)
print(f"  Only in A: {len(only_a):,}   Only in B: {len(only_b):,}   Common: {len(common_genes):,}", flush=True)
if only_a: print(f"    A-only examples: {list(only_a)[:8]}", flush=True)
if only_b: print(f"    B-only examples: {list(only_b)[:8]}", flush=True)

var_idx_a = {g: i for i, g in enumerate(var_a)}
var_idx_b = {g: i for i, g in enumerate(var_b)}

# ── 5. Cell barcode overlap ──────────────────────────────────────────────────

sep("5. Cell barcode overlap for VELMESHEV cells")

barcodes_a = set(obs_a_vel.index)
barcodes_b = set(obs_b_vel.index)
in_both   = barcodes_a & barcodes_b
only_in_a = barcodes_a - barcodes_b
only_in_b = barcodes_b - barcodes_a
print(f"  Run A VELMESHEV: {len(barcodes_a):,}", flush=True)
print(f"  Run B VELMESHEV: {len(barcodes_b):,}", flush=True)
print(f"  Shared:          {len(in_both):,}", flush=True)
print(f"  Only in A:       {len(only_in_a):,}", flush=True)
print(f"  Only in B:       {len(only_in_b):,}", flush=True)
if only_in_a: print(f"    A-only examples: {list(only_in_a)[:5]}", flush=True)
if only_in_b: print(f"    B-only examples: {list(only_in_b)[:5]}", flush=True)
mem("after barcode overlap")

# ── 6. Raw counts stats (h5py CSR reads, sampled) ────────────────────────────

sep("6. Raw counts layer stats for VELMESHEV cells (sampled)")

def counts_sample_stats(path, mask, n_genes, label, n=SAMPLE_N):
    idx = np.where(mask)[0]
    chosen = np.sort(RNG.choice(idx, size=min(n, len(idx)), replace=False))
    mem(f"  before counts read ({label})")
    mat = read_csr_rows(path, 'layers/counts', chosen, n_genes)
    mem(f"  after counts read  ({label})")
    cell_sums = mat.sum(axis=1)
    print(f"  [{label}] n_sampled={len(chosen)}", flush=True)
    print(f"    min={mat.min():.3f}  max={mat.max():.3f}  mean={mat.mean():.4f}", flush=True)
    print(f"    Is integer? {np.allclose(mat, mat.astype(int))}", flush=True)
    print(f"    Sparsity: {(mat == 0).mean():.3f}", flush=True)
    print(f"    Cell sums: min={cell_sums.min():.0f}  max={cell_sums.max():.0f}  "
          f"mean={cell_sums.mean():.1f}  median={np.median(cell_sums):.1f}", flush=True)
    del mat, cell_sums; gc.collect()

counts_sample_stats(PATH_A, mask_a, n_genes_a, "Run A")
counts_sample_stats(PATH_B, mask_b, n_genes_b, "Run B")

# ── 7. Direct expression comparison on shared barcodes ──────────────────────

sep("7. Direct expression comparison on shared barcodes")

if not in_both:
    print("  No shared barcodes — skipping comparison.", flush=True)
else:
    n_compare = min(200, len(in_both))
    sample_bcs = RNG.choice(sorted(in_both), size=n_compare, replace=False).tolist()
    print(f"  Comparing {n_compare} shared cells × {len(common_genes):,} common genes", flush=True)

    # Map barcodes to sorted row positions in each file
    pos_a = pd.Series(np.arange(len(obs_a)), index=obs_a.index)
    pos_b = pd.Series(np.arange(len(obs_b)), index=obs_b.index)
    orig_rows_a = pos_a[sample_bcs].values.astype(int)
    orig_rows_b = pos_b[sample_bcs].values.astype(int)
    sort_a = np.argsort(orig_rows_a);  rows_a = orig_rows_a[sort_a]
    sort_b = np.argsort(orig_rows_b);  rows_b = orig_rows_b[sort_b]
    mem("after row index mapping")

    cols_a = np.array([var_idx_a[g] for g in common_genes])
    cols_b = np.array([var_idx_b[g] for g in common_genes])

    mem("before counts read (comparison A)")
    mat_a = read_csr_rows(PATH_A, 'layers/counts', rows_a, n_genes_a)[:, cols_a]
    mem("after counts read (comparison A)")
    mat_b = read_csr_rows(PATH_B, 'layers/counts', rows_b, n_genes_b)[:, cols_b]
    mem("after counts read (comparison B)")

    # Restore original sample order for paired comparison
    inv_a = np.argsort(sort_a);  mat_a = mat_a[inv_a]
    inv_b = np.argsort(sort_b);  mat_b = mat_b[inv_b]

    diff = mat_a.astype(np.float32) - mat_b.astype(np.float32)
    n_diff = int((diff != 0).sum())
    total  = diff.size
    print(f"  Matrix shape: {mat_a.shape}", flush=True)
    print(f"  Identical: {total-n_diff:,}/{total:,} ({100*(total-n_diff)/total:.2f}%)", flush=True)
    print(f"  Different: {n_diff:,}/{total:,} ({100*n_diff/total:.2f}%)", flush=True)
    if n_diff > 0:
        print(f"  Max |diff|:  {np.abs(diff).max():.4f}", flush=True)
        print(f"  Mean |diff|: {np.abs(diff).mean():.6f}", flush=True)

    sums_a = mat_a.sum(axis=1)
    sums_b = mat_b.sum(axis=1)
    sd = sums_a - sums_b
    print(f"\n  Per-cell sums (common genes):", flush=True)
    print(f"    Run A: mean={sums_a.mean():.1f}  min={sums_a.min():.0f}  max={sums_a.max():.0f}", flush=True)
    print(f"    Run B: mean={sums_b.mean():.1f}  min={sums_b.min():.0f}  max={sums_b.max():.0f}", flush=True)
    print(f"    Diff:  mean={sd.mean():.4f}  max_abs={np.abs(sd).max():.4f}", flush=True)
    print(f"    -> {'IDENTICAL' if np.allclose(sums_a, sums_b) else 'DIFFER'}", flush=True)
    del mat_a, mat_b, diff; gc.collect()
    mem("after comparison cleanup")

# ── 8. X stats for VELMESHEV cells ──────────────────────────────────────────

sep("8. adata.X stats for VELMESHEV cells (sampled)")

def x_sample_stats(path, mask, n_genes, label, n=SAMPLE_N):
    idx = np.where(mask)[0]
    chosen = np.sort(RNG.choice(idx, size=min(n, len(idx)), replace=False))
    mem(f"  before X read ({label})")
    mat = read_csr_rows(path, 'X', chosen, n_genes)
    mem(f"  after X read  ({label})")
    cell_sums = mat.sum(axis=1)
    print(f"  [{label}] n_sampled={len(chosen)}", flush=True)
    print(f"    min={mat.min():.4f}  max={mat.max():.4f}  mean={mat.mean():.6f}", flush=True)
    print(f"    Is integer? {np.allclose(mat, mat.astype(int))}", flush=True)
    print(f"    Sparsity: {(mat == 0).mean():.3f}", flush=True)
    print(f"    Cell sums: min={cell_sums.min():.1f}  max={cell_sums.max():.1f}  "
          f"mean={cell_sums.mean():.2f}", flush=True)
    del mat, cell_sums; gc.collect()

x_sample_stats(PATH_A, mask_a, n_genes_a, "Run A")
x_sample_stats(PATH_B, mask_b, n_genes_b, "Run B")

# ── 9. GRN C3+ projection on shared cells ────────────────────────────────────

sep("9. AHBA C3+ GRN projection on shared-barcode sample")

grn_file = os.path.join(env['ref_dir'], "ahba_dme_hcp_top8kgenes_weights.csv")
print(f"  GRN file: {grn_file}", flush=True)
mem("before GRN load")

from regulons import get_ahba_GRN
ahba_GRN = get_ahba_GRN(path_to_ahba_weights=grn_file, use_weights=True)
# Extract C3+ weights: Series indexed by gene symbol
grn_pivot = ahba_GRN.pivot_table(index='Network', columns='Gene',
                                  values='Importance', fill_value=0)
c3plus_series = grn_pivot.loc['C3+']   # gene symbol → weight
print(f"  GRN C3+ genes with non-zero weight: {(c3plus_series > 0).sum()}", flush=True)
mem("after GRN load")

def build_c3_weights_h5(path, c3plus_series):
    """
    Build C3+ weight vector aligned to gene positions in an h5ad file.

    Handles two formats found in the var/gene_symbol categorical:
      - Run A style: clean symbols  e.g. 'WNT7A'
      - Run B style: mangled        e.g. 'SAMD11_ENSG00000187634'
        (duplicate-resolved by appending Ensembl ID during pipeline build)

    The fix: strip the '_ENSG...' suffix before matching against the GRN.
    Returns float32 array shape (n_genes,).
    """
    with h5py.File(path, 'r') as f:
        vg = f['var']
        for key in ('gene_symbol', 'feature_name'):
            if key in vg and isinstance(vg[key], h5py.Group) and 'codes' in vg[key]:
                cats  = vg[key]['categories'][:].astype(str)
                codes = vg[key]['codes'][:]
                raw_syms = cats[codes]   # one per gene position
                break
        else:
            raw_syms = vg['_index'][:].astype(str)

    n_genes = len(raw_syms)
    weights = np.zeros(n_genes, dtype=np.float32)
    n_matched = 0

    grn_set = set(c3plus_series.index)
    for i, sym in enumerate(raw_syms):
        # Strip mangled suffix: 'SAMD11_ENSG00000187634' → 'SAMD11'
        if '_ENSG' in sym:
            clean = sym.split('_ENSG')[0]
        elif sym.startswith('ENSG') or sym.startswith('hsa-'):
            continue   # no usable symbol
        else:
            clean = sym

        if clean in grn_set:
            w = float(c3plus_series[clean])
            if w > 0:
                weights[i] = w
                n_matched += 1

    print(f"    GRN C3+ genes matched: {n_matched} / {n_genes}", flush=True)
    return weights

def cpm_c3plus(path, row_indices_sorted, n_genes, c3_weights):
    """
    Load raw counts for given rows, CPM-normalise, project C3+.
    Returns (n_cells,) float32 array of C3+ CPM scores.
    """
    mat = read_csr_rows(path, 'layers/counts', row_indices_sorted, n_genes)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat_cpm = mat / row_sums * 1e6
    scores = mat_cpm @ c3_weights
    del mat, mat_cpm; gc.collect()
    return scores

# Build weight vectors for both datasets (needed by sections 9 and 10)
print("  Building C3+ weight vector for Run A:", flush=True)
c3w_a = build_c3_weights_h5(PATH_A, c3plus_series)
print("  Building C3+ weight vector for Run B:", flush=True)
c3w_b = build_c3_weights_h5(PATH_B, c3plus_series)
mem("after weight vector build")

if not in_both:
    print("  No shared barcodes — skipping projection.", flush=True)
else:
    print(f"  Using the same {n_compare} shared cells as section 7.", flush=True)
    mem("before GRN projection reads")

    # rows_a / rows_b from section 7 (already sorted global indices for sample_bcs)
    scores_a = cpm_c3plus(PATH_A, rows_a, n_genes_a, c3w_a)
    mem("after Run A projection")
    scores_b = cpm_c3plus(PATH_B, rows_b, n_genes_b, c3w_b)
    mem("after Run B projection")

    # Re-order to match sample_bcs order (same inv_a / inv_b as section 7)
    scores_a = scores_a[inv_a]
    scores_b = scores_b[inv_b]

    print(f"\n  C3+ CPM score summary (n={n_compare} shared cells):", flush=True)
    print(f"    Run A: mean={scores_a.mean():.4f}  median={np.median(scores_a):.4f}  "
          f"min={scores_a.min():.4f}  max={scores_a.max():.4f}", flush=True)
    print(f"    Run B: mean={scores_b.mean():.4f}  median={np.median(scores_b):.4f}  "
          f"min={scores_b.min():.4f}  max={scores_b.max():.4f}", flush=True)

    ratio_means = scores_a.mean() / scores_b.mean() if scores_b.mean() != 0 else np.nan
    print(f"    Run A / Run B mean ratio: {ratio_means:.4f}", flush=True)

    cell_ratios = scores_a / np.where(scores_b == 0, np.nan, scores_b)
    print(f"    Per-cell A/B ratio: mean={np.nanmean(cell_ratios):.4f}  "
          f"median={np.nanmedian(cell_ratios):.4f}", flush=True)

    del scores_a, scores_b, cell_ratios; gc.collect()

# ── 10. V2 vs V3 chemistry — Excitatory cells, age 5–25 ─────────────────────

sep("10. V2 vs V3 chemistry — C3+ CPM ratio\n"
    "    (VELMESHEV Excitatory cells, age 5–25 only)")

# Run A: cell_class is broken (only Other/Glia/OPC/Microglia from bad scANVI).
# Use cell_class_original instead, which has correct Velmeshev labels.
# Run B: cell_class is correct.
EXCIT_COL = {'Run A': 'cell_class_original', 'Run B': 'cell_class'}
AGE_MIN, AGE_MAX = 5, 25

for (label, path, obs_all, mask, n_genes, c3w) in [
        ("Run A", PATH_A, obs_a, mask_a, n_genes_a, c3w_a),
        ("Run B", PATH_B, obs_b, mask_b, n_genes_b, c3w_b)]:

    obs_vel = obs_all[mask]
    exc_col = EXCIT_COL[label]

    # Chemistry labels
    chem = None
    for col in ('chemistry', 'source-chemistry'):
        if col in obs_vel.columns:
            chem = obs_vel[col].astype(str).str.extract(r'(V[23])', expand=False)
            print(f"  [{label}] chemistry from '{col}': "
                  f"{sorted(chem.dropna().unique())}", flush=True)
            break
    if chem is None:
        print(f"  [{label}] no chemistry column — skipping", flush=True)
        continue

    # Excitatory filter
    if exc_col not in obs_vel.columns:
        print(f"  [{label}] '{exc_col}' not found — skipping", flush=True)
        continue
    excit_mask = np.array(obs_vel[exc_col].astype(str)) == 'Excitatory'

    # Age filter
    age_mask = ((obs_vel['age_years'].astype(float) >= AGE_MIN) &
                (obs_vel['age_years'].astype(float) <= AGE_MAX)).values

    base_mask = excit_mask & age_mask
    global_idx = np.where(mask)[0]
    print(f"  [{label}] Excitatory & age {AGE_MIN}–{AGE_MAX}: "
          f"{base_mask.sum():,} / {len(obs_vel):,} VELMESHEV cells", flush=True)

    results = {}
    for v in ('V2', 'V3'):
        v_mask = base_mask & (np.array(chem) == v)
        v_global = global_idx[v_mask]
        n_avail = len(v_global)
        if n_avail == 0:
            print(f"    {v}: 0 cells after filtering", flush=True)
            continue

        n_sample = min(N_CHEM, n_avail)
        chosen = np.sort(RNG.choice(v_global, size=n_sample, replace=False))
        mem(f"  before counts read ({label} {v}, n={n_sample})")
        scores = cpm_c3plus(path, chosen, n_genes, c3w)
        mem(f"  after  counts read ({label} {v})")
        results[v] = scores
        print(f"    {v}: n_avail={n_avail:,}  sampled={n_sample}  "
              f"C3+ mean={scores.mean():.4f}  median={np.median(scores):.4f}  "
              f"sd={scores.std():.4f}", flush=True)

    if 'V2' in results and 'V3' in results:
        ratio = results['V2'].mean() / results['V3'].mean()
        print(f"  [{label}] V2/V3 mean C3+ ratio = {ratio:.4f}  "
              f"({'V2 > V3' if ratio > 1 else 'V2 < V3'})", flush=True)

    del results; gc.collect()
    mem(f"  after {label} chemistry comparison")


# ── 11. Age distribution of V2/V3 Excitatory cells (Run A, section 10 context) ─
sep("11. Age distribution of V2 vs V3 Excitatory cells in Run A (age 5–25)")

# Reuse obs_a and mask_a from earlier sections.
obs_vel_a = obs_a[mask_a]

# Reconstruct filters from section 10 for Run A.
chem_a = obs_vel_a['chemistry'].astype(str).str.extract(r'(V[23])', expand=False)
excit_a = (np.array(obs_vel_a['cell_class_original'].astype(str)) == 'Excitatory'
           if 'cell_class_original' in obs_vel_a.columns else
           np.zeros(len(obs_vel_a), dtype=bool))
age_a   = obs_vel_a['age_years'].astype(float)
age_mask_a = (age_a >= AGE_MIN) & (age_a <= AGE_MAX)
base_a  = excit_a & age_mask_a.values

print(f"  Run A Excitatory cells, age {AGE_MIN}–{AGE_MAX}: {base_a.sum():,}", flush=True)
for v in ('V2', 'V3'):
    vmask = base_a & (np.array(chem_a) == v)
    if vmask.sum() == 0:
        print(f"  {v}: 0 cells", flush=True)
        continue
    ages_v = age_a.values[vmask]
    print(f"  {v}: n={vmask.sum():,}  age mean={ages_v.mean():.1f}  "
          f"median={np.median(ages_v):.1f}  "
          f"range=[{ages_v.min():.1f}, {ages_v.max():.1f}]", flush=True)
    # Histogram bins
    bins = [5, 10, 15, 20, 25]
    for lo, hi in zip(bins[:-1], bins[1:]):
        n_bin = ((ages_v >= lo) & (ages_v < hi)).sum()
        print(f"    [{lo},{hi}): {n_bin:,} ({100*n_bin/len(ages_v):.1f}%)", flush=True)

# ── 12. Pseudobulk by_cell_class.h5ad — V2 vs V3 C3+ comparison ─────────────
sep("12. Pseudobulk analysis — V2 vs V3 C3+ CPM (by_cell_class.h5ad)")

PB_PATH = (rds_dir + "/Cam_snRNAseq/integrated/"
           "Vel_prepost_noage_tuning5/pseudobulk_output/by_cell_class.h5ad")
print(f"  Pseudobulk file: {PB_PATH}", flush=True)

def pb_read_obs(path):
    """Read obs from pseudobulk h5ad: index, age_years, chemistry, cell_class, n_cells."""
    cols = ['age_years', 'chemistry', 'cell_class', 'individual', 'source']
    return h5_read_obs(path, cols)

obs_pb = pb_read_obs(PB_PATH)
var_pb = h5_var_names(PB_PATH)
n_genes_pb = len(var_pb)
print(f"  Pseudobulk: {len(obs_pb):,} donor×cell_class rows, {n_genes_pb:,} genes", flush=True)

if 'cell_class' in obs_pb.columns:
    print(f"  cell_class categories: {sorted(obs_pb['cell_class'].astype(str).unique())}", flush=True)
if 'chemistry' in obs_pb.columns:
    print(f"  chemistry categories: {sorted(obs_pb['chemistry'].astype(str).unique())}", flush=True)

# Build C3+ weight vector for pseudobulk (same function, reads var/gene_symbol)
print("  Building C3+ weight vector for pseudobulk:", flush=True)
c3w_pb = build_c3_weights_h5(PB_PATH, c3plus_series)
mem("after pseudobulk weight build")

# CPM project: read all rows (253 donors × cell_class = small)
mem("before pseudobulk counts read")
all_rows_pb = np.arange(len(obs_pb))
mat_pb = read_csr_rows(PB_PATH, 'layers/counts', all_rows_pb, n_genes_pb)
mem("after pseudobulk counts read")

row_sums_pb = mat_pb.sum(axis=1, keepdims=True)
row_sums_pb[row_sums_pb == 0] = 1
cpm_pb = mat_pb / row_sums_pb * 1e6
scores_pb = cpm_pb @ c3w_pb
del mat_pb, cpm_pb; gc.collect()

obs_pb['c3_score'] = scores_pb
obs_pb['age_years_f'] = obs_pb['age_years'].astype(float)
obs_pb['chem'] = obs_pb['chemistry'].astype(str).str.extract(r'(V[23])', expand=False)

print(f"\n  All donors (all cell classes, all ages):", flush=True)
for chem, grp in obs_pb.groupby('chem'):
    print(f"    {chem}: n={len(grp)}  C3+ mean={grp['c3_score'].mean():.1f}  "
          f"median={grp['c3_score'].median():.1f}", flush=True)

# Filter age 5–25
pb_25 = obs_pb[(obs_pb['age_years_f'] >= AGE_MIN) & (obs_pb['age_years_f'] <= AGE_MAX)]
print(f"\n  Age {AGE_MIN}–{AGE_MAX} ({len(pb_25)} rows):", flush=True)
for chem, grp in pb_25.groupby('chem'):
    print(f"    {chem}: n={len(grp)}  C3+ mean={grp['c3_score'].mean():.1f}  "
          f"median={grp['c3_score'].median():.1f}  "
          f"age mean={grp['age_years_f'].mean():.1f}", flush=True)

grp_means = pb_25.groupby('chem')['c3_score'].mean()
if 'V2' in grp_means and 'V3' in grp_means:
    pb_ratio = grp_means['V2'] / grp_means['V3']
    print(f"\n  [Pseudobulk] V2/V3 mean C3+ ratio = {pb_ratio:.4f}  "
          f"({'V2 > V3' if pb_ratio > 1 else 'V2 < V3'})", flush=True)

# Breakdown by cell_class
if 'cell_class' in pb_25.columns:
    print(f"\n  Per cell_class (age {AGE_MIN}–{AGE_MAX}):", flush=True)
    for cc, grp_cc in pb_25.groupby('cell_class'):
        for chem, grp_c in grp_cc.groupby('chem'):
            print(f"    {cc} / {chem}: n={len(grp_c)}  "
                  f"C3+={grp_c['c3_score'].mean():.1f}", flush=True)

mem("after pseudobulk analysis")

sep()
print("DONE", flush=True)
mem("final")
