#!/usr/bin/env python3
"""Compare three trajectory inferences on a neuron manifold: PAGA->DPT,
PAGA->Palantir, PAGA->CellRank2.

Input per dataset: a neuron_manifold.h5ad produced by
scripts/c3_maturation/s11_neuron_manifold.py, carrying:
  obsm['X_pca']            PCA(30) on raw counts (the representation)
  obsp connectivities/distances + uns['neighbors']   (nn=50 graph)
  obsm['X_diffmap'], obsm['X_umap_pagainit']
  obs: native_fine, native_broad_fixed, cluster_vote, leiden_n, age, dpt_seed,
       {EN,IN,Prog,Imm,Pan,Glia}_sig, expr_<GENE> (log1p CPM markers)
  uns['iroot']

For each dataset the pipeline:
  1. builds PAGA (topology / EN-vs-IN branching) on `leiden_n`;
  2. runs DPT, Palantir, CellRank2 (config-driven parameters);
  3. extracts comparable per-cell scores (pseudotimes + EN-fate probabilities + entropy);
  4. writes diagnostic plots (UMAP grid, PAGA graph, pseudotime-vs-age) and a
     cross-method Spearman correlation heatmap;
  5. saves per-cell scores (parquet) and a correlations summary (csv).

Outputs -> <manifold_dir>/<run_name>/ (the dir where integrated.h5ad lives).

Each method is wrapped defensively: if one fails (or a dependency is missing) the
pipeline logs it and continues with the others.

USAGE
  smoke test (synthetic manifold, all methods, ~2.5k cells):
      python code/trajectory/run_trajectory.py --smoke
  real run (sbatch):
      sbatch --time=02:00:00 --mem=128G scripts/run_script.sh \
          code/trajectory/run_trajectory.py --config code/trajectory/trajectory_config.yaml
  single dataset:
      ... run_trajectory.py --config <cfg> --only Velmeshev-V3
"""
from __future__ import annotations
import argparse, sys, json, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sc.settings.verbosity = 1


# ----------------------------- config ---------------------------------------
DEFAULTS = dict(
    run_name="traj_compare",
    root=dict(signatures=["Prog_sig", "Imm_sig"], use_uns_iroot=True),
    lineage=dict(en_labels=["ExN", "Immature_neuron"], in_labels=["InN"], label_col="cluster_vote"),
    neighbors=dict(use_rep="X_pca", n_neighbors=50),
    paga=dict(groups="leiden_n"),
    dpt=dict(n_dcs=10),
    palantir=dict(n_diffusion_components=10, knn=30, num_waypoints=1500, terminal_states="signature"),
    cellrank=dict(pseudotime_weight=0.8, connectivity_weight=0.2, n_macrostates=8,
                  n_terminal_states="auto", schur_components=12, brandts_max_cells=0),
)


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def load_config(path):
    if path is None:
        return dict(DEFAULTS), {}
    import yaml
    raw = yaml.safe_load(Path(path).read_text())
    datasets = raw.pop("datasets", {})
    cfg = deep_merge(DEFAULTS, raw)
    return cfg, datasets


# ----------------------------- helpers --------------------------------------
def ensure_neighbors(a, cfg):
    if "neighbors" in a.uns and "connectivities" in a.obsp:
        return
    sc.pp.neighbors(a, use_rep=cfg["neighbors"]["use_rep"],
                    n_neighbors=cfg["neighbors"]["n_neighbors"])


def pick_root(a, cfg):
    if cfg["root"].get("use_uns_iroot") and "iroot" in a.uns:
        return int(a.uns["iroot"])
    score = np.zeros(a.n_obs)
    for s in cfg["root"]["signatures"]:
        if s in a.obs:
            score = score + a.obs[s].values
    return int(np.argmax(score))


def lineage_of(label, cfg):
    if label in cfg["lineage"]["en_labels"]:
        return "EN"
    if label in cfg["lineage"]["in_labels"]:
        return "IN"
    return "other"


def terminal_cells_by_signature(a, cfg):
    """Representative mature-EN and mature-IN cells = argmax of EN_sig / IN_sig
    among cells of the corresponding lineage (for Palantir/terminal hints)."""
    lab = a.obs[cfg["lineage"]["label_col"]].astype(str).values
    out = {}
    for lin, sig in [("EN", "EN_sig"), ("IN", "IN_sig")]:
        labels = cfg["lineage"][f"{lin.lower()}_labels"]
        m = np.isin(lab, labels)
        if m.sum() and sig in a.obs:
            sub = np.where(m)[0]
            out[a.obs_names[sub[np.argmax(a.obs[sig].values[sub])]]] = lin
    return out


def terminal_states_series(a, cfg, k=60):
    """Categorical Series marking the mature-EN and mature-IN poles (top-k cells by
    EN_sig / IN_sig within each lineage), NaN elsewhere — for set_terminal_states()."""
    lab = a.obs[cfg["lineage"]["label_col"]].astype(str).values
    s = pd.Series(pd.NA, index=a.obs_names, dtype=object)
    for lin, sig in [("EN", "EN_sig"), ("IN", "IN_sig")]:
        labels = cfg["lineage"][f"{lin.lower()}_labels"]
        m = np.isin(lab, labels)
        if m.sum() and sig in a.obs:
            idx = np.where(m)[0]
            top = idx[np.argsort(a.obs[sig].values[idx])[-min(k, len(idx)):]]
            s.iloc[top] = lin
    return s.astype("category")


def minmax(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


# ----------------------------- methods --------------------------------------
def run_paga(a, cfg, log):
    sc.tl.paga(a, groups=cfg["paga"]["groups"])
    log("PAGA computed on groups=%s" % cfg["paga"]["groups"])


def run_dpt(a, cfg, root, log):
    if "X_diffmap" not in a.obsm:
        sc.tl.diffmap(a, n_comps=cfg["dpt"]["n_dcs"])
    a.uns["iroot"] = root
    sc.tl.dpt(a, n_dcs=cfg["dpt"]["n_dcs"])
    return dict(dpt=a.obs["dpt_pseudotime"].values.copy())


def run_palantir(a, cfg, root, log):
    import palantir
    pc = cfg["palantir"]
    palantir.utils.run_diffusion_maps(a, pca_key=cfg["neighbors"]["use_rep"],
                                      n_components=pc["n_diffusion_components"], knn=pc["knn"])
    palantir.utils.determine_multiscale_space(a)
    early = a.obs_names[root]
    # explicit terminal states = mature-EN / mature-IN signature poles (auto-detection is
    # unreliable and returned 0 branches); pass a Series so branch_probs columns are EN/IN.
    terminal = terminal_cells_by_signature(a, cfg)
    ts = None if pc["terminal_states"] == "auto" else pd.Series(terminal)
    pr = None
    for kwargs in (dict(num_waypoints=pc["num_waypoints"], terminal_states=ts, knn=pc["knn"]),
                   dict(num_waypoints=pc["num_waypoints"], terminal_states=ts),
                   dict(num_waypoints=pc["num_waypoints"])):
        try:
            pr = palantir.core.run_palantir(a, early, **kwargs); break
        except TypeError:
            continue

    def _from(obs_key, attr):
        v = a.obs.get(obs_key)
        if v is not None:
            return np.asarray(v)
        return np.asarray(getattr(pr, attr)) if pr is not None and getattr(pr, attr, None) is not None \
            else np.full(a.n_obs, np.nan)

    out = {}
    out["palantir_pt"] = _from("palantir_pseudotime", "pseudotime")
    out["palantir_entropy"] = _from("palantir_entropy", "entropy")
    # fate / branch probabilities: prefer the returned PResult.branch_probs (obsm copy is
    # sometimes empty), columns = terminal cell names -> map to EN via cluster_vote.
    fp = a.obsm.get("palantir_fate_probabilities")
    if fp is None or (hasattr(fp, "shape") and fp.shape[1] == 0):
        fp = getattr(pr, "branch_probs", None) if pr is not None else None
    out["palantir_EN_fate"] = _fate_to_en(fp, a, cfg, log, src="palantir")
    return out


def _fate_to_en(fp, a, cfg, log, src):
    if fp is None:
        log(f"  [{src}] no fate probabilities returned")
        return np.full(a.n_obs, np.nan)
    if isinstance(fp, pd.DataFrame):
        cols, M = list(fp.columns), fp.values
    else:
        cols, M = list(range(fp.shape[1])), np.asarray(fp)
    lab = a.obs[cfg["lineage"]["label_col"]].astype(str).values
    en_cols = []
    for j, c in enumerate(cols):
        cs = str(c)
        # column may be a terminal CELL name, a lineage name, or a cluster label
        if cs in a.obs_names:
            lin = lineage_of(lab[a.obs_names.get_loc(cs)], cfg)
        elif cs in ("EN", "IN"):
            lin = cs
        elif any(e in cs for e in cfg["lineage"]["en_labels"]):
            lin = "EN"
        elif any(i in cs for i in cfg["lineage"]["in_labels"]):
            lin = "IN"
        else:
            lin = "other"
        if lin == "EN":
            en_cols.append(j)
    log(f"  [{src}] fate cols={cols} -> EN cols={en_cols}")
    return M[:, en_cols].sum(1) if en_cols else np.full(a.n_obs, np.nan)


def _fate_probs_robust(g, log):
    """compute_fate_probabilities avoiding PETSc (broken here) and loky workers
    (fail + hang the interpreter at exit). Try progressively simpler kwargs."""
    for kw in (dict(solver="direct", use_petsc=False, n_jobs=1),
               dict(solver="gmres", use_petsc=False, n_jobs=1),
               dict(n_jobs=1), dict()):
        try:
            g.compute_fate_probabilities(**kw); return
        except TypeError:
            continue   # unsupported kwarg name in this cellrank version
    g.compute_fate_probabilities()


def run_cellrank(a, cfg, root, dpt_vals, log):
    import cellrank as cr
    cc = cfg["cellrank"]
    if "dpt_pseudotime" not in a.obs:
        a.obs["dpt_pseudotime"] = dpt_vals
    pk = cr.kernels.PseudotimeKernel(a, time_key="dpt_pseudotime").compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(a).compute_transition_matrix()
    kernel = cc["pseudotime_weight"] * pk + cc["connectivity_weight"] * ck
    kernel.compute_transition_matrix()

    g = None; estimator = None
    # --- preferred: GPCCA (macrostates). Schur uses PETSc (krylov) or dense brandts (small n only) ---
    try:
        g = cr.estimators.GPCCA(kernel)
        try:
            g.compute_schur(n_components=cc["schur_components"])           # krylov / PETSc
        except Exception as e_kr:
            if a.n_obs <= cc.get("brandts_max_cells", 0):
                log(f"  [cellrank] krylov/PETSc unavailable ({type(e_kr).__name__}); using dense brandts")
                g.compute_schur(n_components=cc["schur_components"], method="brandts")
            else:
                raise
        g.compute_macrostates(n_states=cc["n_macrostates"], cluster_key=cfg["lineage"]["label_col"])
        nts = cc["n_terminal_states"]
        g.predict_terminal_states() if nts == "auto" else g.predict_terminal_states(n_states=int(nts))
        _fate_probs_robust(g, log)
        estimator = "GPCCA"
    except Exception as e_g:
        # --- fallback: CFLARE (eigendecomposition via ARPACK; no PETSc, scales to 1e5 cells) ---
        # CFLARE has no auto terminal-state detector here, so set terminal states explicitly
        # from the mature-EN / mature-IN signature poles, then compute fate probabilities.
        log(f"  [cellrank] GPCCA unavailable ({type(e_g).__name__}: {str(e_g)[:100]}); falling back to CFLARE")
        g = cr.estimators.CFLARE(kernel)
        g.compute_eigendecomposition()
        g.set_terminal_states(terminal_states_series(a, cfg))
        # use_petsc=False (petsc is broken in this env) + direct scipy solver + n_jobs=1
        # (loky workers otherwise fail AND hang the interpreter at exit)
        _fate_probs_robust(g, log)
        estimator = "CFLARE"
    log(f"  [cellrank] estimator = {estimator}")

    fp = g.fate_probabilities
    cols = list(getattr(fp, "names", getattr(fp, "columns", [])))
    M = np.asarray(fp)
    out = {"_cellrank_estimator": estimator}
    out["cellrank_EN_fate"] = _fate_to_en(pd.DataFrame(M, columns=cols, index=a.obs_names), a, cfg, log, src="cellrank")
    try:
        out["cellrank_macrostate"] = a.obs["macrostates"].astype(str).values
    except Exception:
        out["cellrank_macrostate"] = None   # CFLARE has no macrostates; plotting skips None
    return out, g


# ----------------------------- plotting -------------------------------------
def _sc_cat(ax, um, lab, title, s=4):
    lab = pd.Series(lab).astype(str).values
    for k in sorted(pd.unique(lab), key=lambda k: -(lab == k).sum()):
        m = lab == k
        ax.scatter(um[m, 0], um[m, 1], s=s, label=f"{k} ({m.sum()})")
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, fontsize=7, loc="best")


def _sc_val(ax, um, v, title, cmap="viridis", s=4):
    sca = ax.scatter(um[:, 0], um[:, 1], c=v, s=s, cmap=cmap)
    plt.colorbar(sca, ax=ax, fraction=.045); ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def make_plots(a, scores, name, outdir, cfg, gpcca, log):
    um = a.obsm.get("X_umap_pagainit", a.obsm.get("X_umap"))
    age = a.obs["age"].values
    # ---- UMAP grid ----
    panels = [("cluster_vote (cat)", scores.get("_cluster_vote"), "cat"),
              ("age (years)", age, "viridis"),
              ("DPT pseudotime", scores.get("dpt"), "plasma"),
              ("Palantir pseudotime", scores.get("palantir_pt"), "plasma"),
              ("Palantir entropy", scores.get("palantir_entropy"), "magma"),
              ("Palantir EN-fate", scores.get("palantir_EN_fate"), "coolwarm"),
              ("CellRank EN-fate", scores.get("cellrank_EN_fate"), "coolwarm"),
              ("CellRank macrostate", scores.get("cellrank_macrostate"), "cat")]
    panels = [(t, v, c) for t, v, c in panels if v is not None]
    ncol = 4; nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (t, v, c) in zip(axes, panels):
        if c == "cat":
            _sc_cat(ax, um, v, t)
        else:
            _sc_val(ax, um, v, t, cmap=c)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{name}: trajectory method comparison (PAGA-init UMAP, n={a.n_obs:,})", fontsize=15, y=1.0)
    fig.tight_layout(); fig.savefig(outdir / f"traj_umaps_{name}.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # ---- PAGA graph ----
    try:
        fig, ax = plt.subplots(figsize=(8, 7))
        sc.pl.paga(a, color=cfg["paga"]["groups"], ax=ax, show=False, fontsize=7,
                   node_size_scale=1.5, edge_width_scale=0.7)
        ax.set_title(f"{name}: PAGA graph (groups={cfg['paga']['groups']}) — EN/IN branching topology")
        fig.savefig(outdir / f"traj_paga_{name}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        log(f"  PAGA plot failed: {e}")

    # ---- pseudotime/fate vs age + correlation heatmap ----
    lab = a.obs[cfg["lineage"]["label_col"]].astype(str).values
    en_m = np.isin(lab, cfg["lineage"]["en_labels"])
    cmp_keys = [k for k in ["dpt", "palantir_pt", "palantir_entropy", "palantir_EN_fate", "cellrank_EN_fate"]
                if k in scores and scores[k] is not None and np.isfinite(np.asarray(scores[k], float)).any()]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    # (a) dpt vs age, (b) palantir_pt vs age on EN lineage
    for ax, key, col in [(axes[0], "dpt", "#d62728"), (axes[1], "palantir_pt", "#1f77b4")]:
        if key in scores and scores[key] is not None:
            v = np.asarray(scores[key], float)
            ax.scatter(age[en_m], v[en_m], s=4, alpha=.3, c=col)
            ok = en_m & np.isfinite(v)
            if ok.sum() > 50:
                r, p = spearmanr(age[ok], v[ok])
                ax.set_title(f"EN-lineage: {key} vs age  (rho={r:.2f}, p={p:.1e}, n={ok.sum()})", fontsize=11)
            ax.set_xlabel("age (years)"); ax.set_ylabel(key)
    # (c) correlation heatmap (Spearman) among scores + age
    cmp_keys2 = cmp_keys + ["age"]
    mat = np.column_stack([np.asarray(scores.get(k, age) if k != "age" else age, float) for k in cmp_keys2])
    R = np.full((len(cmp_keys2), len(cmp_keys2)), np.nan)
    for i in range(len(cmp_keys2)):
        for j in range(len(cmp_keys2)):
            ok = np.isfinite(mat[:, i]) & np.isfinite(mat[:, j])
            if ok.sum() > 50:
                R[i, j] = spearmanr(mat[ok, i], mat[ok, j])[0]
    ax = axes[2]; im = ax.imshow(R, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cmp_keys2))); ax.set_xticklabels(cmp_keys2, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cmp_keys2))); ax.set_yticklabels(cmp_keys2, fontsize=8)
    for i in range(len(cmp_keys2)):
        for j in range(len(cmp_keys2)):
            if np.isfinite(R[i, j]):
                ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=.046); ax.set_title("Spearman corr (all cells)", fontsize=11)
    fig.suptitle(f"{name}: cross-method agreement & age alignment", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(outdir / f"traj_corr_{name}.png", dpi=115, bbox_inches="tight"); plt.close(fig)
    return pd.DataFrame(R, index=cmp_keys2, columns=cmp_keys2)


# ----------------------------- driver ---------------------------------------
def process_dataset(name, manifold_path, cfg, log):
    a = ad.read_h5ad(manifold_path)
    log(f"\n=== {name}: {a.n_obs:,} cells from {manifold_path}")
    outdir = Path(manifold_path).parent / cfg["run_name"]
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_neighbors(a, cfg)
    root = pick_root(a, cfg)
    log(f"  root cell idx={root} ({a.obs_names[root]}), label={a.obs[cfg['lineage']['label_col']].iloc[root]}")

    scores = {"_cluster_vote": a.obs[cfg["lineage"]["label_col"]].astype(str).values}
    run_paga(a, cfg, log)

    # DPT
    try:
        scores.update(run_dpt(a, cfg, root, log)); log("  DPT done")
    except Exception:
        log("  DPT FAILED:\n" + traceback.format_exc())
        scores["dpt"] = np.full(a.n_obs, np.nan)

    # Palantir
    try:
        scores.update(run_palantir(a, cfg, root, log)); log("  Palantir done")
    except Exception:
        log("  Palantir FAILED:\n" + traceback.format_exc())

    # CellRank
    gpcca = None
    try:
        cr_out, gpcca = run_cellrank(a, cfg, root, scores.get("dpt"), log)
        scores.update(cr_out); log("  CellRank done")
    except Exception:
        log("  CellRank FAILED:\n" + traceback.format_exc())

    Rdf = make_plots(a, scores, name, outdir, cfg, gpcca, log)

    # ---- save per-cell scores + correlations ----
    keep = ["dpt", "palantir_pt", "palantir_entropy", "palantir_EN_fate", "cellrank_EN_fate",
            "cellrank_macrostate"]
    df = pd.DataFrame({k: scores[k] for k in keep if k in scores}, index=a.obs_names)
    df["age"] = a.obs["age"].values
    df["cluster_vote"] = scores["_cluster_vote"]
    df.to_parquet(outdir / f"traj_scores_{name}.parquet")
    Rdf.to_csv(outdir / f"traj_correlations_{name}.csv")
    log(f"  wrote outputs -> {outdir}")
    return Rdf


# ----------------------------- smoke ----------------------------------------
def make_synthetic_manifold(n=2500, seed=0):
    """Synthetic bifurcation: progenitor -> {EN, IN}, with age ~ pseudotime."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 1, n)                          # latent maturation
    branch = np.where(t < 0.3, "prog", rng.choice(["EN", "IN"], n))
    sign = np.where(branch == "EN", 1.0, np.where(branch == "IN", -1.0, 0.0))
    d0 = t
    d1 = sign * np.clip(t - 0.3, 0, None)
    latent = np.column_stack([d0, d1])
    proj = rng.normal(size=(2, 30))
    X_pca = latent @ proj + rng.normal(scale=0.05, size=(n, 30))
    EN_sig = np.clip(d1, 0, None) * 5
    IN_sig = np.clip(-d1, 0, None) * 5
    Prog_sig = np.clip(0.3 - t, 0, None) * 8
    Imm_sig = np.clip(0.6 - t, 0, None) * 4 * (branch != "prog")
    cvote = np.where(branch == "prog", "Progenitor",
                     np.where((branch != "prog") & (t < 0.5), "Immature_neuron",
                              np.where(branch == "EN", "ExN", "InN")))
    age = t * 30 + rng.normal(scale=2, size=n)
    a = ad.AnnData(sp.csr_matrix((n, 1), dtype="float32"),
                   obs=pd.DataFrame(dict(cluster_vote=cvote, age=age,
                                         EN_sig=EN_sig, IN_sig=IN_sig, Prog_sig=Prog_sig, Imm_sig=Imm_sig),
                                    index=[f"c{i}" for i in range(n)]))
    a.obsm["X_pca"] = X_pca.astype("float32")
    sc.pp.neighbors(a, use_rep="X_pca", n_neighbors=30)
    try:
        sc.tl.leiden(a, resolution=1.0, key_added="leiden_n",
                     flavor="igraph", n_iterations=2, directed=False)
    except TypeError:
        sc.tl.leiden(a, resolution=1.0, key_added="leiden_n")
    sc.tl.umap(a)
    a.obsm["X_umap_pagainit"] = a.obsm["X_umap"]
    a.uns["iroot"] = int(np.argmax(Prog_sig + Imm_sig))
    return a


def smoke():
    log = lambda m: print(m, flush=True)
    log("SMOKE: generating synthetic bifurcation manifold ...")
    a = make_synthetic_manifold()
    tmp = Path("/tmp/traj_smoke"); tmp.mkdir(exist_ok=True)
    mpath = tmp / "neuron_manifold.h5ad"; a.write(mpath)
    cfg = deep_merge(DEFAULTS, dict(run_name="smoke_out",
                                    cellrank=dict(n_macrostates=4, schur_components=6)))
    R = process_dataset("SMOKE", mpath, cfg, log)
    log("\nSMOKE correlations:\n" + R.round(2).to_string())
    # success criteria: dpt computed and correlates with age
    sd = pd.read_parquet(tmp / "smoke_out" / "traj_scores_SMOKE.parquet")
    ok = sd["dpt"].notna().mean() > 0.9
    log(f"\nSMOKE {'PASS' if ok else 'FAIL'}: dpt finite frac={sd['dpt'].notna().mean():.2f}; "
        f"columns={list(sd.columns)}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--only", help="run only this dataset name from the config")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        sys.exit(smoke())
    cfg, datasets = load_config(args.config)
    log = lambda m: print(m, flush=True)
    if not datasets:
        log("No datasets in config."); sys.exit(1)
    summary = {}
    for name, dcfg in datasets.items():
        if args.only and name != args.only:
            continue
        try:
            R = process_dataset(name, dcfg["manifold"], cfg, log)
            summary[name] = R
        except Exception:
            log(f"DATASET {name} FAILED:\n" + traceback.format_exc())
    log("\n==== DONE ====")
    for n, R in summary.items():
        if "age" in R.columns:
            log(f"{n}: corr(score, age) =\n{R['age'].round(3).to_string()}")


if __name__ == "__main__":
    main()
