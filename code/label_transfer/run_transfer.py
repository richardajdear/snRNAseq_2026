"""Standalone label-transfer job.

Assigns cell subclass labels to Velmeshev cells via weighted kNN in scVI
latent space, using AGING + HBCC + WANG cells (with specific, non-broad
subclass labels) as the reference.

Usage
-----
    PYTHONPATH=code python3 -m label_transfer.run_transfer \
        --input  path/to/integrated.h5ad \
        --output_dir path/to/label_transfer/

Outputs
-------
    transferred_labels.csv  — Velmeshev cells: cell_id, old/new class+subclass,
                              confidence, h5ad_pos (for expression indexing)
    all_cell_labels.csv     — ALL cells: cell_id, old/new class+subclass,
                              UMAP coords
"""

import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc

from label_transfer.transfer import (derive_subclass, knn_transfer,
                                     subclass_to_class, BROAD_LABELS)


def main():
    p = argparse.ArgumentParser(description='kNN label transfer in scVI space')
    p.add_argument('--input', required=True, help='integrated.h5ad path')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--k', type=int, default=50)
    p.add_argument('--confidence_threshold', type=float, default=0.5)
    p.add_argument('--embedding_key', default='X_scVI')
    p.add_argument('--umap_key', default='X_umap_scvi')
    p.add_argument('--target_sources', nargs='+', default=None,
                   help='Default: all non-reference sources')
    p.add_argument('--reference_sources', nargs='+', default=['WANG'],
                   help='Default: WANG only')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load ─────────────────────────────────────────────────────────────
    print(f"Loading {args.input} (backed) …")
    adata = sc.read_h5ad(args.input, backed='r')
    obs  = adata.obs.copy()
    emb  = np.array(adata.obsm[args.embedding_key])
    umap = np.array(adata.obsm[args.umap_key])
    print(f"  {adata.shape[0]} cells, embedding dim={emb.shape[1]}")

    # Preserve original cell IDs and positional index
    obs['cell_id']   = obs.index.values
    obs['h5ad_pos']  = np.arange(len(obs))

    ref_sources = args.reference_sources
    target_sources = args.target_sources or sorted(
        s for s in obs['source'].unique() if s not in set(ref_sources))

    # ── derive subclass ──────────────────────────────────────────────────
    print("Deriving cell_subclass (per-source mappers) …")
    obs['old_subclass'] = derive_subclass(obs)

    # ── build reference ──────────────────────────────────────────────────
    ref_mask    = (obs['source'].isin(ref_sources)
                   & ~obs['old_subclass'].isin(BROAD_LABELS)).values
    target_mask = obs['source'].isin(target_sources).values

    ref_labels = obs.loc[ref_mask, 'old_subclass'].values
    print(f"\nReference: {ref_mask.sum()} cells from {ref_sources}")
    print(f"  {len(np.unique(ref_labels))} unique labels")
    for lbl, n in (pd.Series(ref_labels)
                   .value_counts().items()):
        print(f"    {lbl:25s} {n:6d}")

    print(f"\nTarget: {target_mask.sum()} cells from {target_sources}")
    for _src in target_sources:
        print(f"  {_src}: {(obs['source'] == _src).sum()} cells")

    # ── kNN transfer ─────────────────────────────────────────────────────
    print(f"\nRunning kNN transfer (k={args.k}, "
          f"embedding={args.embedding_key}) …")
    transferred, confidence, mean_dist = knn_transfer(
        emb[target_mask], emb[ref_mask], ref_labels, k=args.k)
    print("  Done.")

    # ── save Velmeshev detail ────────────────────────────────────────────
    cols = ['cell_id', 'h5ad_pos', 'source', 'cell_class', 'cell_type',
            'age_years']
    results = obs.loc[target_mask, cols].copy()
    results['old_cell_class']       = results['cell_class']
    results['old_subclass']         = obs.loc[target_mask, 'old_subclass'].values
    results['transferred_subclass'] = transferred
    results['new_cell_class']       = pd.Series(transferred).map(subclass_to_class).values
    results['transfer_confidence']  = np.round(confidence, 4)
    results['mean_knn_distance']    = np.round(mean_dist, 4)
    results['is_low_confidence']    = confidence < args.confidence_threshold
    results['is_class_remapped']    = results['old_cell_class'] != results['new_cell_class']
    results = results.reset_index(drop=True)

    detail_path = os.path.join(args.output_dir, 'transferred_labels.csv')
    results.to_csv(detail_path, index=False)
    print(f"\nSaved: {detail_path}")

    # ── save all-cell labels + UMAP coords ───────────────────────────────
    all_cols = ['cell_id', 'h5ad_pos', 'source', 'cell_class', 'cell_type',
                'age_years']
    all_df = obs[all_cols].copy()
    all_df['old_cell_class'] = all_df['cell_class']
    all_df['old_subclass']   = obs['old_subclass']
    all_df['new_subclass']   = obs['old_subclass'].copy()
    all_df.loc[target_mask, 'new_subclass'] = transferred
    all_df['new_cell_class'] = all_df['new_subclass'].map(subclass_to_class)
    all_df['transfer_confidence'] = np.nan
    all_df.loc[target_mask, 'transfer_confidence'] = np.round(confidence, 4)
    all_df['umap_1'] = umap[:, 0]
    all_df['umap_2'] = umap[:, 1]
    all_df = all_df.reset_index(drop=True)

    all_path = os.path.join(args.output_dir, 'all_cell_labels.csv')
    all_df.to_csv(all_path, index=False)
    print(f"Saved: {all_path}")

    # ── summary ──────────────────────────────────────────────────────────
    low  = results['is_low_confidence']
    rmap = results['is_class_remapped']
    sep  = '=' * 60
    print(f"\n{sep}\nTRANSFER SUMMARY\n{sep}")
    print(f"Reference: {ref_sources}  →  Target: {target_sources}")
    print(f"Low confidence (<{args.confidence_threshold}): "
          f"{low.sum()} / {len(results)} ({100 * low.mean():.1f}%)")
    print(f"Class-remapped: "
          f"{rmap.sum()} / {len(results)} ({100 * rmap.mean():.1f}%)")

    print("\nPer-source breakdown:")
    for _src in target_sources:
        sm = results['source'] == _src
        if sm.sum() == 0:
            continue
        slc = results.loc[sm, 'is_low_confidence']
        srm = results.loc[sm, 'is_class_remapped']
        print(f"  {_src:12s}: n={sm.sum():6d}  low_conf={slc.sum()} ({100*slc.mean():.1f}%)  "
              f"class_remapped={srm.sum()} ({100*srm.mean():.1f}%)")

    print("\nTransferred label distribution:")
    for lbl, n in results['transferred_subclass'].value_counts().items():
        c = results.loc[
            results['transferred_subclass'] == lbl, 'transfer_confidence'
        ].mean()
        print(f"  {lbl:25s} {n:6d}  (mean conf {c:.3f})")

    # Age-stratified confidence
    print("\nConfidence by age group:")
    age = results['age_years']
    for (lo, hi), tag in [
        ((-np.inf, 0),   'Fetal (<0 y)'),
        ((0, 18),        'Postnatal (0–18 y)'),
        ((18, np.inf),   'Adult (>18 y)'),
    ]:
        m = (age >= lo) & (age < hi)
        if m.sum() == 0:
            continue
        c = results.loc[m, 'transfer_confidence']
        lc = results.loc[m, 'is_low_confidence']
        print(f"  {tag:20s}: n={m.sum():6d}  mean_conf={c.mean():.3f}  "
              f"low_conf={lc.sum()} ({100 * lc.mean():.1f}%)")

    print("\nConfidence by cell_class:")
    for cls in sorted(results['cell_class'].unique()):
        m = results['cell_class'] == cls
        c = results.loc[m, 'transfer_confidence']
        lc = results.loc[m, 'is_low_confidence']
        print(f"  {cls:15s}: n={m.sum():6d}  mean_conf={c.mean():.3f}  "
              f"low_conf={lc.sum()} ({100 * lc.mean():.1f}%)")

    # Class remapping summary
    if rmap.sum() > 0:
        print("\nClass remapping breakdown:")
        for (oc, nc), n in (results.loc[rmap]
                            .groupby(['old_cell_class', 'new_cell_class'])
                            .size().sort_values(ascending=False).items()):
            c = results.loc[rmap & (results['old_cell_class'] == oc) &
                            (results['new_cell_class'] == nc),
                            'transfer_confidence'].mean()
            print(f"  {oc:15s} → {nc:15s}: {n:5d}  (mean conf {c:.3f})")


if __name__ == '__main__':
    main()
