"""Shared-vocabulary cell-type labels for semi-supervised scANVI training.

This module owns the loading/validation/application of `shared_fine_labels.csv`,
the curated mapping from each dataset's native fine-label column onto a shared
vocabulary. See `reference/shared_fine_labels.csv` for the file format.

When the pipeline's `--use_shared_labels` flag is true, `downsample.py` consults
this module to translate native fine labels (Velmeshev `cell_type`, Wang
`Type-updated` / `cell_type_raw`, PsychAD `subclass`) into the shared
vocabulary BEFORE scANVI training. Cells whose native label doesn't have a
clean mapping remain "Unknown" and are treated as unlabelled by scANVI.

Motivation: in the Wang-only-supervised configuration scANVI's classifier
learns only from Wang's perinatal cell-type distribution and mis-routes young
PsychAD donors away from Excitatory entirely. Supervising training with all
three datasets' biological labels (via this CSV) anchors the latent space to
each dataset's structure.

See scripts/integration_qc/outputs/composition/interpretation.md for the
empirical observation that motivated this change.
"""
from __future__ import annotations

import os

import anndata as ad
import pandas as pd


VALID_BROAD_CLASSES = frozenset([
    'Excitatory', 'Inhibitory', 'Astrocytes', 'Oligos', 'OPC',
    'Microglia', 'Endothelial', 'Other',
])

DATASET_COL = {
    'Wang':      'wang_type_updated',
    'Velmeshev': 'vel_cell_type',
    'PsychAD':   'psychad_subclass',
}

REQUIRED_COLUMNS = [
    'shared_label', 'broad_class',
    'vel_cell_type', 'wang_type_updated', 'psychad_subclass',
]


def load_shared_label_map(csv_path: str) -> pd.DataFrame:
    """Load and validate the shared-label CSV.

    Raises ValueError with an informative message on any schema problem.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'shared_fine_labels.csv not found: {csv_path}')
    df = pd.read_csv(csv_path).fillna('')

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f'shared_fine_labels.csv missing required columns: {missing}. '
            f'Got: {list(df.columns)}')

    dup = df['shared_label'][df['shared_label'].duplicated()].tolist()
    if dup:
        raise ValueError(f'duplicate shared_label values: {dup}')

    bad_broad = sorted(set(df['broad_class']) - VALID_BROAD_CLASSES)
    if bad_broad:
        raise ValueError(
            f'invalid broad_class values: {bad_broad}. '
            f'Allowed: {sorted(VALID_BROAD_CLASSES)}')

    for col in ('vel_cell_type', 'wang_type_updated', 'psychad_subclass'):
        for i, entry in enumerate(df[col]):
            for frag in str(entry).split('|'):
                if frag != frag.strip():
                    raise ValueError(
                        f'whitespace around fragment in row {i} of {col!r}: '
                        f'{entry!r}')

    return df


def _native_to_shared(mapping: pd.DataFrame, dataset_type: str) -> dict[str, str]:
    """Return {native_fine_label: shared_label} for the requested dataset."""
    col = DATASET_COL[dataset_type]
    table: dict[str, str] = {}
    for _, row in mapping.iterrows():
        shared = row['shared_label']
        entry = str(row[col])
        for frag in entry.split('|'):
            if not frag:
                continue
            if frag in table and table[frag] != shared:
                raise ValueError(
                    f'ambiguous mapping for {dataset_type} native label '
                    f'{frag!r}: '
                    f'{table[frag]!r} vs {shared!r}')
            table[frag] = shared
    return table


def apply_shared_labels(
    adata: ad.AnnData,
    dataset_type: str,
    fine_label_col: str,
    mapping: pd.DataFrame,
    unlabeled_token: str = 'Unknown',
) -> tuple[pd.Series, dict]:
    """Translate adata's native fine labels into the shared vocabulary.

    Parameters
    ----------
    adata : AnnData
    dataset_type : "Wang" | "Velmeshev" | "PsychAD"
        Selects which mapping column to consult.
    fine_label_col : str
        Column in adata.obs holding the native fine labels.
    mapping : DataFrame
        As returned by load_shared_label_map().
    unlabeled_token : str
        Value for cells whose native label has no clean mapping.

    Returns
    -------
    labels : pd.Series
        Aligned to adata.obs.index, values in the shared vocabulary or
        `unlabeled_token`.
    summary : dict
        Diagnostics for logging: n_cells, n_mapped, n_unmapped,
        coverage_fraction, unmapped_top10, n_shared_labels.
    """
    if dataset_type not in DATASET_COL:
        raise ValueError(
            f'dataset_type {dataset_type!r} not one of {sorted(DATASET_COL)}')
    if fine_label_col not in adata.obs.columns:
        raise KeyError(
            f'fine_label_col {fine_label_col!r} not in adata.obs '
            f'(have: {list(adata.obs.columns)[:20]}...)')

    table = _native_to_shared(mapping, dataset_type)
    native = adata.obs[fine_label_col].astype(str)
    labels = native.map(table).fillna(unlabeled_token)
    labels.index = adata.obs.index

    is_mapped = labels != unlabeled_token
    unmapped = native[~is_mapped].value_counts().head(10).to_dict()
    summary = dict(
        n_cells=len(native),
        n_mapped=int(is_mapped.sum()),
        n_unmapped=int((~is_mapped).sum()),
        coverage_fraction=float(is_mapped.mean()),
        unmapped_top10=unmapped,
        n_shared_labels=int(labels[is_mapped].nunique()),
    )
    return labels, summary
