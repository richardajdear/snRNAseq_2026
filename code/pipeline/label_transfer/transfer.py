"""Core label transfer functions using kNN in scVI embedding space.

The scVI embedding removes batch (source) effects while preserving biological
variation including age and maturation state.  This means:
  - Adult Velmeshev cells naturally neighbour adult AGING/HBCC reference cells
  - Fetal/immature Velmeshev cells neighbour fetal WANG reference cells
  - No explicit age conditioning is required
  - Low kNN confidence flags cells without a good reference match
    (e.g. developmental types absent from the reference)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Wang 'Type-updated' values that are too broad to serve as informative reference.
# NOTE: inspect Wang obs['Type-updated'].value_counts() to verify / extend this list.
BROAD_LABELS = frozenset({
    'Unknown', 'unknown',
    # Canonical broad labels that may appear in other datasets:
    'Excitatory', 'Inhibitory', 'Glia', 'Other',
})


def aligned_to_class(cell_type_aligned):
    """Map a cell_type_aligned label (Wang Type-updated vocabulary) to a broad cell_class.

    Handles both Wang-style labels (EN-L2_3-IT, IN-MGE-PV, Astrocyte-Fibrous, ...)
    and the canonical vocabulary used by older pipeline versions (EN_L2_3_IT, Astro, ...).

    Returns one of: Excitatory, Inhibitory, Astrocytes, Oligos, OPC, Microglia,
                    Endothelial, Glia, Other
    """
    if not isinstance(cell_type_aligned, str):
        return 'Other'

    s = cell_type_aligned

    # Wang-style: EN-* → Excitatory
    if s.startswith('EN-'):
        return 'Excitatory'
    # Wang-style: IN-* → Inhibitory
    if s.startswith('IN-'):
        return 'Inhibitory'
    # Wang-style: Astrocyte-*
    if s.startswith('Astrocyte'):
        return 'Astrocytes'
    # Wang-style: Oligodendrocyte*
    if s.startswith('Oligodendrocyte'):
        return 'Oligos'
    # Wang-style: IPC-* / RG-*
    if s.startswith('IPC-') or s.startswith('RG-'):
        return 'Glia'

    # Canonical labels (EN_* / IN_* / Astro / Oligo / ...)
    if s.startswith('EN_'):
        return 'Excitatory'
    if s.startswith('IN_'):
        return 'Inhibitory'

    _MAP = {
        # Wang-style
        'OPC':               'OPC',
        'Microglia':         'Microglia',
        'Vascular':          'Endothelial',
        'Cajal-Retzius cell': 'Other',
        # Canonical
        'Astro':             'Astrocytes',
        'Astro_Immature':    'Astrocytes',
        'Oligo':             'Oligos',
        'Micro':             'Microglia',
        'Endo':              'Endothelial',
        'PC':                'Endothelial',
        'PVM':               'Endothelial',
        'SMC':               'Endothelial',
        'VLMC':              'Endothelial',
        'Progenitors':       'Glia',
        'Glial_progenitors': 'Glia',
        'Radial_glia':       'Glia',
        'CR_cell':           'Other',
        'Adaptive':          'Microglia',
        # Pass-through broad labels
        'Excitatory':        'Excitatory',
        'Inhibitory':        'Inhibitory',
        'Glia':              'Glia',
        'Other':             'Other',
        'Unknown':           'Other',
    }
    return _MAP.get(s, 'Other')


def knn_transfer(query_emb, ref_emb, ref_labels, k=50):
    """Weighted kNN label transfer.

    For each query cell, finds *k* nearest reference cells and assigns the label
    with the highest inverse-distance-weighted vote.

    Returns
    -------
    labels : ndarray of str
    confidences : ndarray of float  (0–1, fraction of weight for winning label)
    mean_distances : ndarray of float
    """
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nn.fit(ref_emb)
    distances, indices = nn.kneighbors(query_emb)

    weights = 1.0 / (distances + 1e-8)

    # Vectorised weighted vote
    unique = np.unique(ref_labels)
    lbl2idx = {l: i for i, l in enumerate(unique)}
    ref_idx = np.array([lbl2idx[l] for l in ref_labels])

    nbr_idx = ref_idx[indices]                        # (n_query, k)
    n_query, n_labels = len(query_emb), len(unique)
    votes = np.zeros((n_query, n_labels), dtype=np.float64)
    for j in range(k):
        np.add.at(votes, (np.arange(n_query), nbr_idx[:, j]), weights[:, j])

    winner = votes.argmax(axis=1)
    total  = votes.sum(axis=1)
    conf   = votes[np.arange(n_query), winner] / total

    return unique[winner], conf, distances.mean(axis=1)
