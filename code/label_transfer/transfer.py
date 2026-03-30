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

# Labels too broad to serve as informative reference
BROAD_LABELS = frozenset({'Excitatory', 'Inhibitory', 'Glia', 'Other', 'Unknown'})


def derive_subclass(obs):
    """Derive cell_subclass from cell_type using per-source mappers.

    - VELMESHEV: map_velmeshev_subclass  (source-specific labels in cell_type)
    - All others: map_cellxgene_subclass (CellxGene ontology terms)
    - PsychAD EN sub-variants collapsed  (EN_L3_5_IT_1/2/3 → EN_L3_5_IT)
    """
    from read_data import (map_cellxgene_subclass, map_velmeshev_subclass,
                           collapse_en_subclass)

    subclass = obs['cell_type'].map(map_cellxgene_subclass)
    vel = obs['source'] == 'VELMESHEV'
    subclass.loc[vel] = obs.loc[vel, 'cell_type'].map(map_velmeshev_subclass)
    subclass = subclass.map(collapse_en_subclass)
    return subclass


def subclass_to_class(subclass):
    """Map a cell subclass label to a broad cell class.

    Returns the same vocabulary used in the cell_class obs column so that
    old_cell_class and new_cell_class are directly comparable.
    """
    if subclass.startswith('EN_'):
        return 'Excitatory'
    if subclass.startswith('IN_'):
        return 'Inhibitory'
    _MAP = {
        'Astro': 'Astrocytes', 'Astro_Immature': 'Astrocytes',
        'Oligo': 'Oligos', 'OPC': 'OPC', 'Micro': 'Microglia',
        'Endo': 'Endothelial', 'PC': 'Endothelial', 'PVM': 'Endothelial',
        'SMC': 'Endothelial', 'VLMC': 'Endothelial',
        'Progenitors': 'Glia', 'Glial_progenitors': 'Glia',
        'Radial_glia': 'Glia', 'CR_cell': 'Other',
        'Adaptive': 'Microglia',
        'Excitatory': 'Excitatory', 'Inhibitory': 'Inhibitory',
        'Glia': 'Glia', 'Other': 'Other', 'Unknown': 'Other',
    }
    return _MAP.get(subclass, 'Other')


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
