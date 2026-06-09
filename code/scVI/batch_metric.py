"""Batch-integration quality metric for scVI latents.

A kNN-based mixing score normalised for batch count: for a sample of cells,
the observed fraction of nearest neighbours from a *different* batch divided by
the batch-random expectation (1 - sum p_b^2). 1.0 = neighbours are batch-random
(well integrated); ~0 = neighbours are same-batch (poor integration). Logged at
the end of scVI training so a poorly-mixed run is flagged automatically.
"""
import numpy as np
import pandas as pd


def batch_mixing_score(Z, batch, k=30, n_sample=40000, seed=0,
                       warn_threshold=0.20, logger=None):
    """Return mixing score in ~[0,1]; log a WARNING if below warn_threshold."""
    from sklearn.neighbors import NearestNeighbors
    Z = np.asarray(Z)
    batch = np.asarray(batch).astype(str)
    n = Z.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, min(n_sample, n), replace=False)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    _, ind = nn.kneighbors(Z[idx])
    b = pd.factorize(batch)[0]
    same = (b[ind[:, 1:]] == b[idx][:, None]).mean(1)
    obs_cross = 1.0 - float(same.mean())
    p = pd.Series(batch).value_counts(normalize=True).values
    exp_cross = 1.0 - float(np.sum(p ** 2))
    score = obs_cross / exp_cross if exp_cross > 0 else np.nan

    # per-batch mixing (which batch fails to integrate)
    per = {}
    for bi, name in enumerate(pd.factorize(batch)[1]):
        sel = b[idx] == bi
        if sel.sum():
            per[str(name)] = round(1.0 - float(same[sel].mean()), 3)

    msg = (f"batch-mixing score = {score:.3f} "
           f"(obs cross-batch kNN {obs_cross:.2f} / expected {exp_cross:.2f}; "
           f"per-batch cross-frac {per}); 1=well mixed, 0=separated")
    if logger is not None:
        logger.info(msg)
        if np.isfinite(score) and score < warn_threshold:
            logger.warning(
                f"LOW BATCH-MIXING ({score:.3f} < {warn_threshold}): batches are "
                f"poorly integrated in the scVI latent — inspect before trusting "
                f"this embedding (e.g. balance batch cell counts).")
    else:
        print(msg)
    return float(score)
