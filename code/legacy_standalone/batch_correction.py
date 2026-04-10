"""Batch correction for projection scores via OLS residualisation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def correct_projection_scores(
    adata,
    obsm_key='X_ahba',
    batch_key='source',
    covariates=None,
    corrected_key=None,
):
    """
    Regress out batch effects from projection scores using OLS,
    preserving the effects of biological covariates.

    For each component, fits:  score ~ batch + covariates
    Returns residuals + covariate effects + intercept, so that
    biological signal (e.g. age) is retained but batch shifts are removed.

    Parameters
    ----------
    adata : AnnData
        Must contain adata.obsm[obsm_key] (the raw projection scores).
    obsm_key : str
        Key in obsm holding the projection matrix.
    batch_key : str
        Column in adata.obs encoding the batch/dataset.
    covariates : list of str, optional
        Columns in adata.obs for biological variables to preserve.
        Numeric columns are used as-is; categorical/object/bool columns
        are one-hot encoded automatically.
    corrected_key : str, optional
        Key under which to store corrected scores in adata.obsm.
        Defaults to obsm_key + '_corrected'.

    Returns
    -------
    None — modifies adata.obsm in place.
    """
    if corrected_key is None:
        corrected_key = obsm_key + '_corrected'

    scores = adata.obsm[obsm_key].copy()  # (n_cells, n_components)
    n_cells, n_comp = scores.shape

    # --- build design matrix ---
    # batch dummies (drop_first to avoid collinearity with intercept)
    batch = adata.obs[batch_key].values.reshape(-1, 1)
    enc_batch = OneHotEncoder(drop='first', sparse_output=False)
    X_batch = enc_batch.fit_transform(batch)

    # covariate columns: numeric as-is, categorical one-hot encoded
    cov_parts = []
    if covariates:
        for col in covariates:
            series = adata.obs[col]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                cov_parts.append(series.values.astype(float).reshape(-1, 1))
            else:
                enc = OneHotEncoder(drop='first', sparse_output=False)
                cov_parts.append(enc.fit_transform(series.values.reshape(-1, 1)))

    X_cov = np.hstack(cov_parts) if cov_parts else np.empty((n_cells, 0))
    n_cov_cols = X_cov.shape[1]

    # intercept + covariates + batch
    X = np.column_stack([np.ones(n_cells), X_cov, X_batch])

    # --- OLS per component ---
    beta = np.linalg.lstsq(X, scores, rcond=None)[0]  # (n_features, n_comp)

    # keep intercept + covariate effects, remove batch effects
    keep_cols = list(range(1 + n_cov_cols))  # intercept + all covariate columns

    corrected = scores - X @ beta                           # full residuals
    corrected += X[:, keep_cols] @ beta[keep_cols, :]       # add back intercept + covariate effects

    adata.obsm[corrected_key] = corrected
    print(f"Stored corrected scores in adata.obsm['{corrected_key}'] "
          f"(batch_key='{batch_key}', covariates={covariates})")
