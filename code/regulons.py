import numpy as np
import pandas as pd
import anndata as ad

def get_ahba_GRN(
        path_to_ahba_weights="../data/ahba_dme_hcp_top8kgenes_weights.csv",
        use_weights=False
    ):
    """
    Load the AHBA Gene Regulatory Network (GRN) from a CSV file.
    Parameters:
    - path_to_ahba_weights: Path to the CSV file containing AHBA weights.
    - use_weights: If True, keep all genes with original weights (zeroing out
      wrong-sign genes). If False (default), take top/bottom 1000 genes with
      Importance set to 1.
    Returns:
    - ahba_GRN: DataFrame with columns 'Network', 'Gene', and 'Importance'.
    """
    ahba = pd.read_csv(path_to_ahba_weights, index_col=0)

    ahba_melt = (ahba
                .reset_index()
                .rename(columns={'index': 'Gene'})
                .melt(id_vars=['Gene'], var_name='Network', value_name='Importance'))

    if use_weights:
        ahba_GRNpos = (ahba_melt
                    .assign(Importance=lambda x: x['Importance'].clip(lower=0))
                    .assign(Network=lambda x: x['Network'] + '+'))

        ahba_GRNneg = (ahba_melt
                    .assign(Importance=lambda x: (-x['Importance']).clip(lower=0))
                    .assign(Network=lambda x: x['Network'] + '-'))
    else:
        ahba_GRNpos = (ahba_melt
                    .sort_values('Importance', ascending=False)
                    .loc[lambda x: x.groupby('Network')['Importance'].rank(ascending=False)<1000]
                    .assign(Importance=1)
                    .assign(Network=lambda x: x['Network'] + '+'))

        ahba_GRNneg = (ahba_melt
                    .sort_values('Importance', ascending=True)
                    .loc[lambda x: x.groupby('Network')['Importance'].rank(ascending=True)<1000]
                    .assign(Importance=1)
                    .assign(Network=lambda x: x['Network'] + '-'))

    ahba_GRN = pd.concat([ahba_GRNpos, ahba_GRNneg])
    return ahba_GRN


def project_GRN(adata, GRN, GRN_name='GRN', use_raw=False, use_residuals=False, normalize=False, use_highly_variable=True, log_transform=False):
    """
    Project a Gene Regulatory Network (GRN) onto an AnnData object.
    The default and recommended way to use this function is to apply it to expression data that has been normalized but NOT log transformed.
    Instead, apply the log transformation after the projection.
    Parameters:
    - GRN: DataFrame containing the GRN with columns 'TF', 'Gene', and 'Importance'.
    - GRN_name: Name to use for the resulting obsm key.
    - adata: AnnData object containing the data to project onto.
    - use_raw: If True, uses the raw data from adata.raw. Defaults to False.
    - use_residuals: If True, uses residuals from pearson residuals normalization instead of raw data. Defaults to False.
    - normalize: If True, normalizes the projected data to have a total sum of 1e6. Defaults to False.
    - log_transform: If True, applies log1p transformation to the projected data. Defaults to True.
    Returns:
    - None: The function modifies adata.obsm in place.
    """

    GRN_pivot = GRN.pivot_table(
        index='Network',
        columns='Gene', 
        values='Importance', 
        fill_value=0
    )

    if use_residuals:
        # Use the pearson residuals normalization 
        X = adata.uns['pearson_residuals_normalization']['pearson_residuals_df']
        # Matched genes are the intersection of the GRN genes and the normalized genes
        matched_genes = np.intersect1d(GRN_pivot.columns, X.columns)
        X = X.loc[:, lambda x: ~x.columns.duplicated()].loc[:, matched_genes]
        Y = GRN_pivot.loc[:, matched_genes]
        projected = (X @ Y.T)
        
    elif use_raw:
        # Use the raw data
        X = pd.DataFrame(adata.raw.X.todense(), index=adata.obs_names, columns=adata.var_names)
        # Matched genes are the intersection of the GRN genes and the adata var genes
        matched_genes = np.intersect1d(GRN_pivot.columns, adata.var.index)
        X = X.loc[:, lambda x: ~x.columns.duplicated()].loc[:, matched_genes]
        Y = GRN_pivot.loc[:, matched_genes]
        projected = (X @ Y.T)
        
    else:
        # Use whatever is in adata.X
        
        # Check for gene overlap and handle Ensembl ID vs Symbol mismatch
        # GRN columns are typically Symbols. adata.var_names might be Ensembl IDs.
        grn_genes = GRN_pivot.columns
        overlap = np.intersect1d(grn_genes, adata.var_names)
        
        if len(overlap) == 0:
            print("Warning: Zero overlap between GRN genes and adata.var_names.")
            # Try to find a better column
            potential_cols = ['feature_name', 'gene_symbols', 'symbol', 'gene_name']
            best_col = None
            max_overlap = 0
            
            for col in potential_cols:
                if col in adata.var.columns:
                    # Check overlap if we were to use this column
                    col_values = adata.var[col].astype(str).values
                    curr_overlap = len(np.intersect1d(grn_genes, col_values))
                    print(f"Checking column '{col}': {curr_overlap} matching genes")
                    if curr_overlap > max_overlap:
                        max_overlap = curr_overlap
                        best_col = col
            
            if best_col and max_overlap > 0:
                print(f"Swapping adata.var_names with '{best_col}' ({max_overlap} matches)...")
                adata.var[best_col] = adata.var[best_col].astype(str)
                adata.var_names = adata.var[best_col]
                # Make unique just in case
                adata.var_names_make_unique()
            else:
                print("ERROR: Could not find any column with significant overlap with GRN genes.")
        else:
            print(f"Found {len(overlap)} matching genes in var_names.")

        # Identify available genes in adata
        available_genes = adata.var_names
        
        # Optionally filter for highly variable genes
        if use_highly_variable:
            if 'highly_variable' in adata.var.columns:
                available_genes = available_genes[adata.var['highly_variable']]
            else:
                print("Warning: use_highly_variable=True but 'highly_variable' not in adata.var. Using all genes.")
            
        # Matched genes are the intersection of the GRN genes and the adata available genes
        matched_genes = np.intersect1d(GRN_pivot.columns, available_genes)
        
        print(f"Aligning GRN weights to {len(matched_genes)} matched genes for projection...")
        # To avoid scipy fancy indexing bugs and memory blowouts on huge sparse matrices (e.g. 2.1 million cells),
        # we align the small GRN matrix Y to full adata.X genes and do a sparse-dense dot product directly.
        Y_aligned = pd.DataFrame(0.0, index=adata.var_names, columns=GRN_pivot.index)
        Y_aligned = Y_aligned[~Y_aligned.index.duplicated(keep='first')]  # Safety against invalid duplicates
        Y_aligned.loc[matched_genes, :] = GRN_pivot.loc[:, matched_genes].T
        
        # Ensure ordered mapping
        Y_weights = Y_aligned.loc[adata.var_names].values
        
        print("Computing sparse-dense dot product...")
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            projected_vals = adata.X.dot(Y_weights)
        else:
            projected_vals = np.dot(adata.X, Y_weights)
            
        projected = pd.DataFrame(projected_vals, index=adata.obs_names, columns=GRN_pivot.index)

    if normalize:
        # Normalize the projected data to have a total sum of 1e6
        projected = (projected.T / projected.sum(axis=1)).T * 1e4
    if log_transform:
        # Apply log1p transformation
        projected = np.log1p(projected)

    # Add to adata.obsm as array to work with scanpy plotting
    adata.obsm[GRN_name] = projected.values
    adata.uns[GRN_name + '_names'] = GRN_pivot.index.tolist()
