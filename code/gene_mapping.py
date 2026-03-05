import mygene
import numpy as np
import pandas as pd

def map_ensembl_to_symbol(adata):
    """
    Maps Ensembl IDs (in adata.var_names) to Gene Symbols using mygene.
    Handles duplicates by taking the first match and ensuring uniqueness.
    """
    mg = mygene.MyGeneInfo()
    ensembl_ids = adata.var_names.tolist()
    print(f"Querying mygene for {len(ensembl_ids)} genes...")
    
    # Query mygene
    results = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')
    
    # Create a mapping dict
    mapping = {}
    for res in results:
        query = res.get('query')
        symbol = res.get('symbol')
        if query and symbol and query not in mapping:
            mapping[query] = symbol
            
    # Apply mapping
    new_names = [mapping.get(idx, idx) for idx in adata.var_names]
    
    adata.var['gene_symbol'] = new_names
    adata.var_names = new_names
    adata.var_names_make_unique()
    
    print("Gene mapping complete. var_names updated to symbols (unique).")
    return adata


def map_grn_symbols_to_ensembl(grn, adata):
    """
    Remap the 'Gene' column of a GRN DataFrame from gene symbols to Ensembl IDs,
    using adata.var as the primary mapping and mygene as a fallback.

    Parameters:
    - grn: DataFrame with a 'Gene' column containing gene symbols.
    - adata: AnnData whose var_names are Ensembl IDs and var has a 'gene_symbol' column.

    Returns:
    - A copy of grn with 'Gene' replaced by Ensembl IDs. Rows that could not be
      mapped are dropped.
    """
    symbols = grn['Gene'].unique()

    # 1) Build mapping from adata.var (symbol -> ensembl ID)
    #    For duplicates, keep the first occurrence.
    var = adata.var[['gene_symbol']].copy()
    var['ensembl_id'] = var.index
    var_dedup = var.drop_duplicates(subset='gene_symbol', keep='first')
    local_map = dict(zip(var_dedup['gene_symbol'], var_dedup['ensembl_id']))

    mapped = {s: local_map[s] for s in symbols if s in local_map}
    unmapped = [s for s in symbols if s not in mapped]
    print(f"Mapped {len(mapped)}/{len(symbols)} symbols via adata.var")

    # 2) Fallback: query mygene for unmapped symbols
    if unmapped:
        print(f"Querying mygene for {len(unmapped)} unmapped symbols...")
        mg = mygene.MyGeneInfo()
        results = mg.querymany(unmapped, scopes='symbol', fields='ensembl.gene',
                               species='human', returnall=False)
        for res in results:
            symbol = res.get('query')
            ensembl = res.get('ensembl')
            if symbol and ensembl:
                # ensembl can be a list of dicts or a single dict
                if isinstance(ensembl, list):
                    eid = ensembl[0].get('gene')
                else:
                    eid = ensembl.get('gene')
                if eid and eid in adata.var_names:
                    mapped[symbol] = eid

        still_unmapped = len(symbols) - len(mapped)
        print(f"After mygene: {len(mapped)}/{len(symbols)} mapped, {still_unmapped} dropped")

    # 3) Remap and drop unmapped rows
    grn_out = grn.copy()
    grn_out['Gene'] = grn_out['Gene'].map(mapped)
    grn_out = grn_out.dropna(subset=['Gene'])

    return grn_out
