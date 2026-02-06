import mygene
import numpy as np

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
