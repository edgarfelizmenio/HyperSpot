import scanpy as sc
import pandas as pd
import numpy as np

from ..metadata import ProjectMetadata

def extract_sc_patient_data(adata, patient_id):
    adata = adata[adata.obs['patient'] == patient_id].copy()
    return adata

def find_conflicting_genes(adata_sc, adata_st, geneset: pd.DataFrame):
    st_map = adata_st.var[['SYMBOL']].copy()
    st_map = st_map[st_map.index.isin(geneset.index)]
    st_map = st_map[st_map['SYMBOL'].notna()]
    
    dupes = st_map[st_map['SYMBOL'].duplicated(keep=False)].copy()
    
    if len(dupes) == 0:
        print("No duplicate symbols found.")
        # return empty dataframe
        return pd.DataFrame()

    print(dupes)
    
    counts = np.array(adata_st[:, dupes.index].X.sum(axis=0)).flatten()
    dupes['total_counts'] = counts
    # percentage of spots that have nonzero expression of the gene
    dupes['pct_nonzero'] = (
        np.array((adata_st[:, dupes.index].X > 0).sum(axis=0)).flatten() / adata_st.n_obs
    )
    
    dupes['would_keep'] = False
    
    dupes.loc[dupes.groupby('SYMBOL', observed=True)['total_counts'].idxmax(), 'would_keep'] = True
    
    dupes['in_sc'] = dupes['SYMBOL'].isin(adata_sc.var_names)
    
    n_conflicts = dupes['SYMBOL'].nunique()
    n_relevant = dupes[dupes['in_sc']]['SYMBOL'].nunique()
    
    print(f"Found {n_conflicts} symbols with duplicate Ensembl IDs")
    print(f"  → {n_relevant} overlap with scRNA-seq and require manual inspection")
    print(f"  → {n_conflicts - n_relevant} are absent from scRNA-seq and will be handled automatically\n")
    
    return dupes.sort_values(['in_sc', 'SYMBOL', 'total_counts'], ascending=[False, True, False])    

def use_ensembl_gene_ids(
    adata_sc, 
    adata_st, 
    geneset: set,                     
    drop_cols: list[str] = [], 
    rename_cols: dict[str, str] = {}
):
    st_map = adata_st.var[['SYMBOL']].copy()
    st_map = st_map[st_map.index.isin(geneset.index)]
    st_map = st_map[st_map['SYMBOL'].notna()]
    
    st_map = st_map[~st_map.index.isin(drop_cols)]
    # st_map['SYMBOL'] = st_map.index.map(rename_cols).fillna(st_map['SYMBOL'].values)
    mask = st_map.index.isin(rename_cols)
    new_symbols = st_map.loc[mask].index.map(rename_cols)
    # symbol is categorical, make new categories first before assignment
    st_map['SYMBOL'] = st_map['SYMBOL'].cat.add_categories(
        [s for s in new_symbols if s not in st_map['SYMBOL'].cat.categories]
    )
    st_map.loc[mask, 'SYMBOL'] = new_symbols
    
    dupes = st_map[st_map['SYMBOL'].duplicated(keep=False)]
    if len(dupes):
        raise ValueError(
            f"Unresolved duplicate symbols remain after applying drop_cols/rename_cols. "
            f"Run check_symbol_conflicts() and update drop_cols/rename_cols accordingly.\n"
            f"{dupes.sort_values('SYMBOL')}"
        )
    
    common = st_map[st_map['SYMBOL'].isin(adata_sc.var_names)]
    
    new_adata_st = adata_st[:, common.index].copy()
    new_adata_sc = adata_sc[:, common['SYMBOL'].astype(str).tolist()].copy()
    new_adata_sc.var.index = common.index
    new_adata_sc.var_names_make_unique()
    
    n_genes = len(common)
    print(f"Retained {n_genes} genes common to both datasets")
    print(f"  scRNA-seq: {adata_sc.n_vars} → {n_genes} genes")
    print(f"  Spatial:   {adata_st.n_vars} → {n_genes} genes")
    
    return new_adata_sc, new_adata_st

def st_samples_in_name(sample_ids, name):
    return any(sample_id in name for sample_id in sample_ids)

def st_samples_not_in_name(sample_ids, name):
    return all(sample_id not in name for sample_id in sample_ids)

def extract_st_patient_data(adata, patient_id):
    adata = adata[adata.obs['donor_id'] == patient_id].copy()
    
    target_sample_ids = adata[adata.obs['donor_id'] == patient_id].obs['sample'].unique().tolist()
    columns_to_keep = ['feature_types', 'genome', 'SYMBOL', 'mt']
 
    columns_to_keep.extend([colname for colname in adata.var.columns if st_samples_in_name(target_sample_ids, colname)])
    
    columns_to_drop = [col for col in adata.var.columns if col not in columns_to_keep]
    adata.var.drop(columns=columns_to_drop, inplace=True)

    libraries_to_remove = [key for key in adata.uns['spatial'].keys() if st_samples_not_in_name(target_sample_ids, key)]
    
    for library_key in libraries_to_remove:
        adata.uns['spatial'].pop(library_key, None)
        
    return adata