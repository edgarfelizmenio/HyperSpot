import torch

import scanpy as sc

import pandas as pd

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

def filter(adata, is_mt_gene=None, min_genes=20, min_locs=20, max_genes=None, max_locs=None):
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # remove mitochondrial genes    
    if is_mt_gene:
        non_mt_genes = adata.var[~is_mt_gene(adata)].index.values
        adata = adata[:, non_mt_genes]
    
    # Use scanpy's filter_cells and filter_genes
    sc.pp.filter_cells(adata, min_genes=min_genes, max_genes=max_genes)
    sc.pp.filter_genes(adata, min_cells=min_locs, max_cells=max_locs)
    
    return adata

def compute_hvgs(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)

def preprocess(adata, target_sum = None, max_value = None, scale=False):
    adata.layers['original'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, zero_center=False, max_value=max_value)
        adata.X = adata.X.astype('float32')

def use_ensembl_gene_ids(adata, gene_names_col):
    annot = sc.queries.biomart_annotations("hsapiens",
                                       ["ensembl_gene_id", 
                                        "external_gene_name"])
    new_var = pd.merge(adata.var, annot, left_index=True, right_on='external_gene_name', how='inner')
    new_var = new_var.rename(columns={'name': 'SYMBOL', 'ensembl_gene_id': 'ENSEMBL'})
    new_var = new_var.reset_index().set_index('ENSEMBL')
    new_var = new_var[['SYMBOL']]
    
    new_adata = adata[:, new_var['SYMBOL']].copy()
    new_adata.var = new_var
    new_adata.var_names_make_unique()

    return new_adata

def get_common_geneset(adata_st, adata_sc):
    st_hvg = adata_st.var[adata_st.var['highly_variable']].index
    sc_hvg = adata_sc.var[adata_sc.var['highly_variable']].index

    geneset = st_hvg.intersection(sc_hvg)

    print(f"Number of overlap genes: {len(geneset)}")
    return geneset

def get_feature_matrix(adata):
    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
       feat = adata.X.toarray()
    else:
       feat = adata.X
    print(type(feat))
    return torch.tensor(feat)