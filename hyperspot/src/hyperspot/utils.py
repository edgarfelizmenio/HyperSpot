import scanpy as sc
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pathlib import Path

from .metadata import ProjectMetadata


# --- spot-duplicate primitives (single source of truth) ------------------------
# Spotiphy-style deconvolution assigns one per-type profile per spot, so same-spot
# same-CT cells are exact duplicates (identical expression -> identical embedding).
# The dedup used in clustering + ctSVG detection, and the standalone verification,
# all rest on the two functions below so the check can never silently drift.

def within_spot_max_deviation(X, spot_labels, sample=None, seed=0):
    """Max abs deviation of any cell from its spot's first cell, over features.

    ~0 => same-spot cells are (near-)identical duplicates. `sample` limits the check
    to that many randomly chosen multi-cell spots (for wide/sparse expression
    matrices where materialising all cells is costly); None checks every cell.
    """
    spot_labels = np.asarray(spot_labels)
    n = len(spot_labels)
    if n == 0:
        return 0.0
    _, inv = np.unique(spot_labels, return_inverse=True)
    if sample is None:
        Xd = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
        rep = pd.Series(np.arange(n)).groupby(inv).first().sort_index().to_numpy()
        return float(np.abs(Xd - Xd[rep][inv]).max())
    multi = np.where(np.bincount(inv) >= 2)[0]
    if len(multi) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    check = multi if len(multi) <= sample else rng.choice(multi, sample, replace=False)
    dev = 0.0
    for g in check:
        members = np.where(inv == g)[0]
        Xm = X[members]
        Xm = np.asarray(Xm.todense()) if sp.issparse(Xm) else np.asarray(Xm)
        dev = max(dev, float(np.abs(Xm - Xm[0]).max()))
    return dev


def guarded_spot_reps(X, spot_labels, dedup=True, tol=1e-3, sample=None):
    """(rep_idx, inv) collapsing same-spot cells to one representative -- but ONLY
    when they are (near-)identical (max within-spot deviation < tol). Otherwise, or
    when dedup=False / no spot labels, returns the identity mapping (every cell its
    own unit). rep_idx[r] is a representative cell of unique spot r; inv[c] is the
    unit id of cell c (labels[inv] broadcasts unit labels back to cells).
    """
    n = X.shape[0]                                      # one unit per cell (row of X)
    inv = np.arange(n)                                  # identity: no dedup
    if dedup and n > 0 and spot_labels is not None:
        _, cand = np.unique(np.asarray(spot_labels), return_inverse=True)
        if within_spot_max_deviation(X, spot_labels, sample=sample) < tol:
            inv = cand
    rep_idx = pd.Series(np.arange(n)).groupby(inv).first().sort_index().to_numpy()
    return rep_idx, inv
# -------------------------------------------------------------------------------

def save_orig_st_data(adata_st):
    adata_st.write_h5ad(ProjectMetadata().project_folder / 'ST_orig.h5ad')

def load_orig_st_data():
    adata_st = sc.read_h5ad(ProjectMetadata().project_folder / 'ST_orig.h5ad')
    return adata_st

def save_decomposed_st_data(adata_decomp):
    adata_decomp.write_h5ad(ProjectMetadata().project_folder / 'ST_decomposition.h5ad')

def load_decomposed_st_data():
    adata_decomp = sc.read_h5ad(ProjectMetadata().project_folder / 'ST_decomposition.h5ad')
    return adata_decomp

def save_aggr_st_data(adata_aggr):
    adata_aggr.write_h5ad(ProjectMetadata().project_folder / 'ST_aggregated.h5ad')

def load_aggr_st_data():
    adata_aggr = sc.read_h5ad(ProjectMetadata().project_folder / 'ST_aggregated.h5ad')
    if 'domain' in adata_aggr.obs:   # ensure discrete for plotting (squidpy/scanpy)
        adata_aggr.obs['domain'] = adata_aggr.obs['domain'].astype('category')
    return adata_aggr

def save_ct_spatial_data(adata_ct, cell_type: str):
    # save adata for cell type
    adata_ct.write_h5ad(ProjectMetadata().ct_data_dir / f'ST_cell_{cell_type}.h5ad')

def load_ct_spatial_data(cell_type: str):
    # load adata for cell type
    adata_ct = sc.read_h5ad(ProjectMetadata().ct_data_dir / f"ST_cell_{cell_type}.h5ad")
    if 'domain' in adata_ct.obs:      # ensure discrete for plotting (squidpy/scanpy)
        adata_ct.obs['domain'] = adata_ct.obs['domain'].astype('category')
    return adata_ct


def get_gene_minmax(adata, gene_name):
    index = adata.var_names.get_loc(gene_name)
    gene_values = adata.X[:,index]
    gene_min = gene_values.min()
    gene_max = gene_values.max()

    return gene_min, gene_max

def build_incidence_matrix(hyperedge_index):
    # reindex hyperedges to make compact incidence matrix for a batch
    unique_cell_idx, map_to_unique_cell = hyperedge_index[0].unique(return_inverse=True)
    unique_neighborhood_idx, map_to_unique_neighborhood = hyperedge_index[1].unique(return_inverse=True)

    n_unique_cells = unique_cell_idx.shape[0]
    n_unique_neighborhoods = unique_neighborhood_idx.shape[0]
    
    incidence_matrix = torch.zeros((n_unique_cells, n_unique_neighborhoods), dtype=torch.float)
    incidence_matrix[map_to_unique_cell, map_to_unique_neighborhood] = 1

    return incidence_matrix