import anndata as ad
import pickle
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

from ..data import STHypergraphData
from ..metadata import ProjectMetadata
from ..utils import load_ct_spatial_data, save_ct_spatial_data, load_decomposed_st_data


def build_hyperedge_feats(adata_st, spot_name_to_index, n_hyperedge_feats=3000):
    """Per-spot pseudobulk target for the hyperedge reconstruction head.

    Hyperedge s is 1:1 with spot s. Its members are the cells of spot s and of
    its k-1 nearest spots (scipy cKDTree, k nearest incl. s). The decoder pools the
    embeddings of all members and predicts spot s's own pseudobulk from them.
    Spot s's cells are among the pooled members, so the target is partly
    reconstructed from those cells' own embeddings; it is not a prediction of
    the centre spot from its neighbours alone.

    Steps: sum the decomposed cells of each spot in the decomposition gene space
    -> divide by the spot's cell count, then log1p (mean expression per cell, so
    the target does not scale with how many cells the spot has) -> keep the
    top-N highly variable genes across spots (scanpy, seurat flavor, computed on
    this pseudobulk so the genes vary at the spot level). Spots without
    decomposed cells get an all-zero row, which the loss excludes.

    Returns (feats  [n_spots × N], selected_genes  [N]).
    """
    adata_dec = load_decomposed_st_data()
    X = adata_dec.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = X.astype(np.float32)
    genes = np.asarray(adata_dec.var_names)

    M = adata_st.n_obs
    idx = np.array([spot_name_to_index.get(s, -1) for s in adata_dec.obs['spot_name'].values])
    valid = idx >= 0

    pb = np.zeros((M, X.shape[1]), dtype=np.float32)     # per-spot summed expression
    counts = np.zeros(M, dtype=np.float32)               # cells per spot
    np.add.at(pb, idx[valid], X[valid])
    np.add.at(counts, idx[valid], 1.0)

    pb_norm = np.log1p(pb / np.where(counts > 0, counts, 1.0)[:, None])   # composition
    pb_ad = ad.AnnData(pb_norm)
    sc.pp.highly_variable_genes(pb_ad, n_top_genes=n_hyperedge_feats, flavor="seurat")
    top = np.where(pb_ad.var["highly_variable"].values)[0]
    return torch.tensor(pb_norm[:, top], dtype=torch.float), list(genes[top])


def create_hypergraph_data(adata_st, n_features=3000, num_neighbor_spots=7,
                           n_hyperedge_feats=3000):
    project_metadata = ProjectMetadata()
    cell_type_mapping = project_metadata.ct_mapping

    # map spot names to numerical position in adata_st
    spot_name_to_index = {spot_name: i for i, spot_name in enumerate(adata_st.obs_names)}

    # create st graph. This graph will be the basis of our hyperedges.
    # Each spot s gets its num_neighbor_spots nearest spots, s itself included (distance 0): for k = 7,
    # s and its 6 nearest. Ties at the k-th neighbour are broken by scipy's KD-tree order (deterministic
    # for the same coordinates). edge_index is 2 x (n*k): row 0 = neighbour q, row 1 = spot s (ascending).
    coords = np.asarray(adata_st.obsm['spatial'], dtype=np.float64)
    k = min(num_neighbor_spots, len(coords))
    _, nn_idx = cKDTree(coords).query(coords, k=k)
    nn_idx = np.asarray(nn_idx, dtype=np.int64).reshape(len(coords), k)
    edge_index = torch.stack([torch.from_numpy(nn_idx.ravel()),
                              torch.arange(len(coords), dtype=torch.int64).repeat_interleave(k)])

    ct_list = project_metadata.ct_list

    cell_ids = []
    cell_feats = []
    cell_to_spot_id = []
    cell_types = []
    cell_offset = 0
    for ct in ct_list:
        adata_ct = load_ct_spatial_data(ct)

        n_ct = len(adata_ct)
        ct_cell_ids = torch.arange(cell_offset, cell_offset + n_ct, dtype=torch.int)
        adata_ct.obs['cell_idx'] = ct_cell_ids.numpy()
        save_ct_spatial_data(adata_ct, ct)
        cell_offset += n_ct
        cell_ids.append(ct_cell_ids)
        
        ct_feature_filename = project_metadata.project_folder / "ct_features.pkl"
        with open(ct_feature_filename, "rb") as ctff:
            feature_genes = pickle.load(ctff)

        ct_feature_genes = feature_genes[ct]
        adata_ct_feats = adata_ct[:, ct_feature_genes].copy()
        X = adata_ct_feats.X
        ct_cell_feats = torch.tensor(X.toarray() if sp.issparse(X) else X, dtype=torch.float)
        # pad cell feats if it is less than n_features
        pad_size = n_features - ct_cell_feats.shape[1] if n_features - ct_cell_feats.shape[1] > 0 else 0
        padding = torch.zeros((ct_cell_feats.shape[0], pad_size), dtype=ct_cell_feats.dtype)
        ct_cell_feats = torch.cat((ct_cell_feats, padding), dim=1)
        del padding
        cell_feats.append(ct_cell_feats)

        # map cells to their corresponding spots
        # resulting tensor has the following
        # index: cell id, value: spot id
        ct_cell_to_spot_id = adata_ct.obs['spot_name'].map(spot_name_to_index).to_numpy()
        ct_cell_to_spot_id = torch.from_numpy(ct_cell_to_spot_id)
        cell_to_spot_id.append(ct_cell_to_spot_id)
        
        ct_cell_types = adata_ct.obs[project_metadata.ct_key].map(cell_type_mapping).to_numpy()
        ct_cell_types = torch.from_numpy(ct_cell_types)
        cell_types.append(ct_cell_types)

    cell_ids = torch.cat(cell_ids, dim = 0)
    cell_feats = torch.cat(cell_feats, dim=0)
    cell_to_spot_id = torch.cat(cell_to_spot_id, axis=0)
    cell_types = torch.cat(cell_types, dim=0)

    # cell_ids are already sequential (0..N-1) so sort is a no-op, kept for consistency
    cell_ids, sort_order = torch.sort(cell_ids)
    cell_feats = cell_feats[sort_order]
    cell_to_spot_id = cell_to_spot_id[sort_order]
    cell_types = cell_types[sort_order]

    # hyperedge s = cells of spot s and of its num_neighbor_spots-1 nearest spots:
    # spot q is in hyperedge s when q is one of s's nearest spots (edge_index[0] = q, edge_index[1] = s),
    # and every cell belongs to the hyperedges its spot is in
    spot_in_hyperedge = torch.zeros((adata_st.n_obs, adata_st.n_obs), dtype=torch.bool)
    spot_in_hyperedge[edge_index[0], edge_index[1]] = True
    hyperedge_index = spot_in_hyperedge[cell_to_spot_id].nonzero().t()

    # hyperedge feats: spot-basis pseudobulk target for hyperedge-embedding
    # reconstruction (global HVG; cell-less spots -> all-zero rows, masked
    # out in the loss). Indexed by hyperedge id == spot index (length num_spots).
    hyperedge_feats, hyperedge_feat_genes = build_hyperedge_feats(
        adata_st, spot_name_to_index, n_hyperedge_feats=n_hyperedge_feats)
    with open(project_metadata.project_folder / "hyperedge_feat_genes.pkl", "wb") as f:
        pickle.dump(hyperedge_feat_genes, f)
    
    n_cell_types = project_metadata.n_cell_types
    n_spots = adata_st.n_obs

    return STHypergraphData(cell_feats,
                            hyperedge_feats,
                            cell_to_spot_id,
                            cell_types,
                            hyperedge_index,
                            n_cell_types,
                            n_spots)