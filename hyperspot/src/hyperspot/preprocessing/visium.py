import anndata as ad
import pickle
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from torch_geometric.nn import knn_graph

from ..data import STHypergraphData
from ..metadata import ProjectMetadata
from ..utils import load_ct_spatial_data, save_ct_spatial_data, load_decomposed_st_data


def build_hyperedge_feats(adata_st, spot_name_to_index, n_hyperedge_feats=256):
    """Spot-basis pseudobulk reconstruction target for hyperedge embeddings.

    A hyperedge is 1:1 with a spot; its embedding is decoded to predict that
    *center spot's own* composition (not the smoothed neighborhood — see design
    notes: predicting the center from the neighborhood is a non-circular pretext
    and avoids reinforcing tissue-level over-smoothing).

    Steps: sum decomposed cells per spot in the shared decomposition gene space
    -> size-normalize per spot (÷ its cell count, then log1p, so the target is
    composition not neighborhood size) -> keep the top-N genes by variance across
    spots (a small *global* HVG set, selected on the pseudobulk itself so the
    genes actually vary at the level being reconstructed).

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
    
    # small global HVG set: top-N by variance of the (normalized) pseudobulk
    # n_hvg = min(n_hyperedge_feats, pb_norm.shape[1])
    # top = np.sort(np.argsort(pb_norm.var(axis=0))[::-1][:n_hvg])
    sc.pp.highly_variable_genes(pb_ad, n_top_genes=n_hyperedge_feats, flavor="seurat")
    top = np.where(pb_ad.var["highly_variable"].values)[0]
    return torch.tensor(pb_norm[:, top], dtype=torch.float), list(genes[top])


def create_hypergraph_data(adata_st, n_features=3000, num_neighbor_spots=7,
                           intra_weight=1.0, inter_weight=1.0, scale=False,
                           n_hyperedge_feats=256):
    project_metadata = ProjectMetadata()
    cell_type_mapping = project_metadata.ct_mapping

    # map spot names to numerical position in adata_st
    spot_name_to_index = {spot_name: i for i, spot_name in enumerate(adata_st.obs_names)}

    # create st graph. This graph will be the basis of our hyperedges
    # when self loops are also computed in knn graph the self loop is also included when counting the number of nearest
    # neighbors. For example, if k = 6, 5 edges are incident to other nodes and 1 is a self loop.
    # The resulting edge index is a 2x(nk) matrix, with the 1st row indicating the nodes that are nearest to the node
    # 2nd row. The 2nd row is sorted in ascending order.
    coords = torch.from_numpy(adata_st.obsm['spatial'])
    edge_index = knn_graph(coords, k=num_neighbor_spots, loop=True)
    adj_list = edge_index[0].reshape(-1, num_neighbor_spots)
    
    ct_list = project_metadata.ct_list

    cell_ids = []
    cell_feats = []
    cell_to_spot_id = []
    cell_types = []
    hyperedge_indexes = []
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
        if scale:
            sc.pp.scale(adata_ct_feats, max_value=None)
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
        
        ct_cell_hyperedge_list = adj_list[ct_cell_to_spot_id]
        # create hyperedge index
        # hyperedge_members are cells belonging to incident hyperedges
        ct_hyperedge_members = ct_cell_ids.repeat_interleave(num_neighbor_spots)
        # incident_hyperedges are the hyperedges incident to each hyperedge_member
        ct_incident_hyperedges = ct_cell_hyperedge_list.flatten()
        ct_hyperedge_index = torch.stack((ct_hyperedge_members, ct_incident_hyperedges))
        hyperedge_indexes.append(ct_hyperedge_index)

    cell_ids = torch.cat(cell_ids, dim = 0)
    cell_feats = torch.cat(cell_feats, dim=0)
    cell_to_spot_id = torch.cat(cell_to_spot_id, axis=0)
    cell_types = torch.cat(cell_types, dim=0)
    hyperedge_index = torch.cat(hyperedge_indexes, dim=1)

    # cell_ids are already sequential (0..N-1) so sort is a no-op, kept for consistency
    cell_ids, sort_order = torch.sort(cell_ids)
    cell_feats = cell_feats[sort_order]
    cell_to_spot_id = cell_to_spot_id[sort_order]
    cell_types = cell_types[sort_order]

    # derive incidence matrix from hyperedge index
    # intra-spot entries (cell's spot == hyperedge) get intra_weight; inter-spot get inter_weight
    incidence_matrix = torch.zeros((cell_ids.shape[0], adata_st.n_obs), dtype=torch.float)
    is_intra = (cell_to_spot_id[hyperedge_index[0]] == hyperedge_index[1])
    weights = torch.where(is_intra,
                          torch.tensor(intra_weight, dtype=torch.float),
                          torch.tensor(inter_weight, dtype=torch.float))
    incidence_matrix[hyperedge_index[0], hyperedge_index[1]] = weights

    # hyperedge feats: spot-basis pseudobulk target for hyperedge-embedding
    # reconstruction (small global HVG; cell-less spots -> all-zero rows, masked
    # out in the loss). Indexed by hyperedge id == spot index (length num_spots).
    # n_hyperedge_feats=0 keeps the legacy degree vector (hyperedge optimization
    # then cannot be enabled at train time).
    if n_hyperedge_feats:
        hyperedge_feats, hyperedge_feat_genes = build_hyperedge_feats(
            adata_st, spot_name_to_index, n_hyperedge_feats=n_hyperedge_feats)
        with open(project_metadata.project_folder / "hyperedge_feat_genes.pkl", "wb") as f:
            pickle.dump(hyperedge_feat_genes, f)
    else:
        hyperedge_feats = incidence_matrix.sum(dim=0)   # legacy: hyperedge degree
    
    n_cell_types = project_metadata.n_cell_types
    n_spots = adata_st.n_obs

    # return cell_ids, cell_feats, cell_to_spot_id, cell_types, hyperedge_index, incidence_matrix, n_cell_types, n_spots
    return STHypergraphData(cell_feats,
                            hyperedge_feats,
                            cell_to_spot_id,
                            cell_types,
                            hyperedge_index,
                            incidence_matrix,
                            n_cell_types,
                            n_spots)