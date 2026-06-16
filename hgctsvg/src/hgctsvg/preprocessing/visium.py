import pickle
import torch
from torch_geometric.nn import knn_graph

from ..data import STHypergraphData
from ..metadata import ProjectMetadata
from ..utils import load_ct_spatial_data, save_ct_spatial_data

def create_hypergraph_data(adata_st, n_features=3000, num_neighbor_spots=7):
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
        ct_cell_feats = torch.tensor(adata_ct[:, ct_feature_genes].X.toarray(), dtype=torch.float)
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
    # create incidence matrix
    incidence_matrix = torch.zeros((cell_ids.shape[0], adata_st.n_obs), dtype=torch.float)
    incidence_matrix[hyperedge_index[0], hyperedge_index[1]] = 1

    # set hyperedge feats as number of nodes in hyperedge
    # to be removed later
    hyperedge_feats = incidence_matrix.sum(dim=0)
    
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