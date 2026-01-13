import torch
from torch_geometric.nn import knn_graph

def extract_sc_patient_data(adata, patient_id):
    adata = adata[adata.obs['patient'] == patient_id].copy()
    return adata

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

def create_cell_type_mapping(adata_decomp, key_type='cell_type'):
    type_list = sorted(list(adata_decomp.obs[key_type].unique().astype(str)))

    cell_type_mapping = {
        cell_type:i for i, cell_type in enumerate(type_list) 
    }

    return type_list, cell_type_mapping


def create_hypergraph_data(adata_decomp, adata_st, num_neighbor_spots=7):
    cell_feats = torch.tensor(adata_decomp.X, dtype=torch.float)
    spot_feats = torch.tensor(adata_st.X.toarray(), dtype=torch.float)

    # map spot names to numerical position in adata_st
    spot_name_to_index = {spot_name: i for i, spot_name in enumerate(adata_st.obs_names)}

    # map cells to their corresponding spots
    # resulting tensor has the following
    # index: cell id, value: spot id
    cell_to_spot_id = adata_decomp.obs['spot_name'].map(spot_name_to_index).to_numpy()
    cell_to_spot_id = torch.from_numpy(cell_to_spot_id)

    type_list, cell_type_mapping = create_cell_type_mapping(adata_decomp)
    cell_types = adata_decomp.obs['cell_type'].map(cell_type_mapping).to_numpy()
    cell_types = torch.from_numpy(cell_types)
    
    # create st graph. This graph will be the basis of our hyperedges
    # when self loops are also computed in knn graph the self loop is also included when counting the number of nearest
    # neighbors. For example, if k = 6, 5 edges are incident to other nodes and 1 is a self loop.
    # The resulting edge index is a 2x(nk) matrix, with the 1st row indicating the nodes that are nearest to the node
    # 2nd row. The 2nd row is sorted in ascending order.
    coords = torch.from_numpy(adata_st.obsm['spatial'])
    edge_index = knn_graph(coords, k=num_neighbor_spots, loop=True)
    # edge_index = add_self_loops(edge_index)[0]

    adj_list = edge_index[0].reshape(-1, num_neighbor_spots)
    cell_hyperedge_list = adj_list[cell_to_spot_id]

    # create hyperedge index
    # hyperedge_members are cells belonging to incident hyperedges
    hyperedge_members = torch.arange(adata_decomp.n_obs).repeat_interleave(num_neighbor_spots)
    # incident_hyperedges are the hyperedges incident to each hyperedge_member
    incident_hyperedges = cell_hyperedge_list.flatten()
    hyperedge_index = torch.stack((hyperedge_members, incident_hyperedges))

    # derive incidence matrix from hyperedge index
    # create incidence matrix
    incidence_matrix = torch.zeros((adata_decomp.n_obs, adata_st.n_obs), dtype=torch.float)
    incidence_matrix[hyperedge_index[0], hyperedge_index[1]] = 1

    # given incidence matrix, derive hyperedge features as mean expression of all cells in hyperedge index
    # incidence_matrix: n_cells * n_spots
    # cell_feats: n_cells * n_feats
    # hyperedge_feats: n_spots * n_feats
    hyperedge_feat_sum = incidence_matrix.T @ cell_feats
    hyperedge_node_count = incidence_matrix.sum(dim=0).clamp(min=1)
    hyperedge_feats = hyperedge_feat_sum / hyperedge_node_count.unsqueeze(1)
    
    # optional: sort the hyperedge index
    # by sorting cell_spot_id, sorted values become the hyperedge ids sorted in ascending order,
    # and incides indicate the cell ids grouped by their increasing hyperedge id
    # _, idx = torch.sort(hyperedge_index[0])
    # _, idx2 = torch.sort(hyperedge_index[1,idx], stable=True)
    # idx_final = idx[idx2]
    # hyperedge_index = hyperedge_index[:, idx_final]
    cell_order = torch.argsort(hyperedge_index[0])
    hyperedge_index = hyperedge_index[:, cell_order]

    n_cell_types = len(type_list)
    n_spots = adata_st.n_obs
    
    return cell_feats, hyperedge_feats, cell_to_spot_id, cell_types, hyperedge_index, incidence_matrix, n_cell_types, n_spots
