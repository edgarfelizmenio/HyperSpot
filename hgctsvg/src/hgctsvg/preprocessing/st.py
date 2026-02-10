import torch
import pandas as pd

from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from tqdm.auto import tqdm

def build_graph(adata, k, loop=False):
    if isinstance(adata.obsm['spatial'], pd.DataFrame):
        coords = torch.from_numpy(adata.obsm['spatial'].values)
    else:
        coords = torch.tensor(adata.obsm['spatial'])
    edge_index = knn_graph(coords, k=k, loop=loop)
    edge_index = to_undirected(edge_index, num_nodes=adata.n_obs)
    return edge_index

def create_graph_data(adata, k=6, self_loops=False, layer=None):
    # cell_types = adata.uns['cell_types']
    if layer:
        feats = torch.tensor(adata.layers[layer].toarray())
    else:
        feats = torch.tensor(adata.X.toarray())
    # probs = torch.tensor(adata.obs[cell_types].values, dtype=torch.float)
    edge_index = build_graph(adata, k, self_loops)

    data = Data(x = feats, edge_index = edge_index)
    # data = Data(x = torch.cat((feats, probs),1), edge_index=edge_index)
    return data

def create_st_edge_index(adata_st, k=7):
    coords = torch.from_numpy(adata_st.obsm['spatial'])
    edge_index = knn_graph(coords, k=k, loop=True)
    return edge_index

def create_incidence_matrix(adata_decomp, adata_st, edge_index):
    n_nodes = adata_decomp.n_obs
    n_hyperedge = adata_st.n_obs
    incidence_mat = torch.zeros((adata_decomp.n_obs, adata_st.n_obs), dtype=torch.int)
    
    for i in tqdm(range(adata_st.n_obs)):
        spot_name = adata_st.obs.index[i]
        neighbors = edge_index[0,edge_index[1] == i]
        # display(neighbors[0])
        # neighbors = neighbors[1]
        # display(i, neighbors)
        neighbor_names = adata_st.obs.index[neighbors]
        # display(neighbor_names)
        cells = adata_decomp[adata_decomp.obs['spot_name'] == spot_name].obs_names.astype(int)
        # display(cells)
        neighbor_ids = adata_decomp[adata_decomp.obs['spot_name'].isin(neighbor_names)].obs_names.astype(int)
        # display(neighbor_ids, neighbor_ids.shape)
        incidence_mat[neighbor_ids,i] = 1
        # display(incidence_mat[neighbor_ids,i])

    return incidence_mat

def create_hyperedge_index(node_ids, incidence_mat):
    spot_ids = cell_spot_id[node_ids]

    # get spot neighborhood
    spot_neighborhood = edge_index[0, torch.isin(edge_index[1], spot_ids)]
    
    # get all nodes from spot neighborhood
    cell_neighbors = incidence_mat[:,spot_neighborhood].nonzero()[:,0].unique()
    # TODO: sample nodes with probability
    
    # get all incident edges from each spot in the neighborhood
    hyperedge_indices = incidence_mat[cell_neighbors].nonzero()[:,1]
    # TODO: sample hyperedges with probability
    # combine incident edges from original spot neighborhood
    hyperedge_indices = torch.cat((spot_neighborhood, hyperedge_indices)).unique()

    # create batch incidence matrix
    incidence_sub = incidence_mat[cell_neighbors][:,hyperedge_indices]
    row_idx, col_idx = torch.nonzero(incidence_sub, as_tuple=True)
    cell_idx = cell_neighbors[row_idx]
    hyp_idx = hyperedge_indices[col_idx]
    hyperedge_index = torch.stack((cell_idx, hyp_idx))
    return hyperedge_index

