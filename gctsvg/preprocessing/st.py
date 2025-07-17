# import torch
import numpy as np

# from torch_geometric.nn import knn_graph
# from torch_geometric.data import Data
# from torch_geometric.utils import to_undirected

def get_image_array(adata, to_uint8=True):
    key = next(iter(adata.uns["spatial"].keys()))
    image_arr = adata.uns["spatial"][key]["images"]["hires"]
    if to_uint8:
        image_arr = (255 * image_arr).clip(0, 255).astype(np.uint8)
    return image_arr

# def build_graph(adata, k, loop=False):
#     coords = torch.tensor(adata.obsm['spatial'])
#     edge_index = knn_graph(coords, k=k, loop=loop)
#     edge_index = to_undirected(edge_index, num_nodes=adata.n_obs)
#     return edge_index

# def create_graph_data(adata, k=18, self_loops=False, layer=None, device="cpu"):
#     # cell_types = adata.uns['cell_types']

#     if layer:
#         feats = torch.tensor(adata.layers['original'].toarray()).to(device).to(torch.double)
#     else:
#         feats = torch.tensor(adata.X.toarray()).to(device)
#     # probs = torch.tensor(adata.obs[cell_types].values, dtype=torch.float)
#     edge_index = build_graph(adata, k, self_loops).to(device)

#     data = Data(x = feats, edge_index = edge_index)
#     # data = Data(x = torch.cat((feats, probs),1), edge_index=edge_index)
#     return data