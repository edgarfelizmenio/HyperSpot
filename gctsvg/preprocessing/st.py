import torch
import numpy as np
import pandas as pd
import squidpy as sq
from PIL import Image

from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def get_image(adata, to_uint8=True, scale=False):
    key = next(iter(adata.uns["spatial"].keys()))
    hires_image = adata.uns["spatial"][key]["images"]["hires"]

    image_container = sq.im.ImageContainer(hires_image)
    image_arr = image_container.data["image"].compute().values
    image_arr = np.squeeze(image_arr, axis=2)

    if to_uint8:
        image_arr = (255 * image_arr).clip(0, 255).astype(np.uint8)
    
    image = Image.fromarray(image_arr)
    
    if scale:
        scalefactor = adata.uns["spatial"][key]["scalefactors"]["tissue_hires_scalef"]
        height, width = image_container.shape
        new_height = int(height / scalefactor)
        new_width = int (width / scalefactor)
        image = image.resize((new_width, new_height), resample=Image.BICUBIC)

    return image

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