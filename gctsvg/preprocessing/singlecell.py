import torch
import torch.nn.functional as F
import scanpy as sc
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset

def sample_cells_per_type(adata, cell_type_column, num_samples_per_type):
    adata = adata[adata.obs.index.isin(adata.obs.groupby(cell_type_column).apply(
        lambda x: x.sample(frac=num_samples_per_type/x.shape[0],replace=False) if x.shape[0] > num_samples_per_type else x).reset_index(level=0,drop=True).index)].copy()
    return adata

def get_marker_genes(adata, cell_type_column, n_top_hvgs, n_top_markers):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_hvgs)
    sc.tl.rank_genes_groups(adata, groupby=cell_type_column, method='wilcoxon')
    
    markers_df = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).iloc[0:n_top_markers :]
    
    markers = list(np.unique(markers_df.melt().value.values))
    markers = list(set(adata.var.loc[adata.var['highly_variable']==1].index)|set(markers)) # highly variable genes + cell type marker genes

    return markers

def get_cell_type_centroids(adata, geneset, class_name='broad_type', method="mean",device="cpu"):
    adata_centroids = sc.get.aggregate(adata[:, geneset], by=class_name, func=[method], layer="original")    
    return torch.tensor(adata_centroids.layers["mean"], device=device)

def create_classifier_dataset(adata, class_name='broad_type', device="cpu"):
    data = torch.tensor(adata.X.toarray())
    labels = adata.obs[class_name].values    
    n_classes = labels.categories.nunique()

    cell_type_codes = torch.from_numpy(labels.codes.copy()).to(torch.long)

    data = data.to(device)
    cell_type_codes = cell_type_codes.to(device)
    
    dataset = TensorDataset(data, cell_type_codes)
    return dataset

def create_denoiser_dataset(adata, device="cpu"):
    data = torch.tensor(adata.X.toarray())
    data = data.to(device)
    
    dataset = TensorDataset(data)
    return dataset
    

def get_cell_types(adata, class_name):
    cell_types = adata.obs[class_name].values.categories.to_list()
    n_classes = adata.obs[class_name].values.categories.nunique()
    return cell_types, n_classes