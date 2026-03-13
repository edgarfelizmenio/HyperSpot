import pickle
import torch
import torch.nn.functional as F
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np

from pathlib import Path
from torch.utils.data import TensorDataset

from ..metadata import ProjectMetadata
from ..utils import save_ct_spatial_data, load_ct_spatial_data

# Make cell type-specific adata objects
def make_ct_spatial_adata(adata_decomp):
    project_metadata = ProjectMetadata() 
    
    cell_types = project_metadata.ct_list
    cell_type_key = project_metadata.ct_key
    
    for ct in cell_types:
        adata_ct = adata_decomp[adata_decomp.obs[cell_type_key] == ct].copy()

        # drop genes that have 0 expression
        sc.pp.filter_genes(adata_ct, min_cells=1)

        # preprocess cell type-specific data
        adata_ct.raw = adata_ct.copy()
        # Spotiphy output is already in CPM so no need to normalize
        # Compute log normalization
        sc.pp.log1p(adata_ct)

        assert adata_ct.X.max() < adata_ct.raw.X.max()
        
        save_ct_spatial_data(adata_ct, ct)

# Perform feature selection on each cell type.
def perform_ct_feature_selection(n_features = 3000):
    project_metadata = ProjectMetadata() 
    cell_types = project_metadata.ct_list

    ct_feats = {}
    
    for ct in cell_types:
        adata_ct = load_ct_spatial_data(ct)

        feature_genes = compute_feature_genes(adata_ct, n_features)

        ct_feats[ct] = feature_genes.tolist()
        save_ct_spatial_data(adata_ct, ct)

    # save features for each cell type
    ct_feature_filename = project_metadata.project_folder / "ct_features.pkl"
    with open(ct_feature_filename, "wb") as ctff:
        pickle.dump(ct_feats, ctff)

def compute_feature_genes(adata_ct, n_features=3000):
    sc.pp.filter_genes(adata_ct, min_cells=1)
    genes = adata_ct.var_names

    sc.pp.highly_variable_genes(adata_ct, flavor="seurat")
    
    sq.gr.spatial_neighbors(adata_ct)
    sq.gr.spatial_autocorr(adata_ct, mode='moran', genes = adata_ct.var_names)

    var = adata_ct.var
    corr_df = adata_ct.uns['moranI'].loc[adata_ct.var_names]

    valid_hvg = (var["highly_variable"] == True)
    hvg_genes = var.index[valid_hvg]
    hvg_pct = var.loc[hvg_genes, 'dispersions_norm'].rank(pct=True)
    
    variance = adata_ct.X.var(axis=0)
    moran_abs = corr_df['I'].abs()
    
    moran_z = (moran_abs - moran_abs.mean()) / moran_abs.std()
    var_z = (variance - variance.mean()) / variance.std()

    spatial_score = moran_z * var_z
    spatial_pct = spatial_score.rank(pct=True)

    combined_score = np.sqrt(hvg_pct * spatial_pct.loc[hvg_genes])

    adata_ct.var['hvg_scores'] = np.nan
    adata_ct.var.loc[hvg_genes, 'hvg_scores'] = hvg_pct

    adata_ct.var['spatial_scores'] = spatial_pct
    
    adata_ct.var['combined_scores'] = np.nan
    adata_ct.var.loc[hvg_genes, 'combined_scores'] = combined_score

    top_genes = combined_score.sort_values(ascending=False).index[:n_features]

    n_spatial_genes = n_features - len(top_genes)

    if n_spatial_genes > 0:
        # get the genes that are not yet selected
        remaining_genes = var.index.difference(top_genes)
        remaining_spatial_pct = spatial_pct.loc[remaining_genes]

        top_genes = top_genes.append(remaining_spatial_pct.sort_values(ascending=False).index[:n_spatial_genes])

    adata_ct.var['feature_gene'] = False
    adata_ct.var.loc[top_genes, 'feature_gene'] = True
    return top_genes


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