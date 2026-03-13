import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from pathlib import Path

from ..metadata import ProjectMetadata
from ..utils import load_ct_spatial_data, save_ct_spatial_data

class HGCTSVGClustering(object):
    def __init__(self):
        pass
        
    def compute_spatial_clusters(self,
                            adata_st,
                            embedding_key="embedding",
                            resolution=1):
        
        # sc.pp.neighbors(adata_st, use_rep='embedding', metric='euclidean', knn=False, method='gauss')
        sc.pp.neighbors(adata_st, use_rep='embedding', metric='euclidean', knn=True, method='umap')        
        sc.tl.leiden(adata_st, flavor="igraph", n_iterations=-1, resolution=resolution)

        cluster_results = adata_st.obs['leiden'].to_numpy(dtype=str)
        
        adata_st.write_h5ad(self.output_dir / "ST_init.h5ad")
        np.save(self.output_dir / "spatial_domains.npy", cluster_results)

        print(adata_st)
    
    def compute_celltype_specific_spatial_clusters(self, 
                                                   type_list = [],
                                                   embedding_key="embedding", 
                                                   resolution=1):

        project_metadata = ProjectMetadata()
        
        if len(type_list) == 0:
            type_list = project_metadata.ct_list

        cluster_results = {}
        for ct in type_list:
            adata_ct = load_ct_spatial_data(ct)
            sc.pp.neighbors(adata_ct, use_rep=embedding_key, metric='euclidean', knn=False, method='gauss')
            # sc.pp.neighbors(adata_ct, use_rep=embedding_key, metric='euclidean', knn=True, method='umap')
            
            sc.tl.leiden(adata_ct, flavor="igraph", n_iterations=-1, resolution=resolution)

            self.__merge_clusters(adata_ct, embedding_key=embedding_key, label_key='leiden')
            
            save_ct_spatial_data(adata_ct, ct)
            cluster_results[ct] = adata_ct.obs['leiden'].to_numpy(dtype=str) 
            print(f"{ct}:", adata_ct)

        np.savez(project_metadata.clustering_dir / "cell_type_spatial_domains.npz", *cluster_results)

    def load_embedding_to_ct_spatial_data(self, embedding_key="embedding"):
        project_metadata = ProjectMetadata()

        # assumes that this is embedding for entire dataset
        embedding = torch.load(project_metadata.project_folder / "node_embeddings.pth")
        
        for ct in project_metadata.ct_list:
            adata_ct = load_ct_spatial_data(ct)
            cell_ids = torch.from_numpy(adata_ct.obs_names.astype(int).to_numpy())
            adata_ct.obsm[embedding_key] = embedding[cell_ids].numpy() 
            save_ct_spatial_data(adata_ct, ct)

    def __merge_clusters(self, adata_ct, threshold=0.05,
                         embedding_key="embedding",
                         label_key="leiden"):

        embedding = adata_ct.obsm[embedding_key]
        labels = adata_ct.obs[label_key].astype(str).copy()
    
        while True:
            cluster_ratio = labels.value_counts() / len(labels)
            if not (cluster_ratio < threshold).any():
                break
    
            smallest = cluster_ratio.idxmin()
    
            # compute centroids
            unique = labels.unique()
            centroids = np.vstack([
                embedding[labels.values == c].mean(axis=0)
                for c in unique
            ])
            centroid_labels = np.array(unique)
    
            # remove smallest centroid
            mask = centroid_labels != smallest
            centroid_matrix = centroids[mask]
            centroid_labels = centroid_labels[mask]
    
            # cells belonging to smallest cluster
            idx = np.where(labels.values == smallest)[0]
            cells = embedding[idx]
    
            # fast distance computation
            cell_norm = (cells**2).sum(axis=1, keepdims=True)
            centroid_norm = (centroid_matrix**2).sum(axis=1)
            dists = cell_norm + centroid_norm - 2 * cells @ centroid_matrix.T
    
            nearest = centroid_labels[dists.argmin(axis=1)]
    
            # reassign all cells
            labels.iloc[idx] = nearest
    
        # renumber clusters
        mapping = {c: str(i) for i, c in enumerate(sorted(labels.unique()))}
        adata_ct.obs[label_key] = labels.map(mapping)
        # remove old plot colors of label_key if it exists
        adata_ct.uns.pop(f'{label_key}_colors', None)
