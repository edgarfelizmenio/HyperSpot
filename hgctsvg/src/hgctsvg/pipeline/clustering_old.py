import anndata as ad
import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from pathlib import Path

from ..metadata import ProjectMetadata
from ..utils import (
    load_orig_st_data,
    load_decomposed_st_data,
    load_ct_spatial_data, 
    save_ct_spatial_data,
    load_aggr_st_data,
    save_aggr_st_data
)

# normalize embeddings first?
class HGCTSVGClustering(object):
    def __init__(self):
        pass
        
    def compute_spatial_clusters(
        self,
        embedding_key="embedding",
        min_cluster_pct=-1
    ):

        project_metadata = ProjectMetadata()

        adata_aggr = load_aggr_st_data()
        
        # sc.pp.neighbors(
        #     adata_aggr, 
        #     use_rep=embedding_key, 
        #     metric='euclidean', 
        #     knn=False, 
        #     method='gauss'
        # )
        
        sc.pp.neighbors(
            adata_aggr, 
            use_rep=embedding_key, 
            metric='euclidean', 
            knn=True, 
            method='umap'
        )
        
        sc.tl.leiden(
            adata_aggr, 
            flavor="igraph", 
            n_iterations=-1, 
            resolution=1
        )

        self.__merge_clusters(
            adata_aggr, 
            threshold = min_cluster_pct,
            embedding_key = embedding_key, 
            label_key='leiden'
        )
        
        cluster_results = adata_aggr.obs['leiden'].to_numpy(dtype=str)
        
        save_aggr_st_data(adata_aggr)

        np.save(
            project_metadata.clustering_dir / "spatial_domains.npy", 
            cluster_results
        )

        print(adata_aggr)
    
    def compute_ct_spatial_clusters(self, 
                                    type_list = [],
                                    embedding_key="embedding",
                                    min_cluster_pct=0.05):

        project_metadata = ProjectMetadata()
        
        if len(type_list) == 0:
            type_list = project_metadata.ct_list

        cluster_results = {}
        for ct in type_list:
            adata_ct = load_ct_spatial_data(ct)

            sc.pp.neighbors(adata_ct, use_rep=embedding_key, metric='euclidean', knn=False, method='gauss')
            # sc.pp.neighbors(adata_ct, use_rep=embedding_key, metric='euclidean', knn=True, method='umap')
            
            sc.tl.leiden(adata_ct, flavor="igraph", n_iterations=-1, resolution=1)

            self.__merge_clusters(adata_ct, 
                                  threshold = min_cluster_pct,
                                  embedding_key = embedding_key, 
                                  label_key='leiden')
            
            save_ct_spatial_data(adata_ct, ct)
            cluster_results[ct] = adata_ct.obs['leiden'].to_numpy(dtype=str) 
            print(f"{ct}:", adata_ct)

        np.savez(project_metadata.clustering_dir / "cell_type_spatial_domains.npz", *cluster_results)

    def create_aggr_st_data(self):
        adata_decomp = load_decomposed_st_data()
        adata_st = load_orig_st_data()

        # Aggregate decomposed cells to spot level
        adata_aggr = sc.get.aggregate(
            adata_decomp,
            by="spot_name",
            func="sum"
        )
        
        # Spots present in both datasets
        common_spots = adata_st.obs_names.intersection(adata_aggr.obs_names)
        
        # Align spot order to ST automatically
        adata_aggr = adata_aggr[common_spots]

        var = pd.DataFrame(
            {"SYMBOL": adata_decomp.var['SYMBOL']},
            index=adata_decomp.var_names
        )
        
        # Build final AnnData
        adata_final = ad.AnnData(
            X=adata_aggr.layers["sum"],
            obs=pd.DataFrame(index=common_spots),
            var=var,
        )
        
        # Copy spatial coordinates (automatic order alignment)
        spatial_df = pd.DataFrame(
            adata_st.obsm["spatial"],
            index=adata_st.obs_names
        )
        
        adata_final.obsm["spatial"] = spatial_df.loc[common_spots].values
        
        # Copy Visium metadata
        adata_final.uns["spatial"] = adata_st.uns["spatial"].copy()
        
        # Preprocessing
        adata_final.raw = adata_final.copy()
        sc.pp.log1p(adata_final)

        # Save
        save_aggr_st_data(adata_final)

    def load_embedding_to_aggr_data(self,
                                    embedding_key="embedding"):

        project_metadata = ProjectMetadata()

        embedding = torch.load(project_metadata.project_folder / "hyperedge_embeddings.pth")

        nan_mask = torch.isnan(embedding)
        nan_rows = torch.any(nan_mask, dim=1)
        embedding = embedding[~nan_rows]

        adata_aggr = load_aggr_st_data()

        assert embedding.shape[0] == adata_aggr.n_obs
        
        adata_aggr.obsm[embedding_key] = embedding.numpy()

        save_aggr_st_data(adata_aggr)
        
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
        """
        Merge clusters so that each the size of each cluster is (threshold)*(num_cells)
        """

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
