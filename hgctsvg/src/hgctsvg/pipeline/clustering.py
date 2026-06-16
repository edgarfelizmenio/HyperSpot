import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from sklearn.cluster import OPTICS
from typing import Literal

from ..metadata import ProjectMetadata
from ..utils import (
    load_orig_st_data,
    load_decomposed_st_data,
    load_ct_spatial_data,
    save_ct_spatial_data,
    load_aggr_st_data,
    save_aggr_st_data
)

class HGCTSVGClustering(object):
    def __init__(self):
        pass
        
    def _cluster(
        self,
        adata,
        embedding_key: str,
        algorithm: Literal["leiden", "optics"] = "leiden",
        use_gauss_neighbors: bool = False,
        min_cluster_pct: float = -1,
        seed: int = 0,
    ) -> str:

        if algorithm == "leiden":
            if use_gauss_neighbors:
                sc.pp.neighbors(adata, use_rep=embedding_key, metric='euclidean', knn=False, method='gauss')
            else:
                sc.pp.neighbors(adata, use_rep=embedding_key, metric='euclidean', knn=True, method='umap')
            sc.tl.leiden(adata, flavor="igraph", n_iterations=-1, resolution=1, key_added="domain",
                         random_state=seed)
            self.__merge_clusters(adata, threshold=min_cluster_pct, embedding_key=embedding_key, label_key="domain")
        else:
            embedding = adata.obsm[embedding_key]
            optics_kwargs = {"min_cluster_size": min_cluster_pct} if min_cluster_pct > 0 else {}
            labels = OPTICS(metric='euclidean', **optics_kwargs).fit_predict(embedding)
            adata.obs["domain"] = pd.Categorical(labels.astype(str))
        return "domain"

    def compute_spatial_clusters(self,
                                 embedding_key="embedding",
                                 algorithm: Literal["leiden", "optics"] = "leiden",
                                 min_cluster_pct=-1,
                                 seed: int = 0):
        project_metadata = ProjectMetadata()

        adata_aggr = load_aggr_st_data()

        label_key = self._cluster(adata_aggr, embedding_key,
                                  algorithm,
                                  use_gauss_neighbors=False, min_cluster_pct=min_cluster_pct,
                                  seed=seed)

        cluster_results = adata_aggr.obs[label_key].to_numpy(dtype=str)

        save_aggr_st_data(adata_aggr)

        np.save(
            project_metadata.clustering_dir / "spatial_domains.npy",
            cluster_results
        )

        print(f"Tissue: num spots = {adata_aggr.n_obs}, num domains = {adata_aggr.obs[label_key].nunique()}")

    def compute_ct_spatial_clusters(self,
                                    type_list=[],
                                    embedding_key="embedding",
                                    algorithm: Literal["leiden", "optics"] = "leiden",
                                    min_cluster_pct=0.05,
                                    seed: int = 0):
        project_metadata = ProjectMetadata()

        if len(type_list) == 0:
            type_list = project_metadata.ct_list

        cluster_results = {}
        for ct in type_list:
            adata_ct = load_ct_spatial_data(ct)

            label_key = self._cluster(adata_ct, embedding_key,
                                      algorithm,
                                      use_gauss_neighbors=True, min_cluster_pct=min_cluster_pct,
                                      seed=seed)

            save_ct_spatial_data(adata_ct, ct)
            cluster_results[ct] = adata_ct.obs[label_key].to_numpy(dtype=str)
            print(f"{ct}: num cells = {adata_ct.n_obs}, num domains = {adata_ct.obs[label_key].nunique()}")

        np.savez(project_metadata.clustering_dir / "cell_type_spatial_domains.npz", *cluster_results)

    def create_aggr_st_data(self, var_cols: list[str] | None = None):
        adata_decomp = load_decomposed_st_data()
        adata_st = load_orig_st_data()

        adata_aggr = sc.get.aggregate(adata_decomp, by="spot_name", func="sum")

        common_spots = adata_st.obs_names.intersection(adata_aggr.obs_names)
        adata_aggr = adata_aggr[common_spots]

        if var_cols:
            existing = [c for c in var_cols if c in adata_decomp.var.columns]
            var = adata_decomp.var[existing].copy()
        else:
            var = pd.DataFrame(index=adata_decomp.var_names)

        adata_final = ad.AnnData(
            X=adata_aggr.layers["sum"],
            obs=pd.DataFrame(index=common_spots),
            var=var,
        )

        spatial_df = pd.DataFrame(adata_st.obsm["spatial"], index=adata_st.obs_names)
        adata_final.obsm["spatial"] = spatial_df.loc[common_spots].values

        adata_final.uns["spatial"] = adata_st.uns["spatial"].copy()

        adata_final.raw = adata_final.copy()
        sc.pp.log1p(adata_final)

        save_aggr_st_data(adata_final)

    def load_embedding_to_aggr_data(self, embedding_key="embedding"):
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

        embedding = torch.load(project_metadata.project_folder / "node_embeddings.pth")
        total = 0
        for ct in project_metadata.ct_list:
            adata_ct = load_ct_spatial_data(ct)
            cell_ids = torch.from_numpy(
                adata_ct.obs['cell_idx'].to_numpy() if 'cell_idx' in adata_ct.obs.columns
                else adata_ct.obs_names.astype(int).to_numpy()
            )
            # total += adata_ct.n_obs
            # print(adata_ct.shape)
            # print(cell_ids, cell_ids.shape)
            # print(embedding.shape)
            # print(total)
            adata_ct.obsm[embedding_key] = embedding[cell_ids].numpy()
            # print('lol')
            save_ct_spatial_data(adata_ct, ct)

    def __merge_clusters(self, adata_ct, threshold=0.05,
                         embedding_key="embedding",
                         label_key="domain",
                         merge_mode="cluster"):
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

            unique = labels.unique()
            centroids = np.vstack([
                embedding[labels.values == c].mean(axis=0)
                for c in unique
            ])
            centroid_labels = np.array(unique)

            mask = centroid_labels != smallest
            centroid_matrix = centroids[mask]
            centroid_labels = centroid_labels[mask]

            idx = np.where(labels.values == smallest)[0]
            cells = embedding[idx]

            if merge_mode == "cluster":                  # ← added branch
                small_centroid = cells.mean(axis=0, keepdims=True)
                centroid_norm  = (centroid_matrix ** 2).sum(axis=1)
                small_norm     = (small_centroid ** 2).sum(axis=1, keepdims=True)
                dists = small_norm + centroid_norm - 2 * small_centroid @ centroid_matrix.T
                nearest_label  = centroid_labels[dists.argmin()]
                labels.iloc[idx] = nearest_label
            else:                                        # original sample-based logic
                cell_norm = (cells**2).sum(axis=1, keepdims=True)
                centroid_norm = (centroid_matrix**2).sum(axis=1)
                dists = cell_norm + centroid_norm - 2 * cells @ centroid_matrix.T
                nearest = centroid_labels[dists.argmin(axis=1)]
                labels.iloc[idx] = nearest


        mapping = {c: str(i) for i, c in enumerate(sorted(labels.unique()))}
        adata_ct.obs[label_key] = labels.map(mapping)
        adata_ct.uns.pop(f'{label_key}_colors', None)
