import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import diptest
from threadpoolctl import threadpool_limits
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
        algorithm: Literal["leiden", "agglomerative", "gmm"] = "leiden",
        use_gauss_neighbors: bool = False,
        min_cluster_pct: float = -1,
        seed: int = 0,
        n_pca_components: int = 50,
        n_clusters: int = 20,
        n_neighbors: int = 15,
        min_cluster_size: int = 3,
        select_pcs: bool = False,
        pc_var_floor: float = 0.04,
        pc_dip_alpha: float = 0.05,
        select_min_cells: int = 500,
    ) -> str:

        # Cap OpenBLAS threads around the BLAS-heavy clustering ops (PCA,
        # neighbors, GMM). On high-core machines the default pools
        # over-subscribe past OpenBLAS's compiled 128-thread limit, producing
        # the "precompiled NUM_THREADS exceeded" warning and "Bad memory
        # unallocation!" errors. Scoped to the BLAS API only.
        with threadpool_limits(limits=16, user_api="blas"):
            cluster_key = embedding_key
            n_comps = min(n_pca_components, adata.obsm[embedding_key].shape[1] - 1, adata.n_obs - 1)
            if n_comps > 0:
                pca_key = embedding_key + "_pca"
                adata.obsm[pca_key] = PCA(n_components=n_comps, random_state=seed).fit_transform(adata.obsm[embedding_key])
                cluster_key = pca_key

                # gated-select: keep only PCs with real discrete structure
                # (significantly multimodal by Hartigan's dip test, above a variance
                # floor), then cluster on those. Drops the continuous/noise PCs that
                # otherwise drown the signal.
                if select_pcs and adata.n_obs < select_min_cells:
                    # too few cells to estimate modality reliably -> don't gate
                    print(f"  select-pcs: {adata.n_obs} cells < {select_min_cells}; skipping gating (all PCs)")
                elif select_pcs:
                    keep = self.__select_multimodal_pcs(adata.obsm[pca_key], pc_var_floor, pc_dip_alpha)
                    if keep:
                        sub = adata.obsm[pca_key][:, keep]
                        sel_key = pca_key + "_sel"
                        adata.obsm[sel_key] = (sub - sub.mean(0)) / (sub.std(0) + 1e-8)  # z-score kept PCs
                        cluster_key = sel_key
                        print(f"  select-pcs: kept {len(keep)}/{n_comps} PCs {keep}")
                    else:
                        print("  select-pcs: no PC passed the dip+variance filter; using all PCs")

            # size floor: a domain must clear BOTH the fractional min_cluster_pct
            # and an absolute min_cluster_size (# cells). The absolute floor stops
            # tiny cell types over-segmenting (e.g. 13 cells -> 13 domains); the
            # requested cluster count is capped by population, and we bail to a
            # single domain when the population can't support even two valid ones.
            eff_thresh = min_cluster_pct
            max_k = n_clusters
            if min_cluster_size > 0:
                eff_thresh = max(min_cluster_pct, min_cluster_size / adata.n_obs)
                max_k = min(n_clusters, adata.n_obs // min_cluster_size)
                if max_k < 2:
                    adata.obs["domain"] = pd.Categorical(["0"] * adata.n_obs)
                    return "domain"

            if algorithm == "leiden":
                for key in ['connectivities', 'distances']:
                    if key in adata.obsp:
                        del adata.obsp[key]
                if 'neighbors' in adata.uns:
                    del adata.uns['neighbors']
                # scale k with spot occupancy. For high-population cell types many
                # same-spot cells are near-identical (spot-pooled), so a small k
                # fills up with same-spot neighbors and never reaches other spots,
                # fragmenting the graph into per-spot components (resolution then
                # can't merge them). k must exceed the same-spot cell count to make
                # inter-spot edges.
                n_neigh = n_neighbors
                if 'spot_name' in adata.obs:
                    occ = adata.obs['spot_name'].value_counts()
                    n_neigh = int(min(adata.n_obs - 1, n_neighbors + occ.quantile(0.90)))
                if use_gauss_neighbors:
                    sc.pp.neighbors(adata, n_neighbors=n_neigh, use_rep=cluster_key, metric='euclidean', knn=False, method='gauss')
                else:
                    sc.pp.neighbors(adata, n_neighbors=n_neigh, use_rep=cluster_key, metric='euclidean', knn=True, method='umap')

                sc.tl.leiden(adata, flavor="igraph", n_iterations=-1, resolution=1, key_added="domain",
                             random_state=seed)
                self.__merge_clusters(adata, threshold=eff_thresh, embedding_key=cluster_key, label_key="domain")
            elif algorithm == "agglomerative":
                embedding = adata.obsm[cluster_key]
                k = min(max_k, adata.n_obs)
                labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(embedding)
                adata.obs["domain"] = pd.Categorical(labels.astype(str))
                # over-segment into n_clusters, then absorb sub-threshold clusters
                # into their hierarchical sibling (min Ward-cost neighbor) so the
                # merge follows the dendrogram rather than an arbitrary centroid.
                self.__merge_clusters(adata, threshold=eff_thresh,
                                      embedding_key=cluster_key, label_key="domain",
                                      merge_mode="sibling")
            elif algorithm == "gmm":
                embedding = adata.obsm[cluster_key]
                n_max = max(2, min(max_k, adata.n_obs))
                # BIC-select the mixture, constrained so the smallest component
                # clears the size floor -> "merging" is choosing a coarser model,
                # not reassigning cells. Ternary search over the (U-shaped) BIC.
                labels, feasible = self.__gmm_bic_select(embedding, n_max, eff_thresh, seed)
                adata.obs["domain"] = pd.Categorical(labels.astype(str))
                if not feasible:
                    # even the coarsest model leaves a sub-threshold component ->
                    # fall back to centroid merge to enforce the size floor
                    self.__merge_clusters(adata, threshold=eff_thresh,
                                          embedding_key=cluster_key, label_key="domain",
                                          merge_mode="cluster")
            else:
                raise ValueError(f"unknown clustering algorithm: {algorithm}")
        return "domain"

    @staticmethod
    def __select_multimodal_pcs(pca, var_floor, dip_alpha):
        """Indices of PCs that carry discrete structure: significantly multimodal
        by Hartigan's dip test (p < dip_alpha) AND above a fractional-variance
        floor. The dip test is bandwidth-free (robust across cell-type sizes,
        unlike a KDE peak count); the floor drops low-variance noise PCs."""
        ratio = pca.var(axis=0) / pca.var(axis=0).sum()
        keep = []
        for k in range(pca.shape[1]):
            if ratio[k] <= var_floor:
                continue
            if diptest.diptest(pca[:, k])[1] < dip_alpha:   # (dip_stat, p_value)
                keep.append(k)
        return keep

    @staticmethod
    def __gmm_bic_select(embedding, n_max, min_frac, seed, n_min=2, linear_window=8):
        """Pick a diagonal-GMM component count by BIC, subject to every component
        having population fraction >= min_frac (so the size floor is satisfied by
        model choice, not by reassigning cells).

        Hybrid search: ternary-narrow the range while it is wide (BIC(n) is *often*
        U-shaped), then **linear-scan the small remaining window** (<= linear_window)
        around the optimum. Linear near the minimum avoids a pure ternary settling
        in a local BIC dip, since BIC(n) is not guaranteed unimodal (EM local optima,
        hierarchical structure). If the whole range is already small, it is a full
        linear scan. Returns (labels, feasible); feasible is False if even the
        coarsest model has a sub-threshold component (caller then does a cleanup
        merge)."""
        cache = {}

        def fit(n):
            n = int(min(max(n, n_min), n_max))
            if n not in cache:
                gm = GaussianMixture(n_components=n, covariance_type="diag",
                                     random_state=seed, n_init=1).fit(embedding)
                lab = gm.predict(embedding)
                frac = np.bincount(lab).min() / embedding.shape[0]
                cache[n] = (gm.bic(embedding), frac, lab)
            return cache[n]

        lo, hi = n_min, n_max
        while hi - lo > linear_window:                       # ternary narrowing
            m1, m2 = lo + (hi - lo) // 3, hi - (hi - lo) // 3
            if fit(m1)[0] <= fit(m2)[0]:
                hi = m2
            else:
                lo = m1
        for n in range(lo, hi + 1):                          # linear scan the window
            fit(n)

        feasible = {n: v for n, v in cache.items() if v[1] >= min_frac}
        pool = feasible if feasible else cache
        best = min(pool, key=lambda n: pool[n][0])           # lowest BIC in the pool
        return cache[best][2], bool(feasible)

    def compute_spatial_clusters(self,
                                 embedding_key="embedding",
                                 algorithm: Literal["leiden", "agglomerative", "gmm"] = "leiden",
                                 min_cluster_pct=-1,
                                 seed: int = 0,
                                 n_pca_components: int = 50,
                                 n_clusters: int = 20,
                                 n_neighbors: int = 15,
                                 min_cluster_size: int = 3,
                                 select_pcs: bool = False,
                                 pc_var_floor: float = 0.04,
                                 pc_dip_alpha: float = 0.05,
                                 select_min_cells: int = 500):
        project_metadata = ProjectMetadata()

        adata_aggr = load_aggr_st_data()

        label_key = self._cluster(adata_aggr, embedding_key,
                                  algorithm,
                                  use_gauss_neighbors=False, min_cluster_pct=min_cluster_pct,
                                  seed=seed, n_pca_components=n_pca_components, n_clusters=n_clusters,
                                  n_neighbors=n_neighbors, min_cluster_size=min_cluster_size,
                                  select_pcs=select_pcs, pc_var_floor=pc_var_floor,
                                  pc_dip_alpha=pc_dip_alpha, select_min_cells=select_min_cells)

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
                                    algorithm: Literal["leiden", "agglomerative", "gmm"] = "leiden",
                                    min_cluster_pct=0.05,
                                    seed: int = 0,
                                    n_pca_components: int = 50,
                                    n_clusters: int = 20,
                                    n_neighbors: int = 15,
                                    min_cluster_size: int = 3,
                                    select_pcs: bool = False,
                                    pc_var_floor: float = 0.04,
                                    pc_dip_alpha: float = 0.05,
                                    select_min_cells: int = 500):
        project_metadata = ProjectMetadata()

        if len(type_list) == 0:
            type_list = project_metadata.ct_list

        cluster_results = {}
        for ct in type_list:
            adata_ct = load_ct_spatial_data(ct)

            label_key = self._cluster(adata_ct, embedding_key,
                                      algorithm,
                                      use_gauss_neighbors=False, min_cluster_pct=min_cluster_pct,
                                      seed=seed, n_pca_components=n_pca_components, n_clusters=n_clusters,
                                      n_neighbors=n_neighbors, min_cluster_size=min_cluster_size,
                                      select_pcs=select_pcs, pc_var_floor=pc_var_floor,
                                      pc_dip_alpha=pc_dip_alpha, select_min_cells=select_min_cells)

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
                         merge_mode="connectivity"):
        embedding = adata_ct.obsm[embedding_key]
        labels = adata_ct.obs[label_key].astype(str).copy()
        connectivities = adata_ct.obsp.get('connectivities')

        while labels.nunique() > 1:          # never merge below a single cluster
            cluster_ratio = labels.value_counts() / len(labels)
            if not (cluster_ratio < threshold).any():
                break

            smallest = cluster_ratio.idxmin()
            small_idx = np.where(labels.values == smallest)[0]
            other_labels = [c for c in labels.unique() if c != smallest]

            if merge_mode == "connectivity" and connectivities is not None:
                # merge into most graph-connected neighbor (consistent with Leiden)
                small_to_all = np.asarray(connectivities[small_idx].sum(axis=0)).flatten()
                scores = {c: small_to_all[labels.values == c].sum() for c in other_labels}
                nearest_label = max(scores, key=scores.get)
                labels.iloc[small_idx] = nearest_label

            elif merge_mode == "cluster":
                # merge into nearest cluster by centroid distance
                centroid_matrix = np.vstack([embedding[labels.values == c].mean(axis=0) for c in other_labels])
                small_centroid = embedding[small_idx].mean(axis=0, keepdims=True)
                dists = ((small_centroid - centroid_matrix) ** 2).sum(axis=1)
                labels.iloc[small_idx] = other_labels[dists.argmin()]

            elif merge_mode == "sibling":
                # merge into the cluster with the smallest Ward merge cost — the
                # single cluster this one agglomerates with next up the dendrogram
                # (its hierarchical sibling), rather than a plain nearest centroid.
                # Ward cost(A,B) = 2|A||B|/(|A|+|B|) * ||mean_A - mean_B||^2
                ns = len(small_idx)
                small_centroid = embedding[small_idx].mean(axis=0)
                sizes = np.array([(labels.values == c).sum() for c in other_labels], dtype=float)
                centroids = np.vstack([embedding[labels.values == c].mean(axis=0) for c in other_labels])
                sq = ((small_centroid - centroids) ** 2).sum(axis=1)
                ward = (2.0 * ns * sizes / (ns + sizes)) * sq
                labels.iloc[small_idx] = other_labels[int(ward.argmin())]

            else:
                # assign each cell to its nearest centroid individually
                centroid_matrix = np.vstack([embedding[labels.values == c].mean(axis=0) for c in other_labels])
                cells = embedding[small_idx]
                dists = ((cells[:, None] - centroid_matrix[None]) ** 2).sum(axis=2)
                labels.iloc[small_idx] = np.array(other_labels)[dists.argmin(axis=1)]

        mapping = {c: str(i) for i, c in enumerate(sorted(labels.unique()))}
        # keep it categorical -> plotting (squidpy/scanpy) treats it as discrete
        adata_ct.obs[label_key] = pd.Categorical(labels.map(mapping))
        adata_ct.uns.pop(f'{label_key}_colors', None)
