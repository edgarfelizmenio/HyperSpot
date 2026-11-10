import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch

from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from ..metadata import ProjectMetadata
from ..utils import (
    load_orig_st_data,
    load_decomposed_st_data,
    load_ct_spatial_data,
    save_ct_spatial_data,
    load_aggr_st_data,
    save_aggr_st_data,
    guarded_spot_reps,
)

class HyperSpotClustering(object):
    def __init__(self):
        pass
        
    def _cluster(
        self,
        adata,
        embedding_key: str,
        min_cluster_pct: float = -1,
        seed: int = 42,
        n_pca_components: int = 50,
        resolution: float = 1,
        max_clusters: int = None,
        dedup_spots: bool = True,    # same-spot same-CT cells are exact duplicates: cluster unique spots, broadcast
        min_cluster_size: int = 3,
    ) -> str:

        # Cap OpenBLAS threads around the BLAS-heavy clustering ops (PCA). 
        # On high-core machines the default pools over-subscribe past 
        # OpenBLAS's compiled 128-thread limit, producing the "precompiled 
        # NUM_THREADS exceeded" warning and "Bad memory unallocation!" 
        # errors. Scoped to the BLAS API only.
        with threadpool_limits(limits=16, user_api="blas"):
            cluster_key = embedding_key
            dim = adata.obsm[embedding_key].shape[1]
            n_comps = min(n_pca_components, dim - 1, adata.n_obs - 1)
            # PCA only helps when it actually reduces dimensionality. A learned
            # embedding already within the clustering algorithms' comfortable range
            # (dim <= n_pca_components) is clustered as-is: PCA on it would be a
            # near-identity rotation (distance-preserving) with no benefit, and for
            # dim <= n_pca_components there is no low-variance tail worth truncating.
            if dim > n_pca_components and n_comps > 0:
                pca_key = embedding_key + "_pca"
                adata.obsm[pca_key] = PCA(n_components=n_comps, random_state=seed).fit_transform(adata.obsm[embedding_key])
                cluster_key = pca_key
            else:
                print(f"  pca: embedding dim {dim} <= n_pca_components {n_pca_components}; clustering on embedding as-is")
                
            # DEDUP: same-spot same-CT cells are EXACT duplicates
            # (the deconvolution gives every cell of a type in a spot the same profile
            # -> identical embedding, verified within-spot deviation = 0). Clustering the
            # UNIQUE representatives is lossless and removes the distance-0 duplicate mass
            # that fragments Leiden (a dense spot's duplicates form an isolated clique). 
            # Guarded no-op on genuine per-cell variation or dedup_spots=False. Cluster 
            # reps, then broadcast labels to cells.
            rep_idx, inv = self.__spot_reps(adata, cluster_key, dedup_spots)
            
            # bail to a single domain when the representatives can't form two valid
            # domains (also guards leiden/kNN against a single representative)
            if len(rep_idx) < max(2, 2 * min_cluster_size):
                adata.obs["domain"] = pd.Categorical(["0"] * adata.n_obs)
                return "domain"
            
            Xrep = np.asarray(adata.obsm[cluster_key])[rep_idx]
            rep = self.__leiden(Xrep, seed, resolution)

            # size floor, applied on the representatives with Leiden's kNN graph:
            # min_cluster_size counts representatives; min_cluster_pct is a fraction of
            # CELLS (each representative weighted by the number of cells it stands for).
            self.__merge_clusters(
                rep,
                min_size=min_cluster_size,
                min_frac=min_cluster_pct,
                weights=np.bincount(inv),
                max_clusters=max_clusters,
            )

            # broadcast rep -> cells; rep (and its graph) is discarded on return
            adata.obs["domain"] = pd.Categorical(rep.obs["domain"].astype(str).values[inv])
            adata.uns.pop("domain_colors", None)

        return "domain"

    def __spot_reps(self, adata, cluster_key, dedup_spots=True, dup_tol=1e-3):
        """Map cells to unique (spot) representatives for dedup.

        Returns (rep_idx, inv): rep_idx[r] is a representative cell of unique spot r,
        inv[c] is the unique-spot id of cell c, so labels[inv] broadcasts rep labels
        back to all cells.

        WHY (a workaround for a deconvolution limitation, not a general method):
        Spotiphy-style deconvolution assigns one per-type expression profile per spot,
        so every cell of a given type in a spot has *identical* expression -> identical
        embedding (verified: max within-spot deviation = 0 across all genes/CTs on
        Kidney). Those exact duplicates sit at distance 0: Leiden fragments (a dense
        spot's cells form an isolated distance-0 clique -> its own domain; RCC gave 316
        clusters, dedup -> 11). Clustering the unique reps is LOSSLESS *because the 
        cells are identical*.

        GUARDED -- conditional on that duplication. On genuine per-cell variation
        (single-cell resolution, or a per-cell deconvolution) same-spot cells are NOT
        identical and clustering all cells is correct; deduping would discard real
        resolution. So dedup only when max within-spot deviation < dup_tol (per spot);
        else identity mapping (cluster all cells). dedup_spots=False
        forces all cells. Per-spot ST data (unique spots) is a natural no-op.
        """
        spots = adata.obs['spot_name'].to_numpy() if 'spot_name' in adata.obs else None
        # embedding is narrow (~20-d) -> full check (no sampling needed)
        return guarded_spot_reps(np.asarray(adata.obsm[cluster_key]), spots,
                                 dedup=dedup_spots, tol=dup_tol)

    def __leiden(self, X, seed, resolution=1, n_neighbors=15):
        """Leiden on representative rows X (already deduped). Returns a temporary AnnData
        with obs['domain'], obsm['_rep'] (= X) and the kNN graph obsp['connectivities']."""
        rep = ad.AnnData(np.zeros((len(X), 1), dtype=np.float32))
        rep.obsm['_rep'] = np.asarray(X)
        # cap k at what the data can supply (scanpy's k counts the point itself);
        # a single representative never gets here (bail-out in _cluster)
        k = min(n_neighbors, max(2, rep.n_obs - 1))
        sc.pp.neighbors(rep, n_neighbors=k, use_rep='_rep', metric='euclidean', knn=True, method='umap')
        sc.tl.leiden(rep, flavor="igraph", n_iterations=-1, resolution=resolution, key_added="domain",
                     random_state=seed)
        return rep

    def compute_spatial_clusters(
        self,
        embedding_key="embedding",
        seed: int = 42,
        n_pca_components: int = 50,
        min_cluster_size: int = 3,
    ):
        # tissue level: one row per spot, so there is nothing to dedup and no cell
        # fraction floor -- Leiden on the spots, then the min_cluster_size merge.
        project_metadata = ProjectMetadata()
        adata_aggr = load_aggr_st_data()
        label_key = self._cluster(
            adata_aggr,
            embedding_key,
            seed=seed,
            n_pca_components=n_pca_components,
            min_cluster_size=min_cluster_size,
            dedup_spots=False,
        )

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
                                    min_cluster_pct=-1,
                                    seed: int = 42,
                                    n_pca_components: int = 50,
                                    min_cluster_size: int = 3,
                                    dedup_spots: bool = True):
        project_metadata = ProjectMetadata()

        if len(type_list) == 0:
            type_list = project_metadata.ct_list

        cluster_results = {}
        for ct in type_list:
            adata_ct = load_ct_spatial_data(ct)

            label_key = self._cluster(
                adata_ct, 
                embedding_key,                      
                min_cluster_pct=min_cluster_pct,   
                seed=seed, 
                n_pca_components=n_pca_components, 
                min_cluster_size=min_cluster_size,
                dedup_spots=dedup_spots
            )

            save_ct_spatial_data(adata_ct, ct)
            cluster_results[ct] = adata_ct.obs[label_key].to_numpy(dtype=str)
            print(f"{ct}: num cells = {adata_ct.n_obs}, num domains = {adata_ct.obs[label_key].nunique()}")

        np.savez(project_metadata.clustering_dir / "cell_type_spatial_domains.npz", **cluster_results)

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
        for ct in project_metadata.ct_list:
            adata_ct = load_ct_spatial_data(ct)
            cell_ids = torch.from_numpy(
                adata_ct.obs['cell_idx'].to_numpy() if 'cell_idx' in adata_ct.obs.columns
                else adata_ct.obs_names.astype(int).to_numpy()
            )
            adata_ct.obsm[embedding_key] = embedding[cell_ids].numpy()
            save_ct_spatial_data(adata_ct, ct)

    def __merge_clusters(self, rep, min_size=3, min_frac=-1, weights=None,
                         label_key="domain", max_clusters=None):
        """Merge small domains on the representatives (output of __leiden), each WHOLE
        domain into its most graph-connected neighbour on Leiden's kNN graph.

        A domain is too small if it has < min_size representatives, or covers < min_frac
        of the CELLS (weights[r] = number of cells representative r stands for; all ones
        when nothing was deduped). With max_clusters set, merging also continues until at
        most that many domains remain. Each step dissolves the smallest offending domain.
        """
        X = rep.obsm['_rep']
        C = rep.obsp['connectivities']
        w = np.ones(rep.n_obs) if weights is None else np.asarray(weights, dtype=float)
        labels = rep.obs[label_key].astype(str).copy()

        while labels.nunique() > 1:          # never merge below a single domain
            n_reps = labels.value_counts()
            frac = pd.Series(w).groupby(labels.values).sum()[n_reps.index] / w.sum()
            bad = n_reps.index[(n_reps < min_size) | (frac < min_frac)]
            if len(bad) == 0 and (max_clusters is None or labels.nunique() <= max_clusters):
                break

            pool = bad if len(bad) else n_reps.index
            smallest = frac[pool].idxmin()   # smallest (in cells) among the offending domains
            small_idx = np.where(labels.values == smallest)[0]
            others = [c for c in labels.unique() if c != smallest]

            to_all = np.asarray(C[small_idx].sum(axis=0)).ravel()
            scores = {c: to_all[labels.values == c].sum() for c in others}
            if max(scores.values()) > 0:
                target = max(scores, key=scores.get)        # most graph-connected neighbour
            else:                                           # disconnected from all: nearest centroid
                cent = np.vstack([X[labels.values == c].mean(axis=0) for c in others])
                target = others[int(((X[small_idx].mean(axis=0) - cent) ** 2).sum(axis=1).argmin())]
            labels.iloc[small_idx] = target                 # the whole domain merges

        mapping = {c: str(i) for i, c in enumerate(sorted(labels.unique()))}
        rep.obs[label_key] = pd.Categorical(labels.map(mapping))
