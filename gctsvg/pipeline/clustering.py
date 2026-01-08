import numpy as np
import scanpy as sc
import os

from pathlib import Path

class STClustering(object):
    def __init__(self, output_dir):
        self.output_dir = f"{output_dir}/clustering"

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        
    def compute_clusters(self, adata, embedding_key="embedding", resolution=0.6):
        adata_clusters = adata.copy()

        print("Building neighborhood graph...")
        sc.pp.neighbors(adata_clusters, use_rep=embedding_key)

        print("Performing clustering...")
        sc.tl.leiden(adata_clusters, flavor="igraph", n_iterations=-1, resolution=resolution)

        np_clusters = adata_clusters.obs["leiden"].to_numpy().astype("int")

        cluster_path = f"{self.output_dir}/cluster_labels.npy"
        adata_path = f"{self.output_dir}/adata_clusters.h5ad"

        sc.write(adata_path, adata_clusters)
        np.save(cluster_path, np_clusters)        
        
        print(f"Saved clustering results to {cluster_path}.")
        print(f"Saved AnnData with clustering to {adata_path}.")

class HGCTSVGClustering(object):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "clustering"

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def compute_spatial_clusters(self,
                            adata_st,
                            embedding_key="embedding",
                            resolution=1):
        
        sc.pp.neighbors(adata_st, use_rep='embedding', metric='euclidean', knn=False, method='gauss')
        sc.tl.leiden(adata_st, flavor="igraph", n_iterations=-1, resolution=resolution)

        cluster_results = adata_st.obs['leiden'].to_numpy(dtype=str)
        
        adata_st.write_h5ad(self.output_dir / "ST_init.h5ad")
        np.save(self.output_dir / "spatial_domains.npy", cluster_results)

        print(adata_st)
    
    def compute_celltype_specific_spatial_clusters(self, 
                                           adata_decomp,
                                           type_list,
                                           cell_type_key='cell_type',
                                           embedding_key="embedding", 
                                           resolution=1):


        # merge small classes with larger ones
        adata_celltypes = []
        cluster_results = {}
        for i, cell_type in enumerate(type_list):
            adata_celltype = adata_decomp[adata_decomp.obs[cell_type_key] == cell_type].copy()
            sc.pp.neighbors(adata_celltype, use_rep=embedding_key, metric='euclidean', knn=False, method='gauss')
            sc.tl.leiden(adata_celltype, flavor="igraph", n_iterations=-1, resolution=resolution)
            
            adata_celltype.write_h5ad(self.output_dir / f"ST_{cell_type}.h5ad")
            
            cluster_results[cell_type] = adata_celltype.obs['leiden'].to_numpy(dtype=str) 

            print(adata_celltype)
            
        np.savez(self.output_dir / "cell_type_spatial_domains.npz", *cluster_results)