import numpy as np
import scanpy as sc
import os


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
        sc.tl.leiden(adata_clusters, flavor="igraph", n_iterations=2, resolution=resolution)

        np_clusters = adata_clusters.obs["leiden"].to_numpy().astype("int")

        cluster_path = f"{self.output_dir}/cluster_labels.npy"
        adata_path = f"{self.output_dir}/adata_clusters.h5ad"

        sc.write(adata_path, adata_clusters)
        np.save(cluster_path, np_clusters)        
        
        print(f"Saved clustering results to {cluster_path}.")
        print(f"Saved AnnData with clustering to {adata_path}.")
        