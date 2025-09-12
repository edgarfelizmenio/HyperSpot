import scanpy as sc

from cyclopts import App

from gctsvg.pipeline.clustering import STClustering

app = App()

@app.default
def main(project: str, resolution: float = 1.0):
    project_dir = f"gctsvg_output/{project}"
    embedding_dir = f"gctsvg_output/{project}/embedding" 
    embedding_filename = f"{embedding_dir}/adata_embedding.h5ad"

    adata_embedding = sc.read(embedding_filename)
    
    clustering = STClustering(project_dir)
    clustering.compute_clusters(adata_embedding, resolution=resolution)

if __name__ == "__main__":
    app()
    
    

