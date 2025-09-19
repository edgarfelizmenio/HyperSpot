import scanpy as sc

from cyclopts import App

from gctsvg.pipeline.clustering import STClustering
from gctsvg.pipeline import utils

app = App()

@app.default
def main(project: str, resolution: float = 1.0, version: str | int = "latest"):
    project_dir = f"gctsvg_output/{project}"

    run_dir = utils.get_experiment_run(project_dir, "st_embedder", version=version)
    embedding_dir = run_dir / "embedding" 
    embedding_filename = embedding_dir / "adata_embedding.h5ad"
    adata_embedding = sc.read(embedding_filename)
    
    clustering = STClustering(project_dir)
    clustering.compute_clusters(adata_embedding, resolution=resolution)

if __name__ == "__main__":
    app()
    
    

