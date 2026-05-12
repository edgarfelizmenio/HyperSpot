from sklearn.metrics import silhouette_score

from ..data import STHypergraphData

from ..metadata import ProjectMetadata

from ..model.hypergraph import HypergraphEmbedderArgs

from ..utils import (
    load_ct_spatial_data, 
    load_aggr_st_data
)

from . import (
    HypergraphEmbedderDriver,
    HGCTSVGClustering,
    compute_domain_svgs,
    compute_domain_ct_svgs,
    create_svg_list,
)

def HGCTSVG_eval_pipeline(
    run_number: int,
    args: HypergraphEmbedderArgs,
    num_epochs,
    batch_size,
    accelerator,
    embedding_key="embedding",
    min_cluster_pct=0.05
):
    
    project_metadata = ProjectMetadata()
    
    # representation learning
    hypergraph = STHypergraphData.load()
    embedder = HypergraphEmbedderDriver()
    embedder.fit(
        hypergraph,         
        args,
        num_epochs, 
        batch_size,         
        accelerator
    )
    embedder.generate_embedding(hypergraph,
                                args,
                                batch_size,
                                accelerator)

    clustering_driver = HGCTSVGClustering()

    # clustering tissue domains
    clustering_driver.create_aggr_st_data()
    clustering_driver.load_embedding_to_aggr_data(embedding_key)
    clustering_driver.compute_spatial_clusters(
        embedding_key=embedding_key,
        min_cluster_pct=-1
    )
    
    # clustering cell type domains
    clustering_driver.load_embedding_to_ct_spatial_data(embedding_key)
    clustering_driver.compute_ct_spatial_clusters(
        embedding_key=embedding_key,                                          
        min_cluster_pct=min_cluster_pct
    )
    
    run_dir = project_metadata.eval_dir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # save clusters to eval_dir
    adata_aggr = load_aggr_st_data()
    adata_aggr.obs['leiden'].astype(int).to_csv(run_dir / "clusters_ST.csv")

    top_genes = compute_domain_svgs(rank_by='scores')
    top_gene_list = create_svg_list(top_genes)
    top_gene_list.to_csv(run_dir / "svgs_ST.csv")
    
    for ct in project_metadata.ct_list:
        adata_ct = load_ct_spatial_data(ct)
        adata_ct.obs['leiden'].astype(int).to_csv(run_dir / f"clusters_ct_{ct}.csv")

        top_ct_genes = compute_domain_ct_svgs(ct, rank_by='scores')
        top_ct_gene_list = create_svg_list(top_ct_genes)
        top_ct_gene_list.to_csv(run_dir / f"svgs_ct_{ct}.csv")

def run_eval(
    args: HypergraphEmbedderArgs,
    num_epochs,
    batch_size,
    accelerator,
    embedding_key="embedding",
    min_cluster_pct=0.05,
    n = 10,
):
    
    for i in range(n):
        HGCTSVG_eval_pipeline(
            i,
            args,
            num_epochs,
            batch_size,
            accelerator,
            embedding_key="embedding",
            min_cluster_pct=0.05
        )


# compute silhouette score across runs
def compute_ct_silhouette_score(adata):
    pass

def compute_mean_ct_silhouette_score():
    silhouette_scores = [compute_ct_silhouette_score(ct) for ct in cell_types]
        

# compute jaccard score across runs

def compute_mean_jaccard_score():
    pass