
import cyclopts
import numpy as np
import scanpy as sc
import torch

from cyclopts import App, Parameter
from dataclasses import dataclass, astuple
from typing import Optional, Annotated, List

from pathlib import Path
from tqdm.auto import tqdm

from . import config

from .data import STHypergraphData
from .dataset.hypergraph import STHypergraphListDataset

from .pipeline import HypergraphEmbedderDriver, HypergraphEmbedderArgs, HGCTSVGClustering, SCEmbeddingSpatialClassifierDriver, Explainer, ExplainerArgs

app = App(name="hgctsvg")

@Parameter(name="*")
@dataclass
class ProjectArgs:
    name: Annotated[str, Parameter(name="project", help="Name of the project.")]

    @property
    def project_dir(self):
        return config.project_files_dir / self.name

@dataclass
class EmbedderCLIArgs:
    model_name: Annotated[str, Parameter(help=f"Name of hypergraph embedder to use. Possible values: {HypergraphEmbedderDriver.get_available_embedders()}")]
    input_dim: Annotated[int, Parameter(help="Number of input features.")]
    embedding_dim: Annotated[int, Parameter(help="Size of embedding.")]
    dropout: Annotated[Optional[float], Parameter(help="Dropout parameter.")] = 0.1
    latent_dim: Annotated[Optional[int], Parameter(help="Size of hidden embedding (for transformer and fusion embedders only).")] = None
    num_heads: Annotated[Optional[int], Parameter(help="Number of attention heads (for transformers only).")] = None
    num_blocks: Annotated[Optional[int], Parameter(help="Number of transformer blocks (for transformers only).")] = None
    num_hyperedges: Annotated[Optional[int], Parameter(help="Number of hyperedges (for transformers only).")] = None
    hyperedge_feats: Annotated[Optional[str], Parameter(help="Initial value of hyperedge features (for transformers only).")] = None

@dataclass
class ClassifierCLIArgs:
    embedding_dim: Annotated[int, Parameter(help="The size of the embedding.")]
    num_layers: Annotated[int, Parameter(help="The number of layers of the classifier")] = 3
    
@dataclass
class TrainerArgs:
    num_epochs: int = 100
    batch_size: int = 8192
    accelerator: config.AcceleratorOptions = config.AcceleratorOptions.CUDA    

@dataclass
class ClusterDataArgs:
    cell_type_key: str
    embedding_key: str = "embedding"

@dataclass
class ClusterArgs:
    resolution: float = 1.0
    # TODO: include leiden clustering options

@dataclass
class ExplainerCLIArgs:
    n_steps: Annotated[int, Parameter(help="Number of steps performed by Integrated Gradients")] = 1000
    batch_size: Annotated[int, Parameter(help="Number of input samples for each batch of Integrated Gradients.")] = None
    accelerator: config.AcceleratorOptions = config.AcceleratorOptions.CUDA

@app.command(help="Create a project files directory.",
             sort_key=1)
def init_project_files_dir():
    if not config.project_files_dir.exists():
        config.project_files_dir.mkdir()

@app.command(help="Fit a hypergraph model for making embeddings.",
             sort_key=2)
def train_embedder(project: ProjectArgs,
         model: EmbedderCLIArgs,
         trainer: TrainerArgs):

    hypergraph = STHypergraphData.load(project.project_dir)

    print("hyperedges: ", hypergraph["hyperedge_feats"].shape[0])
    print(vars(model))
    
    embedder = HypergraphEmbedderDriver(project.project_dir, 
                                    HypergraphEmbedderArgs(**vars(model)))
    
    embedder.fit(hypergraph,
                 HypergraphEmbedderArgs(**vars(model)),
                 trainer.num_epochs, 
                 trainer.batch_size,
                 trainer.accelerator)

@app.command(help="Use the trained hypergraph model to generate the embeddings.",
             sort_key=3)
def make_embedding(project: ProjectArgs,
                   model: EmbedderCLIArgs,
                   trainer: TrainerArgs):

    hypergraph = STHypergraphData.load(project.project_dir)
    embedder = HypergraphEmbedderDriver(project.project_dir)

    embedder.generate_embedding(hypergraph,
                                HypergraphEmbedderArgs(**vars(model)),
                                trainer.batch_size,
                                trainer.accelerator)

@app.command(help="Using hyperedge embeddings, find the spatial domains present in the Spatial Transcriptomics tissue.",
             sort_key=4)
def compute_spatial_domains(project: ProjectArgs,
                            embedding_key = "embedding",
                            clustering_args: ClusterArgs = ClusterArgs()):

    adata_st = sc.read_h5ad(project.project_dir / "ST_init.h5ad")
    spot_embeddings = torch.load(project.project_dir / "hyperedge_embeddings.pth")
    adata_st.obsm[embedding_key] = spot_embeddings.numpy()

    clustering = HGCTSVGClustering(project.project_dir)
    clustering.compute_spatial_clusters(adata_st, 
                                        embedding_key, 
                                        clustering_args.resolution)

@app.command(help="Using hypergraph node embeddings, find the spatial domains present for each cell type in the Spatial Transcriptomics tissue.",
            sort_key=5)
def compute_celltype_spatial_domains(project: ProjectArgs,
                             data: ClusterDataArgs,
                             clustering_args: ClusterArgs = ClusterArgs()):

    adata_decomp = sc.read_h5ad(project.project_dir / "ST_decomposition.h5ad")
    embedding = torch.load(project.project_dir / "node_embeddings.pth")
    type_list = np.loadtxt(project.project_dir / "cell_types.txt", dtype=str)
 
    adata_decomp.obsm[data.embedding_key] = embedding.numpy()
    
    clustering = HGCTSVGClustering(project.project_dir)
    clustering.compute_celltype_specific_spatial_clusters(adata_decomp, 
                                                  type_list,
                                                  data.cell_type_key,
                                                  data.embedding_key,
                                                  clustering_args.resolution)

@app.command(help="Fit a classifier that maps embedding features into cell type-specific spatial domains", 
             sort_key=6)
def train_celltype_domain_classifiers(project: ProjectArgs,
                     classifier: ClassifierCLIArgs,
                     trainer: TrainerArgs):

    driver = SCEmbeddingSpatialClassifierDriver(project.project_dir)
    driver.train_sc_embedding_classifiers(classifier.embedding_dim,
                                          classifier.num_layers,
                                          trainer.num_epochs,
                                          trainer.batch_size,
                                          trainer.accelerator)

@app.command(help="Given the trained classifier trained on embeddings, identify embedding dimensions that are highly attributed into each celltype-specific spatial domains.",
             sort_key=7)
def compute_cell_embedding_attributions(project: ProjectArgs,
                                        classifier: ClassifierCLIArgs,
                                        explainer: ExplainerCLIArgs,
                                        cell_types: Optional[List[str]] = None):
    
    driver = Explainer(project.project_dir)
    driver.compute_cell_type_spatial_embedding_attribution(cell_types,
                                                           *astuple(classifier),
                                                           ExplainerArgs(**vars(explainer)))



@app.command(help="Check if project folder exists.",
             sort_key=8)
def folder_test(project: ProjectArgs):
    if project.project_dir.exists():
        print(f"Folder {project.project_dir} exists!")
        for child in project.project_dir.iterdir(): 
              print(f"\t{child}")
    else:
        print(f"Folder {project.project_dir} does not exist!")

if __name__ == "__main__":
    app()