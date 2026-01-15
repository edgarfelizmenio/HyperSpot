
import cyclopts
import numpy as np
import scanpy as sc
import torch

from cyclopts import App, Parameter
from dataclasses import dataclass, field
from typing import Optional, Annotated

from pathlib import Path
from tqdm.auto import tqdm

from . import config

from .data import STHypergraphData
from .dataset.hypergraph import STHypergraphListDataset

from .pipeline import STHypergraphEmbedder, HGCTSVGClustering, load_sc_classifier_inputs, load_sc_classifier_input, train_sc_embedding_classifiers, Explainer

app = App(name="hgctsvg")

@Parameter(name="*")
@dataclass
class ProjectArgs:
    name: Annotated[str, Parameter(name="project", help="Name of the project.")]

    @property
    def project_dir(self):
        return config.project_files_dir / self.name

@dataclass
class ModelArgs:
     input_dim: Annotated[int, Parameter(help="Number of input features.")]
     embedding_dim: Annotated[int, Parameter(help="Size of embedding.")]
     dropout: Annotated[float, Parameter(help="Dropout parameter.")] = 0.1

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
class ClassifierArgs:
    embedding_dim: int
    num_layers: int = 3

@dataclass
class AttributionArgs:
    n_steps: int = 1000
    accelerator: config.AcceleratorOptions = config.AcceleratorOptions.CUDA

@app.command(help="Fit a hypergraph model for making embeddings.",
             sort_key=1)
def train_embedder(project: ProjectArgs,
         model: ModelArgs,
         trainer: TrainerArgs):

    hypergraph = STHypergraphData.load(project.project_dir)

    print("hyperedges: ", hypergraph["hyperedge_feats"].shape[0])
    
    embedder = STHypergraphEmbedder(project_dir, 
                                    model.input_dim,
                                    model.embedding_dim,
                                    model.dropout)
    
    embedder.fit(hypergraph, 
                 trainer.num_epochs, 
                 trainer.batch_size,
                 trainer.accelerator)

@app.command(help="Use the trained hypergraph model to generate the embeddings.",
             sort_key=2)
def make_embedding(project: ProjectArgs,
                   model: ModelArgs,
                   trainer: TrainerArgs):

    hypergraph = STHypergraphData.load(project.project_dir)
    embedder = STHypergraphEmbedder(project_dir, 
                                    model.input_dim,
                                    model.embedding_dim,
                                    model.dropout)
    
    embedder.generate_embedding(hypergraph,
                                trainer.batch_size,
                                trainer.accelerator)

@app.command(help="Using hyperedge embeddings, find the spatial domains present in the Spatial Transcriptomics tissue.",
             sort_key=3)
def compute_spatial_domains(project: ProjectArgs,
                            embedding_key = "embedding",
                            clustering_args: ClusterArgs = ClusterArgs()):

    adata_st = sc.read_h5ad(project_dir / "ST_init.h5ad")
    spot_embeddings = torch.load(project_dir / "hyperedge_embeddings.pth")
    adata_st.obsm[embedding_key] = spot_embeddings.numpy()

    clustering = HGCTSVGClustering(project.project_dir)
    clustering.compute_spatial_clusters(adata_st, 
                                        embedding_key, 
                                        clustering_args.resolution)

@app.command(help="Using hypergraph node embeddings, find the spatial domains present for each cell type in the Spatial Transcriptomics tissue.",
            sort_key=4)
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
             sort_key=5)
def train_celltype_domain_classifier(project: ProjectArgs,
                     classifier: ClassifierArgs,
                     trainer: TrainerArgs):
    
    cell_types, cell_type_embeddings, cell_type_classes, cell_type_num_classes = load_sc_classifier_inputs(project.project_dir)

    train_sc_embedding_classifiers(project.project_dir, 
                                   classifier.embedding_dim,
                                   cell_types,
                                   cell_type_num_classes,
                                   classifier.num_layers,
                                   cell_type_embeddings,
                                   cell_type_classes,
                                   trainer.num_epochs,
                                   trainer.batch_size,
                                   trainer.accelerator)

@app.command(help="Given the trained classifier trained on embeddings, identify embedding dimensions that are highly attributed into each celltype-specific spatial domains.",
             sort_key=6)
def compute_cell_embedding_attributions(project: ProjectArgs,
                                        cell_type: str,
                                        classifier: ClassifierArgs,
                                        attribution: AttributionArgs):

    embeddings, classes, num_classes = load_sc_classifier_input(project.project_dir, cell_type)
    
    explainer = Explainer(project.project_dir)
    explainer.compute_cell_type_spatial_embedding_attribution(cell_type,
                                                              classifier.embedding_dim,
                                                              classifier.num_layers,
                                                              num_classes,
                                                              embeddings,
                                                              classes,
                                                              attribution.n_steps,
                                                              attribution.accelerator)

@app.command(help="Check if project folder exists.",
             sort_key=7)
def folder_test(project: ProjectArgs):
    if project.project_dir.exists():
        print(f"Folder {project.project_dir} exists!")
        for child in project.project_dir.iterdir(): 
              print(f"\t{child}")
    else:
        print(f"Folder {project.project_dir} does not exist!")

if __name__ == "__main__":
    app()