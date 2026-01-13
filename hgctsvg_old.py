import torch

import numpy as np
import scanpy as sc

from cyclopts import App
from dataclasses import dataclass, field
from typing import Optional

from pathlib import Path
from tqdm.auto import tqdm

from hgctsvg.data import STHypergraphData
from hgctsvg.dataset.hypergraph import STHypergraphListDataset

from hgctsvg.pipeline import STHypergraphEmbedder, HGCTSVGClustering, load_sc_classifier_inputs, load_sc_classifier_input, train_sc_embedding_classifiers, Explainer
from hgctsvg.config import AcceleratorOptions

app = App()

@dataclass
class ModelArgs:
     input_dim: int
     embedding_dim: int
     dropout: float = 0.1

@dataclass
class TrainerArgs:
    num_epochs: int = 100
    batch_size: int = 8192
    accelerator: AcceleratorOptions = AcceleratorOptions.CUDA    

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
    accelerator: AcceleratorOptions = AcceleratorOptions.CUDA

@app.command(help="Fit a hypergraph model for making embeddings.")
def train_embedder(project: str,
         model: ModelArgs,
         trainer: TrainerArgs):

    project_dir = Path("hgctsvg_output") / project

    hypergraph = STHypergraphData.load(project_dir)

    print("hyperedges: ", hypergraph["hyperedge_feats"].shape[0])
    
    embedder = STHypergraphEmbedder(project_dir, 
                                    model.input_dim,
                                    model.embedding_dim,
                                    model.dropout)
    
    embedder.fit(hypergraph, 
                 trainer.num_epochs, 
                 trainer.batch_size,
                 trainer.accelerator)

@app.command(help="Use the trained hypergraph model to generate the embeddings.")
def make_embedding(project: str,
                   model: ModelArgs,
                   trainer: TrainerArgs):
    project_dir = Path("hgctsvg_output") / project

    hypergraph = STHypergraphData.load(project_dir)
    embedder = STHypergraphEmbedder(project_dir, 
                                    model.input_dim,
                                    model.embedding_dim,
                                    model.dropout)
    
    embedder.generate_embedding(hypergraph,
                                trainer.batch_size,
                                trainer.accelerator)

@app.command(help="Using hyperedge embeddings, find the spatial domains present in the Spatial Transcriptomics tissue.")
def compute_spatial_domains(project:str,
                            embedding_key = "embedding",
                            clustering_args: ClusterArgs = ClusterArgs()):
    project_dir =  Path("hgctsvg_output") / project

    adata_st = sc.read_h5ad(project_dir / "ST_init.h5ad")
    spot_embeddings = torch.load(project_dir / "hyperedge_embeddings.pth")
    adata_st.obsm[embedding_key] = spot_embeddings.numpy()

    clustering = HGCTSVGClustering(project_dir)
    clustering.compute_spatial_clusters(adata_st, 
                                        embedding_key, 
                                        clustering_args.resolution)

@app.command(help="Using hypergraph node embeddings, find the spatial domains present for each cell type in the Spatial Transcriptomics tissue.")
def compute_celltype_spatial_domains(project: str,
                             data: ClusterDataArgs,
                             clustering_args: ClusterArgs = ClusterArgs()):
    project_dir = Path("hgctsvg_output") / project

    adata_decomp = sc.read_h5ad(project_dir / "ST_decomposition.h5ad")
    embedding = torch.load(project_dir / "node_embeddings.pth")
    type_list = np.loadtxt(project_dir / "cell_types.txt", dtype=str)

    adata_decomp.obsm[data.embedding_key] = embedding.numpy()
    
    clustering = HGCTSVGClustering(project_dir)
    clustering.compute_celltype_specific_spatial_clusters(adata_decomp, 
                                                  type_list,
                                                  data.cell_type_key,
                                                  data.embedding_key,
                                                  clustering_args.resolution)

@app.command
def train_classifier(project: str,
                     classifier: ClassifierArgs,
                     trainer: TrainerArgs):
    project_dir = Path("hgctsvg_output") / project
    
    cell_types, cell_type_embeddings, cell_type_classes, cell_type_num_classes = load_sc_classifier_inputs(project_dir)

    train_sc_embedding_classifiers(project_dir, 
                                   classifier.embedding_dim,
                                   cell_types,
                                   cell_type_num_classes,
                                   classifier.num_layers,
                                   cell_type_embeddings,
                                   cell_type_classes,
                                   trainer.num_epochs,
                                   trainer.batch_size,
                                   trainer.accelerator)

@app.command(help="Given a classifier trained on embeddings, compute the attribution scores of each embedding dimension")
def compute_cell_embedding_attributions(project:str,
                                        cell_type: str,
                                        classifier: ClassifierArgs,
                                        attribution: AttributionArgs):
    project_dir = Path("hgctsvg_output") / project

    embeddings, classes, num_classes = load_sc_classifier_input(project_dir, cell_type)
    
    explainer = Explainer(project_dir)
    explainer.compute_cell_type_spatial_embedding_attribution(cell_type,
                                                              classifier.embedding_dim,
                                                              classifier.num_layers,
                                                              num_classes,
                                                              embeddings,
                                                              classes,
                                                              attribution.n_steps,
                                                              attribution.accelerator)
    
    

if __name__ == "__main__":
    app()