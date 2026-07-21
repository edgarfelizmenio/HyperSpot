import numpy as np
import scanpy as sc
import torch

from cyclopts import Parameter
from dataclasses import astuple
from typing import Optional, Annotated, List

from pathlib import Path

from .. import config
from ..data import STHypergraphData

from ..pipeline import HypergraphEmbedderDriver, HypergraphEmbedderArgs, HGCTSVGClustering, SCEmbeddingSpatialClassifierDriver, Explainer, ExplainerArgs

@Parameter(name="*")
@dataclass
class ProjectArgs:
    name: Annotated[str, Parameter(name="project", help="Name of the project.")]

    @property
    def project_dir(self):
        return project_files_dir / self.name

@dataclass
class LossFxnCLIArgs:
    name: Annotated[str, Paramter(help=f"Name of loss function.")]

def convert_cliaargs_to_args(cfgargs_cls, args_cls):
    return args_cls(**vars(cfgargs_cls))

@dataclass
class EmbedderCLIArgs:
    model_name: Annotated[str, Parameter(help=f"Name of hypergraph embedder to use. Possible values: {HypergraphEmbedderDriver.get_available_embedders()}")]
    input_dim: Annotated[int, Parameter(help="Number of input features.")]
    embedding_dim: Annotated[int, Parameter(help="Size of embedding.")]
    dropout: Annotated[Optional[float], Parameter(help="Dropout parameter.")] = 0.1
    hyperedge_pool: str = "spot"          # "spot" (own-spot cells) | "neighborhood"

@dataclass
class ClassifierCLIArgs:
    embedding_dim: Annotated[int, Parameter(help="The size of the embedding.")]
    num_layers: Annotated[int, Parameter(help="The number of layers of the classifier")] = 3
    
@dataclass
class TrainerArgs:
    num_epochs: int = 100
    batch_size: int = 8192
    accelerator: AcceleratorOptions = AcceleratorOptions.CUDA

@dataclass
class RegularizerArgs:
    """Opt-in, per-cell-type training regularizers (all default off)."""
    lambda_var: float = 0.0
    lambda_cov: float = 0.0
    lambda_nbr: float = 0.0
    temp_nbr: float = 0.5
    edge_preserving: bool = True
    reg_weight: float = 0.0
    optimize_hyperedges: bool = False
    rho_h: float = 1.0
    lambda_var_h: float = 0.0
    lambda_cov_h: float = 0.0
    lambda_nbr_h: float = 0.0

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
    accelerator: AcceleratorOptions = AcceleratorOptions.CUDA

def init_project_files_dir():
    if not config.project_files_dir.exists():
        config.project_files_dir.mkdir()

def train_embedder(project: ProjectArgs,
         model: EmbedderCLIArgs,
         trainer: TrainerArgs,
         regularizer: RegularizerArgs = RegularizerArgs()):

    hypergraph = STHypergraphData.load(project.project_dir)

    print("hyperedges: ", hypergraph["hyperedge_feats"].shape[0])
    print(vars(model))
    
    embedder = HypergraphEmbedderDriver(project.project_dir, 
                                    HypergraphEmbedderArgs(**vars(model)))
    
    embedder.fit(hypergraph,
                 HypergraphEmbedderArgs(**vars(model)),
                 trainer.num_epochs,
                 trainer.batch_size,
                 trainer.accelerator,
                 lambda_var=regularizer.lambda_var,
                 lambda_cov=regularizer.lambda_cov,
                 lambda_nbr=regularizer.lambda_nbr,
                 temp_nbr=regularizer.temp_nbr,
                 edge_preserving=regularizer.edge_preserving,
                 reg_weight=regularizer.reg_weight,
                 optimize_hyperedges=regularizer.optimize_hyperedges,
                 rho_h=regularizer.rho_h,
                 lambda_var_h=regularizer.lambda_var_h,
                 lambda_cov_h=regularizer.lambda_cov_h,
                 lambda_nbr_h=regularizer.lambda_nbr_h)

def make_embedding(project: ProjectArgs,
                   model: EmbedderCLIArgs,
                   trainer: TrainerArgs):

    hypergraph = STHypergraphData.load(project.project_dir)
    embedder = HypergraphEmbedderDriver(project.project_dir)

    embedder.generate_embedding(hypergraph,
                                HypergraphEmbedderArgs(**vars(model)),
                                trainer.batch_size,
                                trainer.accelerator)

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

def train_celltype_domain_classifiers(project: ProjectArgs,
                     classifier: ClassifierCLIArgs,
                     trainer: TrainerArgs):

    driver = SCEmbeddingSpatialClassifierDriver(project.project_dir)
    driver.train_sc_embedding_classifiers(classifier.embedding_dim,
                                          classifier.num_layers,
                                          trainer.num_epochs,
                                          trainer.batch_size,
                                          trainer.accelerator)

def compute_cell_embedding_attributions(project: ProjectArgs,
                                        classifier: ClassifierCLIArgs,
                                        explainer: ExplainerCLIArgs,
                                        cell_types: Optional[List[str]] = None):
    
    driver = Explainer(project.project_dir)
    driver.compute_cell_type_spatial_embedding_attribution(cell_types,
                                                           *astuple(classifier),
                                                           ExplainerArgs(**vars(explainer)))

def compute_metagene_to_region_attribution(project: ProjectArgs,
                                           cell_type: str,
                                           region: int,
                                           classifier: ClassifierCLIArgs,
                                           explainer: ExplainerCLIArgs):

    driver = Explainer(project.project_dir)
    driver.compute_metagene_to_region_attribution(cell_type,
                                                 region,
                                                 *astuple(classifier),
                                                 ExplainerArgs(**vars(explainer)))

def compute_gene_to_cell_metagene_attribution(project: ProjectArgs,
                                         model: EmbedderCLIArgs,
                                         explainer: ExplainerCLIArgs,
                                         idx: Annotated[Optional[List[str]], Parameter(help="Indexes of metagenes where attributions have to be computed. If nothing is provided, computes the attributions of all features to all metagenes.")] = None):
    
    hypergraph = STHypergraphData.load(project.project_dir)
    driver = Explainer(project.project_dir)
    driver.compute_cell_feature_to_cell_metagene_attribution(hypergraph,
                                                             HypergraphEmbedderArgs(**vars(model)),
                                                             ExplainerArgs(**vars(explainer)),
                                                             idx)

def compute_gene_to_celltype_domain_attribution(project: ProjectArgs,
                                              cell_type: str,
                                              model: EmbedderCLIArgs,
                                              classifier: ClassifierCLIArgs,
                                              explainer: ExplainerCLIArgs):
    
    hypergraph = STHypergraphData.load(project.project_dir)

    assert model.embedding_dim == classifier.embedding_dim
    
    driver = Explainer(project.project_dir)
    driver.compute_g2ctd_attribution(cell_type,
                                     hypergraph,
                                     HypergraphEmbedderArgs(**vars(model)),
                                     classifier.num_layers,
                                     ExplainerArgs(**vars(explainer)))

def folder_test(project: ProjectArgs):
    if project.project_dir.exists():
        print(f"Folder {project.project_dir} exists!")
        for child in project.project_dir.iterdir(): 
              print(f"\t{child}")
    else:
        print(f"Folder {project.project_dir} does not exist!")