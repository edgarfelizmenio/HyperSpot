
# Cap BLAS/OpenMP threads BEFORE numpy/BLAS loads. OpenBLAS sizes its thread
# pool at import based on the machine's core count; on >128-core hosts this
# overflows its precompiled 128-thread limit ("Bad memory unallocation"), which
# a runtime threadpool_limits can no longer undo. setdefault preserves any value
# the user already exported.
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import cyclopts
import numpy as np
import scanpy as sc
import torch

from cyclopts import App, Parameter
from dataclasses import dataclass, astuple
from typing import Optional, Annotated, List, Literal

from pathlib import Path
from tqdm.auto import tqdm

from . import config

from .data import STHypergraphData
from .dataset.hypergraph import STHypergraphListDataset

from .metadata import ProjectMetadata

from .pipeline import (
    HypergraphEmbedderDriver, 
    HypergraphEmbedderArgs, 
    HyperSpotClustering, 
    run_eval,
)

app = App(name="hyperspot")

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
    hyperedge_pool: Annotated[str, Parameter(help="Pooling for the opt-in hyperedge branch: 'spot' (own-spot cells only, sharper) or 'neighborhood' (spot + neighbors, smoother).")] = "spot"

@dataclass
class TrainerArgs:
    num_epochs: int = 100
    batch_size: int = 8192
    accelerator: config.AcceleratorOptions = config.AcceleratorOptions.CUDA
    seed: int = 42

@dataclass
class RegularizerArgs:
    """Opt-in, per-cell-type training regularizers (all default off)."""
    lambda_var: Annotated[float, Parameter(help="Weight for per-cell-type variance (anti-collapse) regularizer.")] = 0.0
    lambda_cov: Annotated[float, Parameter(help="Weight for per-cell-type covariance (decorrelation) regularizer.")] = 0.0
    lambda_nbr: Annotated[float, Parameter(help="Weight for neighborhood contrastive regularizer.")] = 0.0
    temp_nbr: Annotated[float, Parameter(help="Temperature for the neighborhood contrastive regularizer.")] = 0.5
    edge_preserving: Annotated[bool, Parameter(help="Expression-gated (bilateral) neighborhood pull: smooths within regions but not across expression boundaries. Set False for the isotropic-blur ablation.")] = True
    reg_weight: Annotated[float, Parameter(help="Weight for the model-side regularizer (e.g. v4 gate L1).")] = 0.0
    optimize_hyperedges: Annotated[bool, Parameter(help="Enable the hyperedge autoencoder branch (reconstruct each spot's own pseudobulk + regularize hyperedge embeddings). Requires a hypergraph built with n_hyperedge_feats>0.")] = False
    rho_h: Annotated[float, Parameter(help="Hyperedge reconstruction weight (only used when optimize_hyperedges).")] = 1.0
    lambda_var_h: Annotated[float, Parameter(help="Variance (anti-collapse) weight on hyperedge embeddings.")] = 0.0
    lambda_cov_h: Annotated[float, Parameter(help="Covariance (decorrelation) weight on hyperedge embeddings.")] = 0.0
    lambda_nbr_h: Annotated[float, Parameter(help="Neighborhood-contrastive weight on hyperedge embeddings (positives = hyperedges sharing cells).")] = 0.0

@dataclass
class ClusterArgs:
    # minimum number of samples in a cluster as a percentage of the total number of samples.
    embedding_key: str = "embedding"
    algorithm: Literal["leiden", "agglomerative", "gmm"] = "leiden"
    min_cluster_pct: float = 0.05   # fractional size floor: domain must be >= this fraction of cells
    min_cluster_size: int = 3       # absolute size floor: domain must have >= this many cells (< 3 is statistically insignificant)
    seed: int = 42
    n_pca_components: int = 50
    n_clusters: int = 20          # agglomerative: over-segment before min_cluster_pct merge
    n_neighbors: int = 15         # leiden: base kNN; auto-bumped by per-spot cell count
    select_pcs: bool = False      # gated-select: cluster only on multimodal PCs
    pc_var_floor: float = 0.04    # min fractional variance for a PC to be considered
    pc_dip_alpha: float = 0.05    # PC kept if Hartigan dip-test p < this (multimodal)
    select_min_cells: int = 500   # skip gating below this many cells (can't estimate modality)

@app.command(help="Create a project files directory.",
             sort_key=1)
def init_project_files_dir():
    if not config.project_files_dir.exists():
        config.project_files_dir.mkdir()

@app.command(help="Fit a hypergraph model for making embeddings.",
             sort_key=2)
def train_embedder(project: ProjectArgs,
         model: EmbedderCLIArgs,
         trainer: TrainerArgs,
         regularizer: RegularizerArgs = RegularizerArgs()):

    ProjectMetadata(project.project_dir)
    hypergraph = STHypergraphData.load()

    print("hyperedges: ", hypergraph["hyperedge_feats"].shape[0])
    print(vars(model))
    
    embedder = HypergraphEmbedderDriver()
    
    embedder.fit(hypergraph,
                 HypergraphEmbedderArgs(**vars(model)),
                 trainer.num_epochs,
                 trainer.batch_size,
                 trainer.accelerator,
                 seed=trainer.seed,
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

@app.command(help="Use the trained hypergraph model to generate the embeddings.",
             sort_key=3)
def make_embedding(project: ProjectArgs,
                   model: EmbedderCLIArgs,
                   trainer: TrainerArgs):

    ProjectMetadata(project.project_dir)
    
    hypergraph = STHypergraphData.load()
    embedder = HypergraphEmbedderDriver()

    embedder.generate_embedding(hypergraph,
                                HypergraphEmbedderArgs(**vars(model)),
                                trainer.batch_size,
                                trainer.accelerator)

@app.command(help="Using hyperedge embeddings, find the spatial domains present in the Spatial Transcriptomics tissue.",
             sort_key=4)
def compute_spatial_domains(project: ProjectArgs,
                            clustering: ClusterArgs = ClusterArgs(),
                            copy_var_cols: Annotated[Optional[List[str]], Parameter(help="var columns to copy from the decomposed AnnData into the aggregated AnnData.")] = None):

    ProjectMetadata(project.project_dir)

    clustering_driver = HyperSpotClustering()
    clustering_driver.create_aggr_st_data(var_cols=copy_var_cols)
    clustering_driver.load_embedding_to_aggr_data(clustering.embedding_key)
    clustering_driver.compute_spatial_clusters(**vars(clustering))

@app.command(help="Using hypergraph node embeddings, find the spatial domains present for each cell type in the Spatial Transcriptomics tissue.",
            sort_key=5)
def compute_celltype_spatial_domains(project: ProjectArgs,
                                     clustering: ClusterArgs = ClusterArgs()):

    ProjectMetadata(project.project_dir)
    
    clustering_driver = HyperSpotClustering()
    clustering_driver.load_embedding_to_ct_spatial_data(clustering.embedding_key)
    clustering_driver.compute_ct_spatial_clusters(**vars(clustering))

@app.command(help="Run evaluation test.", sort_key=6)
def perform_model_evaluation(
    project: ProjectArgs,         
    model: EmbedderCLIArgs,
    trainer: TrainerArgs,
    clustering: ClusterArgs = ClusterArgs(),
    num_runs: int = 10,
    ):

    ProjectMetadata(project.project_dir)

    run_eval(
        HypergraphEmbedderArgs(**vars(model)),
        trainer.num_epochs,
        trainer.batch_size,
        trainer.accelerator,
        embedding_key=clustering.embedding_key,
        algorithm=clustering.algorithm,
        min_cluster_pct=clustering.min_cluster_pct,
        n_pca_components=clustering.n_pca_components,
        n=num_runs,
    )

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