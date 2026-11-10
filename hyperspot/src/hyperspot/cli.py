
# Cap BLAS/OpenMP threads BEFORE numpy/BLAS loads. OpenBLAS sizes its thread
# pool at import based on the machine's core count; on >128-core hosts this
# overflows its precompiled 128-thread limit ("Bad memory unallocation"), which
# a runtime threadpool_limits can no longer undo. setdefault preserves any value
# the user already exported.
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

# Show HyperSpot's own log messages on the console. Without a handler Python only prints
# WARNING and above, so logger.info() calls were silently dropped. Configured here (module
# level) because the `hyperspot` entry point calls `app` directly and never runs __main__.
# Only the "hyperspot" logger is configured, so INFO chatter from scanpy/anndata stays hidden.
# Set HYPERSPOT_LOG_LEVEL=DEBUG for more detail (or WARNING for less).
import logging
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
_hs_logger = logging.getLogger("hyperspot")
_hs_logger.addHandler(_log_handler)
_hs_logger.setLevel(os.environ.get("HYPERSPOT_LOG_LEVEL", "INFO").upper())
_hs_logger.propagate = False
logger = logging.getLogger(__name__)

from cyclopts import App, Parameter
from dataclasses import dataclass
from typing import Optional, Annotated, List

from pathlib import Path

from . import config, utils

from .data import STHypergraphData

from .metadata import ProjectMetadata

from .pipeline import (
    HypergraphEmbedderDriver,
    HypergraphEmbedderArgs,
    HyperSpotClustering,
    run_eval,
    compute_domain_svgs,
    compute_domain_ct_svgs,
    create_svg_list,
)

app = App(name="hyperspot")

@Parameter(name="*")
@dataclass
class ProjectArgs:
    name: Annotated[str, Parameter(name="project", help="Name of the project.")]

    @property
    def project_dir(self):
        return config.project_files_dir / self.name

@Parameter(name="*")
@dataclass
class CommonArgs:
    seed: Annotated[int, Parameter(help="Seed for reproducibility")] = 42

@dataclass(kw_only=True)
class EmbedderCLIArgs:
    model_name: Annotated[str, Parameter(help=f"Name of hypergraph embedder to use. Possible values: {HypergraphEmbedderDriver.get_available_embedders()}")] = "CTHGCAEv6Embedder"
    embedding_dim: Annotated[int, Parameter(help="Size of embedding.")] = 20
    dropout: Annotated[float, Parameter(help="Dropout parameter.")] = 0.2

@Parameter(name="*")
@dataclass(kw_only=True)
class RuntimeArgs:                  # how a run executes on the hardware (not what it computes)
    batch_size: Annotated[int, Parameter(help="Cells per batch; lower it if the GPU runs out of memory.")] = 8192
    accelerator: config.AcceleratorOptions = config.AcceleratorOptions.CUDA

# kw_only: cyclopts would otherwise also accept the fields positionally, so a bare value
# after a boolean flag (e.g. "--ct-clustering.dedup-spots False") would land in embedding_key.
@dataclass(kw_only=True)
class ClusterArgs:                  # tissue domains (one row per spot)
    embedding_key: str = "embedding"
    min_cluster_size: int = 3       # absolute size floor: domain must have >= this many spots
    n_pca_components: int = 50      # PCA only if the embedding dim exceeds this

@dataclass(kw_only=True)
class CTClusterArgs:                # cell-type domains
    embedding_key: str = "embedding"
    min_cluster_size: int = 3       # absolute size floor: domain must have >= this many representatives
    n_pca_components: int = 50      # PCA only if the embedding dim exceeds this
    min_cluster_pct: float = -1     # min fraction of the cell type's CELLS per domain; OFF by default (dedup fixes the over-clustering it patched)
    dedup_spots: bool = True        # collapse same-spot same-CT duplicate cells before clustering (Spotiphy); guarded -> auto no-op on real per-cell variation. Set False for true-counts / per-cell data.

# Command order in --help (sort_key), in steps of 10 so new commands slot in:
#   10 init-project-files-dir   20 train-embedder   30 make-embedding
#   40 compute-spatial-domains  50 compute-celltype-spatial-domains
#   60 compute-svgs             70 compute-ct-svgs
#   80 (reserved: ct-SVG scoring)  90 (reserved: run-all)
#  100 perform-model-evaluation
# 1000 folder-test (always last)
@app.command(help="Create a project files directory.",
             sort_key=10)
def init_project_files_dir():
    if not config.project_files_dir.exists():
        config.project_files_dir.mkdir()

@app.command(help="Fit a hypergraph model for making embeddings.",
             sort_key=20)
def train_embedder(
    project: ProjectArgs,
    common: CommonArgs = CommonArgs(),
    num_epochs: Annotated[int, Parameter(help="Training epochs.")] = 1000,
    model: EmbedderCLIArgs = EmbedderCLIArgs(),
    runtime: RuntimeArgs = RuntimeArgs(),
):
    ProjectMetadata(project.project_dir)

    utils.set_seed(common.seed)
    hypergraph = STHypergraphData.load()

    logger.debug(f"hyperedges: {hypergraph['hyperedge_feats'].shape[0]}; model args: {vars(model)}")
    
    embedder = HypergraphEmbedderDriver()
    args = HypergraphEmbedderArgs(
        **vars(model),
        input_dim = hypergraph["cell_feats"].shape[1]
    )
    
    embedder.fit(
        hypergraph,
        args,
        num_epochs,
        runtime.batch_size,
        runtime.accelerator,
    )

@app.command(help="Use the trained hypergraph model to generate the embeddings.",
             sort_key=30)
def make_embedding(
    project: ProjectArgs,
    K: Annotated[float, Parameter(help="Fraction of neighborhood hyperedges kept per hop at inference (fan-out cap, avoids OOM). 1.0 = full neighborhood (deterministic, reproduces training-time embeddings).")] = 0.7,
    M: Annotated[float, Parameter(help="Fraction of neighborhood cells kept per hyperedge expansion at inference (fan-out cap, avoids OOM). 1.0 = full neighborhood.")] = 0.7,
    common: CommonArgs = CommonArgs(),
    model: EmbedderCLIArgs = EmbedderCLIArgs(),
    runtime: RuntimeArgs = RuntimeArgs(),
):

    ProjectMetadata(project.project_dir)
    utils.set_seed(common.seed)
    
    hypergraph = STHypergraphData.load()
    embedder = HypergraphEmbedderDriver()
    args = HypergraphEmbedderArgs(
        **vars(model),
        input_dim = hypergraph["cell_feats"].shape[1]
    )
    
    embedder.generate_embedding(hypergraph,
                                args,
                                runtime.batch_size,
                                runtime.accelerator,
                                K=K, M=M)

@app.command(help="Using hyperedge embeddings, find the spatial domains present in the Spatial Transcriptomics tissue.",
             sort_key=40)
def compute_spatial_domains(
    project: ProjectArgs,            
    clustering: ClusterArgs = ClusterArgs(),
    copy_var_cols: Annotated[
        Optional[List[str]], 
        Parameter(help="var columns to copy from the decomposed AnnData into the aggregated AnnData.")] = None,
    common: CommonArgs = CommonArgs(),
):

    ProjectMetadata(project.project_dir)
    utils.set_seed(common.seed)
    
    clustering_driver = HyperSpotClustering()
    clustering_driver.create_aggr_st_data(var_cols=copy_var_cols)
    clustering_driver.load_embedding_to_aggr_data(clustering.embedding_key)
    clustering_driver.compute_spatial_clusters(
        **vars(clustering),
        seed=common.seed,
    )

@app.command(help="Using hypergraph node embeddings, find the spatial domains present for each cell type in the Spatial Transcriptomics tissue.",
            sort_key=50)
def compute_celltype_spatial_domains(
    project: ProjectArgs,
    ct_clustering: CTClusterArgs = CTClusterArgs(),
    common: CommonArgs = CommonArgs(),
):
    ProjectMetadata(project.project_dir)
    utils.set_seed(common.seed)

    clustering_driver = HyperSpotClustering()
    clustering_driver.load_embedding_to_ct_spatial_data(ct_clustering.embedding_key)
    clustering_driver.compute_ct_spatial_clusters(
        **vars(ct_clustering),
        seed = common.seed,
    )

@app.command(help="Detect tissue-level spatially variable genes (SVGs) for the ST spatial domains -> svgs_ST.csv. Requires spatial domains first (compute-spatial-domains).",
             sort_key=60)
def compute_svgs(
    project: ProjectArgs,
    key: Annotated[str, Parameter(help="obs column holding the domain labels.")] = 'domain',
    min_pts: Annotated[float, Parameter(help="Keep genes expressed in >= this fraction of the domain.")] = 0.8,
    min_logfc: Annotated[float, Parameter(help="Keep genes with >= this log fold-change vs the rest.")] = 1.5,
    rank_by: Annotated[str, Parameter(help="Ranking statistic for the SVG list.")] = 'scores',
    dedup_spots: Annotated[bool, Parameter(help="Collapse same-spot duplicate cells before the test (Spotiphy).")] = True,
    output_dir: Annotated[Optional[str], Parameter(help="Directory for the SVG CSV (default <project>/svgs).")] = None,
):
    ProjectMetadata(project.project_dir)
    out = Path(output_dir) if output_dir else ProjectMetadata().project_folder / "svgs"
    out.mkdir(parents=True, exist_ok=True)

    svgs = compute_domain_svgs(key=key, min_pts=min_pts, min_logfc=min_logfc,
                               rank_by=rank_by, dedup_spots=dedup_spots)
    create_svg_list(svgs, rank_by=rank_by).to_csv(out / "svgs_ST.csv")
    logger.info(f"SVGs -> {out / 'svgs_ST.csv'}")

@app.command(
    help="Detect cell-type-specific SVGs (ctSVGs): domain-marker genes per cell type -> svgs_ct_<ct>.csv. Default: all cell types; --cell-types limits to a subset. Requires per-CT domains first (compute-celltype-spatial-domains).",         
    sort_key=70)
def compute_ct_svgs(
    project: ProjectArgs,
    cell_types: Annotated[Optional[List[str]], Parameter(help="Restrict ctSVG detection to these cell types (default: all).")] = None,
    key: Annotated[str, Parameter(help="obs column holding the domain labels.")] = 'domain',
    min_pts: Annotated[float, Parameter(help="Keep genes expressed in >= this fraction of the domain.")] = 0.8,
    min_logfc: Annotated[float, Parameter(help="Keep genes with >= this log fold-change vs the rest.")] = 1.5,
    rank_by: Annotated[str, Parameter(help="Ranking statistic for the SVG list.")] = 'scores',
    dedup_spots: Annotated[bool, Parameter(help="Collapse same-spot duplicate cells before the test (Spotiphy).")] = True,
    output_dir: Annotated[Optional[str], Parameter(help="Directory for the ctSVG CSVs (default <project>/svgs).")] = None,
):
    ProjectMetadata(project.project_dir)
    pm = ProjectMetadata()
    out = Path(output_dir) if output_dir else pm.project_folder / "svgs"
    out.mkdir(parents=True, exist_ok=True)

    types = cell_types if cell_types else pm.ct_list
    for ct in types:
        ct_svgs = compute_domain_ct_svgs(ct, key=key, min_pts=min_pts, min_logfc=min_logfc,
                                         rank_by=rank_by, dedup_spots=dedup_spots)
        create_svg_list(ct_svgs, rank_by=rank_by).to_csv(out / f"svgs_ct_{ct}.csv")
        logger.info(f"ctSVGs [{ct}] -> {out / ('svgs_ct_' + ct + '.csv')}")

@app.command(help="Run evaluation test.", sort_key=100)
def perform_model_evaluation(
    project: ProjectArgs,         
    num_epochs: Annotated[int, Parameter(help="Training epochs.")] = 1000,
    model: EmbedderCLIArgs = EmbedderCLIArgs(),
    runtime: RuntimeArgs = RuntimeArgs(),
    clustering: ClusterArgs = ClusterArgs(),
    ct_clustering: CTClusterArgs = CTClusterArgs(),
    num_runs: int = 10,
):

    ProjectMetadata(project.project_dir)

    hypergraph = STHypergraphData.load()
    
    args = HypergraphEmbedderArgs(
        **vars(model),
        input_dim = hypergraph["cell_feats"].shape[1]
    )
    
    run_eval(
        hypergraph,
        args,
        num_epochs,
        runtime.batch_size,
        runtime.accelerator,
        clustering=vars(clustering),
        ct_clustering=vars(ct_clustering),
        n=num_runs,
    )

@app.command(help="Check if project folder exists.",
             sort_key=1000)   # always last
def folder_test(project: ProjectArgs):
    if project.project_dir.exists():
        print(f"Folder {project.project_dir} exists!")
        for child in project.project_dir.iterdir(): 
              print(f"\t{child}")
    else:
        print(f"Folder {project.project_dir} does not exist!")

if __name__ == "__main__":
    app()