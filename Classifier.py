import logging
import scanpy as sc
import torch
from cyclopts import App
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from gctsvg.pipeline.classifier import ClassifierDataModule, STClassification

logger = logging.getLogger(__name__)

@dataclass
class ModelArgs:
    n_in: int
    n_out: int

@dataclass
class TrainerArgs:
    accelerator: str
    strategy: str
    num_nodes: int
    devices: int | str
    num_epochs: int
    batch_size: int

app = App()

@app.default
def main(project: str, 
         model: ModelArgs,
         trainer: TrainerArgs,
         version: str | int = "latest"):

    accelerator = trainer.accelerator
    strategy = trainer.strategy
    devices = trainer.devices
    num_nodes = trainer.num_nodes
    num_epochs = trainer.num_epochs
    batch_size = trainer.batch_size
    
    classifier_kwargs = {
        "n_in": model.n_in,
        "n_out": model.n_out
    }

    # load clustering and embedding data
    project_dir = Path("gctsvg_output") / project
    clustering_dir = project_dir / "clustering"
    clustering_filename = clustering_dir / "adata_clusters.h5ad"
    adata_clustering = sc.read(clustering_filename)

    logger.info("Initializing model...")
    stc = STClassification(project = project,
                           num_epochs = num_epochs,
                           accelerator = accelerator,
                          devices = devices,
                          strategy = strategy,
                          num_nodes = num_nodes)

    logger.info("Initializing data...")
    
    # TODO set num_workers = num_devices
    
    dm = ClassifierDataModule(adata_clustering, project, batch_size=batch_size, num_workers=4)
    logger.debug(dm)
    
    torch.set_float32_matmul_precision("high")
    
    stc.train(classifier_kwargs, dm)

    # stc.compute_attributions(dm, version)
    # scorer.compute_attributions(dm, version="latest")
    logger.info("Classifier training complete.")

if __name__ == "__main__":
    app()
