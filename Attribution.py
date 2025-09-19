import logging
import scanpy as sc
import torch

from cyclopts import App
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from gctsvg.pipeline.attribution import STAttribution, StackedClassifierDataModule

@dataclass
class TrainerArgs:
    accelerator: str
    strategy: str
    num_nodes: int
    devices: int | str
    batch_size: int

@dataclass
class AttributionArgs:
    method: str

app = App()

@app.default
def main(project: str, 
         trainer: TrainerArgs,
         attribution: AttributionArgs,
         encoder_version: str | int = "latest",
         classifier_version: str | int = "latest"):

    batch_size = trainer.batch_size
    accelerator = trainer.accelerator
    devices = trainer.devices
    num_nodes = trainer.num_nodes
    strategy = trainer.strategy
    
    attribution_method = attribution.method

    project_dir = Path("gctsvg_output") / project
    clustering_dir = project_dir / "clustering"
    clustering_filename = clustering_dir / "adata_clusters.h5ad"
    adata_clustering = sc.read(clustering_filename)

    datamodule = StackedClassifierDataModule(adata_clustering, project, batch_size)
    sta = STAttribution(project, 
                        attribution_method, 
                        accelerator, 
                        devices, 
                        strategy, 
                        num_nodes)
    
    torch.set_float32_matmul_precision("high")
    
    sta.compute_attributions(datamodule, encoder_version, classifier_version)
    # explanations = sta.compute_attributions(encoder_version, classifier_version)

    # explanation = explanations[0]
    # print(explanation)
    # print(explanation.node_mask[torch.nonzero(explanation.node_mask, as_tuple=True)])
    # print()
    # print(explanation.x[torch.nonzero(explanation.x, as_tuple=True)])

    # print(explanation.node_mask[torch.nonzero(explanation.node_mask, as_tuple=True)].shape)
    # print()
    # print(explanation.x[torch.nonzero(explanation.x, as_tuple=True)].shape)

    # print(torch.nonzero(explanation.x, as_tuple=True))

    # nonzero_columns_mask = torch.sum(torch.abs(explanation.node_mask), dim=0) > 0
    # print(torch.nonzero(nonzero_columns_mask), torch.nonzero(nonzero_columns_mask).shape)
    # print(explanation.node_mask[:, nonzero_columns_mask].sum(dim=0), explanation.node_mask[:, nonzero_columns_mask].sum(dim=0).shape)
    # print(explanation.x[:, nonzero_columns_mask].sum(dim=0), explanation.x[:, nonzero_columns_mask].sum(dim=0).shape)

    # nonzero_explanation_mask = torch.nonzero(explanation.node_mask[0])
    # print(nonzero_explanation_mask, nonzero_explanation_mask.shape)
    # print(explanation.node_mask[0, nonzero_explanation_mask])
    # nonzero_x_mask = torch.nonzero(explanation.x[0])
    # print(nonzero_x_mask, nonzero_x_mask.shape)
    # print(explanation.x[0, nonzero_x_mask])

if __name__ == "__main__":
    app()