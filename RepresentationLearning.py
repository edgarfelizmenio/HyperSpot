import scanpy as sc
import torch
from cyclopts import App
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from gctsvg.preprocessing import common, st
from gctsvg.pipeline.data import STDataModule
from gctsvg.pipeline.embedding import STEmbedding

@dataclass
class ModelArgs:
    n_in: int
    n_out: int

@dataclass
class PreprocessArgs:
    target_sum: Optional[float] = None
    max_value: Optional[float] = None
    scale: bool = False
    
@dataclass
class DataArgs:
    filename: str
    preprocess: bool = True
    preprocess_args: Optional[PreprocessArgs] = field(default_factory=PreprocessArgs)

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
         data: DataArgs,
         model: ModelArgs, 
         trainer: TrainerArgs):
    
    data_filename = Path(data.filename)
    adata_decomp = sc.read(data_filename)

    if data.preprocess:
        common.preprocess(adata_decomp,
                         target_sum = data.preprocess_args.target_sum,
                         max_value = data.preprocess_args.max_value,
                         scale = data.preprocess_args.scale)
    
    accelerator = trainer.accelerator
    strategy = trainer.strategy
    
    num_epochs = trainer.num_epochs
    batch_size = trainer.batch_size
    
    devices = trainer.devices
    num_nodes = trainer.num_nodes
    
    ae_train_kwargs = {
        "n_in": model.n_in,
        "n_out": model.n_out,
    }
    
    embedder = STEmbedding(num_epochs = num_epochs,
                          project = project,
                          accelerator = accelerator,
                          devices = devices,
                          strategy = strategy,
                          num_nodes = num_nodes)

    # TODO set num_workers = num_devices
    dm = STDataModule(adata_decomp, project, batch_size=batch_size, num_workers=4)
    dm.prepare_data()
    
    torch.set_float32_matmul_precision("high")
    embedder.train(ae_train_kwargs, dm)
    print("Training finished.")
    
    embedder.generate_embedding(dm)
    print("Embedding generation finished.")

    # save graph data for future use
    torch.save(dm.graph_data, f"gctsvg_output/{project}/graph.pt")
    print("Saved graph data.")

if __name__ == "__main__":
    # project = "Human Kidney Cancer Interface"
    # decomp_dir = 'spatialscope_output/Human Kidney Cancer Interface'

    # model_kwargs = {
    #     "n_in": 2047,
    #     "n_out": 200
    # }
    
    # trainer_kwargs = {
    #     "accelerator": "gpu",
    #     "strategy": "ddp",
    #     "devices": 1,
    #     "num_epochs": 1000,
    #     "batch_size": 512,
    # }

    # main(project, decomp_dir, model_kwargs, trainer_kwargs)
    app()
