import scanpy as sc
import torch
from pathlib import Path

from lightning.fabric.accelerators import find_usable_cuda_devices

from gctsvg.preprocessing import common, st
from gctsvg.pipeline.embedding import STDataModule, STEmbedding

from tqdm.auto import tqdm

def main(project, decomp_dir, model_kwargs, trainer_kwargs):
    decomp_dir = Path(decomp_dir)
    adata_decomp = sc.read(decomp_dir / 'sp_adata.h5ad')
    
    common.preprocess(adata_decomp)
    
    accelerator = "gpu"
    
    strategy = "ddp"
    
    n_feats_st = adata_decomp.n_vars
    n_code_st = 200
    
    num_epochs=1000
    batch_size=512
    
    devices = 4
    num_nodes = 1
    
    ae_train_kwargs = {
        "n_in": n_feats_st,
        "n_out": n_code_st,
    }
    
    embedder = STEmbedding(num_epochs = num_epochs,
                          project = project,
                          accelerator = accelerator,
                          devices = devices,
                          strategy = strategy,
                          num_nodes = num_nodes)
    
    dm = STDataModule(adata_decomp, batch_size=batch_size, num_workers=10)
    dm.prepare_data()
    
    torch.set_float32_matmul_precision("high")
    embedder.train(ae_train_kwargs, dm)
    
    embedder.generate_embedding(dm)

if __name__ == "__main__":
    project = "Kidney Cancer Interface"
    decomp_dir = 'spatialscope_output/Human Kidney Cancer Interface'

    model_kwargs = {
        "n_in": 2047,
        "n_out": 200
    }
    
    trainer_kwargs = {
        "accelerator": "gpu",
        "strategy": "ddp",
        "devices": 1,
        "num_epochs": 1000,
        "batch_size": 512,
    }

    main(project, decomp_dir, model_kwargs, trainer_kwargs)
