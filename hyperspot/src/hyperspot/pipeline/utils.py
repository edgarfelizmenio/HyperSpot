import os
import torch
import pickle
import scanpy as sc

from pathlib import Path

from ..config import AcceleratorOptions

def get_device(accelerator: AcceleratorOptions):
    if accelerator == AcceleratorOptions.CUDA:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU is available. Use --accelerator cpu, or check the "
                               "GPU driver and CUDA_VISIBLE_DEVICES.")
        n_devices = torch.cuda.device_count()
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible_devices:
            device_ids = cuda_visible_devices.split(",")
        else:
            device_ids = list(range(n_devices))
        # for i, device_id in enumerate(device_ids):
            # print(f"device mapping: cuda:{i} -> {device_id}")
        
        if len(device_ids) > 1:
            raise NotImplementedError("Multi-GPU support not implemented. Set CUDA_VISIBLE_DEVICES before running.")
        else:
            device = 'cuda:0'
    else:
        device = 'cpu'

    return device

def load_celltype_data(project: Path, 
                       cell_type: str):
    # load adata for cell type
    adata_ct = sc.read_h5ad(project / "clustering" / f"ST_cell_{cell_type}.h5ad")

    return adata_ct