import os
import torch
import pickle

from ..config import AcceleratorOptions

def get_device(accelerator: AcceleratorOptions):
    if accelerator == AcceleratorOptions.CUDA:
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

def create_celltype_mapping(project_folder, adata_decomp, key_type='cell_type'):
    ct_map_filename = project_folder / "ctmap.pkl"
    
    int2ct = sorted(list(adata_decomp.obs[key_type].unique().astype(str)))

    ct2int = {
        cell_type:i for i, cell_type in enumerate(int2ct) 
    }

    with open(ct_map_filename, "wb") as ct_map_file:
        pickle.dump(int2ct, ct_map_file)
        pickle.dump(ct2int, ct_map_file)

def get_celltype_mapping(project_folder, key_type='cell_type'):
    ct_map_filename = project_folder / "ctmap.pkl"

    if ct_map_filename.exists():
        with open(ct_map_filename, "rb") as ct_map_file:
            int2ct = pickle.load(ct_map_file)
            ct2int = pickle.load(ct_map_file)
    else:
        raise FileNotFoundError(f'File "{ct_map_filename}" does not exist!')

    return int2ct, ct2int