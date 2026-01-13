import torch

import lightning as L

from pathlib import Path

from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from torch_geometric.loader import NeighborLoader, GraphSAINTNodeSampler

from ..preprocessing import st

def make_baseline_data(n_obs, n_vars):
    return torch.zeros(1, 1, n_vars)

class STDataModule(L.LightningDataModule):
    def __init__(self, adata, project,
                 batch_size = 512,
                 num_workers = 1,
                 add_baseline = False):
        super(STDataModule, self).__init__()
        self.adata = adata
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_dir = Path(f"gctsvg_output") / project
        self.graph_data = None
        self.add_baseline = add_baseline

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if not self.graph_data:
            graph_file = self.save_dir / "graph.pt"
            if graph_file.exists():
                with torch.serialization.safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
                    self.graph_data = torch.load(graph_file)
            else:
                self.graph_data = st.create_graph_data(self.adata)
        if self.add_baseline:
            self.graph_data.baseline = make_baseline_data(self.adata.n_obs, self.adata.n_vars)
    
    def train_dataloader(self):
        # loader = NeighborLoader(self.graph_data,
        #                     num_neighbors = [-1, -1],
        #                     batch_size = self.batch_size,
        #                     replace = False,
        #                     shuffle = True,
        #                     num_workers = self.num_workers)
        # return DataLoader([self.graph_data_train], num_workers = self.num_workers)
        loader = GraphSAINTNodeSampler(self.graph_data, batch_size=self.batch_size,
                                       num_steps=1, sample_coverage=1000,
                                       save_dir=self.save_dir,
                                       shuffle=True,
                                       num_workers=self.num_workers)
        return loader
    
    def predict_dataloader(self):
        loader = NeighborLoader(self.graph_data,
                            num_neighbors = [-1, -1],
                            batch_size = 10, # memory considerations
                            replace = False,
                            shuffle = False,
                            num_workers = self.num_workers)
        # loader = GraphSAINTNodeSampler(self.graph_data, batch_size=self.batch_size,
        #                                num_steps=5, sample_coverage=100,
        #                                save_dir=self.save_dir,
        #                                shuffle=False,
        #                                num_workers=self.num_workers)
        return loader