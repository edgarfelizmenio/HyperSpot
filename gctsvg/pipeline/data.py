import lightning as L

from torch_geometric.loader import NeighborLoader, GraphSAINTNodeSampler

from ..preprocessing import st

class STDataModule(L.LightningDataModule):
    def __init__(self, adata, project,
                 batch_size=512,
                 num_workers=1):
        super(STDataModule, self).__init__()
        self.adata = adata
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_dir = f"gctsvg_output/{project}"
        # self.prepare_data_per_node = True

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.graph_data = st.create_graph_data(self.adata)
    
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
                            batch_size = self.batch_size,
                            replace = False,
                            shuffle = False,
                            num_workers = self.num_workers)
        # loader = GraphSAINTNodeSampler(self.graph_data, batch_size=self.batch_size,
        #                                num_steps=5, sample_coverage=100,
        #                                save_dir=self.save_dir,
        #                                shuffle=False,
        #                                num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        loader = NeighborLoader(self.graph_data,
                            num_neighbors = [-1, -1],
                            batch_size = self.batch_size,
                            replace = False,
                            shuffle = False,
                            num_workers = self.num_workers)
        return loader