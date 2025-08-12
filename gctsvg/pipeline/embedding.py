import os
import scanpy as sc
import torch
import torch.optim as optim

import lightning as L

from lightning.pytorch import loggers
from lightning.pytorch.callbacks import Callback, BasePredictionWriter

from torch_geometric.loader import DataLoader, NeighborLoader

from ..preprocessing import st
from ..model.ae import GCNDGIAE

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print(f"Process {trainer.global_rank}: Training...")

    def on_train_end(self, trainer, pl_module):
        print(f"Process {trainer.global_rank}: Done training.")

    def on_predict_start(self, trainer, pl_module):
        print(f"Process {trainer.global_rank}: Predicting...")

    def on_predict_end(self, trainer, pl_module):
        print(f"Process {trainer.global_rank}: Done predicting.")

class EmbeddingWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super(EmbeddingWriter, self).__init__(write_interval)
        self.output_dir = f"{output_dir}/embedding"

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # print(predictions)
        node_ids = torch.cat([prediction[0] for prediction in predictions])
        embedding = torch.cat([prediction[1] for prediction in predictions])
        reconstruction = torch.cat([prediction[2] for prediction in predictions])

        torch.save(node_ids, f"{self.output_dir}/{trainer.global_rank}_node_ids.pt")
        torch.save(embedding, f"{self.output_dir}/{trainer.global_rank}_embedding.pt")
        torch.save(reconstruction, f"{self.output_dir}/{trainer.global_rank}_reconstruction.pt")
        
        trainer.strategy.barrier("write predictions")
        print(f"Finished writing embedding from process {trainer.global_rank}")        
        if trainer.global_rank == 0:
            print("Merging embedding...")
            print(f"Number of processes: {trainer.world_size}")

            # load the results saved by each process
            node_id_list = [torch.load(f"{self.output_dir}/{i}_node_ids.pt") for i in range(trainer.world_size)]
            embedding_list = [torch.load(f"{self.output_dir}/{i}_embedding.pt") for i in range(trainer.world_size)]
            reconstruction_list = [torch.load(f"{self.output_dir}/{i}_reconstruction.pt") for i in range(trainer.world_size)]

            # combine the results 
            id_all = torch.cat(node_id_list)
            embedding_all = torch.cat(embedding_list)
            reconstruction_all = torch.cat(reconstruction_list)

            # sort the results by node id
            orig_order = torch.argsort(id_all)
            id_all = id_all[orig_order]
            embedding_all = embedding_all[orig_order]
            reconstruction_all = reconstruction_all[orig_order]

            # create new adata object with embedding
            adata_embedding = trainer.datamodule.adata.copy()
            adata_embedding.obsm["embedding"] = embedding_all.numpy()
            adata_embedding.layers["reconstruction"] = reconstruction_all.numpy()
            
            # save the new adata with embedding
            sc.write(f"{self.output_dir}/adata_embedding.h5ad", adata_embedding)
            print(adata_embedding)

        # print(predictions_list)

class STDataModule(L.LightningDataModule):
    def __init__(self, adata,
                 batch_size=512,
                 num_workers=1):
        super(STDataModule, self).__init__()
        self.adata = adata
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.prepare_data_per_node = True

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.graph_data = st.create_graph_data(self.adata)

    def train_dataloader(self):
        loader = NeighborLoader(self.graph_data,
                            num_neighbors = [-1, -1],
                            batch_size = self.batch_size,
                            replace = False,
                            shuffle = True,
                            num_workers = self.num_workers)
        # return DataLoader([self.graph_data_train], num_workers = self.num_workers)
        return loader
    
    def predict_dataloader(self):
        loader = NeighborLoader(self.graph_data,
                            num_neighbors = [-1, -1],
                            batch_size = self.batch_size,
                            replace = False,
                            shuffle = False,
                            num_workers = self.num_workers)
        return loader

class STEmbedding(object):
    def __init__(self, 
                 num_epochs,
                 project,
                 accelerator="auto",
                 devices=1,
                 strategy = "ddp",
                 num_nodes=1):
        self.num_epochs = num_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.project = project
        self.output_dir = f"gctsvg_output/{project}"

    def train(self, ae_kwargs, datamodule):
        datamodule.setup("fit")

        ae = GCNDGIAE(**ae_kwargs)
        
        trainer = L.Trainer(accelerator=self.accelerator,
                     max_epochs = self.num_epochs, 
                     devices = self.devices,
                     strategy = self.strategy,
                     log_every_n_steps=1,
                     num_nodes=self.num_nodes,
                     default_root_dir=self.output_dir,
                     logger=loggers.CSVLogger(f'{self.output_dir}/logs', name='st_embedder'),
                     callbacks=[PrintCallback()])

        trainer.fit(model = ae, 
                     datamodule = datamodule)
        trainer.save_checkpoint(f"{self.output_dir}/gatae_model.ckpt")

    def generate_embedding(self, datamodule):
        datamodule.setup("predict")

        ae = GCNDGIAE.load_from_checkpoint(f"{self.output_dir}/gatae_model.ckpt")

        prediction_writer = EmbeddingWriter(output_dir=self.output_dir, write_interval="epoch")
        
        trainer = L.Trainer(accelerator=self.accelerator,
                     max_epochs = self.num_epochs, 
                     devices = self.devices,
                     strategy = self.strategy,
                     log_every_n_steps=1,
                     num_nodes=self.num_nodes,
                     default_root_dir=self.output_dir,
                     logger=loggers.CSVLogger(f'{self.output_dir}/logs', name='st_embedder'),
                     callbacks=[prediction_writer, PrintCallback()])

        trainer.predict(ae, datamodule = datamodule, return_predictions=False)