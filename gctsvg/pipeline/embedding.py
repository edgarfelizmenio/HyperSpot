import os
import scanpy as sc
import torch
import torch.optim as optim

import lightning as L

from lightning.pytorch import loggers
from lightning.pytorch.callbacks import Callback, BasePredictionWriter, ModelCheckpoint

from ..preprocessing import st
from ..model.ae import GCNDGIAE, GCNDGIAELoss, graph_embedding, shuffle_attributes


class GCNDGIAEEmbedder(L.LightningModule):
    def __init__(self,
                 n_in,
                 n_out,
                 Activation = "leakyrelu",
                 summary = graph_embedding,
                 corruption = shuffle_attributes,
                 alpha=1,
                 beta=0.1,
                 lr = 1e-4):
        super(GCNDGIAEEmbedder, self).__init__()
        self.save_hyperparameters()
        self.model = GCNDGIAE(n_in, 
                              n_out,
                              Activation,
                              summary,
                              corruption,
                              alpha,
                              beta)
        self.criterion = GCNDGIAELoss(alpha, beta)
        self.lr = lr

    def forward(self, x, edge_index):
        code, reconstruction, dgi_loss = self.model(x, edge_index)
        return code, reconstruction, dgi_loss
    
    def training_step(self, batch, batch_idx):
        X = batch.x
        edge_index = batch.edge_index

        _, reconstruction, dgi_loss = self(X, edge_index)
        # code, reconstruction = self(X, edge_index)
        
        # only compute the loss from the seed nodes of batch
        # TODO: only compute DGI loss from seed nodes of batch
        # code = code[:batch.batch_size]
        # reconstruction = reconstruction[:batch.batch_size]
        # X = X[:batch.batch_size]

        # total_loss = self.criterion(reconstruction, X)
        total_loss, reconstruction_loss, contrastive_loss = self.criterion(reconstruction, X, dgi_loss)

        # sync_dist synchronizes the loss across all processes, 
        # which is important when saving and loading the best model
        self.log('total loss', total_loss, sync_dist=True, prog_bar=True)
        self.log('reconstruction loss', reconstruction_loss, sync_dist=True, prog_bar=True)
        self.log('contrastive loss', contrastive_loss, sync_dist=True, prog_bar=True)
 
        return total_loss
    
    def predict_step(self, batch, batch_idx):
        X =  batch.x
        edge_index = batch.edge_index
        code, reconstruction, _ = self(X, edge_index)
        node_ids = batch.n_id
        
        # only return predictions from seed nodes
        code = code[:batch.batch_size]
        reconstruction = reconstruction[:batch.batch_size]
        
        # also return node ids of seed for aggregation purposes
        node_ids = node_ids[:batch.batch_size]
        
        return node_ids, code, reconstruction
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print(f"Process {trainer.global_rank}: Training...")

    def on_train_end(self, trainer, pl_module):
        print(f"Process {trainer.global_rank}: Done training.")

    def on_predict_start(self, trainer, pl_module):
        print(f"Process {trainer.global_rank}: Predicting...")

    def on_predict_end(self, trainer, pl_module):
        print(f"Process {trainer.global_rank}: Done predicting.")

class BestCheckpointSync(Callback):
    """Broadcasts the best_model_path from rank 0 to all ranks at the end of training."""    

    def __init__(self):
        self.best_checkpoint_path = None

    def on_train_end(self, trainer, pl_module):
        path_list = [None]
        if pl_module.global_rank == 0:
            # Find the best ModelCheckpoint callback
            for cb in trainer.callbacks:
                if hasattr(cb, "best_model_path") and cb.best_model_path:
                    path_list[0] = cb.best_model_path
                    break

        # Broadcast to all ranks
        if trainer.strategy.launcher is not None or trainer.strategy.is_distributed:
            torch.distributed.broadcast_object_list(path_list, src=0)

        self.best_checkpoint_path = path_list[0]
        if pl_module.global_rank == 0:
            print(f"[Rank 0] Best checkpoint path synced: {self.best_checkpoint_path}")
    
class EmbeddingWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super(EmbeddingWriter, self).__init__(write_interval)
        # self.output_dir = f"{output_dir}/embedding"

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        
        self.output_dir = f"{trainer.log_dir}/embedding"
        
        if pl_module.global_rank == 0 and not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

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

        self.configure_callbacks()
        self.configure_trainer()

    def configure_callbacks(self):
        self.checkpoint_callback = ModelCheckpoint(
            # dirpath=f"{self.output_dir}/checkpoints",
            filename='{epoch}-{total loss:.4f}',
            monitor='total loss',
            save_top_k=10,
            mode='min',
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_weights_only=True,
            auto_insert_metric_name=True)
        # self.checkpoint_sync_callback = BestCheckpointSync()
    
    def configure_trainer(self):
        prediction_writer = EmbeddingWriter(output_dir=self.output_dir, write_interval="epoch")
        self.trainer = L.Trainer(accelerator=self.accelerator,
                 max_epochs = self.num_epochs, 
                 devices = self.devices,
                 strategy = self.strategy,
                 log_every_n_steps=1,
                 num_nodes=self.num_nodes,
                 default_root_dir=self.output_dir,
                 logger=loggers.CSVLogger(f'{self.output_dir}', name='st_embedder'),
                 callbacks=[prediction_writer, 
                            PrintCallback(), 
                            self.checkpoint_callback,
                            # self.checkpoint_sync_callback,
                           ])
    
    def train(self, ae_kwargs, datamodule):
        datamodule.setup("fit")

        ae = GCNDGIAEEmbedder(**ae_kwargs)
        
        self.trainer.fit(model = ae, 
                     datamodule = datamodule)

        self.trainer.strategy.barrier("training end")        
        print(f"Training finished. Best model score: {self.checkpoint_callback.best_model_score}. Path: {self.checkpoint_callback.best_model_path}")
        print("Saving best models...")
        # self.trainer.checkpoint_callback.to_yaml()

        for cb in self.trainer.callbacks:
            if hasattr(cb, "to_yaml"):
                cb.to_yaml()
                # path_list[0] = cb.best_model_path
                break
        
        # best_ae = GCNDGIAE.load_from_checkpoint(checkpoint_callback.best_model_path)
        # torch.save(best_ae.state_dict(), f"{self.output_dir}/gatae_model.ckpt")
        # trainer.save_checkpoint(f"{self.output_dir}/gatae_model.ckpt")

    def generate_embedding(self, datamodule):
        datamodule.setup("predict")

        # ae = GCNDGIAE.load_from_checkpoint(f"{self.output_dir}/gatae_model.ckpt")

        print("Predicting using best model...")
        # best_ckpt_path = self.checkpoint_sync_callback.best_checkpoint_path
        best_ckpt_path = self.checkpoint_callback.best_model_path
        self.trainer.predict(ckpt_path=best_ckpt_path, datamodule = datamodule, return_predictions=False)

        self.trainer.strategy.barrier("prediction end")