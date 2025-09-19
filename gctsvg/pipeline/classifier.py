import copy
import logging
import torch

import torch.nn as nn
import torch.optim as optim

import lightning as L

from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import CaptumExplainer, GNNExplainer
from torch_geometric.loader import NeighborLoader, GraphSAINTNodeSampler
from torch_geometric.nn.models import GCN, GAT

from lightning.pytorch import loggers
from lightning.pytorch.callbacks import BasePredictionWriter, ModelCheckpoint

from .utils import get_experiment_run, TrainerStatusLogCallback

logger = logging.getLogger(__name__)

# ST Classifier
#  train: learning
#  predict: attributions

class ClassifierDataModule(L.LightningDataModule):
    def __init__(self, adata, project, batch_size = 425,
                 num_workers = 4):
        super(ClassifierDataModule, self).__init__()
        self.save_dir = Path(f"gctsvg_output") / project
        self.adata = adata
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.graph_train_data = None
        self.graph_pred_data = None
    
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if not self.graph_train_data and not self.graph_pred_data:
            graph_file = self.save_dir / "graph.pt"
            if graph_file.exists():
                with torch.serialization.safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
                    graph_data = torch.load(graph_file)
            else:
                raise FileNotFoundError("Graph file graph.pt does not exist!")
            self.graph_train_data = self.__create_classifier_data(graph_data)

    def train_dataloader(self):
        loader = NeighborLoader(self.graph_train_data,
                            num_neighbors = [-1, -1],
                            batch_size = self.batch_size, # memory considerations
                            replace = False,
                            shuffle = False,
                            num_workers = self.num_workers)
        return loader
    
    def __create_classifier_data(self, graph_data, embedding_key = 'embedding', clustering_key = 'leiden'):
        # get embedding from adata_clusters
        x = torch.from_numpy(self.adata.X.toarray())
        embedding = torch.from_numpy(self.adata.obsm[embedding_key])
        
        # get edge_index from existing graph data
        edge_index = graph_data.edge_index
    
        clustering = self.adata.obs[clustering_key].apply(int).to_numpy()
        clustering = torch.from_numpy(clustering)

        classifier_data = Data(x = embedding, edge_index = edge_index, y = clustering, num_nodes=self.adata.n_obs)


        return classifier_data
        
class AttributionWriter(BasePredictionWriter):
    def __init__(self):
        super(AttributionWriter, self).__init__(write_interval="epoch")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        output_dir = Path(trainer.log_dir) / "attribution"
        
        if pl_module.global_rank == 0 and not self.output_dir.exists():
            output_dir.mkdir()

        # print(predictions)
        node_ids = torch.cat([prediction[0] for prediction in predictions])
        attributions = torch.cat([prediction[1] for prediction in predictions])
        delta = torch.cat([prediction[2] for prediction in predictions])
        
        torch.save(node_ids, f"{self.output_dir}/{trainer.global_rank}_node_ids.pt")
        torch.save(attributions, f"{self.output_dir}/{trainer.global_rank}_attributions.pt")
        torch.save(reconstruction, f"{self.output_dir}/{trainer.global_rank}_delta.pt")
        
        trainer.strategy.barrier("write predictions")
        logger.info(f"Finished writing embedding from process {trainer.global_rank}")        
        if trainer.global_rank == 0:
            logger.info("Merging attributions...")
            logger.debug(f"Number of processes: {trainer.world_size}")

            # load the results saved by each process
            node_id_list = [torch.load(f"{self.output_dir}/{i}_node_ids.pt") for i in range(trainer.world_size)]
            attributions_list = [torch.load(f"{self.output_dir}/{i}_attributions.pt") for i in range(trainer.world_size)]
            delta_list = [torch.load(f"{self.output_dir}/{i}_delta.pt") for i in range(trainer.world_size)]

            # combine the results 
            id_all = torch.cat(node_id_list)
            attributions_all = torch.cat(attributions_list)
            delta_all = torch.cat(delta_list)

            # sort the results by node id
            orig_order = torch.argsort(id_all)
            id_all = id_all[orig_order]
            attributions_all = attributions_all[orig_order]
            delta_all = delta_all[orig_order]

            # create new adata object with embedding
            adata_attribution = trainer.datamodule.adata.copy()
            adata_attribution.layers["attributions"] = embedding_all.numpy()
            adata_attribution.layers["delta"] = reconstruction_all.numpy()
            
            # save the new adata with embedding
            sc.write(output_dir / "adata_attributions.h5ad", adata_embedding)
            logger.debug(adata_attribution)

class SpatialClassifier(L.LightningModule):
    def __init__(self,
                 n_in,
                 n_out,
                 Activation = "leakyrelu",
                 lr = 1e-4):
        super(SpatialClassifier, self).__init__()
        self.save_hyperparameters()
        n_hidden = (n_in + n_out) // 2
        self.classifier = GCN(n_in, n_hidden, 3, n_out, dropout=0.2, act=Activation, jk="cat")
    
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.lr = lr
    
    def forward(self, x, edge_index):
        z = self.classifier(x, edge_index)
        return z

    def training_step(self, batch, batch_idx):
        X = batch.x
        y = batch.y
        edge_index = batch.edge_index
        node_ids = batch.n_id
        
        probs = self(X, edge_index)        
        total_loss = self.criterion(probs, y)

        self.log('total_loss', total_loss, 
                 on_step=False,
                 on_epoch=True,
                 batch_size=batch.batch_size,
                 sync_dist=True, 
                 prog_bar=True)
        
        return total_loss
        
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = self.lr, weight_decay=1e-5)         
        
class STClassification(object):
    def __init__(self, 
                 project,
                 num_epochs,
                 attribution="ig",
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
            filename='{epoch}-{total_loss:.4f}',
            monitor='total_loss',
            save_top_k=10,
            mode='min',
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_weights_only=True,
            auto_insert_metric_name=True)
        self.attribution_writer = AttributionWriter()
    
    def configure_trainer(self):
        self.trainer = L.Trainer(accelerator=self.accelerator,
                 max_epochs = self.num_epochs, 
                 devices = self.devices,
                 strategy = self.strategy,
                 log_every_n_steps=1,
                 num_nodes=self.num_nodes,
                 default_root_dir=self.output_dir,
                 logger=loggers.CSVLogger(f'{self.output_dir}', name='st_classification'),
                 callbacks=[TrainerStatusLogCallback(logger),
                            self.checkpoint_callback,
                            self.attribution_writer])
    
    def train(self, classifier_kwargs, datamodule):
        datamodule.setup("fit")
        
        classifier = SpatialClassifier(**classifier_kwargs)
        self.trainer.fit(classifier, 
                         datamodule = datamodule)

        self.trainer.strategy.barrier("training end")        
        logger.info(f"Training finished. Best model score: {self.checkpoint_callback.best_model_score}. Path: {self.checkpoint_callback.best_model_path}")

        for cb in self.trainer.callbacks:
            if hasattr(cb, "to_yaml"):
                logger.info("Saving best models...")
                cb.to_yaml()
                best_model_path = Path(cb.best_model_path)
                best_link_path = best_model_path.parent / "best.ckpt"
                best_link_path.symlink_to(best_model_path)
                # path_list[0] = cb.best_model_path
                break
    
    