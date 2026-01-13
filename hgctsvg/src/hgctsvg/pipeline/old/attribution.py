import copy
import logging
import scanpy as sc
import torch
import torch.nn as nn
import lightning as L

from pathlib import Path

from captum.attr import IntegratedGradients

from lightning.pytorch import loggers
from lightning.pytorch.callbacks import BasePredictionWriter

from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import CaptumExplainer, GNNExplainer
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GAT

from tqdm.auto import tqdm

from .embedding import GCNDGIAEEmbedder
from .classifier import SpatialClassifier
from .utils import get_experiment_run

logger = logging.getLogger(__name__)

class StackedClassifierDataModule(L.LightningDataModule):
    def __init__(self, adata, project, batch_size = 425,
                 num_workers = 1):
        super(StackedClassifierDataModule, self).__init__()
        self.save_dir = Path(f"gctsvg_output") / project
        self.adata = adata
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.graph_predict_data = None
    
    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if not self.graph_predict_data:
            graph_file = self.save_dir / "graph.pt"
            if graph_file.exists():
                with torch.serialization.safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
                    graph_data = torch.load(graph_file)
            else:
                raise FileNotFoundError("Graph file graph.pt does not exist!")
            self.graph_predict_data = self.__create_stacked_classifier_data(graph_data)

    def predict_dataloader(self):
        dataloader = NeighborLoader(self.graph_predict_data,
                            num_neighbors = [-1, -1, -1, -1],
                            batch_size = self.batch_size, # memory considerations
                            replace = False,
                            shuffle = False,
                            num_workers = self.num_workers)
        return dataloader
    
    def __create_stacked_classifier_data(self, graph_data, clustering_key = 'leiden'):        
        x = torch.from_numpy(self.adata.X.toarray())
        edge_index = graph_data.edge_index
        
        clustering = self.adata.obs[clustering_key].apply(int).to_numpy()
        clustering = torch.from_numpy(clustering)

        stacked_classifier_data = Data(x = x, edge_index = edge_index, y = clustering, num_nodes=self.adata.n_obs)
        return stacked_classifier_data

class StackedClassifier(L.LightningModule):
    def __init__(self,
                 encoder: GAT,
                 classifier: SpatialClassifier):
        super(StackedClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

        self.explainer_algorithm = GNNExplainer(50)
        self.explainer = Explainer(
            model=self,
            algorithm=self.explainer_algorithm,
            explanation_type='phenomenon',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw',
            ),
            node_mask_type='attributes',
            edge_mask_type=None
        )

        # self.encoder = copy.copy(encoder)
        # self.classifier = copy.copy(classifier)

        # for param in self.encoder.parameters():
        #     param.requires_grad = True
        # for param in self.classifier.parameters():
        #     param.requires_grad = True
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        z = self.classifier(z, edge_index)
        return z

    def predict_step(self, batch, batch_idx):
        with torch.enable_grad():
            X = batch.x
            edge_index = batch.edge_index
            # y = batch.y
            node_ids = batch.n_id[:batch.batch_size]
            # mask_type="node"
    
            # print(X.device)
            # print(node_id, X.shape, y.shape, batch.n_id[0])
            # print(torch.arange(batch.batch_size))
            explanation = self.explainer(X, edge_index, index=torch.arange(batch.batch_size))
            # todo only store node mask and index
            # explanations.append(explanation)
    
            # nonzero_x_mask = torch.nonzero(X[int(node_id)])
            # print(nonzero_x_mask, nonzero_x_mask.shape)
            # print(X[0, nonzero_x_mask])
            # print(batch.batch_size, X.shape)
            # print(explanation.target)
            # break
            node_ids = node_ids[:batch.batch_size]
            node_mask = explanation.node_mask[:batch.batch_size]
            
            return node_ids, node_mask
        
class AttributionWriter(BasePredictionWriter):
    def __init__(self):
        super(AttributionWriter, self).__init__("epoch")

    def write_on_batch_end(trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        output_dir = Path(trainer.log_dir) / "attribution"
        
        if pl_module.global_rank == 0 and not output_dir.exists():
            output_dir.mkdir()
            
        node_ids = torch.cat([prediction[0] for prediction in predictions])
        node_mask = torch.cat([prediction[1] for prediction in predictions])

        torch.save(node_ids,  output_dir / f"{dataloader_idx}_{batch_idx}_node_ids.pt")
        torch.save(node_mask, output_dir / f"{dataloader_idx}_{batch_idx}_node_mask.pt")

        # trainer.strategy.barrier("write explanations")
        # logger.info(f"Finished writing attribution batch {batch_idx} from dataloader {dataloader_idx}")        
    
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        output_dir = Path(trainer.log_dir) / "attribution"
        
        if pl_module.global_rank == 0 and not output_dir.exists():
            output_dir.mkdir()
            
        node_ids = torch.cat([prediction[0] for prediction in predictions])
        node_mask = torch.cat([prediction[1] for prediction in predictions])

        torch.save(node_ids, f"{output_dir}/{trainer.global_rank}_node_ids.pt")
        torch.save(node_mask, f"{output_dir}/{trainer.global_rank}_node_mask.pt")

        trainer.strategy.barrier("write explanations")
        logger.info(f"Finished writing attribution from process {trainer.global_rank}")        
        if trainer.global_rank == 0:
            logger.info("Merging embedding...")
            logger.info(f"Number of processes: {trainer.world_size}")

            # load the results saved by each process
            node_id_list = [torch.load(output_dir / f"{i}_node_ids.pt") for i in range(trainer.world_size)]
            node_mask_list = [torch.load(output_dir / f"{i}_node_mask.pt") for i in range(trainer.world_size)]

            # combine the results 
            id_all = torch.cat(node_id_list)
            node_mask_all = torch.cat(node_mask_list)
            
            # sort the results by node id
            orig_order = torch.argsort(id_all)
            id_all = id_all[orig_order]
            node_mask_all = node_mask_all[orig_order]

            # create new adata object with embedding
            adata_attribution = trainer.datamodule.adata.copy()
            adata_attribution.layers["attribution"] = node_mask_all.numpy()
            
            # save the new adata with embedding
            sc.write(output_dir / "adata_attribution.h5ad", adata_attribution)
            logger.info(adata_attribution)
        
class STAttribution(object):
    def __init__(self, 
                 project, 
                 attribution="ig",
                 accelerator="auto",
                 devices="1,",
                 strategy = "ddp",
                 num_nodes=1):
        self.output_dir = Path("gctsvg_output") / project
        self.attribution = attribution
        self.accelerator = accelerator
        self.devices = "1,"
        self.strategy = strategy
        self.num_nodes = 1

        self.configure_callbacks()
        self.configure_trainer()
        
    # def __create_stacked_classifier_data(self, clustering_key = 'leiden'):
    #     graph_file = self.output_dir / "graph.pt"
    #     if graph_file.exists():
    #         with torch.serialization.safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
    #             graph_data = torch.load(graph_file)
    #     else:
    #         raise FileNotFoundError("Graph file graph.pt does not exist!")
        
    #     x = torch.from_numpy(self.adata.X.toarray())
    #     edge_index = graph_data.edge_index
        
    #     clustering = self.adata.obs[clustering_key].apply(int).to_numpy()
    #     clustering = torch.from_numpy(clustering)

    #     stacked_classifier_data = Data(x = x, edge_index = edge_index, y = clustering, num_nodes=self.adata.n_obs)
    #     return stacked_classifier_data

    def configure_callbacks(self):
        self.attribution_writer = AttributionWriter()
        
    def configure_trainer(self):
        self.trainer = L.Trainer(
            accelerator = self.accelerator,
            devices = self.devices,
            strategy = self.strategy,
            log_every_n_steps=1,
            num_nodes=self.num_nodes,
            default_root_dir=self.output_dir,
            logger=loggers.CSVLogger(f'{self.output_dir}', name='st_attribution'),
            inference_mode=False,
            callbacks=[self.attribution_writer])
    
    def compute_attributions(self, datamodule, encoder_version="latest", classifier_version="latest"):
        # load the best encoder
        logger.info("Loading the best encoder...")
        embedder_run = get_experiment_run(self.output_dir, "st_embedder", version=encoder_version)
        embedder_model_path = embedder_run / "checkpoints" / "best.ckpt"        
        logger.info(embedder_model_path)
        embedder = GCNDGIAEEmbedder.load_from_checkpoint(embedder_model_path.readlink()) 
        encoder = embedder.cpu().model.encoder

        logger.info("Loading the best classifier...")
        classifier_run = get_experiment_run(self.output_dir, "st_classification", version=classifier_version)
        classifier_model_path = classifier_run / "checkpoints" / "best.ckpt"
        spatial_classifier = SpatialClassifier.load_from_checkpoint(classifier_model_path.readlink())
        classifier = spatial_classifier.cpu().classifier

        logger.info("Combining encoder and classifier...")
        stacked_classifier = StackedClassifier(encoder, classifier)
        logger.info(stacked_classifier)
        torch.compile(stacked_classifier)
        logger.info("Computing attributions using best combined model...")
        print(stacked_classifier)
        print(self.trainer.strategy.root_device)
        self.trainer.predict(stacked_classifier, 
                             datamodule = datamodule,
                             return_predictions=False)        
    
    # def compute_attributions(self, encoder_version, classifier_version):
    #     # load the best encoder
    #     logger.info("Loading the best encoder...")
    #     embedder_run = get_experiment_run(self.output_dir, "st_embedder", version=encoder_version)
    #     embedder_model_path = embedder_run / "checkpoints" / "best.ckpt"        
    #     logger.info(embedder_model_path)
    #     embedder = GCNDGIAEEmbedder.load_from_checkpoint(embedder_model_path.readlink()) 
    #     encoder = embedder.cpu().model.encoder

    #     logger.info("Loading the best classifier...")
    #     classifier_run = get_experiment_run(self.output_dir, "st_classification", version=classifier_version)
    #     classifier_model_path = classifier_run / "checkpoints" / "best.ckpt"
    #     spatial_classifier = SpatialClassifier.load_from_checkpoint(classifier_model_path.readlink())
    #     classifier = spatial_classifier.cpu().classifier

    #     logger.info("Combining encoder and classifier...")
    #     stacked_classifier = StackedClassifier(encoder, classifier).to("cuda:1")
    #     logger.info(stacked_classifier)
        
    #     logger.info("Computing attributions using best combined model...")
    #     # self.trainer.predict(stacked_classifier, datamodule = datamodule, return_predictions=False)

    #     explainer = Explainer(
    #         model=stacked_classifier,
    #         algorithm=GNNExplainer(50).to("cuda:1"),
    #         explanation_type='model',
    #         model_config=dict(
    #             mode='multiclass_classification',
    #             task_level='node',
    #             return_type='raw',
    #         ),
    #         node_mask_type='attributes',
    #         edge_mask_type=None
    #     )

    #     data = self.__create_stacked_classifier_data()
    #     data.pin_memory()
    #     data.to("cuda:1")

    #     dataloader = NeighborLoader(data,
    #                         num_neighbors = [-1, -1, -1, -1],
    #                         batch_size = 200, # memory considerations
    #                         replace = False,
    #                         shuffle = False,
    #                         num_workers = self.num_workers)
        
    #     print(embedder)
    #     print(spatial_classifier)
    #     print(embedder.device)
    #     print(spatial_classifier.device)
    #     print(stacked_classifier.device)
    #     explanations = []
    #     with tqdm(dataloader) as pbar:
    #         for batch in pbar:
    #             X = batch.x
    #             edge_index = batch.edge_index
    #             y = batch.y
    #             # node_id = batch.n_id[:batch.batch_size]
    #             mask_type="node"
    
    #             # print(X.device)
    #             # print(node_id, X.shape, y.shape, batch.n_id[0])
    #             print(torch.arange(batch.batch_size))
    #             explanation = explainer(X, edge_index, index=torch.arange(batch.batch_size))
    #             # todo only store node mask and index
    #             explanations.append(explanation)
    
    #             # nonzero_x_mask = torch.nonzero(X[int(node_id)])
    #             # print(nonzero_x_mask, nonzero_x_mask.shape)
    #             # print(X[0, nonzero_x_mask])
    #             print(batch.batch_size, X.shape)
    #             print(explanation.target)
    #             # break
        
    #     return explanations