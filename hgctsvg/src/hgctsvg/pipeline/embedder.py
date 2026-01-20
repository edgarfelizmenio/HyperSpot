import copy
import gc
import os
import torch
import torch.optim as optim

import logging

from collections import namedtuple

from pathlib import Path
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

from .utils import get_device

from ..data import STHypergraphData
from ..dataset.hypergraph import STHypergraphMatrixDataset, STHypergraphBatchMatrixData

from ..utils import build_incidence_matrix

# from ..model.hypergraph.transformer import STHypergraphTransformer
# from ..model.hypergraph.hgnn import HyFiHGNN
from ..model.hypergraph.fusion import HyFiHGNN, SpatialFusionNetwork
from ..model.cl.hyfi import HyFi, HyFiLoss, IntraClassContrastiveLoss
# from ..model.cl.tricl import TriCL, TriCLLoss
# from ..model.classifier import Classifier

from ..config import AcceleratorOptions

logger = logging.getLogger(__name__)

# save node embeddings and hyperedge embeddings
class STHypergraphEmbedder(object):
    def __init__(self, 
                 project: Path,
                 input_dim,
                 embedding_dim,
                 dropout,
                 sigma=1):
        self.project = project
      
        logger.info("Initializing model...")
        # self.model = STHypergraphTransformer(
        #     input_dim, 
        #     embedding_dim, 
        #     num_spots,
        #     num_heads, 
        #     dropout=0.1
        # )
        self.model = HyFiHGNN(
            input_dim,
            embedding_dim,
            dropout=0.1
        )

        self.cl_model = HyFi(self.model, embedding_dim)
        # self.criterion = HyFiLoss(embedding_dim)
        self.criterion = IntraClassContrastiveLoss(embedding_dim)
        self.__configure_optimizer()
    
    def __setup_model(self, accelerator: AcceleratorOptions):
        self.device = get_device(accelerator)
        
        self.cl_model.to(self.device)
        self.criterion.to(self.device)
    
    def fit(self, 
            hypergraph: STHypergraphData,
            num_epochs: int,
            batch_size: int,
            accelerator: AcceleratorOptions):
        self.__setup_model(accelerator)
        self.cl_model.train()
        # use random sampler in training
        logger.info("Learning node and hyperedge embeddings...")

        hgdataset = STHypergraphMatrixDataset(hypergraph)
        
        logger.info(f"Number of cells: {len(hgdataset)}")
        logger.info(f"Batch Size: {batch_size}")
        
        hgtrainsampler = BatchSampler(
            # Use SequentialSampler for testing and validation
            RandomSampler(hgdataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )
        hgloader = DataLoader(hgdataset, sampler=hgtrainsampler, batch_size=None)

        with tqdm(range(num_epochs)) as e_pbar:
            max_epoch_loss = None
            min_epoch_loss = None
            for epoch in e_pbar:
                # create function for 1 epoch that returns the loss
                num_batches = len(hgloader)
                epoch_loss = 0.0
                with tqdm(hgloader, leave=False, position=1) as b_pbar:
                    for batch in b_pbar:
                        self.optimizer.zero_grad()
                        loss = self.__step_batch(batch)
                        loss.backward()
                        self.optimizer.step()
                        epoch_loss += loss.item()

                        num_nodes = batch.batch_subgraph_cells.shape[0]
                        num_hyperedges = batch.batch_subgraph_hyperedges.shape[0]
                        
                        b_pbar.set_postfix({'loss': loss.item(),
                                            'n_cells': batch.sample_n_cells,
                                            'n_hyperedges': batch.sample_n_hyperedges,
                                            'cl_n_cells': batch.neighborhood_n_cells,
                                            'cl_n_hyperedges': batch.neighborhood_n_hyperedges,
                                            'nodes': num_nodes,
                                            'hyperedges': num_hyperedges})
                        gc.collect()
                epoch_loss /= num_batches

                # TODO save best model
                if not max_epoch_loss or epoch_loss > max_epoch_loss:
                    max_epoch_loss = epoch_loss
                if not min_epoch_loss or epoch_loss < min_epoch_loss:
                    min_epoch_loss = epoch_loss
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': copy.deepcopy(self.cl_model.state_dict()),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': min_epoch_loss
                    }
                    torch.save(checkpoint, self.project / 'embedder.pth')

                e_pbar.set_postfix({
                    'epoch_loss': epoch_loss, 
                    'max_loss': max_epoch_loss,
                    'min_loss': min_epoch_loss})

                    
    def __configure_optimizer(self):
        self.optimizer = optim.Adam(list(self.cl_model.parameters()) + list(self.criterion.parameters()), lr=0.0001)
                
    def __step_batch(self, 
                     batch: namedtuple):

        if hasattr(batch, 'batch_incidence_matrix'):
            batch_incidence_matrix = batch.batch_incidence_matrix.to(self.device)
            if self.cl_model.training:
                neighborhood_incidence_matrix = batch.neighborhood_incidence_matrix.to(self.device)
        else:
            batch_incidence_matrix = build_incidence_matrix(batch.batch_incidence_matrix.to(self.device))
        

        if self.cl_model.training:
            node_embeddings, hyperedge_embeddings, augmented_node_embeddings, augmented_hyperedge_embeddings = self.cl_model(batch.batch_cell_feats.to(self.device), 
                                                                                                                             batch.batch_subgraph_hyperedges.to(self.device),
                                                                                                                             batch_incidence_matrix)
            contrastive_loss = self.criterion(node_embeddings[:batch.neighborhood_n_cells], 
                                              hyperedge_embeddings[:batch.neighborhood_n_hyperedges], 
                                              augmented_node_embeddings[:,:batch.neighborhood_n_cells],
                                              augmented_hyperedge_embeddings[:,:batch.neighborhood_n_hyperedges], 
                                              neighborhood_incidence_matrix,
                                              batch.neighborhood_cell_types.to(self.device),
                                              batch.sample_n_cells,
                                              batch.sample_n_hyperedges)
            return contrastive_loss
        else:
            node_embeddings, hyperedge_embeddings = self.model(batch.batch_cell_feats.to(self.device),
                                                               batch_incidence_matrix)
            return node_embeddings, hyperedge_embeddings
            

        # suppose hyperedge_feats are derived from node feats (eg., average pooling). In a contrastive learning scheme where node augmentations are performed by adding a random noise to nodes, is it also necessary to update hyperedge feats?
        
    def generate_embedding(self,
                           hypergraph: STHypergraphData,
                           batch_size: int,
                           accelerator: AcceleratorOptions):
        
        self.__setup_model(accelerator)
        self.cl_model.eval()

        model_checkpoint = torch.load(self.project / 'embedder.pth', weights_only=True)
        cl_model_state = model_checkpoint['model_state_dict']
        self.cl_model.load_state_dict(cl_model_state)
        
        hgdataset = STHypergraphMatrixDataset(hypergraph)
        
        hgtestsampler = BatchSampler(
            SequentialSampler(hgdataset),
            batch_size=batch_size,
            drop_last=False
        )
        
        hgloader = DataLoader(hgdataset, sampler=hgtestsampler, batch_size=None)
        node_embedding_batches = []
        hyperedge_embedding_batches = []
        node_id_batches = []
        hyperedge_id_batches = []
        for batch in tqdm(hgloader, leave=False, disable=True):
            n_cells, n_hyperedges = batch.batch_incidence_matrix.shape
            node_embedding_batch, hyperedge_embedding_batch = self.__step_batch(batch)
            node_embedding_batches.append(node_embedding_batch[:n_cells].detach().cpu())
            hyperedge_embedding_batches.append(hyperedge_embedding_batch[:n_hyperedges].detach().cpu())
            node_id_batches.append(batch.batch_subgraph_cells[:n_cells].detach().cpu())
            hyperedge_id_batches.append(batch.batch_subgraph_hyperedges[:n_hyperedges].detach().cpu())
            print(node_embedding_batch.shape, node_embedding_batch[:n_cells].shape)
            print(hyperedge_embedding_batch.shape, hyperedge_embedding_batch[:n_hyperedges].shape)

        logger.info("Saving embeddings...")
        node_embeddings = torch.cat(node_embedding_batches, dim=0)
        hyperedge_embeddings = torch.cat(hyperedge_embedding_batches, dim=0)

        print(node_embeddings.shape)
        
        node_ids = torch.cat(node_id_batches, dim=0)
        hyperedge_ids = torch.cat(hyperedge_id_batches, dim=0)
        
        node_order = torch.argsort(node_ids)
        hyperedge_order = torch.argsort(hyperedge_ids)

        node_embeddings = node_embeddings[node_order]
        hyperedge_embeddings = hyperedge_embeddings[hyperedge_order]

        print(node_embeddings.shape, hyperedge_embeddings.shape)
        torch.save(node_embeddings, self.project / "node_embeddings.pth")
        torch.save(hyperedge_embeddings, self.project / "hyperedge_embeddings.pth")
        