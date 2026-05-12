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

from ..metadata import ProjectMetadata

from ..model.hypergraph import HypergraphEmbedder, HypergraphEmbedderArgs
from ..model.cl.hyfi import HyFi, HyFiAE
from ..model.cl.hgctsvgloss import HGCTCAELoss

from ..utils import build_incidence_matrix


# from ..model.cl.tricl import TriCL, TriCLLoss

from ..config import AcceleratorOptions

logger = logging.getLogger(__name__)

# save node embeddings and hyperedge embeddings
class HypergraphEmbedderDriver(object):
    def __init__(self,
                 sigma=1):
        
        logger.info("Initializing model...")
    
    def __setup_model(self, 
                      args: HypergraphEmbedderArgs, 
                      accelerator: AcceleratorOptions):
        
        self.device = get_device(accelerator)

        self.model = HypergraphEmbedder.build(args)
        self.cl_model = HyFiAE(self.model, args.embedding_dim) 
        self.cl_model.to(self.device)
        
    def fit(self, 
            hypergraph: STHypergraphData,
            embedder_args: HypergraphEmbedderArgs,
            num_epochs: int,
            batch_size: int,
            accelerator: AcceleratorOptions):

        project_metadata = ProjectMetadata()
        
        self.__setup_model(embedder_args,
                           accelerator)
        self.criterion = HGCTCAELoss(embedder_args.embedding_dim,
                                     n_classes = 12,
                                     temp_intraclass = 1.5,
                                     temp_interclass = 0.5,
                                     alpha=0.0,
                                     beta=0.0,
                                     gamma=0.0,
                                     rho=1.0)
        self.__configure_optimizer()
        self.criterion.to(self.device)
        
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
                        'embedder_state_dict': copy.deepcopy(self.model.state_dict()),
                        'criterion_state_dict': copy.deepcopy(self.criterion.state_dict()),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': min_epoch_loss
                    }
                    torch.save(checkpoint, project_metadata.project_folder / f'{self.model.checkpoint_name()}.pth')

                e_pbar.set_postfix({
                    'epoch_loss': epoch_loss, 
                    'max_loss': max_epoch_loss,
                    'min_loss': min_epoch_loss})

                    
    def __configure_optimizer(self):
        self.optimizer = optim.Adam(list(self.cl_model.parameters()) + list(self.criterion.parameters()), lr=0.0001)
                
    def __step_batch(self, batch: STHypergraphBatchMatrixData):

        batch = batch.to_device(self.device)        

        if self.cl_model.training:
            (node_embeddings, 
             hyperedge_embeddings, 
             # augmented_node_embeddings, 
             # augmented_hyperedge_embeddings,
             reconstruction) = self.cl_model(batch)

            contrastive_loss = self.criterion(batch.batch_cell_feats[:batch.neighborhood_n_cells],
                                              reconstruction[:batch.neighborhood_n_cells],
                                              # node_embeddings[:batch.neighborhood_n_cells], 
                                              # hyperedge_embeddings[:batch.neighborhood_n_hyperedges], 
                                              # augmented_node_embeddings[:,:batch.neighborhood_n_cells],
                                              # augmented_hyperedge_embeddings[:,:batch.neighborhood_n_hyperedges], 
                                              # batch.neighborhood_incidence_matrix,
                                              batch.batch_cell_types[:batch.neighborhood_n_cells],
                                              batch.sample_n_cells,
                                              batch.sample_n_hyperedges)
            
            return contrastive_loss
        
        else:
            node_embeddings, hyperedge_embeddings, _ = self.model(batch)
            
            return node_embeddings, hyperedge_embeddings
        
    def generate_embedding(self,
                           hypergraph: STHypergraphData,
                           embedder_args: HypergraphEmbedderArgs,
                           batch_size: int,
                           accelerator: AcceleratorOptions):

        project_metadata = ProjectMetadata()
        
        self.__setup_model(embedder_args, accelerator)
        self.model.load_checkpoint()
        self.cl_model.eval()
        
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
            n_cells = batch.sample_n_cells
            n_hyperedges = batch.sample_n_hyperedges
            
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
        node_order = torch.argsort(node_ids)
        node_embeddings = node_embeddings[node_order]
        
        hyperedge_ids = torch.cat(hyperedge_id_batches, dim=0)
        hyperedge_embeddings = self.__reduce_hyperedge_embeddings(hyperedge_embeddings,
                                                                  hyperedge_ids,
                                                                  hypergraph["num_spots"])

        print(node_embeddings.shape, hyperedge_embeddings.shape)
        torch.save(node_embeddings, project_metadata.project_folder / "node_embeddings.pth")
        torch.save(hyperedge_embeddings, project_metadata.project_folder / "hyperedge_embeddings.pth")

    def __reduce_hyperedge_embeddings(self, hyperedge_embeddings, hyperedge_ids, n_hyperedges):
        hyperedge_ids = hyperedge_ids.squeeze().long()
        unique_ids, emb_to_id = torch.unique(hyperedge_ids, sorted=True, return_inverse=True)
        out = torch.full((n_hyperedges, hyperedge_embeddings.size(1)), float('nan'), device=hyperedge_embeddings.device)
        count = torch.zeros(n_hyperedges, device=hyperedge_embeddings.device)
        
        out.scatter_reduce_(0, 
                            emb_to_id[:, None].expand_as(hyperedge_embeddings), 
                            hyperedge_embeddings, "sum", include_self=False)
        count.scatter_reduce_(0, 
                              emb_to_id, torch.ones_like(emb_to_id, dtype=count.dtype), 
                              "sum", include_self=False)

        out[unique_ids] = out[unique_ids] / count[unique_ids].unsqueeze(1)

        return out
    
    @classmethod
    def get_available_embedders(cls):
        return HypergraphEmbedder.get_available_embedders()