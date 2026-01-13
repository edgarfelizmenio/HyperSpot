import copy
import numpy as np
import scanpy as sc

import torch
import torch.nn as nn
import torch.optim as optim

import logging

from pathlib import Path
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader

from ..model.classifier import EmbeddingClassifier

from ..config import AcceleratorOptions
from .utils import get_device

logger = logging.getLogger(__name__)

class STEmbeddingSpatialClassifierTrainer(object):
    pass

class SCEmbeddingSpatialClassifierTrainer(object):
    def __init__(self,
                 project: Path,
                 cell_type: str,
                 input_dim,
                 num_classes,
                 num_layers):

        self.output_dir = project / "classifiers"

        if not self.output_dir.exists():
            self.output_dir.mkdir()

        
        self.cell_type = cell_type
        
        logger.info("Initializing model...")

        self.model = EmbeddingClassifier(input_dim, num_classes, num_layers)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
        self.__configure_optimizer()
    
    def fit(self, 
            embedding, 
            clusters, 
            num_epochs: int,
            batch_size: int,
            accelerator: AcceleratorOptions,
            pbar_position=0):
        self.device = get_device(accelerator)

        self.model.to(self.device)
        
        # setup dataset and dataloader
        dataset = TensorDataset(embedding, clusters)
        loader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = True)

        with tqdm(range(num_epochs), position=pbar_position + 1) as e_pbar:
            max_epoch_loss = None
            min_epoch_loss = None
            for epoch in e_pbar:
                epoch_loss = self.__step_epoch(loader)
                    
                if not max_epoch_loss or epoch_loss > max_epoch_loss:
                    max_epoch_loss = epoch_loss
                if not min_epoch_loss or epoch_loss < min_epoch_loss:
                    min_epoch_loss = epoch_loss
                    # perform checkpointing 
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': copy.deepcopy(self.model.state_dict()),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': min_epoch_loss
                    }
                    torch.save(checkpoint, self.output_dir / f'cell_{self.cell_type}.pth')
                
                e_pbar.set_postfix({
                    'epoch_loss': epoch_loss,
                    'max_loss': max_epoch_loss,
                    'min_loss': min_epoch_loss})
    
    def __configure_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def __step_epoch(self, dataloader, pbar_position=0):
        num_batches = len(dataloader)
        epoch_loss = 0.0
        with tqdm(dataloader, leave=False, position=pbar_position+1) as b_pbar:
            for batch in b_pbar:
                self.optimizer.zero_grad()
                loss = self.__step_batch(batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                
                b_pbar.set_postfix({'loss': loss.item()})

        epoch_loss /= num_batches

        return epoch_loss
    
    def __step_batch(self, batch):
        batch_X = batch[0].to(self.device)
        batch_y = batch[1].to(self.device)
        batch_out = self.model(batch_X)
        loss = self.criterion(batch_out, batch_y)
        return loss

def load_sc_classifier_input(project: Path, 
                             cell_type: str):
    adata_cell_type = sc.read_h5ad(project / "clustering" / f"ST_cell_{cell_type}.h5ad")
    
    embeddings = torch.from_numpy(adata_cell_type.obsm["embedding"])
    classes = torch.from_numpy(adata_cell_type.obs["leiden"].to_numpy(dtype=int))
    num_classes = adata_cell_type.obs["leiden"].nunique()

    return embeddings, classes, num_classes 

def load_sc_classifier_inputs(project: Path):
    cell_types = np.loadtxt(project / "cell_types.txt", dtype=str)
    cell_type_embeddings = {}
    cell_type_classes = {}
    cell_type_num_classes = {}
    for cell_type in tqdm(cell_types, desc="Loading classifier input for each cell type..."):
        
        embeddings, classes, num_classes = load_sc_classifier_input(project, cell_type)
        
        cell_type_embeddings[cell_type] = embeddings
        cell_type_classes[cell_type] = classes
        cell_type_num_classes[cell_type] = num_classes

    return cell_types, cell_type_embeddings, cell_type_classes, cell_type_num_classes

def train_sc_embedding_classifiers(project, 
                      input_dim,
                      cell_types: list,
                      cell_type_num_classes: dict,
                      num_layers: int,
                      embeddings: dict,
                      clusters: dict,
                      num_epochs: int,
                      batch_size: int,
                      accelerator: AcceleratorOptions):

    with tqdm(cell_types, desc="Training classifiers for each cell type...") as cell_type_pbar:
        for cell_type in cell_type_pbar:
            trainer = SCEmbeddingSpatialClassifierTrainer(project,
                                                          cell_type,
                                                          input_dim,
                                                          cell_type_num_classes[cell_type],
                                                          num_layers)
            trainer.fit(embeddings[cell_type],
                        clusters[cell_type],
                        num_epochs,
                        batch_size,
                        accelerator)