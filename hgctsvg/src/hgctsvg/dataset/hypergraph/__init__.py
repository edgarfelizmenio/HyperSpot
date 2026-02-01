import torch

from dataclasses import dataclass, field

from collections import namedtuple
from torch.utils.data import Dataset
from typing import NamedTuple, Optional

from ...data import STHypergraphData

class STHypergraphBatchMatrixData(NamedTuple):
    sample_n_cells: int
    sample_n_hyperedges: int
    neighborhood_n_cells: int
    neighborhood_n_hyperedges: int
    batch_subgraph_cells: torch.Tensor
    batch_subgraph_hyperedges: torch.Tensor
    batch_cell_feats: torch.Tensor
    batch_hyperedge_feats: torch.Tensor
    batch_cell_types: torch.Tensor
    batch_incidence_matrix: torch.Tensor
    neighborhood_incidence_matrix: torch.Tensor
    
    def to_device(self, device):
        return STHypergraphBatchMatrixData(
            sample_n_cells = self.sample_n_cells,
            sample_n_hyperedges = self.sample_n_hyperedges,
            neighborhood_n_cells = self.neighborhood_n_cells,
            neighborhood_n_hyperedges = self.neighborhood_n_hyperedges,
            batch_subgraph_cells = self.batch_subgraph_cells.to(device),
            batch_subgraph_hyperedges = self.batch_subgraph_hyperedges.to(device),
            batch_cell_feats = self.batch_cell_feats.to(device),
            batch_hyperedge_feats = self.batch_hyperedge_feats.to(device),
            batch_cell_types = self.batch_cell_types.to(device),
            batch_incidence_matrix = self.batch_incidence_matrix.to(device),
            neighborhood_incidence_matrix = self.neighborhood_incidence_matrix.to(device),
        )
    
STHypergraphBatchListData = namedtuple("STHypergraphBatchListData", ["n_cells", 
                                                             "n_hyperedges",
                                                             "batch_cells",
                                                             "batch_hyperedges",
                                                             "cell_feats",
                                                             "hyperedge_feats",
                                                             "cell_types",
                                                             "hyperedge_index"])

class STHypergraphDataset(Dataset):
    def __init__(self, 
                 hypergraph: STHypergraphData):
        super(STHypergraphDataset, self).__init__()

        self.cell_feats = hypergraph['cell_feats']
        self.cell_to_spot_id = hypergraph['cell_to_spot_id']
        self.hyperedge_feats = hypergraph['hyperedge_feats']
        self.cell_types = hypergraph['cell_types']
        self.num_cell_types = hypergraph['num_cell_types']

from .list_dataset import STHypergraphListDataset
from .matrix_dataset import STHypergraphMatrixDataset
from .sparse_dataset import STHypergraphSparseDataset