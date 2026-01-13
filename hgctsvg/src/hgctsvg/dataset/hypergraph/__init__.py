import torch

from dataclasses import dataclass, field

from collections import namedtuple
from torch.utils.data import Dataset

from ...data import STHypergraphData

STHypergraphBatchListData = namedtuple("STHypergraphBatchListData", ["n_cells", 
                                                             "n_hyperedges",
                                                             "batch_cells",
                                                             "batch_hyperedges",
                                                             "node_feats",
                                                             "hyperedge_feats",
                                                             "cell_types",
                                                             "hyperedge_index"])

STHypergraphBatchMatrixData = namedtuple("STHypergraphBatchMatrixData", ["n_cells", 
                                                             "n_hyperedges",
                                                             "cl_n_cells",
                                                             "cl_n_hyperedges",
                                                             "batch_cells",
                                                             "batch_hyperedges",
                                                             "node_feats",
                                                             "hyperedge_feats",
                                                             "cell_types",
                                                             "subgraph_incidence_matrix",
                                                             "batch_incidence_matrix"])

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