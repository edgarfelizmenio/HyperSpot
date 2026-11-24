import math
import torch

from typing import List

from . import STHypergraphDataset, STHypergraphBatchMatrixData

from ...data import STHypergraphData

class STHypergraphMatrixDataset(STHypergraphDataset):
    def __init__(self,
                 hypergraph: STHypergraphData,
                 K: float = 0.7,
                 M: float = 0.7,
                 device: torch.device | str | None = None):
        super(STHypergraphMatrixDataset, self).__init__(hypergraph)

        # Hypergraph membership as (cell, hyperedge) pairs: cell hyperedge_index[0, i] is in hyperedge
        # hyperedge_index[1, i]. Batches are built from these pairs, never from a dense cells x hyperedges
        # matrix, so memory and time grow with the number of memberships (~7 per cell).
        self.hyperedge_index = hypergraph['hyperedge_index'].long()
        self.num_cells = self.cell_feats.shape[0]
        self.num_hyperedges = self.hyperedge_feats.shape[0]

        # Neighborhood-sampling ratios (GraphSAGE-style fan-out control), applied
        # per expansion hop so a minibatch stays bounded instead of inducing the
        # full 2-hop subgraph. Seed cells and their home hyperedges (the loss
        # targets) are never sampled; only the context hops are subsampled.
        #   K -- fraction of neighborhood hyperedges kept per hop
        #   M -- fraction of neighborhood cells kept per hyperedge expansion
        # 1.0 keeps everything (original behavior).
        self.K = K
        self.M = M

        # Optional: keep the graph tensors on `device` (e.g. the training GPU) and build every batch there,
        # so a batch's features need no copy at every step. Same sampling logic as on the CPU; only the
        # RNG stream differs (randperm runs on the device).
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if self.device.type != "cpu":
            self.hyperedge_index = self.hyperedge_index.to(self.device)
            self.cell_feats = self.cell_feats.to(self.device)
            self.hyperedge_feats = self.hyperedge_feats.to(self.device)
            self.cell_to_spot_id = self.cell_to_spot_id.to(self.device)
            self.cell_types = self.cell_types.to(self.device)

    def _sample(self, ids: torch.Tensor, ratio: float) -> torch.Tensor:
        """Randomly keep ceil(ratio * len) of a 1-D index tensor (no-op at ratio >= 1)."""
        n = ids.numel()
        if ratio >= 1.0 or n == 0:
            return ids
        k = max(1, math.ceil(ratio * n))
        if k >= n:
            return ids
        return ids[torch.randperm(n, device=ids.device)[:k]]

    def _hyperedges_of(self, cells: torch.Tensor, exclude: torch.Tensor) -> torch.Tensor:
        """Sorted ids of the hyperedges that contain any cell in the mask `cells`, minus those in `exclude`."""
        ids = self.hyperedge_index[1][cells[self.hyperedge_index[0]]].unique()
        return ids[~exclude[ids]]

    def _cells_of(self, hyperedges: torch.Tensor, exclude: torch.Tensor) -> torch.Tensor:
        """Sorted ids of the cells in any hyperedge in the mask `hyperedges`, minus those in `exclude`."""
        ids = self.hyperedge_index[0][hyperedges[self.hyperedge_index[1]]].unique()
        return ids[~exclude[ids]]

    def __len__(self):
        return self.num_cells

    def __getitem__(self, idx: List[int]) -> STHypergraphBatchMatrixData:
        '''
        Get sub-hypergraph induced by cells indicated by idx
            Params: idx - index of the cells
        '''
        # Initialize filtering flags
        selected_cells = torch.zeros(self.num_cells, dtype=bool, device=self.device)
        selected_hyperedges = torch.zeros(self.num_hyperedges, dtype=bool, device=self.device)

        # 1. Get initial set of cells based on index.
        initial_cell_ids = torch.tensor(idx, device=self.device).unique()
        selected_cells[initial_cell_ids] = True

        # 2. Get initial hyperedges (main spots where the cells are located).
        initial_hyperedge_ids = self.cell_to_spot_id[initial_cell_ids].unique()
        selected_hyperedges[initial_hyperedge_ids] = True

        # 3. One-hop hyperedges: the other hyperedges of the initial cells.
        one_hop_hyperedge_ids = self._hyperedges_of(selected_cells, selected_hyperedges)
        one_hop_hyperedge_ids = self._sample(one_hop_hyperedge_ids, self.K)   # fan-out cap

        # 4. Cells of the initial hyperedges (before the one-hop hyperedges are added).
        initial_hyperedge_cell_ids = self._cells_of(selected_hyperedges, selected_cells)
        initial_hyperedge_cell_ids = self._sample(initial_hyperedge_cell_ids, self.M)   # fan-out cap
        selected_cells[initial_hyperedge_cell_ids] = True

        # only update this here, after nodes of initial hyperedges have been identified
        selected_hyperedges[one_hop_hyperedge_ids] = True

        # 5. One-hop cells: the remaining cells of the initial and one-hop hyperedges.
        one_hop_cell_ids = self._cells_of(selected_hyperedges, selected_cells)
        one_hop_cell_ids = self._sample(one_hop_cell_ids, self.M)   # fan-out cap
        selected_cells[one_hop_cell_ids] = True

        # 6. Two-hop hyperedges: the other hyperedges of all selected cells
        # (two-hop hyperedges are required for edge convolution).
        two_hop_hyperedge_ids = self._hyperedges_of(selected_cells, selected_hyperedges)
        two_hop_hyperedge_ids = self._sample(two_hop_hyperedge_ids, self.K)   # fan-out cap
        selected_hyperedges[two_hop_hyperedge_ids] = True

        # get the induced subgraph using the selected cells and hyperedges
        batch_subgraph_cells = torch.cat((initial_cell_ids, initial_hyperedge_cell_ids, one_hop_cell_ids))
        batch_subgraph_hyperedges = torch.cat((initial_hyperedge_ids, one_hop_hyperedge_ids, two_hop_hyperedge_ids))

        # number of cells and hyperedges for gradient computation
        sample_n_cells = initial_cell_ids.shape[0]
        sample_n_hyperedges = initial_hyperedge_ids.shape[0]
        neighborhood_n_cells = sample_n_cells + initial_hyperedge_cell_ids.shape[0]
        neighborhood_n_hyperedges = sample_n_hyperedges + one_hop_hyperedge_ids.shape[0]

        batch_cell_feats = self.cell_feats[batch_subgraph_cells]
        batch_hyperedge_feats = self.hyperedge_feats[batch_subgraph_hyperedges]
        batch_cell_types = self.cell_types[batch_subgraph_cells]

        # Sparse batch incidence matrix: the membership pairs whose cell and hyperedge are both in the batch,
        # renumbered to their positions in batch_subgraph_cells / batch_subgraph_hyperedges.
        cell_pos = torch.full((self.num_cells,), -1, dtype=torch.long, device=self.device)
        cell_pos[batch_subgraph_cells] = torch.arange(batch_subgraph_cells.shape[0], device=self.device)
        hyperedge_pos = torch.full((self.num_hyperedges,), -1, dtype=torch.long, device=self.device)
        hyperedge_pos[batch_subgraph_hyperedges] = torch.arange(batch_subgraph_hyperedges.shape[0], device=self.device)
        rows = cell_pos[self.hyperedge_index[0]]
        cols = hyperedge_pos[self.hyperedge_index[1]]
        in_batch = (rows >= 0) & (cols >= 0)
        batch_incidence_matrix = torch.sparse_coo_tensor(
            torch.stack((rows[in_batch], cols[in_batch])),
            torch.ones(int(in_batch.sum()), device=self.device),
            size=(batch_subgraph_cells.shape[0], batch_subgraph_hyperedges.shape[0]),
        ).coalesce()

        # sample_* -> initial indicated by batch_size parameter
        # neigborhood_* -> immediate neighborhood of sample cells and hyperedges (rows the training loss uses)
        # batch_* -> entire batch of cells and hyperedges using in forward pass
        return STHypergraphBatchMatrixData(sample_n_cells,
                                     sample_n_hyperedges,
                                     neighborhood_n_cells,
                                     neighborhood_n_hyperedges,
                                     batch_subgraph_cells,
                                     batch_subgraph_hyperedges,
                                     batch_cell_feats,
                                     batch_hyperedge_feats,
                                     batch_cell_types,
                                     batch_incidence_matrix)
