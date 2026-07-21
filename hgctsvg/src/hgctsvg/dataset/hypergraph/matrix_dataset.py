import torch

from typing import List

from . import STHypergraphDataset, STHypergraphBatchMatrixData

from ...data import STHypergraphData

class STHypergraphMatrixDataset(STHypergraphDataset):
    def __init__(self,
                 hypergraph: STHypergraphData):
        super(STHypergraphMatrixDataset, self).__init__(hypergraph)

        self.incidence_matrix = hypergraph['incidence_matrix']

        # optimization notes:
        # 1. getting neighboring hyperedges is faster if you have the edge_index (get 2-hop neighbors)
    
    def __len__(self):
        return self.cell_feats.shape[0]
        
    def __getitem__(self, idx: List[int]) -> STHypergraphBatchMatrixData:
        '''
        Get sub-hypergraph induced by cells indicated by idx
            Params: idx - index of the cells
        '''
        # Initialize filtering flags
        selected_cells = torch.zeros(self.cell_feats.shape[0], dtype=bool)
        selected_hyperedges = torch.zeros(self.hyperedge_feats.shape[0], dtype=bool)       
        
        # 1. Get initial set of cells based on index.
        initial_cell_ids = torch.tensor(idx).unique()
        selected_cells[initial_cell_ids] = True
        
        # 2. Get initial hyperedges (main spots where the cells are located).
        initial_hyperedge_ids = self.cell_to_spot_id[initial_cell_ids].unique()
        selected_hyperedges[initial_hyperedge_ids] = True

        # 3. Get all one-hop Hyperedges. These are hyperedges adjacent to initial nodes.
        # Required data for commanality computation of initial nodes
        one_hop_hyperedges = (self.incidence_matrix[selected_cells].bool() & ~selected_hyperedges).any(dim=0)
        one_hop_hyperedge_ids = one_hop_hyperedges.nonzero(as_tuple=True)[0]
        # selected_hyperedges[one_hop_hyperedge_ids] = True        
        
        # 4. Get all nodes adjacent to initial hyperedges.
        # Required data for commonality computation of initial hyperedges
        initial_hyperedge_cells = (self.incidence_matrix[:,selected_hyperedges].bool() & (~selected_cells)[:, None]).any(dim=1)
        initial_hyperedge_cell_ids = initial_hyperedge_cells.nonzero(as_tuple=True)[0]
        selected_cells[initial_hyperedge_cell_ids] = True        

        # only update this here, after nodes of initial hyperedges have been identified
        selected_hyperedges[one_hop_hyperedge_ids] = True        

        # get first hyperedges - 
        # 4. hyperedges adjacent to node -> torch.isin(self.edge_index[0], cell_ids)
        # 5. hyperedges that are not the spots of indicated cells -> ~torch.isin(self.edge_index[0],hyperedge_ids)
        # one_hop_hyperedges = torch.nonzero(self.incidence_matrix.bool() & (selected_cells[:, None] & ~selected_hyperedges[None, :]), as_tuple=True)
        # incident_hyperedges = one_hop_hyperedges[1]
        # selected_hyperedges[incident_hyperedges] = True

        
        # # 6. get neighboring cells
        # # - cells found in incident hyperedges
        # # - cells not yet in current sample        
        # one_hop_cells = torch.nonzero(self.incidence_matrix.bool() & (~selected_cells[:, None] & selected_hyperedges[None, :]), as_tuple=True)
        # neighboring_cell_ids = one_hop_cells[0]
        # selected_cells[neighboring_cell_ids] = True
        # # print(cell_ids, neighbor_cell_ids)
        one_hop_cells = (self.incidence_matrix[:, selected_hyperedges].bool() & (~selected_cells)[:, None]).any(dim=1)
        one_hop_cell_ids = one_hop_cells.nonzero(as_tuple=True)[0]
        selected_cells[one_hop_cell_ids] = True
        
        # # 7. Get neighbor incident hyperedges
        # # - hyperedges incident to neighbor cells
        # # - hyperedges not yet in current hyperedge list
        # two_hop_hyperedges = torch.nonzero(self.incidence_matrix.bool() & (selected_cells[:, None] & ~selected_hyperedges[None, :]), as_tuple=True)
        # neighboring_cell_hyperedge_ids = two_hop_hyperedges[1]
        # selected_hyperedges[neighboring_cell_hyperedge_ids] = True
        two_hop_hyperedges = (self.incidence_matrix[selected_cells].bool() & ~selected_hyperedges).any(dim=0)
        two_hop_hyperedge_ids = two_hop_hyperedges.nonzero(as_tuple=True)[0]
        selected_hyperedges[two_hop_hyperedge_ids] = True

        # get the induced subgraph using the selected cells and hyperedges
        # two-hop hyperedges are required for edge convolution
        batch_subgraph_cells = torch.cat((initial_cell_ids, initial_hyperedge_cell_ids, one_hop_cell_ids)).squeeze()
        batch_subgraph_hyperedges = torch.cat((initial_hyperedge_ids, one_hop_hyperedge_ids, two_hop_hyperedge_ids)).squeeze()

        # number of cells and hyperedges for gradient computation
        sample_n_cells = initial_cell_ids.shape[0]
        sample_n_hyperedges = initial_hyperedge_ids.shape[0]
        neighborhood_n_cells = sample_n_cells + initial_hyperedge_cell_ids.shape[0]
        neighborhood_n_hyperedges = sample_n_hyperedges + one_hop_hyperedge_ids.shape[0]
        # subgraph_nodes = torch.cat((cell_ids, neighbor_node_ids))        
        # subgraph_edges = torch.cat((edge_ids, neighbor_edge_ids)).unique()
        # spot_expression = self.x_spot[subgraph_hyperedges]

        # print(subgraph_cells)
        # print(subgraph_hyperedges)
        
        # sample edges
        # discard nodes not adjacent to edges
        # if len(all_nodes) < node_sample_size, do not sample
        # sample node_sample_size - len(idx) from all_nodes / node_sample_size 
        
        batch_cell_feats = self.cell_feats[batch_subgraph_cells]
        batch_hyperedge_feats = self.hyperedge_feats[batch_subgraph_hyperedges]
        batch_cell_types = self.cell_types[batch_subgraph_cells]
        
        batch_incidence_matrix = self.incidence_matrix[batch_subgraph_cells][:, batch_subgraph_hyperedges]
        neighborhood_incidence_matrix = self.incidence_matrix[batch_subgraph_cells[:neighborhood_n_cells]][:, batch_subgraph_hyperedges[:neighborhood_n_hyperedges]]

        # TODO move remap on model
        # remapped_cells = hyperedge_index[0].unique(return_inverse=True)[1]
        # remapped_edges = hyperedge_index[1].unique(return_inverse=True)[1]
        # remapped_hyperedge_index = torch.stack((remapped_cells, remapped_edges))

        # sample_* -> initial indicated by batch_size parameter
        # neigborhood_* -> immediate neighborhood of sample cells and hyperedges (used in computing contrastive loss)
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
                                     batch_incidence_matrix,
                                     neighborhood_incidence_matrix)