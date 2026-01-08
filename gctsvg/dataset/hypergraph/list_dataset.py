import gc
import torch

from . import STHypergraphDataset, STHypergraphBatchListData

from ...data import STHypergraphData

class STHypergraphListDataset(STHypergraphDataset):
    def __init__(self, 
                 hypergraph: STHypergraphData):
        super(STHypergraphListDataset, self).__init__(hypergraph)

        self.hyperedge_index = hypergraph['hyperedge_index']

        # optimization notes:
        # 1. getting neighboring hyperedges is faster if you have the edge_index (get 2-hop neighbors)
    
    def __len__(self):
        return self.cell_feats.shape[0]
        
    def __getitem__(self, idx) -> STHypergraphBatchListData:
        '''
        Get sub-hypergraph induced by cells indicated by idx
            Params: idx - index of the cells
        '''

        # 1. get unique set of cells based on index.
        cell_ids = torch.tensor(idx).unique()

        # 3. get the spots associated to cells
        # TODO: remove spots without cells
        hyperedge_ids = self.cell_to_spot_id[cell_ids].unique()
        
        # get first hyperedges - 
        # 4. hyperedges adjacent to node -> torch.isin(self.edge_index[0], cell_ids)
        # 5. hyperedges that are not the spots of indicated cells -> ~torch.isin(self.edge_index[0],spot_ids)
        # hyperedges = self.hyperedge_index[:, torch.isin(self.hyperedge_index[0], cell_ids) & ~torch.isin(self.hyperedge_index[1], spot_ids)]
        # print(hyperedges)
        # update hyperedge list
        # incident_hyperedge_ids = torch.cat((spot_ids, hyperedges[1,:].unique()))
        # print(incident_hyperedge_ids)
        
        # 6. get neighboring cells
        # - cells found in incident hyperedges
        # - cells not yet in current sample
        # neighbor_cell_ids = self.hyperedge_index[0, torch.isin(self.hyperedge_index[1], incident_hyperedge_ids) & ~torch.isin(self.hyperedge_index[0], cell_ids)].unique()
        # print(cell_ids, neighbor_cell_ids)

        # 7. Get neighbor incident hyperedges
        # - hyperedges incident to neighbor cells
        # - hyperedges not yet in current hyperedge list
        # neighbor_hyperedge_ids = self.hyperedge_index[1, torch.isin(self.hyperedge_index[0], neighbor_cell_ids) & ~torch.isin(self.hyperedge_index[1], incident_hyperedge_ids)].unique()

        rows, cols = self.hyperedge_index
        selected_cells = torch.zeros(self.cell_feats.shape[0], dtype=bool)
        selected_hyperedges = torch.zeros(self.hyperedge_feats.shape[0], dtype=bool)
        selected_cells[cell_ids] = True
        selected_hyperedges[hyperedge_ids] = True

        # Given the initial cells and hyperedges (spots), 
        # get all hyperedges adjacent to initial cells that are not spots of the given cells
        one_hop_hyperedges = selected_cells[rows] & ~selected_hyperedges[cols]
        one_hop_hyperedge_ids = cols[one_hop_hyperedges]
        selected_hyperedges[one_hop_hyperedge_ids] = True

        # Given the initial cells and the spots and 1-hop hyperedges,
        # Get all the 1-hop cells, cells that are reachable from the immediate hyperedges.
        one_hop_cells = ~selected_cells[rows] & selected_hyperedges[cols]
        one_hop_cell_ids = rows[one_hop_cells]
        selected_cells[one_hop_cell_ids] = True

        # Get the two-hop hyperedges, the remaining hyperedges that are adjacent to the 1-hop cells. 
        two_hop_hyperedges = selected_cells[rows] & ~selected_hyperedges[cols]
        two_hop_hyperedge_ids = cols[two_hop_hyperedges]
        selected_hyperedges[two_hop_hyperedge_ids] = True

        # Combine all the ids to form the batch.
        # Make sure that the sampled cells and neighborhoods come first.        
        subgraph_cells = torch.cat((cell_ids, one_hop_cell_ids)).squeeze()
        subgraph_hyperedges = torch.cat((hyperedge_ids, one_hop_hyperedge_ids, two_hop_hyperedge_ids)).squeeze()
        
        n_cells = cell_ids.shape[0]
        n_hyperedges = hyperedge_ids.shape[0]
        
        # get the induced subgraph using the selected cells and hyperedges        
        # subgraph_cells = torch.cat((cell_ids, neighbor_cell_ids))
        # subgraph_hyperedges = torch.cat((incident_hyperedge_ids, neighbor_hyperedge_ids))
        
        # subgraph_nodes = torch.cat((cell_ids, neighbor_node_ids))        
        # subgraph_edges = torch.cat((edge_ids, neighbor_edge_ids)).unique()
        # spot_expression = self.x_spot[subgraph_hyperedges]

        # print(subgraph_cells)
        # print(subgraph_hyperedges)

        # sample edges
        # discard nodes not adjacent to edges
        # if len(all_nodes) < node_sample_size, do not sample
        # sample node_sample_size - len(idx) from all_nodes / node_sample_size 
        
        feats = self.cell_feats[subgraph_cells]
        edge_feats = self.hyperedge_feats[subgraph_hyperedges]
        cell_types = self.cell_types[subgraph_cells]
        
        hyperedge_index = self.hyperedge_index[:, 
            torch.isin(self.hyperedge_index[0], subgraph_cells) & torch.isin(self.hyperedge_index[1], subgraph_hyperedges)]

        # TODO move remap on model
        # remapped_cells = hyperedge_index[0].unique(return_inverse=True)[1]
        # remapped_edges = hyperedge_index[1].unique(return_inverse=True)[1]
        # remapped_hyperedge_index = torch.stack((remapped_cells, remapped_edges))

        return STHypergraphBatchListData(n_cells, 
                                     n_hyperedges, 
                                     subgraph_cells, 
                                     subgraph_hyperedges, 
                                     feats, 
                                     edge_feats, 
                                     cell_types, 
                                     hyperedge_index)