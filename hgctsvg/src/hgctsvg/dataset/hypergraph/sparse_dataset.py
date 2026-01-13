import torch

from . import STHypergraphDataset, STHypergraphBatchListData

from ...data import STHypergraphData

def build_hypergraph_sparse_csr_index(hyperedge_index, num_nodes, num_hyperedges):
    rows, cols = hyperedge_index
    
    node2hyperedge_idx = torch.sparse_csr_tensor(
        crow_indices=torch._convert_indices_from_coo_to_csr(rows, size=num_nodes),
        col_indices=cols,
        values=torch.ones_like(cols, dtype=torch.float32),
        size=(num_nodes, num_hyperedges)
    )
    
    hyperedge2node_idx = torch.sparse_csr_tensor(
        crow_indices=torch._convert_indices_from_coo_to_csr(cols, size=num_hyperedges),
        col_indices=rows,
        values=torch.ones_like(rows, dtype=torch.float32),
        size=(num_hyperedges, num_nodes)
    )

    return node2hyperedge_idx, hyperedge2node_idx

def get_index_val(query, index):
    indptr = index.crow_indices()
    colidx = index.col_indices()

    query_ids = torch.nonzero(query).flatten()

    starts = indptr[query_ids]
    ends   = indptr[query_ids + 1]

    # lengths per row
    lengths = ends - starts
    total = lengths.sum()

    # build offsets
    offsets = torch.repeat_interleave(starts, lengths)

    # generate local indices
    local = torch.cat([torch.arange(l) for l in lengths])

    # neighbor hyperedges
    # print("query:", colidx[offsets + local])
    return colidx[offsets + local]

class STHypergraphSparseDataset(STHypergraphDataset):
    def __init__(self, 
                 hypergraph: STHypergraphData):
        super(STHypergraphSparseDataset, self).__init__(hypergraph)

        self.hyperedge_index = hypergraph['hyperedge_index']
        (self.node2hyperedge, 
         self.hyperedge2node) = build_hypergraph_sparse_csr_index(self.hyperedge_index, 
                                                                  self.cell_feats.shape[0],
                                                                  self.hyperedge_feats.shape[0])
    
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


        rows, cols = self.hyperedge_index
        selected_cells = torch.zeros(self.cell_feats.shape[0], dtype=bool)
        selected_hyperedges = torch.zeros(self.hyperedge_feats.shape[0], dtype=bool)
        selected_cells[cell_ids] = True
        selected_hyperedges[hyperedge_ids] = True

        # step 1: find hyperedges incident to all nodes (except from spot)
        one_hop_hyperedge_ids = get_index_val(selected_cells, self.node2hyperedge)
        one_hop_hyperedge_ids = one_hop_hyperedge_ids[~selected_hyperedges[one_hop_hyperedge_ids]]
        selected_hyperedges[one_hop_hyperedge_ids] = True

        # step 2: find cells incident from all hyperedges
        one_hop_cell_ids = get_index_val(selected_hyperedges, self.hyperedge2node)
        one_hop_cell_ids = one_hop_cell_ids[~selected_cells[one_hop_cell_ids]]
        selected_cells[one_hop_cell_ids] = True

        # step 3: find incident hyperedges from each cell found
        two_hop_hyperedge_ids = get_index_val(selected_cells, self.node2hyperedge)
        two_hop_hyperedge_ids = two_hop_hyperedge_ids[~selected_hyperedges[two_hop_hyperedge_ids]]
        selected_hyperedges[two_hop_hyperedge_ids] = True
                
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