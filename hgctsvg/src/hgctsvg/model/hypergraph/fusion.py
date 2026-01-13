import torch

import torch.nn as nn
import torch.nn.functional as F

from .hgnn import HyFiHGNN

class SpatialFusionNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim=None, dropout = 0.1):
        super(SpatialFusionNetwork, self).__init__()
        self.hgnn = HyFiHGNN(input_dim, embedding_dim, dropout = 0)
        
        if not output_dim:
            output_dim = embedding_dim
        
        self.fusion_node = nn.Linear(input_dim + embedding_dim, output_dim)
        self.fusion_hyperedge = nn.Linear(input_dim + embedding_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        self.hgnn.reset_parameters()
        self.fusion_node.reset_parameters()
        self.fusion_hyperedge.reset_parameters()
    
    def forward(self, node_feats, incidence_matrix):
        node_feats = self.dropout(node_feats)
        hyperedge_feats = incidence_matrix.t() @ node_feats

        node_embeddings, hyperedge_embeddings = self.hgnn(node_feats, incidence_matrix)

        fused_node_embeddings = self.fusion_node(torch.cat((node_feats, node_embeddings),dim = 1))
        fused_hyperedge_embeddings = self.fusion_hyperedge(torch.cat((hyperedge_feats, hyperedge_embeddings), dim = 1))

        return fused_node_embeddings, fused_hyperedge_embeddings