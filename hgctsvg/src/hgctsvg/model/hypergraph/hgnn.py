import torch

import torch.nn as nn
import torch.nn.functional as F

class HyFiHGNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout = 0.1):
        super(HyFiHGNN, self).__init__()
        # perform node-edge-node feature transform

        self.hyperedge_emb_weight = nn.Linear(input_dim, embedding_dim, bias=False)
        self.hyperedge_emb_bias = nn.Parameter(torch.zeros((embedding_dim,)))
        self.node_emb_weight = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.node_emb_bias = nn.Parameter(torch.zeros((embedding_dim,)))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.hyperedge_emb_weight.reset_parameters()
        self.node_emb_weight.reset_parameters()

    def forward(self, node_feats, incidence_matrix):
        dv = incidence_matrix.sum(dim=1) 
        de = incidence_matrix.sum(dim=0)
        
        dv_invsqrt = (dv**-0.5).diag()
        dv_invsqrt[dv_invsqrt == float('inf')] = 0
        de_inv = (de**-1).diag()
        de_inv[de_inv == float('inf')] = 0

        node_feats = self.dropout(node_feats)
        hyperedge_embeddings = de_inv @ incidence_matrix.t() @ dv_invsqrt @ self.hyperedge_emb_weight(node_feats) + self.hyperedge_emb_bias
        hyperedge_embeddings = self.act(hyperedge_embeddings)

        node_embeddings = dv_invsqrt @ incidence_matrix @ self.node_emb_weight(hyperedge_embeddings) + self.node_emb_bias
        node_embeddings = self.act(node_embeddings)        

        return node_embeddings, hyperedge_embeddings