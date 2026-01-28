import torch

import torch.nn as nn
import torch.nn.functional as F

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from .hgnn import HyFiHGNN

class SpatialFusionNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, latent_dim=None, dropout = 0.1):
        super(SpatialFusionNetwork, self).__init__()
        self.hgnn = HyFiHGNN(input_dim, latent_dim, dropout = 0)
        
        if not latent_dim:
            latent_dim = embedding_dim
        
        # self.fusion_node = nn.Linear(input_dim + latent_dim, embedding_dim)
        # self.fusion_hyperedge = nn.Linear(input_dim + latent_dim, embedding_dim)

        self.node_linear = nn.Linear(input_dim, latent_dim)
        self.hyperedge_linear = nn.Linear(input_dim, latent_dim)

        self.node_embedding_linear = nn.Linear(latent_dim * 2, embedding_dim)
        self.hyperedge_embedding_linear = nn.Linear(latent_dim * 2, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        
        self.reset_parameters()

    def reset_parameters(self):
        self.hgnn.reset_parameters()
        # self.fusion_node.reset_parameters()
        # self.fusion_hyperedge.reset_parameters()
        self.node_linear.reset_parameters()
        self.hyperedge_linear.reset_parameters()
        self.node_embedding_linear.reset_parameters()
        self.hyperedge_embedding_linear.reset_parameters()

    
    def forward(self, node_feats, incidence_matrix):
        node_feats = self.dropout(node_feats)
        hyperedge_feats = incidence_matrix.t() @ node_feats

        node_embeddings, hyperedge_embeddings = self.hgnn(node_feats, incidence_matrix)

        # fused_node_embeddings = self.fusion_node(torch.cat((node_feats, node_embeddings),dim = 1))
        # fused_hyperedge_embeddings = self.fusion_hyperedge(torch.cat((hyperedge_feats, hyperedge_embeddings), dim = 1))

        # fused_node_embeddings = self.act(fused_node_embeddings)
        # fused_hyperedge_embeddings = self.act(fused_hyperedge_embeddings)
        
        node_h = self.act(self.node_linear(node_feats))
        hyperedge_h = self.act(self.hyperedge_linear(hyperedge_feats))

        fused_node_embeddings = self.act(self.node_embedding_linear(torch.cat((node_embeddings, 
                                                                               node_h), dim = 1)))
        fused_hyperedge_embeddings = self.act(self.hyperedge_embedding_linear(torch.cat((hyperedge_embeddings,
                                                                                         hyperedge_h), dim = 1)))
    
        return fused_node_embeddings, fused_hyperedge_embeddings

@HypergraphEmbedder.register("SpatialFusionEmbedder")
class SpatialFusionEmbedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs):
        model = SpatialFusionNetwork(
            args.input_dim,
            args.embedding_dim,
            args.latent_dim,
            args.dropout
        )
        return cls(model)

    @classmethod
    def checkpoint_name(cls) -> str:
        return "SpatialFusionEmbedder"

    def forward(self, batch):
        feats = batch.batch_cell_feats
        incidence_matrix = batch.batch_incidence_matrix
        return self.model(feats, incidence_matrix)

    def get_attribution_forward(self, index = 0):
        def attribution_forward(self, feats, incidence_matrix):
            return self.model(feats, incidence_matrix)[index]

    def get_attribution_inputs(self, batch):
        inputs = batch.batch_cell_feats
        additional_forward_args = (batch.batch_incidence_matrix,)
        return inputs, additional_forward_args