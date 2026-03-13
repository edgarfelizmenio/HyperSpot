import torch

import torch.nn as nn
import torch.nn.functional as F

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from ..moduledict import LinearDict, LinearDictLora, ParameterDict

class CTHyFiHGNNv1(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout = 0.1, node_act=nn.PReLU, hyperedge_act=nn.PReLU):
        super(CTHyFiHGNNv1, self).__init__()
        # perform node-edge-node feature transform

        # TODO parametrize num cell types
        self.hyperedge_emb_weight = LinearDict(12, input_dim, embedding_dim, bias=False)
        self.hyperedge_emb_bias = nn.Parameter(torch.zeros((embedding_dim,)))
        self.node_emb_weight = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.node_emb_bias = ParameterDict(12, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.node_act = node_act()
        self.hyperedge_act = hyperedge_act()
        
        self.reset_parameters()

    def reset_parameters(self):
        self.hyperedge_emb_weight.reset_parameters()
        self.node_emb_weight.reset_parameters()
        self.node_emb_bias.reset_parameters()

    def forward(self, node_feats, cell_types, incidence_matrix):
        dv = incidence_matrix.sum(dim=1) 
        de = incidence_matrix.sum(dim=0)
        
        dv_invsqrt = (dv**-0.5).diag()
        dv_invsqrt[dv_invsqrt == float('inf')] = 0
        de_inv = (de**-1).diag()
        de_inv[de_inv == float('inf')] = 0

        # node_feats = self.dropout(node_feats)

        hyperedge_embeddings = de_inv @ incidence_matrix.t() @ dv_invsqrt @ self.hyperedge_emb_weight(node_feats, cell_types) + self.hyperedge_emb_bias
        hyperedge_embeddings = self.hyperedge_act(hyperedge_embeddings)
        hyperedge_embeddings = self.dropout(hyperedge_embeddings)
        
        # TODO: check if node activation necessary
        node_embeddings = dv_invsqrt @ incidence_matrix @ self.node_emb_weight(hyperedge_embeddings) + self.node_emb_bias(cell_types)
        node_embeddings = self.node_act(node_embeddings)        

        return node_embeddings, hyperedge_embeddings

@HypergraphEmbedder.register("CTHyFiEmbedder")
class CTHyFiEmbedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs):
        model = CTHyFiHGNNv1(
            args.input_dim,
            args.embedding_dim,
            args.dropout
        )
        return cls(model)

    def forward(self, batch):
        feats = batch.batch_cell_feats
        cell_types = batch.batch_cell_types
        incidence_matrix = batch.batch_incidence_matrix
        return self.model(feats, cell_types, incidence_matrix)

    @classmethod
    def checkpoint_name(cls) -> str:
        return "CTHyFiEmbedder"

    def get_attribution_forward(self, index = 0):
        def attribution_forward(feats, cell_types, incidence_matrix):
            return self.model(feats, cell_types, incidence_matrix)[index]
        return attribution_forward

    def get_attribution_inputs(self, batch):
        inputs = batch.batch_cell_feats
        additional_forward_args = (batch.batch_cell_types, batch.batch_incidence_matrix)
        return inputs, additional_forward_args

    def get_e2e_attribution_forward(self, classifier, index = 0, num_repeats=1):
        class ExplainerClassifier(nn.Module):
            def __init__(self, embedder, classifier, index):
                super(ExplainerClassifier, self).__init__()
                self.embedder = embedder.model
                self.classifier = classifier
                self.index = index
                self.num_repeats = num_repeats
            
            def forward(self, feats, cell_types, incidence_matrix, batch_size):
                
                embedding = self.embedder(feats, cell_types, incidence_matrix)[self.index]

                num_samples = embedding.shape[0] // self.num_repeats
                num_cols = embedding.shape[1]
                
                embedding = embedding.view(self.num_repeats, num_samples, num_cols)
                embedding = embedding[:, :batch_size, :]
                embedding = embedding.reshape(self.num_repeats * batch_size, num_cols)

                classes = self.classifier(embedding)

                return classes
        
        explainer_classifier = ExplainerClassifier(self, classifier, index)
        return explainer_classifier

    def get_e2e_attribution_inputs(self, batch):
        inputs, additional_forward_args = self.get_attribution_inputs(batch)
        cell_types, incidence_matrix = additional_forward_args
        batch_size = batch.sample_n_cells
        additional_forward_args = (cell_types, incidence_matrix, batch_size)
        return inputs, additional_forward_args
        