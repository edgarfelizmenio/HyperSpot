import torch

import torch.nn as nn
import torch.nn.functional as F

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from .cthgnn import CTHyFiHGNNv1
from ..moduledict import LinearDict, ParameterDict

class CTHGCAE(nn.Module):
    def __init__(self, input_dim, embedding_dim, dropout=0.1):
        super(CTHGCAE, self).__init__()
        self.encoder = CTHyFiHGNNv1(input_dim, embedding_dim, dropout)
        self.decoder = CTHyFiHGNNv1(embedding_dim, input_dim, dropout=0)

    def forward(self, node_feats, cell_types, incidence_matrix):
        node_embeddings, hyperedge_embeddings = self.encoder(node_feats,
                                          cell_types,
                                          incidence_matrix)

        node_reconstruction, _ = self.decoder(node_embeddings,
                                              cell_types,
                                              incidence_matrix)

        return node_embeddings, hyperedge_embeddings, node_reconstruction

@HypergraphEmbedder.register("CTHGCAEEmbedder")
class CTHGCAEEmbedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs):
        model = CTHGCAE(
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
        return "CTHGCAEEmbedder"

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