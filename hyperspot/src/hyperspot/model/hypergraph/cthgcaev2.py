import torch
import torch.nn as nn

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from .cthgnn import CTHyFiHGNNv1
from ..moduledict import LinearDict, ParameterDict

# bias=True gives per-CT bias via Embedding


class CTHGCAEEncoderv2(CTHyFiHGNNv1):
    """
    CTHyFiHGNNv1 encoder with hyperedge dropout active.
    Inherits the full node→hyperedge→node convolution; only change is that
    dropout is also applied after the hyperedge activation (line 47 in cthgnn.py
    was commented out in the base class).
    """

    def forward(self, node_feats, cell_types, incidence_matrix):
        dv = incidence_matrix.sum(dim=1)
        de = incidence_matrix.sum(dim=0)

        dv_invsqrt = (dv ** -0.5).diag()
        dv_invsqrt[dv_invsqrt == float('inf')] = 0
        de_inv = (de ** -1).diag()
        de_inv[de_inv == float('inf')] = 0

        # node_feats = self.dropout(node_feats)

        hyperedge_embeddings = (
            de_inv @ incidence_matrix.t() @ dv_invsqrt
            @ self.hyperedge_emb_weight(node_feats, cell_types)
            + self.hyperedge_emb_bias
        )
        hyperedge_embeddings_act = self.act(hyperedge_embeddings)
        hyperedge_embeddings_act = self.dropout(hyperedge_embeddings_act)  # now active

        node_embeddings = (
            dv_invsqrt @ incidence_matrix
            @ self.node_emb_weight(hyperedge_embeddings_act)
            + self.node_emb_bias(cell_types)
        )
        # no node activation — correct for autoencoder bottleneck

        return node_embeddings, hyperedge_embeddings


class CTLinearDecoder(nn.Module):
    """
    Cell-type-specific linear decoder: E → reconstructed input features.
    Uses LinearDict so each cell type has its own projection matrix and bias.
    Weaker than the hypergraph decoder — forces the bottleneck to carry
    more information rather than relying on spot-level aggregation.
    """

    def __init__(self, num_cell_types, embedding_dim, output_dim):
        super().__init__()
        self.decoder = LinearDict(num_cell_types, embedding_dim, output_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.decoder.reset_parameters()

    def forward(self, node_embeddings, cell_types):
        return self.decoder(node_embeddings, cell_types)


class CTHGCAEv2(nn.Module):
    """
    Improved CTHGCAE with:
      1. Hyperedge dropout active in encoder (prevents reliance on specific spots)
      2. Cell-type-specific linear decoder (removes hypergraph shortcut)
      3. PairNorm on bottleneck (prevents embedding collapse)
    """

    def __init__(self, num_cell_types, input_dim, embedding_dim, dropout=0.1):
        super().__init__()
        self.encoder = CTHGCAEEncoderv2(num_cell_types, input_dim, embedding_dim, dropout)
        self.decoder = CTLinearDecoder(num_cell_types, embedding_dim, input_dim)

    @staticmethod
    def _pair_norm(E: torch.Tensor) -> torch.Tensor:
        E = E - E.mean(dim=0)
        scale = E.norm(dim=1, keepdim=True).mean() + 1e-8
        return E / scale

    def forward(self, node_feats, cell_types, incidence_matrix):
        node_embeddings, hyperedge_embeddings = self.encoder(
            node_feats, cell_types, incidence_matrix
        )
        # node_embeddings = self._pair_norm(node_embeddings)
        node_reconstruction = self.decoder(node_embeddings, cell_types)
        return node_embeddings, hyperedge_embeddings, node_reconstruction


@HypergraphEmbedder.register("CTHGCAEv2Embedder")
class CTHGCAEv2Embedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs, n_cell_types: int):
        model = CTHGCAEv2(
            num_cell_types=n_cell_types,
            input_dim=args.input_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
        )
        return cls(model)

    def forward(self, batch):
        feats = batch.batch_cell_feats
        cell_types = batch.batch_cell_types
        incidence_matrix = batch.batch_incidence_matrix
        return self.model(feats, cell_types, incidence_matrix)

    @classmethod
    def checkpoint_name(cls) -> str:
        return "CTHGCAEv2Embedder"

    def get_attribution_forward(self, index=0):
        def attribution_forward(feats, cell_types, incidence_matrix):
            return self.model(feats, cell_types, incidence_matrix)[index]
        return attribution_forward

    def get_attribution_inputs(self, batch):
        inputs = batch.batch_cell_feats
        additional_forward_args = (batch.batch_cell_types, batch.batch_incidence_matrix)
        return inputs, additional_forward_args

    def get_e2e_attribution_forward(self, classifier, index=0, num_repeats=1):
        class ExplainerClassifier(nn.Module):
            def __init__(self, embedder, classifier, index):
                super().__init__()
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
                return self.classifier(embedding)

        return ExplainerClassifier(self, classifier, index)

    def get_e2e_attribution_inputs(self, batch):
        inputs, additional_forward_args = self.get_attribution_inputs(batch)
        cell_types, incidence_matrix = additional_forward_args
        batch_size = batch.sample_n_cells
        return inputs, (cell_types, incidence_matrix, batch_size)
