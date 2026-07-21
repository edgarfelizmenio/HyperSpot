import torch
import torch.nn as nn

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from ..moduledict import LinearDict
from .cthgcaev3 import HyperedgeDecoder


def _norms(H):
    dv = H.sum(dim=1)
    de = H.sum(dim=0)
    dvi = dv ** -0.5
    dvi[dvi == float('inf')] = 0
    dei = de ** -1.0
    dei[dei == float('inf')] = 0
    return dvi, dei


class CTHGEncoderV7(nn.Module):
    """
    Uncompressed variant of v6: the final node embedding is the *concatenation*
    of the neighborhood embedding and the common-space expression, so the two
    factors stay separable for the downstream PCA + clustering (no fusion MLP).

    cell expression --ct transform--> m (cell in common space)
    m               --node conv (node->edge)--> h (hyperedge embeddings)
    h               --hyperedge conv (edge->node)--> u (node embeddings)
    z = concat[u, m]   (2D)   final node embeddings
    """

    def __init__(self, n_cell_types, input_dim, embedding_dim, dropout=0.1):
        super().__init__()
        D = embedding_dim
        self.D = D
        self.W_ct = LinearDict(n_cell_types, input_dim, D, bias=True)  # ct transform
        self.node_conv = nn.Linear(D, D, bias=False)                   # node->edge
        self.edge_bias = nn.Parameter(torch.zeros((D,)))
        self.hyperedge_conv = nn.Linear(D, D, bias=False)              # edge->node
        self.node_bias = nn.Parameter(torch.zeros((D,)))
        self.act = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.W_ct.reset_parameters()
        nn.init.xavier_uniform_(self.node_conv.weight)
        nn.init.xavier_uniform_(self.hyperedge_conv.weight)
        nn.init.zeros_(self.edge_bias)
        nn.init.zeros_(self.node_bias)

    def forward(self, node_feats, cell_types, incidence_matrix):
        H = incidence_matrix
        dvi, dei = _norms(H)
        m = self.W_ct(node_feats, cell_types)                       # cell in common space
        h = dei.unsqueeze(1) * (H.t() @ (dvi.unsqueeze(1) * self.node_conv(m))) + self.edge_bias
        u = dvi.unsqueeze(1) * (H @ self.hyperedge_conv(self.act(h))) + self.node_bias
        z = torch.cat([u, m], dim=1)                                # uncompressed: 2D
        return z, h


class CTHGDecoderV7(nn.Module):
    """
    Split z by slicing (exact, no learned splitter):
        u = z[:, :D]   m = z[:, D:]
    m --ct-specific transform--> decoded node expression
    u --hyperedge_conv (node->edge)--> hyperedge expression
    """

    def __init__(self, n_cell_types, embedding_dim, output_dim, hyperedge_feat_dim=None,
                 dropout=0.1):
        super().__init__()
        D = embedding_dim
        self.D = D
        self.output_proj = LinearDict(n_cell_types, D, output_dim, bias=True)  # m -> ct-specific node expr
        self.he_conv = nn.Linear(D, D, bias=False)                     # node->edge
        self.he_bias = nn.Parameter(torch.zeros((D,)))
        self.he_out = (
            HyperedgeDecoder(D, hyperedge_feat_dim, dropout=dropout) if hyperedge_feat_dim else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.he_conv.weight)
        nn.init.zeros_(self.he_bias)
        self.output_proj.reset_parameters()

    def forward(self, node_embeddings, cell_types, incidence_matrix):
        H = incidence_matrix
        dvi, dei = _norms(H)
        # exact split by slicing
        u, m = node_embeddings[:, :self.D], node_embeddings[:, self.D:]
        # m -> ct-specific transform -> decoded node expression
        node_reconstruction = self.output_proj(m, cell_types)
        # u -> node->edge -> hyperedge expression
        hyperedge_reconstruction = None
        if self.he_out is not None:
            he = dei.unsqueeze(1) * (H.t() @ (dvi.unsqueeze(1) * self.he_conv(u))) + self.he_bias
            hyperedge_reconstruction = self.he_out(he)
        return node_reconstruction, hyperedge_reconstruction


class CTHGCAEv7(nn.Module):
    """v7: v6 with an *uncompressed* concat node embedding (z = [u | m], 2D) and
    a slice-based decoder split. Keeps the two factors linearly separable for the
    downstream PCA + clustering."""

    def __init__(self, num_cell_types, input_dim, embedding_dim, dropout=0.1,
                 hyperedge_feat_dim=None, hyperedge_pool="spot"):
        super().__init__()
        self.encoder = CTHGEncoderV7(num_cell_types, input_dim, embedding_dim, dropout=dropout)
        self.decoder = CTHGDecoderV7(num_cell_types, embedding_dim, input_dim,
                                     hyperedge_feat_dim, dropout=dropout)

    def forward(self, node_feats, cell_types, incidence_matrix):
        z, h = self.encoder(node_feats, cell_types, incidence_matrix)
        node_reconstruction, hyperedge_reconstruction = self.decoder(
            z, cell_types, incidence_matrix
        )
        return z, h, node_reconstruction, hyperedge_reconstruction


@HypergraphEmbedder.register("CTHGCAEv7Embedder")
class CTHGCAEv7Embedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs, n_cell_types: int):
        model = CTHGCAEv7(
            num_cell_types=n_cell_types,
            input_dim=args.input_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            hyperedge_feat_dim=args.hyperedge_feat_dim,
            hyperedge_pool=args.hyperedge_pool,
        )
        return cls(model)

    def forward(self, batch):
        feats = batch.batch_cell_feats
        cell_types = batch.batch_cell_types
        incidence_matrix = batch.batch_incidence_matrix
        return self.model(feats, cell_types, incidence_matrix)

    @classmethod
    def checkpoint_name(cls) -> str:
        return "CTHGCAEv7Embedder"

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
