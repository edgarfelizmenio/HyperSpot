"""
CTGATv2 — hypergraph GATv2 encoder/decoder built on PyG's GATv2Conv.

Two-level *dynamic* attention (GATv2) over the bipartite cell<->hyperedge incidence:
  node->edge : cells  --GATv2Conv--> hyperedge embeddings  (query bootstrapped from a
               uniform member mean, then refined by attention)
  edge->node : hyperedges --GATv2Conv--> cell embeddings

Node (expression) reconstruction, plus an optional hyperedge-reconstruction head
that turns on when hyperedge_feat_dim is set (i.e. when the data carries hyperedge
features and you train with --optimize-hyperedges). The hyperedge head re-aggregates
the decoder's node embeddings to hyperedges with a GATv2 node->edge step. Same
`(feats, cell_types, incidence)` interface and 4-tuple output as v6/CTHGAT, so it
drops into the pipeline and the K/M sampling unchanged. Uses PyG's GATv2Conv for the
attention (distinct from the hand-rolled HGATConv variant, which is kept as-is).
"""
import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter as geo_scatter

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from ..moduledict import LinearDict
from .cthgcaev3 import HyperedgeDecoder
from .cthgcaev6 import CTNorm   # winning normalization ported from v6


class CTGATv2Encoder(nn.Module):
    """Attention only on edge->node -- the direction that forms the cell embedding
    (what gets used downstream). node->edge is a uniform mean-pool: it matches the
    mean pseudobulk hyperedge-reconstruction target and halves the attention params
    (easier to train)."""

    def __init__(self, n_cell_types, input_dim, embedding_dim, dropout=0.1, heads=4):
        super().__init__()
        D = embedding_dim
        assert D % heads == 0, f"embedding_dim {D} not divisible by heads {heads}"
        self.W_ct = LinearDict(n_cell_types, input_dim, D, bias=True)
        # WINNER normalization ported from v6:
        #  - CTNorm before the aggregation (inter-CT pooling) balances CT contributions
        #  - additive LayerNorm skip (per-cell) replaces the fuse MLP, which collapses
        self.in_ct = CTNorm(D, n_cell_types)
        self.edge_to_node = GATv2Conv((D, D), D // heads, heads=heads, concat=True,
                                      dropout=dropout, add_self_loops=False)
        self.act = nn.PReLU()
        self.norm_u, self.norm_m = nn.LayerNorm(D), nn.LayerNorm(D)
        self.W_ct.reset_parameters()

    def forward(self, node_feats, cell_types, incidence_matrix):
        H = incidence_matrix                                       # N cells x M hyperedges
        _, M = H.shape
        ei = H.nonzero(as_tuple=False)                            # E x 2: (cell, hyperedge)
        cell_idx, hedge_idx = ei[:, 0], ei[:, 1]

        m = self.W_ct(node_feats, cell_types)                     # N, D
        m_in = self.in_ct(m, cell_types)                          # per-CT norm before aggregation
        # node->edge: mean-pool of CT-balanced m_in -> hyperedge embedding
        h = geo_scatter(m_in[cell_idx], hedge_idx, dim=0, dim_size=M, reduce='mean')    # M, D
        # edge->node: GATv2 attention over a cell's incident hyperedges (CT-balanced query)
        u = self.edge_to_node((self.act(h), m_in), torch.stack([hedge_idx, cell_idx]))  # N, D
        # additive normalized skip: LayerNorm(u) + LayerNorm(raw m), parallel (not stacked)
        z = self.norm_u(u) + self.norm_m(m)
        return z, h


class CTGATv2Decoder(nn.Module):
    """Node (expression) reconstruction, plus an optional hyperedge-reconstruction
    head (enabled when hyperedge_feat_dim is set). z -> (node_emb | m): m decodes
    per-cell expression; node_emb is re-aggregated to hyperedges via a GATv2
    node->edge step (a real attention aggregation) and decoded to hyperedge feats."""

    def __init__(self, n_cell_types, embedding_dim, output_dim, hyperedge_feat_dim=None,
                 dropout=0.1):
        super().__init__()
        self.D = embedding_dim
        D = embedding_dim
        self.split = nn.Sequential(
            nn.Linear(D, D),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(D, 2 * D),
        )
        self.output_proj = LinearDict(n_cell_types, D, output_dim, bias=True)
        # node->edge is a uniform mean-pool (no attention), matching the mean
        # pseudobulk target; only edge->node in the encoder carries attention.
        self.he_out = (
            HyperedgeDecoder(D, hyperedge_feat_dim, dropout=dropout) if hyperedge_feat_dim else None
        )
        self.output_proj.reset_parameters()

    def forward(self, node_embeddings, cell_types, incidence_matrix):
        split = self.split(node_embeddings)
        node_emb, m = split[:, :self.D], split[:, self.D:]
        node_reconstruction = self.output_proj(m, cell_types)
        hyperedge_reconstruction = None
        if self.he_out is not None:
            H = incidence_matrix
            _, M = H.shape
            ei = H.nonzero(as_tuple=False)
            cell_idx, hedge_idx = ei[:, 0], ei[:, 1]
            he = geo_scatter(node_emb[cell_idx], hedge_idx, dim=0, dim_size=M, reduce='mean')
            hyperedge_reconstruction = self.he_out(he)
        return node_reconstruction, hyperedge_reconstruction


class CTGATv2(nn.Module):
    def __init__(self, num_cell_types, input_dim, embedding_dim, dropout=0.1,
                 hyperedge_feat_dim=None, hyperedge_pool="spot", heads=4):
        super().__init__()
        self.encoder = CTGATv2Encoder(num_cell_types, input_dim, embedding_dim,
                                      dropout=dropout, heads=heads)
        self.decoder = CTGATv2Decoder(num_cell_types, embedding_dim, input_dim,
                                      hyperedge_feat_dim=hyperedge_feat_dim,
                                      dropout=dropout)

    def forward(self, node_feats, cell_types, incidence_matrix):
        z, h = self.encoder(node_feats, cell_types, incidence_matrix)
        node_reconstruction, hyperedge_reconstruction = self.decoder(
            z, cell_types, incidence_matrix
        )
        return z, h, node_reconstruction, hyperedge_reconstruction


@HypergraphEmbedder.register("CTGATv2Embedder")
class CTGATv2Embedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs, n_cell_types: int):
        model = CTGATv2(
            num_cell_types=n_cell_types,
            input_dim=args.input_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            hyperedge_feat_dim=args.hyperedge_feat_dim,
            hyperedge_pool=args.hyperedge_pool,
            heads=args.num_heads or 4,
        )
        return cls(model)

    def forward(self, batch):
        return self.model(batch.batch_cell_feats, batch.batch_cell_types,
                          batch.batch_incidence_matrix)

    @classmethod
    def checkpoint_name(cls) -> str:
        return "CTGATv2Embedder"

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
