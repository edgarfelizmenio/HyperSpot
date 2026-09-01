"""
CTHGConv -- attention-based analog of CTHGCAEv6, built on PyG's native hypergraph
attention operator `HypergraphConv(use_attention=True)`.

Mirrors v6's structure -- CT-specific input transform, fuse MLP, dual-objective
decoder (node + optional hyperedge reconstruction) -- but replaces v6's fixed
degree-normalised node->edge->node aggregation with HypergraphConv's hypergraph
attention. HypergraphConv returns node embeddings only (it fuses node->edge->node
internally and does not expose hyperedge embeddings), so the hyperedge representation
used both as the attention query (`hyperedge_attr`) and by the hyperedge-recon head
is a uniform member mean-pool.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import scatter as geo_scatter, softmax as geo_softmax

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from ..moduledict import LinearDict
from .cthgcaev3 import HyperedgeDecoder


def _incidence_pairs(incidence_matrix):
    # HypergraphConv expects hyperedge_index = [node_idx; hyperedge_idx];
    # incidence.nonzero() yields (node, hyperedge) rows.
    ei = incidence_matrix.nonzero(as_tuple=False)
    return ei[:, 0], ei[:, 1]


class ExposedHypergraphConv(HypergraphConv):
    """HypergraphConv that also returns the intermediate *attention-weighted*
    hyperedge embedding -- the result of the node->edge propagate that the base
    class computes and then discards. Forward mirrors PyG 2.7.0's HypergraphConv.
    Returns (node_out, hyperedge_emb)."""

    def forward(self, x, hyperedge_index, hyperedge_weight=None, hyperedge_attr=None,
                num_edges=None):
        num_nodes = x.size(0)
        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr).view(-1, self.heads, self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == 'node':
                alpha = geo_softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = geo_softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = geo_scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                        dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = geo_scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                        dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        # node -> edge: the attention-weighted hyperedge embedding (kept, not discarded)
        he = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                            size=(num_nodes, num_edges))
        # edge -> node: the usual node output
        out = self.propagate(hyperedge_index.flip([0]), x=he, norm=D, alpha=alpha,
                             size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
            he = he.reshape(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            he = he.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out, he


class CTHGConvEncoder(nn.Module):
    def __init__(self, n_cell_types, input_dim, embedding_dim, dropout=0.1, heads=4,
                 attention_mode="edge"):
        super().__init__()
        D = embedding_dim
        assert D % heads == 0, f"embedding_dim {D} not divisible by heads {heads}"
        self.W_ct = LinearDict(n_cell_types, input_dim, D, bias=True)
        self.conv = ExposedHypergraphConv(D, D // heads, use_attention=True,
                                          attention_mode=attention_mode, heads=heads,
                                          concat=True, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(D, D),
        )
        self.W_ct.reset_parameters()

    def forward(self, node_feats, cell_types, incidence_matrix):
        cell_idx, hedge_idx = _incidence_pairs(incidence_matrix)
        M = incidence_matrix.shape[1]
        hyperedge_index = torch.stack([cell_idx, hedge_idx])
        m = self.W_ct(node_feats, cell_types)                          # N, D
        # bootstrap query for the attention (hyperedges have no embedding yet)
        h_query = geo_scatter(m[cell_idx], hedge_idx, dim=0, dim_size=M, reduce='mean')
        # native hypergraph attention: node embeddings u AND the attention-weighted
        # hyperedge embedding h (exposed by the subclass, not a mean-pool)
        u, h = self.conv(m, hyperedge_index, hyperedge_attr=h_query, num_edges=M)
        z = self.fuse(torch.cat([u, m], dim=1))
        return z, h


class CTHGConvDecoder(nn.Module):
    """z -> (node_emb | m): m decodes per-cell expression; node_emb is mean-pooled to
    hyperedges and decoded to hyperedge feats (optional head)."""

    def __init__(self, n_cell_types, embedding_dim, output_dim, hyperedge_feat_dim=None,
                 dropout=0.1, heads=4, attention_mode="edge"):
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
        # hyperedge recon re-aggregates the decoded node embeddings to hyperedges
        # with the same attention-weighted operator (exposed), then decodes.
        self.he_conv = (
            ExposedHypergraphConv(D, D // heads, use_attention=True,
                                  attention_mode=attention_mode, heads=heads,
                                  concat=True, dropout=dropout) if hyperedge_feat_dim else None
        )
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
            cell_idx, hedge_idx = _incidence_pairs(incidence_matrix)
            M = incidence_matrix.shape[1]
            hyperedge_index = torch.stack([cell_idx, hedge_idx])
            he_query = geo_scatter(node_emb[cell_idx], hedge_idx, dim=0, dim_size=M, reduce='mean')
            _, he = self.he_conv(node_emb, hyperedge_index, hyperedge_attr=he_query, num_edges=M)
            hyperedge_reconstruction = self.he_out(he)
        return node_reconstruction, hyperedge_reconstruction


class CTHGConv(nn.Module):
    def __init__(self, num_cell_types, input_dim, embedding_dim, dropout=0.1,
                 hyperedge_feat_dim=None, hyperedge_pool="spot", heads=4,
                 attention_mode="edge"):
        super().__init__()
        self.encoder = CTHGConvEncoder(num_cell_types, input_dim, embedding_dim,
                                       dropout=dropout, heads=heads,
                                       attention_mode=attention_mode)
        self.decoder = CTHGConvDecoder(num_cell_types, embedding_dim, input_dim,
                                       hyperedge_feat_dim=hyperedge_feat_dim, dropout=dropout,
                                       heads=heads, attention_mode=attention_mode)

    def forward(self, node_feats, cell_types, incidence_matrix):
        z, h = self.encoder(node_feats, cell_types, incidence_matrix)
        node_reconstruction, hyperedge_reconstruction = self.decoder(
            z, cell_types, incidence_matrix
        )
        return z, h, node_reconstruction, hyperedge_reconstruction


@HypergraphEmbedder.register("CTHGConvEmbedder")
class CTHGConvEmbedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs, n_cell_types: int):
        model = CTHGConv(
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
        return "CTHGConvEmbedder"

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
