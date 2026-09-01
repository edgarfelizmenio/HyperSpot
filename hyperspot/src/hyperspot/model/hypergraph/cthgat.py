"""
CTHGAT — hypergraph *attention* encoder/decoder variant of CTHGCAEv6.

Same objective and interfaces as v6 (node + optional hyperedge reconstruction,
dense `(feats, cell_types, incidence)` forward), but the fixed degree-normalised
aggregation is replaced by learned two-level attention:

  node->edge : hyperedge embedding = attention-weighted sum of its member cells
  edge->node : cell embedding      = attention-weighted sum of its incident hyperedges

Attention is GAT-style additive with factorised src/dst scores (no N*M*D tensor),
multi-head, computed masked-dense over the batch incidence. Because the K/M
neighborhood sampling bounds the batch subgraph, the dense N*M*heads score tensor
stays small; full-graph inference is the case to watch (a sparse scatter over
hyperedge_index is the drop-in scale-up). The node-reconstruction head is
unchanged from v6; only the aggregation is attentional.
"""
import torch
import torch.nn as nn

from torch_geometric.utils import softmax as geo_softmax, scatter as geo_scatter

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from ..moduledict import LinearDict
from .cthgcaev3 import HyperedgeDecoder


class HGATConv(nn.Module):
    """Attention aggregation of source units into target units over a binary
    incidence `inc` (Nsrc x Ntgt): target_t = Σ_s α_{s,t} · W·src_s, with α a
    softmax over the sources incident to each target. Multi-head; the per-target
    query bootstraps from a uniform mean of its members (targets have no embedding
    of their own at this layer)."""

    def __init__(self, dim, heads=4, dropout=0.0, negative_slope=0.2):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"
        self.heads = heads
        self.dhead = dim // heads
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dhead))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dhead))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.leaky = nn.LeakyReLU(negative_slope)
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.zeros_(self.bias)

    def forward(self, src, inc):
        # inc: (Nsrc, Ntgt) binary incidence. Sparse message passing over its
        # nonzero (source, target) pairs only -- cost scales with nnz, not Nsrc*Ntgt.
        Nsrc, Ntgt = inc.shape
        ei = inc.nonzero(as_tuple=False)                           # E x 2: (src, tgt)
        s_idx, t_idx = ei[:, 0], ei[:, 1]

        x = self.W(src).view(Nsrc, self.heads, self.dhead)         # Nsrc, H, dh
        xs = x[s_idx]                                              # E, H, dh (message per edge)
        # per-target query: uniform mean of member sources (bootstrap)
        q = geo_scatter(xs, t_idx, dim=0, dim_size=Ntgt, reduce='mean')      # Ntgt, H, dh
        # GAT-style additive score, factorised over the edge endpoints
        e = self.leaky((xs * self.a_src).sum(-1) + (q[t_idx] * self.a_dst).sum(-1))  # E, H
        alpha = geo_softmax(e, t_idx, num_nodes=Ntgt)              # softmax over sources per target
        alpha = self.drop(alpha)
        msg = alpha.unsqueeze(-1) * xs                             # E, H, dh
        out = geo_scatter(msg, t_idx, dim=0, dim_size=Ntgt, reduce='sum')    # Ntgt, H, dh
        return out.reshape(Ntgt, -1) + self.bias                  # Ntgt, D


class CTHGATEncoder(nn.Module):
    """cell expr --ct transform--> common space --attn(node->edge)--> hyperedge emb
    --attn(edge->node)--> node emb; fuse[u, m] -> D."""

    def __init__(self, n_cell_types, input_dim, embedding_dim, dropout=0.1, heads=4,
                 skip_source="common"):
        super().__init__()
        D = embedding_dim
        self.W_ct = LinearDict(n_cell_types, input_dim, D, bias=True)
        self.node_to_edge = HGATConv(D, heads=heads, dropout=dropout)   # node->edge
        self.edge_to_node = HGATConv(D, heads=heads, dropout=dropout)   # edge->node
        self.act = nn.PReLU()
        self.fuse = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(D, D),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.W_ct.reset_parameters()

    def forward(self, node_feats, cell_types, incidence_matrix):
        H = incidence_matrix
        m = self.W_ct(node_feats, cell_types)                      # N, D
        h = self.node_to_edge(m, H)                                # M, D  softmax over member cells
        u = self.edge_to_node(self.act(h), H.t())                  # N, D  softmax over incident hyperedges
        z = self.fuse(torch.cat([u, m], dim=1))
        return z, h


class CTHGATDecoder(nn.Module):
    """z --split--> (node_emb | m); m --ct transform--> node expr;
    node_emb --attn(node->edge)--> hyperedge expr (optional)."""

    def __init__(self, n_cell_types, embedding_dim, output_dim, hyperedge_feat_dim=None,
                 dropout=0.1, heads=4):
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
        self.node_to_edge = HGATConv(D, heads=heads, dropout=dropout)
        self.he_out = (
            HyperedgeDecoder(D, hyperedge_feat_dim, dropout=dropout) if hyperedge_feat_dim else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.output_proj.reset_parameters()

    def forward(self, node_embeddings, cell_types, incidence_matrix):
        H = incidence_matrix
        split = self.split(node_embeddings)
        node_emb, m = split[:, :self.D], split[:, self.D:]
        node_reconstruction = self.output_proj(m, cell_types)
        hyperedge_reconstruction = None
        if self.he_out is not None:
            he = self.node_to_edge(node_emb, H)                    # M, D
            hyperedge_reconstruction = self.he_out(he)
        return node_reconstruction, hyperedge_reconstruction


class CTHGAT(nn.Module):
    def __init__(self, num_cell_types, input_dim, embedding_dim, dropout=0.1,
                 hyperedge_feat_dim=None, hyperedge_pool="spot", heads=4,
                 skip_source="common"):
        super().__init__()
        self.encoder = CTHGATEncoder(num_cell_types, input_dim, embedding_dim,
                                     dropout=dropout, heads=heads, skip_source=skip_source)
        self.decoder = CTHGATDecoder(num_cell_types, embedding_dim, input_dim,
                                     hyperedge_feat_dim, dropout=dropout, heads=heads)

    def forward(self, node_feats, cell_types, incidence_matrix):
        z, h = self.encoder(node_feats, cell_types, incidence_matrix)
        node_reconstruction, hyperedge_reconstruction = self.decoder(
            z, cell_types, incidence_matrix
        )
        return z, h, node_reconstruction, hyperedge_reconstruction


@HypergraphEmbedder.register("CTHGATEmbedder")
class CTHGATEmbedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs, n_cell_types: int):
        model = CTHGAT(
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
        feats = batch.batch_cell_feats
        cell_types = batch.batch_cell_types
        incidence_matrix = batch.batch_incidence_matrix
        return self.model(feats, cell_types, incidence_matrix)

    @classmethod
    def checkpoint_name(cls) -> str:
        return "CTHGATEmbedder"

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
