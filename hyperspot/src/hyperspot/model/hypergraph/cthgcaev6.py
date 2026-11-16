import torch
import torch.nn as nn

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from ..moduledict import LinearDict


class HyperedgeDecoder(nn.Module):
    """Decode a hyperedge embedding back to its spot-basis pseudobulk target.

    Hyperedges are not cell-type-specific, so this is a plain shared MLP
    (embedding_dim → output_dim = small global HVG count). Used to give the
    otherwise-unsupervised hyperedge embeddings a direct reconstruction signal.
    """

    def __init__(self, embedding_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, hyperedge_embeddings):
        return self.net(hyperedge_embeddings)


def _norms(H):
    dv = H.sum(dim=1)
    de = H.sum(dim=0)
    dvi = dv ** -0.5
    dvi[dvi == float('inf')] = 0
    dei = de ** -1.0
    dei[dei == float('inf')] = 0
    return dvi, dei


class CTNorm(nn.Module):
    """Per-cell-type normalization with RUNNING statistics (a per-CT BatchNorm).

    Standardizes each feature within its cell-type group, but — unlike naive
    batch-per-CT stats — accumulates a running mean/var per cell type across
    batches (momentum), so rare cell types build a stable estimate over time
    instead of relying on the few (or single) samples present in one minibatch.

    Robustness to few-samples-per-CT (the whole point):
      * training: normalize with the *current* running stats, then update them
        from this batch. Groups with < min_count samples do NOT update the
        running stats (their batch estimate is too noisy) but are still
        normalized by the existing running stats.
      * a single sample (var undefined) therefore never wipes information — it is
        normalized by the accumulated running var, not a degenerate batch var.
      * eval: always use running stats (no update) -> train/eval consistent, and
        matches full-graph inference.
    """

    def __init__(self, dim, n_cell_types, eps=1e-5, momentum=0.1, min_count=8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.momentum = momentum
        self.min_count = min_count
        self.register_buffer("running_mean", torch.zeros(n_cell_types, dim))
        self.register_buffer("running_var", torch.ones(n_cell_types, dim))
        self.register_buffer("initialized", torch.zeros(n_cell_types, dtype=torch.bool))
        # global (all-CT) running stat: the fallback for a cell type that has not
        # yet accumulated its own stable per-CT estimate. A single rare-CT sample
        # is normalized by this global distribution rather than by itself (which
        # would center it to exactly zero and wipe its information).
        self.register_buffer("global_mean", torch.zeros(dim))
        self.register_buffer("global_var", torch.ones(dim))
        self.register_buffer("global_init", torch.zeros(1, dtype=torch.bool))

    def forward(self, x, cell_types):
        if self.training and x.shape[0] >= 2:           # keep the global fallback fresh
            gmu, gvar = x.mean(0).detach(), x.var(0, unbiased=False).detach()
            if self.global_init:
                self.global_mean.mul_(1 - self.momentum).add_(self.momentum * gmu)
                self.global_var.mul_(1 - self.momentum).add_(self.momentum * gvar)
            else:
                self.global_mean.copy_(gmu); self.global_var.copy_(gvar); self.global_init[0] = True

        out = torch.empty_like(x)
        for c in torch.unique(cell_types):
            ci = int(c)
            mask = cell_types == c
            xc = x[mask]
            n = xc.shape[0]
            if self.training and n >= self.min_count:
                mu, var = xc.mean(0).detach(), xc.var(0, unbiased=False).detach()
                if self.initialized[ci]:
                    self.running_mean[ci].mul_(1 - self.momentum).add_(self.momentum * mu)
                    self.running_var[ci].mul_(1 - self.momentum).add_(self.momentum * var)
                else:                                   # first stable batch seeds the per-CT stat
                    self.running_mean[ci].copy_(mu); self.running_var[ci].copy_(var)
                    self.initialized[ci] = True
            # per-CT running stat if we have it, else the global running stat
            # (never the degenerate single-sample batch stat)
            if self.initialized[ci]:
                mu_use, var_use = self.running_mean[ci], self.running_var[ci]
            else:
                mu_use, var_use = self.global_mean, self.global_var
            out[mask] = (xc - mu_use) / torch.sqrt(var_use + self.eps)
        return out * self.weight + self.bias


class CTHGEncoder(nn.Module):
    """Heterogeneous hypergraph encoder (manuscript Eq. 2).

      P0  = PerCTTransform(X, c)             per-cell-type linear projection to d
      P0~ = PerCTNorm(P0, c)                 CTNorm, applied before the aggregation only
      h   = node -> hyperedge convolution    hyperedge embeddings
      P   = hyperedge -> node convolution    neighbourhood branch
      Z   = LN_P(P) + LN_0(P0)               residual fusion; the skip uses the unnormalised P0

    Returns (Z, h, LN_P(P)).
    """

    def __init__(self, n_cell_types, input_dim, embedding_dim):
        super().__init__()
        D = embedding_dim
        self.W_ct = LinearDict(n_cell_types, input_dim, D, bias=True)  # PerCTTransform
        # PerCTNorm before the aggregation: a hyperedge pools several cell types; without per-CT
        # centring the abundant type dominates the hyperedge embedding.
        self.in_ct = CTNorm(D, n_cell_types)
        self.node_conv = nn.Linear(D, D, bias=False)                   # node->edge
        self.edge_bias = nn.Parameter(torch.zeros((D,)))
        self.hyperedge_conv = nn.Linear(D, D, bias=False)              # edge->node
        self.node_bias = nn.Parameter(torch.zeros((D,)))
        self.act = nn.PReLU()
        # LN_P and LN_0: without them the skip branch enters with ~2x the norm of the
        # neighbourhood branch and dominates Z.
        self.norm_u, self.norm_m = nn.LayerNorm(D), nn.LayerNorm(D)
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

        m = self.W_ct(node_feats, cell_types)                       # P0
        m_in = self.in_ct(m, cell_types)                            # P0~
        h = dei.unsqueeze(1) * (H.t() @ (dvi.unsqueeze(1) * self.node_conv(m_in))) + self.edge_bias
        u = dvi.unsqueeze(1) * (H @ self.hyperedge_conv(self.act(h))) + self.node_bias
        un = self.norm_u(u)                                         # LN_P(P)
        z = un + self.norm_m(m)                                     # + LN_0(P0)
        return z, h, un


class CTHGDecoder(nn.Module):
    """
    z          --shared MLP--> (node_emb | m)    split the compressed code back out
    m          --ct-specific transform-->        decoded node expression
    node_emb   --hyperedge_conv (node->edge)-->  hyperedge expression
    """

    def __init__(self, n_cell_types, embedding_dim, output_dim, hyperedge_feat_dim=None,
                 dropout=0.1):
        super().__init__()
        self.D = embedding_dim
        D = embedding_dim
        self.split = nn.Sequential(                                     # z -> (node_emb | m)
            nn.Linear(D, D),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(D, 2 * D),
        )
        self.output_proj = LinearDict(n_cell_types, D, output_dim, bias=True)  # m -> ct-specific node expr
        self.he_conv = nn.Linear(D, D, bias=False)                     # node->edge
        self.he_bias = nn.Parameter(torch.zeros((D,)))
        # built only for training (the driver sets hyperedge_feat_dim from the hypergraph);
        # inference needs only z, so the hyperedge head is skipped there
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
        # z -> shared MLP -> (node_emb | m)
        split = self.split(node_embeddings)
        node_emb, m = split[:, :self.D], split[:, self.D:]
        # m -> ct-specific transform -> decoded node expression
        node_reconstruction = self.output_proj(m, cell_types)
        # node_emb -> hyperedge_conv (node->edge) -> hyperedge expression
        hyperedge_reconstruction = None
        if self.he_out is not None:
            he = dei.unsqueeze(1) * (H.t() @ (dvi.unsqueeze(1) * self.he_conv(node_emb))) + self.he_bias
            hyperedge_reconstruction = self.he_out(he)
        return node_reconstruction, hyperedge_reconstruction


class CTHGCAEv6(nn.Module):

    def __init__(self, num_cell_types, input_dim, embedding_dim, dropout=0.1,
                 hyperedge_feat_dim=None):
        super().__init__()
        self.encoder = CTHGEncoder(num_cell_types, input_dim, embedding_dim)
        self.decoder = CTHGDecoder(num_cell_types, embedding_dim, input_dim,
                                   hyperedge_feat_dim, dropout=dropout)

    def forward(self, node_feats, cell_types, incidence_matrix):
        z, h, un = self.encoder(node_feats, cell_types, incidence_matrix)
        node_reconstruction, hyperedge_reconstruction = self.decoder(
            z, cell_types, incidence_matrix
        )
        # un (post-norm neighborhood branch) appended at index 4 so indices 0-3
        # (z, h, node_recon, he_recon) stay stable for all positional consumers.
        return z, h, node_reconstruction, hyperedge_reconstruction, un


@HypergraphEmbedder.register("CTHGCAEv6Embedder")
class CTHGCAEv6Embedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs, n_cell_types: int):
        model = CTHGCAEv6(
            num_cell_types=n_cell_types,
            input_dim=args.input_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            hyperedge_feat_dim=args.hyperedge_feat_dim,
        )
        return cls(model)

    def forward(self, batch):
        feats = batch.batch_cell_feats
        cell_types = batch.batch_cell_types
        incidence_matrix = batch.batch_incidence_matrix
        return self.model(feats, cell_types, incidence_matrix)

    @classmethod
    def checkpoint_name(cls) -> str:
        return "CTHGCAEv6Embedder"
