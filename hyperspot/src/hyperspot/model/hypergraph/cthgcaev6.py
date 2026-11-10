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

    def __init__(self, embedding_dim, output_dim, dropout=0.0, act=nn.PReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            act(),
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
    """
    cell expression --ct transform--> cell in common space (m)
    m (input-normalized) --node conv (node->edge)--> hyperedge embeddings
    hyperedge emb   --hyperedge conv (edge->node)--> node/neighborhood branch (u)
    fuse(u, m) --> final node embeddings (D)

    u (neighborhood) and m (own-expression skip) are each normalized (fuse_norm)
    then combined by skip_mode: "add" (residual u+m), "concat" (learned MLP), or
    "cross" (2-token self-attention). input_norm normalizes m before the conv.
    """

    def __init__(self, n_cell_types, input_dim, embedding_dim, dropout=0.1,
                 skip_source="common", skip_mode="add", fuse_norm="layer",
                 input_norm="ct", reuse_input_norm=False, num_heads=2):
        super().__init__()
        assert skip_source in ("common", "expression")
        assert skip_mode in ("concat", "add", "cross")
        assert fuse_norm in ("none", "layer", "ct")
        self.skip_source = skip_source
        self.skip_mode = skip_mode          # how u (neighborhood) and m (skip) combine
        self.fuse_norm = fuse_norm          # per-branch normalization before combining
        self.input_norm = input_norm        # per-CT norm of m BEFORE the aggregation
        # if True, the input-normalized m (m_in) is REUSED for the skip branch and
        # the skip is NOT normalized again at fusion. "normalize m once, share for
        # conv + skip" vs the default which normalizes m separately for conv
        # (input_norm) and skip (fuse_norm).
        self.reuse_input_norm = reuse_input_norm
        D = embedding_dim

        self.W_ct = LinearDict(n_cell_types, input_dim, D, bias=True)  # ct transform
        # Per-CT normalization of m *before* node_conv: a hyperedge pools multiple
        # cell types; without per-CT centering the abundant type dominates the
        # hyperedge embedding, drowning cross-type signal. Affects only what the
        # conv sees; the skip branch uses raw m unless reuse_input_norm.
        self.in_ct = CTNorm(D, n_cell_types) if "ct" in input_norm else None
        self.in_ln = nn.LayerNorm(D) if "layer" in input_norm else None
        self.in_bn = nn.BatchNorm1d(D) if "batch" in input_norm else None
        self.node_conv = nn.Linear(D, D, bias=False)                   # node->edge
        self.edge_bias = nn.Parameter(torch.zeros((D,)))
        self.hyperedge_conv = nn.Linear(D, D, bias=False)              # edge->node
        self.node_bias = nn.Parameter(torch.zeros((D,)))
        self.act = nn.PReLU()

        # Per-branch normalization. Without it the skip branch m (own-expression)
        # enters with ~2x the norm of the neighborhood branch u and dominates z,
        # washing out u's stronger neighborhood signal. "layer" = LayerNorm
        # (per-cell), "ct" = per-cell-type.
        if fuse_norm == "layer":
            self.norm_u, self.norm_m = nn.LayerNorm(D), nn.LayerNorm(D)
        elif fuse_norm == "ct":
            self.norm_u, self.norm_m = CTNorm(D, n_cell_types), CTNorm(D, n_cell_types)
        else:
            self.norm_u, self.norm_m = nn.Identity(), nn.Identity()

        # "concat" -> learned fuse MLP over [u, m]; "add" -> plain residual u + m
        # (forces both branches to contribute equally; no MLP to learn to ignore u).
        self.fuse = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(D, D),
        ) if skip_mode == "concat" else None
        # "cross" -> treat [u, m] as a 2-token sequence and self-attend (each attends
        # to the other), standard transformer block (residual + attn), combined by
        # mean. Tests whether learnable attention routing collapses onto m like concat.
        self.mha = (
            nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)
            if skip_mode == "cross" else None
        )
        self.reset_parameters()

    def _input_norm(self, m, cell_types):
        # applies input_norm to m: "ct", "layer", or "ct_layer" (CTNorm then LayerNorm)
        if self.in_ct is not None:
            m = self.in_ct(m, cell_types)
        if self.in_bn is not None:
            m = self.in_bn(m)
        if self.in_ln is not None:
            m = self.in_ln(m)
        return m

    def _norm(self, branch, x, cell_types):
        mod = self.norm_u if branch == "u" else self.norm_m
        return mod(x, cell_types) if self.fuse_norm == "ct" else mod(x)

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
        # normalize before aggregation (isolates the conv input; skip uses raw m
        # unless reuse_input_norm composes fuse_norm on top of m_in)
        m_in = self._input_norm(m, cell_types) if self.input_norm != "none" else m
        # node convolution (node->edge) -> hyperedge embeddings
        h = dei.unsqueeze(1) * (H.t() @ (dvi.unsqueeze(1) * self.node_conv(m_in))) + self.edge_bias
        # hyperedge convolution (edge->node) -> node embeddings
        u = dvi.unsqueeze(1) * (H @ self.hyperedge_conv(self.act(h))) + self.node_bias
        # skip branch: reuse the input-normalized m_in (normalize m once), or
        # normalize raw m separately at fusion. u always gets its fuse_norm.
        un = self._norm("u", u, cell_types)
        if self.reuse_input_norm and self.input_norm != "none":
            # compose: skip = fuse_norm(input_norm(m)) -- e.g. LayerNorm(CTNorm(m))
            mn = self._norm("m", m_in, cell_types)
        else:
            # parallel: skip = fuse_norm(raw m); input_norm only shapes the conv path
            mn = self._norm("m", m, cell_types)
        if self.skip_mode == "add":
            z = un + mn
        elif self.skip_mode == "concat":
            z = self.fuse(torch.cat([un, mn], dim=1))
        else:  # cross: 2-token self-attention over [u, m], residual, mean-combine
            seq = torch.stack([un, mn], dim=1)          # (N, 2, D)
            seq = seq + self.mha(seq, seq, seq, need_weights=False)[0]
            z = seq.mean(dim=1)                         # (N, D)
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
                 hyperedge_feat_dim=None, hyperedge_pool="spot", skip_source="common",
                 skip_mode="add", fuse_norm="layer", input_norm="ct",
                 reuse_input_norm=False, num_heads=2):
        super().__init__()
        self.encoder = CTHGEncoder(num_cell_types, input_dim, embedding_dim,
                                   dropout=dropout, skip_source=skip_source,
                                   skip_mode=skip_mode, fuse_norm=fuse_norm,
                                   input_norm=input_norm, reuse_input_norm=reuse_input_norm,
                                   num_heads=num_heads)
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
        return "CTHGCAEv6Embedder"

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
