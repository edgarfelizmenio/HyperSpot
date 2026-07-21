import torch
import torch.nn as nn
import torch.nn.functional as F

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from ..moduledict import LinearDict, ParameterDict
from .cthgcaev3 import CTHGMirrorDecoder, HyperedgeDecoder, _own_spot_pool, _neighborhood_pool


class CTTypeAttnEncoder(nn.Module):
    """
    Type-aware encoder with **self-attention over the K type-channels** (K→K).

    Extends the v4 idea (resolve each hyperedge into K per-type channels so rare
    types survive) with three additions that close v4's gaps:

    1. **Cross-type interaction** — a multi-head self-attention over the K type
       fields *within each hyperedge* (params global, attention weights computed
       per hyperedge from field content, absent types masked). Unlike v4's static
       K×K gate, this is data-dependent and models source×source interaction, not
       just a linear remix of marginals.
    2. **Composition / proportions** — a raw-sum blend channel ``E_sum`` (the
       abundance-weighted mixture the per-type *mean* pooling divides out) is fused
       into the hyperedge embedding, so the domain-clustering representation keeps
       cell-type proportion information.
    3. **Self-identity** — a bottlenecked residual skip of the cell's own message
       (``alpha * m_i``) into the node embedding, fighting over-smoothing without
       letting the autoencoder reconstruct through the skip.

    The static K×K ``gate`` is retained for cheap **reader-type conditioning** of
    the node readback (each reader type weights the interaction-refined source
    channels); the attention handles source×source interaction, the gate handles
    reader×source conditioning — one job each.
    """

    def __init__(self, n_cell_types, input_dim, embedding_dim, dropout=0.1,
                 num_heads=4, act=nn.PReLU):
        super().__init__()
        K, D = n_cell_types, embedding_dim
        if D % num_heads != 0:
            # fall back to a head count that divides D
            num_heads = next((h for h in (4, 2, 1) if D % h == 0), 1)
        self.n_cell_types = K

        # CT input adapter W_ct (per-cell-type), shared node refine
        self.hyperedge_emb_weight = LinearDict(K, input_dim, D, bias=False)
        self.hyperedge_emb_bias = nn.Parameter(torch.zeros((D,)))
        self.node_emb_weight = nn.Linear(D, D, bias=False)
        self.node_emb_bias = ParameterDict(K, D)

        # (1) self-attention over the K type-fields (interaction), pre-normed
        self.field_norm = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, num_heads, dropout=dropout, batch_first=True)

        # reader-type conditioning: cheap static K×K gate (identity init)
        self.gate = nn.Parameter(torch.eye(K))

        # (2) raw-sum blend fused into the hyperedge embedding
        self.blend_norm = nn.LayerNorm(D)
        self.hedge_fuse = nn.Linear(2 * D, D)

        # (3) bottlenecked self-identity skip (small init so the pool stays primary)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(dropout)
        self.act = act()
        self.reset_parameters()

    def reset_parameters(self):
        self.hyperedge_emb_weight.reset_parameters()
        nn.init.xavier_uniform_(self.node_emb_weight.weight)
        self.node_emb_bias.reset_parameters()
        nn.init.zeros_(self.hyperedge_emb_bias)
        with torch.no_grad():
            self.gate.copy_(torch.eye(self.n_cell_types))
            self.alpha.fill_(0.1)

    def regularization_loss(self):
        """L1 on the gate off-diagonal (keeps reader-type conditioning sparse)."""
        off = self.gate - torch.diag(torch.diag(self.gate))
        return off.abs().sum()

    def forward(self, node_feats, cell_types, incidence_matrix):
        N = node_feats.shape[0]
        M = incidence_matrix.shape[1]
        K = self.n_cell_types
        H = incidence_matrix

        dv = H.sum(dim=1)
        dv_invsqrt = dv ** -0.5
        dv_invsqrt[dv_invsqrt == float('inf')] = 0

        messages = self.hyperedge_emb_weight(node_feats, cell_types)   # (N × D) = m_i
        m_norm = dv_invsqrt.unsqueeze(1) * messages
        onehot = F.one_hot(cell_types, K).to(messages.dtype)           # (N × K)

        # ── per-type mean pooling → E (M × K × D), + presence mask ──────────────
        channels, present = [], []
        for k in range(K):
            mask = onehot[:, k].unsqueeze(1)
            pooled = H.t() @ (m_norm * mask)                # (M × D)
            de_k = H.t() @ onehot[:, k]                     # (M,)
            de_inv = torch.where(de_k > 0, 1.0 / de_k, torch.zeros_like(de_k))
            channels.append(de_inv.unsqueeze(1) * pooled)
            present.append(de_k > 0)
        E = torch.stack(channels, dim=1)                    # (M × K × D)
        present = torch.stack(present, dim=1)               # (M × K) bool

        # raw-sum blend (abundance / composition), degree-normalized messages
        E_sum = H.t() @ m_norm                              # (M × D)

        # ── (1) self-attention over the K type-fields, absent types masked ─────
        E_in = self.field_norm(E)
        key_padding_mask = ~present                         # True = PAD (absent type)
        valid = present.any(dim=1)                          # (M,) hyperedge has ≥1 type
        # MultiheadAttention NaNs if a row is fully masked; unmask empties, zero after
        kpm = key_padding_mask.clone()
        kpm[~valid] = False
        attn_out, _ = self.attn(E_in, E_in, E_in,
                                key_padding_mask=kpm, need_weights=False)
        attn_out = attn_out * valid.view(M, 1, 1)
        E = E + attn_out                                    # residual, interaction-refined
        E = E * present.unsqueeze(-1)                       # drop absent channels
        E = E + self.hyperedge_emb_bias

        E_act = self.dropout(self.act(E))
        W_e = self.node_emb_weight(E_act)                   # (M × K × D)

        # ── scatter refined channels back to nodes → (N × K × D) ───────────────
        node_contrib = (H @ W_e.reshape(M, K * W_e.shape[-1])).reshape(N, K, -1)
        node_contrib = dv_invsqrt.view(N, 1, 1) * node_contrib

        # ── reader-type gated combine + bottlenecked self-identity skip ────────
        g_rows = self.gate[cell_types]                      # (N × K)
        node_embeddings = (g_rows.unsqueeze(-1) * node_contrib).sum(dim=1)
        node_embeddings = node_embeddings + self.node_emb_bias(cell_types)
        node_embeddings = node_embeddings + self.alpha * messages   # skip

        # ── (2) hyperedge embedding: mean over present refined channels + blend ─
        denom = present.sum(dim=1).clamp_min(1).unsqueeze(1)        # (M × 1)
        h_ct = (E * present.unsqueeze(-1)).sum(dim=1) / denom       # (M × D)
        hyperedge_embeddings = self.hedge_fuse(
            torch.cat([h_ct, self.blend_norm(E_sum)], dim=1)
        )                                                           # (M × D)

        return node_embeddings, hyperedge_embeddings


class CTHGCAEv5(nn.Module):
    """
    v5: type-aware encoder with K→K field self-attention + composition blend +
    self-identity skip, mirror decoder (from v3), optional hyperedge branch.

    Unlike v3/v4 — which discard the encoder's hyperedge output and re-pool the
    *node* embeddings — v5 returns the encoder's **fused hyperedge embedding**
    (attention-refined per-type channels + raw-sum composition blend) as the
    hyperedge representation, since that is the type/interaction/composition-aware
    vector meant for domain clustering. ``hyperedge_pool`` is accepted for API
    compatibility but only used if ``use_node_pool_hyperedge`` is set.
    """

    def __init__(self, num_cell_types, input_dim, embedding_dim, dropout=0.1,
                 hyperedge_feat_dim=None, hyperedge_pool="spot", num_heads=4,
                 use_node_pool_hyperedge=False):
        super().__init__()
        assert hyperedge_pool in ("spot", "neighborhood"), hyperedge_pool
        self.hyperedge_pool = hyperedge_pool
        self.use_node_pool_hyperedge = use_node_pool_hyperedge
        self.encoder = CTTypeAttnEncoder(num_cell_types, input_dim, embedding_dim,
                                         dropout, num_heads=num_heads)
        self.decoder = CTHGMirrorDecoder(num_cell_types, embedding_dim, input_dim, dropout=0.0)
        self.hyperedge_decoder = (
            HyperedgeDecoder(embedding_dim, hyperedge_feat_dim)
            if hyperedge_feat_dim else None
        )

    def forward(self, node_feats, cell_types, incidence_matrix):
        node_embeddings, h_enc = self.encoder(node_feats, cell_types, incidence_matrix)
        node_reconstruction = self.decoder(node_embeddings, cell_types, incidence_matrix)
        if self.use_node_pool_hyperedge:
            pool = _own_spot_pool if self.hyperedge_pool == "spot" else _neighborhood_pool
            hyperedge_embeddings = pool(node_embeddings, incidence_matrix)
        else:
            hyperedge_embeddings = h_enc
        hyperedge_reconstruction = (
            self.hyperedge_decoder(hyperedge_embeddings)
            if self.hyperedge_decoder is not None else None
        )
        return node_embeddings, hyperedge_embeddings, node_reconstruction, hyperedge_reconstruction


@HypergraphEmbedder.register("CTHGCAEv5Embedder")
class CTHGCAEv5Embedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs, n_cell_types: int):
        model = CTHGCAEv5(
            num_cell_types=n_cell_types,
            input_dim=args.input_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            hyperedge_feat_dim=args.hyperedge_feat_dim,
            hyperedge_pool=args.hyperedge_pool,
            num_heads=args.num_heads or 4,
        )
        return cls(model)

    def forward(self, batch):
        feats = batch.batch_cell_feats
        cell_types = batch.batch_cell_types
        incidence_matrix = batch.batch_incidence_matrix
        return self.model(feats, cell_types, incidence_matrix)

    @classmethod
    def checkpoint_name(cls) -> str:
        return "CTHGCAEv5Embedder"

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
