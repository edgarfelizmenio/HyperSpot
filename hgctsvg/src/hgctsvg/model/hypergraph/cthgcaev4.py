import torch
import torch.nn as nn
import torch.nn.functional as F

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from ..moduledict import LinearDict, ParameterDict
from .cthgcaev3 import CTHGMirrorDecoder, HyperedgeDecoder, _own_spot_pool, _neighborhood_pool


class CTTypeAwareEncoder(nn.Module):
    """
    Type-aware hyperedge encoder.

    A standard hypergraph convolution summarises each hyperedge (spatial
    neighborhood) as ONE blended vector over all its member cells, so a cell
    absorbs only the *averaged* neighborhood signal — and abundant cell types
    dominate that average. This encoder instead resolves each hyperedge by
    **source cell type** into K channels ``E[hyperedge, type]``, each normalised
    by that type's own count in the hyperedge (so a rare type's niche signal
    survives), then reads them back with a learned **K×K cross-type gate**
    ``g[a, k]`` = how much a type-``a`` cell weights the type-``k`` signal in its
    neighborhood. The result: a cell's embedding encodes its neighborhood's
    per-type *composition*, decoupled from raw abundance — i.e. niche structure.

    Cell-type-specific parameters: the input adapter ``W_ct``
    (``hyperedge_emb_weight``), the per-CT output bias, and the K×K gate.
    ``node_emb_weight`` is shared. Mirrors the ``CTHyFiHGNNv1`` normalisation
    (``dv^{-1/2}`` on both the node→edge messages and the edge→node scatter).
    """

    def __init__(self, n_cell_types, input_dim, embedding_dim, dropout=0.1, act=nn.PReLU):
        super().__init__()
        self.n_cell_types = n_cell_types
        self.hyperedge_emb_weight = LinearDict(n_cell_types, input_dim, embedding_dim, bias=False)
        self.hyperedge_emb_bias = nn.Parameter(torch.zeros((embedding_dim,)))
        self.node_emb_weight = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.node_emb_bias = ParameterDict(n_cell_types, embedding_dim)
        # K×K cross-type gate. Init to identity: each type starts by reading its
        # own channel; cross-type influence is learned from zero.
        self.gate = nn.Parameter(torch.eye(n_cell_types))
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

    def regularization_loss(self):
        """L1 penalty on the off-diagonal of the K×K cross-type gate.

        Encourages parsimonious cross-type influence: cross-type gates open only
        when the data supports them, keeping the diagonal (own-type) structure
        intact and the gate interpretable. Weighted by the training loop.
        """
        off_diag = self.gate - torch.diag(torch.diag(self.gate))
        return off_diag.abs().sum()

    def forward(self, node_feats, cell_types, incidence_matrix):
        N = node_feats.shape[0]
        M = incidence_matrix.shape[1]
        K = self.n_cell_types
        H = incidence_matrix

        dv = H.sum(dim=1)                                   # (N,) node degrees
        dv_invsqrt = dv ** -0.5
        dv_invsqrt[dv_invsqrt == float('inf')] = 0

        # node-normalised, cell-type-transformed messages
        messages = self.hyperedge_emb_weight(node_feats, cell_types)   # (N × emb)
        messages = dv_invsqrt.unsqueeze(1) * messages

        onehot = F.one_hot(cell_types, K).to(messages.dtype)           # (N × K)

        # ── type-resolved hyperedge pooling → E (M × K × emb) ────────────────
        channels = []
        for k in range(K):
            mask = onehot[:, k].unsqueeze(1)                # (N × 1)
            pooled = H.t() @ (messages * mask)              # (M × emb): sum type-k
            de_k = H.t() @ onehot[:, k]                     # (M,): type-k count/weight
            de_k_inv = torch.where(de_k > 0, 1.0 / de_k, torch.zeros_like(de_k))
            channels.append(de_k_inv.unsqueeze(1) * pooled)
        E = torch.stack(channels, dim=1) + self.hyperedge_emb_bias      # (M × K × emb)

        E_act = self.dropout(self.act(E))
        W_e = self.node_emb_weight(E_act)                   # (M × K × emb)

        # ── scatter each channel back to nodes → (N × K × emb) ───────────────
        emb = W_e.shape[-1]
        node_contrib = (H @ W_e.reshape(M, K * emb)).reshape(N, K, emb)
        node_contrib = dv_invsqrt.view(N, 1, 1) * node_contrib

        # ── K×K gated combine per reader type ────────────────────────────────
        g_rows = self.gate[cell_types]                      # (N × K)
        node_embeddings = (g_rows.unsqueeze(-1) * node_contrib).sum(dim=1)
        node_embeddings = node_embeddings + self.node_emb_bias(cell_types)

        # spot output kept shape-compatible (M × emb) for the downstream
        # hyperedge/spot-embedding pipeline: mean over the type channels.
        hyperedge_embeddings = E.mean(dim=1)

        return node_embeddings, hyperedge_embeddings


class CTHGCAEv4(nn.Module):
    """
    Type-aware encoder + mirror decoder.

    Encoder: ``CTTypeAwareEncoder`` — type-resolved neighborhood channels + K×K
    cross-type gate (rung 3 of type-awareness).
    Decoder: ``CTHGMirrorDecoder`` (from v3) — shared hypergraph de-conv, CT
    output adapter last. Type-specific in, shared in the middle, type-specific
    out.
    """

    def __init__(self, num_cell_types, input_dim, embedding_dim, dropout=0.1,
                 hyperedge_feat_dim=None, hyperedge_pool="spot"):
        super().__init__()
        assert hyperedge_pool in ("spot", "neighborhood"), hyperedge_pool
        self.hyperedge_pool = hyperedge_pool
        self.encoder = CTTypeAwareEncoder(num_cell_types, input_dim, embedding_dim, dropout)
        self.decoder = CTHGMirrorDecoder(num_cell_types, embedding_dim, input_dim, dropout=0.0)
        # optional hyperedge autoencoder branch (built only when a hyperedge
        # feature dim is known; None keeps old checkpoints/behavior unchanged)
        self.hyperedge_decoder = (
            HyperedgeDecoder(embedding_dim, hyperedge_feat_dim)
            if hyperedge_feat_dim else None
        )

    def forward(self, node_feats, cell_types, incidence_matrix):
        node_embeddings, _ = self.encoder(
            node_feats, cell_types, incidence_matrix
        )
        node_reconstruction = self.decoder(
            node_embeddings, cell_types, incidence_matrix
        )
        # Unified hyperedge embedding = pool of the NODE embeddings ("spot" = own
        # cells only, sharper; "neighborhood" = spot + neighbors, smoother). This
        # is reconstructed/regularized AND saved for tissue-domain clustering, so
        # the hyperedge task backprops directly into node embeddings.
        pool = _own_spot_pool if self.hyperedge_pool == "spot" else _neighborhood_pool
        hyperedge_embeddings = pool(node_embeddings, incidence_matrix)
        hyperedge_reconstruction = (
            self.hyperedge_decoder(hyperedge_embeddings)
            if self.hyperedge_decoder is not None else None
        )
        return node_embeddings, hyperedge_embeddings, node_reconstruction, hyperedge_reconstruction


@HypergraphEmbedder.register("CTHGCAEv4Embedder")
class CTHGCAEv4Embedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs, n_cell_types: int):
        model = CTHGCAEv4(
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
        return "CTHGCAEv4Embedder"

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
