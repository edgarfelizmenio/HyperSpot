import torch
import torch.nn as nn

from . import HypergraphEmbedder, HypergraphEmbedderArgs
from .cthgnn import CTHyFiHGNNv1
from ..moduledict import LinearDict


class CTHGMirrorDecoder(nn.Module):
    """
    A *truer mirror* of the ``CTHyFiHGNNv1`` encoder.

    The encoder maps cell-type-heterogeneous inputs into a shared latent space by
    putting the cell-type-specific projection ``W_ct`` **first** (an input
    adapter), then running a shared hypergraph convolution::

        X → [CT input adapter W_ct] → node→edge→node conv → Z   (shared space)

    A symmetric decoder should therefore run the hypergraph de-convolution in the
    shared latent space with **shared** weights, and place the cell-type-specific
    projection **last** (an output adapter: shared latent → per-cell-type feature
    space)::

        Z → node→edge→node conv (shared) → [CT output adapter] → X̂

    This fixes the two problems of the v1 decoder (which reused ``CTHyFiHGNNv1``):
      1. its CT projection sat at the *input* (applied to an already-shared Z) and
         its final feature projection was *shared* — so per-type reconstructions
         could only differ by an additive bias;
      2. it is preserved here that reconstruction still uses the hypergraph
         structure, but the CT split now happens strictly at the output boundary.

    Cell-type specificity lives only in ``output_proj`` (mirroring the encoder,
    where it lives only in the first projection). All intermediate weights and
    biases are shared and operate in the shared latent space.
    """

    def __init__(self, n_cell_types, embedding_dim, output_dim,
                 dropout=0.0, act=nn.PReLU):
        super().__init__()
        # node→hyperedge: SHARED (input Z is already in the shared latent space)
        self.hyperedge_emb_weight = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.hyperedge_emb_bias = nn.Parameter(torch.zeros((embedding_dim,)))
        # hyperedge→node: SHARED, stays in the shared latent space
        self.node_emb_weight = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.node_emb_bias = nn.Parameter(torch.zeros((embedding_dim,)))
        # CT OUTPUT adapter (LAST): shared latent → per-cell-type feature space.
        # Mirrors the encoder's CT input adapter; bias=True → per-CT output bias.
        self.output_proj = LinearDict(n_cell_types, embedding_dim, output_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.act = act()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.hyperedge_emb_weight.weight)
        nn.init.xavier_uniform_(self.node_emb_weight.weight)
        nn.init.zeros_(self.hyperedge_emb_bias)
        nn.init.zeros_(self.node_emb_bias)
        self.output_proj.reset_parameters()

    def forward(self, node_embeddings, cell_types, incidence_matrix):
        dv = incidence_matrix.sum(dim=1)
        de = incidence_matrix.sum(dim=0)

        dv_invsqrt = (dv ** -0.5).diag()
        dv_invsqrt[dv_invsqrt == float('inf')] = 0
        de_inv = (de ** -1).diag()
        de_inv[de_inv == float('inf')] = 0

        # node→hyperedge (shared latent-space projection, then pool)
        hyperedge_embeddings = (
            de_inv @ incidence_matrix.t() @ dv_invsqrt
            @ self.hyperedge_emb_weight(node_embeddings)
            + self.hyperedge_emb_bias
        )
        hyperedge_embeddings_act = self.act(hyperedge_embeddings)
        hyperedge_embeddings_act = self.dropout(hyperedge_embeddings_act)

        # hyperedge→node (shared) — stays in shared latent space, no activation
        latent_reconstruction = (
            dv_invsqrt @ incidence_matrix
            @ self.node_emb_weight(hyperedge_embeddings_act)
            + self.node_emb_bias
        )

        # CT output adapter (LAST): shared latent → per-cell-type feature space
        node_reconstruction = self.output_proj(latent_reconstruction, cell_types)
        return node_reconstruction


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


def _neighborhood_pool(node_embeddings, incidence_matrix):
    """Degree-normalized neighborhood pool of node embeddings onto hyperedges.

    pooled[j] = (sum_i H[i,j] * node_emb[i]) / (sum_i H[i,j])

    H is the (cells × hyperedges) incidence, so this is an incidence-weighted mean
    over each hyperedge's member cells (spot + neighbors; own-spot cells weighted
    more via intra_weight). Empty hyperedges -> 0. Because it pools the *node
    embeddings*, gradients from the hyperedge task flow back into them.
    """
    deg = incidence_matrix.sum(dim=0).clamp_min(1e-6).unsqueeze(1)   # (M × 1)
    return (incidence_matrix.t() @ node_embeddings) / deg


def _own_spot_pool(node_embeddings, incidence_matrix):
    """Pool node embeddings over each hyperedge's OWN-spot cells only.

    A cell's own spot is the incident hyperedge it weights most in the incidence
    (own-spot entries carry ``intra_weight``, neighbors ``inter_weight``), so own
    membership is the per-cell argmax over the incidence row. Tighter match to the
    spot-basis pseudobulk target and avoids neighborhood averaging.

    Requires ``intra_weight > inter_weight``; with equal weights the incidence has
    no own-vs-neighbor signal and this degenerates (use the neighborhood pool
    there instead). Cell-less spots -> 0.
    """
    N = incidence_matrix.shape[0]
    row_max, own = incidence_matrix.max(dim=1)                      # each cell's own spot col
    own_incidence = torch.zeros_like(incidence_matrix)
    idx = torch.arange(N, device=incidence_matrix.device)
    valid = row_max > 0                                             # cell has an in-batch spot
    own_incidence[idx[valid], own[valid]] = 1.0
    deg = own_incidence.sum(dim=0).clamp_min(1e-6).unsqueeze(1)     # own-cell count per spot
    return (own_incidence.t() @ node_embeddings) / deg


class CTHGCAEv3(nn.Module):
    """
    CTHGCAE with a decoder that truly mirrors the encoder.

    Encoder: ``CTHyFiHGNNv1`` — CT input adapter → shared hypergraph conv → Z.
    Decoder: ``CTHGMirrorDecoder`` — shared hypergraph de-conv → CT output adapter.
    """

    def __init__(self, num_cell_types, input_dim, embedding_dim, dropout=0.1,
                 hyperedge_feat_dim=None, hyperedge_pool="spot"):
        super().__init__()
        assert hyperedge_pool in ("spot", "neighborhood"), hyperedge_pool
        self.hyperedge_pool = hyperedge_pool
        self.encoder = CTHyFiHGNNv1(num_cell_types, input_dim, embedding_dim, dropout)
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
        # unified hyperedge embedding = pool of NODE embeddings ("spot"=own cells,
        # "neighborhood"=spot+neighbors): reconstructed/regularized and output for
        # clustering, so the hyperedge task backprops into node embeddings.
        pool = _own_spot_pool if self.hyperedge_pool == "spot" else _neighborhood_pool
        hyperedge_embeddings = pool(node_embeddings, incidence_matrix)
        hyperedge_reconstruction = (
            self.hyperedge_decoder(hyperedge_embeddings)
            if self.hyperedge_decoder is not None else None
        )
        return node_embeddings, hyperedge_embeddings, node_reconstruction, hyperedge_reconstruction


@HypergraphEmbedder.register("CTHGCAEv3Embedder")
class CTHGCAEv3Embedder(HypergraphEmbedder):

    @classmethod
    def _build_impl(cls, args: HypergraphEmbedderArgs, n_cell_types: int):
        model = CTHGCAEv3(
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
        return "CTHGCAEv3Embedder"

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
