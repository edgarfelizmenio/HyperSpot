import torch.nn as nn


class HGCTCAELoss(nn.Module):
    """Training loss of the HyperSpot hypergraph autoencoder: two reconstruction terms.

      node term      = mean over the cell types present in the batch of
                       MSE(cell features, reconstructed cell features) within that type,
                       so abundant cell types do not dominate.
      hyperedge term = MSE(reconstructed hyperedge features, hyperedge features), where a
                       hyperedge's target is its centre spot's own pseudobulk. Hyperedges whose
                       spot has no cells (all-zero target) are excluded.

      loss = node term + hyperedge term   (equal weights)

    forward returns (loss, node term, hyperedge term); the last two are detached and only
    used for the progress bar.
    """

    def __init__(self):
        super().__init__()
        self.reconloss = nn.MSELoss(reduction='mean')

    def forward(self, orig_feats, recon_feats, classes, hyperedge_recon, hyperedge_feats):
        observed_classes = classes.unique()

        node_total = 0
        for c in observed_classes:
            mask = classes == c
            node_total += self.reconloss(orig_feats[mask], recon_feats[mask])
        node_total = node_total / observed_classes.numel()

        hyperedge_total = node_total.new_zeros(())
        valid_h = hyperedge_feats.abs().sum(dim=1) > 0          # spots with at least one cell
        if valid_h.any():
            hyperedge_total = hyperedge_total + self.reconloss(hyperedge_recon[valid_h],
                                                               hyperedge_feats[valid_h])

        loss = node_total + hyperedge_total
        return loss, node_total.detach(), hyperedge_total.detach()
