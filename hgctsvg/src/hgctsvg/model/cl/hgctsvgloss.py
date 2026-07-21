import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from ..moduledict import LinearDict

class IntraClassContrastiveLoss(nn.Module):
    def __init__(self, 
                 input_dim,
                 projection_dim = 100,
                 temp_node = 0.5,
                 temp_hyperedge = 0.5,
                 alpha=1.0):
        super(IntraClassContrastiveLoss, self).__init__()
        # alpha is the edge_loss weight

        self.temp_node = temp_node
        self.temp_hyperedge = temp_hyperedge
        self.alpha = alpha
        
        self.project_nodes = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ELU(),
            nn.Linear(projection_dim , input_dim)
        )
        self.project_hyperedges = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ELU(),
            nn.Linear(projection_dim , input_dim)
        )

    def forward(self, 
                node_embeddings, 
                hyperedge_embeddings, 
                augmented_node_embeddings, 
                augmented_hyperedge_embeddings,
                incidence_matrix,
                classes,
                sample_n_cells = None,
                sample_n_hyperedges = None):

        # project (two-layer MLP with ELU activation)
        node_projections = self.project_nodes(node_embeddings)
        hyperedge_projections = self.project_hyperedges(hyperedge_embeddings)

        augmented_node_projections = self.project_nodes(augmented_node_embeddings)
        augmented_hyperedge_projections = self.project_hyperedges(augmented_hyperedge_embeddings)

        # incidence_matrix = incidence_matrix.to_sparse()        
        same_class_mask = (classes[:, None] == classes[None, :]).int()
        node_commonality = incidence_matrix @ incidence_matrix.t()
        node_level_cl = self.__compute_cl(node_projections, augmented_node_projections, node_commonality, self.temp_node, same_class_mask)
        # node_level_cl = self.__compute_cl_batch(node_projections, augmented_node_projections, incidence_matrix, self.temp_node)

        hyperedge_commonality = incidence_matrix.t() @ incidence_matrix
        hyperedge_level_cl = self.__compute_cl(hyperedge_projections, augmented_hyperedge_projections, hyperedge_commonality, self.temp_hyperedge)

        if sample_n_cells:
            # limit error computation to the sampled batch
            classes = classes[:sample_n_cells]
            node_level_cl = node_level_cl[:sample_n_cells]
            hyperedge_level_cl = hyperedge_level_cl[:sample_n_hyperedges]

        # nodes_per_class = classes.bincount()
        # observed_classes = nodes_per_class > 0
        # total_node_loss_per_class = classes.bincount(weights=node_level_cl)
        # mean_loss_per_class = total_node_loss_per_class[observed_classes] / nodes_per_class[observed_classes]
        # intraclass_node_level_cl = mean_loss_per_class.mean()
        
        # batch_cl = intraclass_node_level_cl + self.alpha * hyperedge_level_cl.mean()
        batch_cl = node_level_cl.mean() + self.alpha * hyperedge_level_cl.mean()
        
        return batch_cl

    def __compute_cl(self, projections, augmented_projections, commonality_matrix, temp, same_class_mask = None): 
        projections_norm = projections / projections.norm(dim=1, keepdim=True)
        aug_projections_norm = augmented_projections / augmented_projections.norm(dim=2, keepdim=True)

        # cross_view_similarity is the similarity between a projection in the original view and its augmented counterpart. 
        # Dimension: 2xN
        cross_view_similarity = torch.einsum("nf,knf->kn", projections_norm, aug_projections_norm)
        
        # compute positive pair score and sum for all projections
        positive_pair_loss = (cross_view_similarity / temp).exp().sum(dim=0)  # (N,)

        if same_class_mask is not None:
            no_common = (commonality_matrix == 0) * same_class_mask
            commonality_matrix *= same_class_mask
        else:
            no_common = (commonality_matrix == 0)
            
        diag = commonality_matrix.diag().unsqueeze(1)
        pos_pair_prob = commonality_matrix/diag
        weak_positive_weight = commonality_matrix * pos_pair_prob
        # no_common = (commonality_matrix == 0)
        
        # similarity is the pair-wise similarity between projections of the original view.
        # Dimension: NxN
        similarity = projections_norm @ projections_norm.t()
        
        # pair-wise loss of a projection vs all projections in the batch
        # Dimension: NxN
        pair_wise_loss = (similarity / temp).exp()
        
        # sum weak positive pair score for each weak positive pair
        # - weak positive weight is always positive since all nodes are adjacent to at least one hyperedge
        # - weak positive weight is always zero for non-adjacent nodes
        # Dimension: Nx1
        weak_positive_pair_loss = (pair_wise_loss * weak_positive_weight).sum(dim=1)

        # sum negative pair score for each negative pair
        # Dimension: Nx1
        negative_pair_loss = (pair_wise_loss * no_common).sum(dim=1)

        # compute node-level cl
        # Dimension: Nx1
        cl = -((positive_pair_loss + weak_positive_pair_loss).log() - (positive_pair_loss + weak_positive_pair_loss + negative_pair_loss).log())

        # print(cl)
        return cl        

    def __compute_cl_batch(self, projections, augmented_projections, hyperedge_index, temp, chunk_size=750):
        N = projections.shape[0]
        device = projections.device
        
        projections_norm = projections / projections.norm(dim=1, keepdim=True)
        aug_projections_norm = augmented_projections / augmented_projections.norm(dim=2, keepdim=True)
        cross_view_similarity = torch.einsum("nf,knf->kn", projections_norm, aug_projections_norm)        
        positive_pair_loss = (cross_view_similarity / temp).exp().sum(dim=0)  # (N,)

        diag = (hyperedge_index * hyperedge_index).sum(dim=1)
        h_idx_sparse = hyperedge_index.to_sparse_csr()

        del augmented_projections, aug_projections_norm, cross_view_similarity, hyperedge_index
        gc.collect()
        
        commonality_matrix = torch.sparse.mm(h_idx_sparse, h_idx_sparse.transpose(0,1).to_sparse_csr())

        rowptr = commonality_matrix.crow_indices()    # shape [N+1]
        col    = commonality_matrix.col_indices()     # shape [E]
        vals   = commonality_matrix.values()

        weak_positive_pair_loss = torch.zeros(N, device=device)
        edge_unweighted_sum = torch.zeros(N, device=device)
        negative_pair_loss = torch.zeros(N, device=device)

        # print(weak_positive_pair_loss.nbytes /  (1024**3))
        # print(edge_unweighted_sum.nbytes /  (1024**3))
        # print(negative_pair_loss.nbytes /  (1024**3))
        # add code to make full neg
        for start_i in tqdm(range(0, N, chunk_size), leave=False):
            end_i = min(start_i + chunk_size, N)

            row_start = rowptr[start_i : end_i]
            row_end   = rowptr[start_i + 1:end_i + 1]
        
            # number of edges per row
            lens = row_end - row_start

            # collect neighbors into a contiguous block
            batch_cols = col[row_start[0]:row_end[-1]]
            batch_vals = vals[row_start[0]:row_end[-1]]

            del row_start, row_end
            
            # build row indices for this batch
            row_idx = torch.repeat_interleave(torch.arange(start_i, end_i, device=device), lens)

            del lens
            
            # everything else identical
            p = batch_vals / diag[row_idx]
            w = batch_vals * p

            del batch_vals, p
            
            sim = (projections_norm[row_idx] * projections_norm[batch_cols]).sum(dim=1)

            del batch_cols
            
            exp_sim = torch.exp(sim / temp)

            edge_unweighted_sum.index_add_(0, row_idx, exp_sim)
            
            del sim
            
            weak_positive_pair_loss.index_add_(0, row_idx, exp_sim * w)

            del row_idx, exp_sim, w
            # weak_positive_pair_loss.index_add_(0, row, pair_vals * weak_positive_weight_vals)
            
            for start_j in range(0, N, chunk_size):
                end_j = min(start_j + chunk_size, N)
                
                similarity_ij = projections_norm[start_i:end_i] @ projections_norm[start_j:end_j].t()

                pair_wise_loss_ij = (similarity_ij / temp).exp()

                del similarity_ij

                negative_pair_loss[start_i:end_i] += pair_wise_loss_ij.sum(dim=1)

                del pair_wise_loss_ij            

            gc.collect()
            
        del diag

        diag_self = torch.exp(torch.tensor(1.0 / temp, device=device))
        
        negative_pair_loss -= (edge_unweighted_sum + diag_self)
        
        cl = -((positive_pair_loss + weak_positive_pair_loss).log() - (positive_pair_loss + weak_positive_pair_loss + negative_pair_loss).log())

        # print(cl)
        return cl

    def forward(self, 
                node_embeddings, 
                hyperedge_embeddings, 
                augmented_node_embeddings, 
                augmented_hyperedge_embeddings,
                incidence_matrix,
                classes,
                sample_n_cells = None,
                sample_n_hyperedges = None):

        # project (two-layer MLP with ELU activation)
        node_projections = self.project_nodes(node_embeddings)
        hyperedge_projections = self.project_hyperedges(hyperedge_embeddings)

        augmented_node_projections = self.project_nodes(augmented_node_embeddings)
        augmented_hyperedge_projections = self.project_hyperedges(augmented_hyperedge_embeddings)

        # incidence_matrix = incidence_matrix.to_sparse()        
        same_class_mask = (classes[:, None] == classes[None, :]).int()
        node_commonality = incidence_matrix @ incidence_matrix.t()
        node_level_cl = self.__compute_cl(node_projections, augmented_node_projections, node_commonality, self.temp_node, same_class_mask)
        # node_level_cl = self.__compute_cl_batch(node_projections, augmented_node_projections, incidence_matrix, self.temp_node)

        hyperedge_commonality = incidence_matrix.t() @ incidence_matrix
        hyperedge_level_cl = self.__compute_cl(hyperedge_projections, augmented_hyperedge_projections, hyperedge_commonality, self.temp_hyperedge)
        
        if sample_n_cells:
            # limit error computation to the sampled batch
            classes = classes[:sample_n_cells]
            node_level_cl = node_level_cl[:sample_n_cells]
            hyperedge_level_cl = hyperedge_level_cl[:sample_n_hyperedges]

        # nodes_per_class = classes.bincount()
        # observed_classes = nodes_per_class > 0
        # total_node_loss_per_class = classes.bincount(weights=node_level_cl)
        # mean_loss_per_class = total_node_loss_per_class[observed_classes] / nodes_per_class[observed_classes]
        # intraclass_node_level_cl = mean_loss_per_class.mean()
        
        # batch_cl = intraclass_node_level_cl + self.alpha * hyperedge_level_cl.mean()
        batch_cl = node_level_cl.mean() + self.alpha * hyperedge_level_cl.mean()
        
        return batch_cl

class CosineReconLoss(nn.Module):
    """Reconstruction loss based on cosine similarity (scale-invariant)."""
    def __init__(self):
        super(CosineReconLoss, self).__init__()
        self._cos = nn.CosineSimilarity(dim=1)

    def forward(self, pred, target):
        return (1 - self._cos(pred, target)).mean()


class HGCTCAELoss(nn.Module):
    """
    Reconstruction loss for the hypergraph CAE, with optional per-cell-type
    regularizers on the bottleneck embedding:

      1. variance/covariance (VICReg-style) — anti-collapse: keeps each embedding
         dimension from collapsing to a constant and decorrelates dimensions.
      2. neighborhood contrastive — pulls together cells that share hyperedges
         (same spatial neighborhood) and pushes apart non-adjacent cells; the
         augmentation-free form of the intra-class hyperedge-commonality CL.

    Both are computed *within each cell type* (the unit that is clustered
    downstream). All regularizer weights default to 0, so the loss is unchanged
    unless they are enabled.
    """

    def __init__(self,
                 rho = 1.0,
                 lambda_var = 0.0,
                 lambda_cov = 0.0,
                 lambda_nbr = 0.0,
                 temp_nbr = 0.5,
                 edge_preserving = True,
                 optimize_hyperedges = False,
                 rho_h = 1.0,
                 lambda_var_h = 0.0,
                 lambda_cov_h = 0.0,
                 lambda_nbr_h = 0.0,
                 var_target = 1.0,
                 recon_loss = None):
        super(HGCTCAELoss, self).__init__()
        self.rho = rho
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.lambda_nbr = lambda_nbr
        self.temp_nbr = temp_nbr
        self.edge_preserving = edge_preserving
        # opt-in hyperedge autoencoder branch (reconstruction + regularizers on
        # the otherwise-unsupervised hyperedge embeddings)
        self.optimize_hyperedges = optimize_hyperedges
        self.rho_h = rho_h
        self.lambda_var_h = lambda_var_h
        self.lambda_cov_h = lambda_cov_h
        self.lambda_nbr_h = lambda_nbr_h
        self.var_target = var_target
        self.reconloss = recon_loss if recon_loss is not None else nn.MSELoss(reduction='mean')

    @staticmethod
    def _variance_covariance(Z, var_target):
        # Z: (n_c × d). VICReg variance hinge + off-diagonal covariance penalty.
        d = Z.shape[1]
        std = torch.sqrt(Z.var(dim=0, unbiased=False) + 1e-4)
        var_loss = torch.relu(var_target - std).mean()
        Zc = Z - Z.mean(dim=0, keepdim=True)
        cov = (Zc.t() @ Zc) / (Z.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / d
        return var_loss, cov_loss

    def _neighborhood_contrastive(self, Z, incidence_c, feats=None):
        # InfoNCE where positives = cells sharing >=1 hyperedge (same neighborhood),
        # negatives = non-adjacent cells. Augmentation-free.
        #
        # Edge-preserving mode (self.edge_preserving, default on): the pull toward a
        # spatial neighbor is *soft-weighted* by expression similarity, turning the
        # isotropic "Gaussian blur" (which rounds domains into artificially convex
        # blobs) into a "bilateral filter" that smooths within homogeneous regions
        # but stops at expression boundaries. A neighbor that is dissimilar in
        # expression gets a small positive weight, so it stays in the denominator
        # and is effectively pushed apart -> the boundary sharpens. The kernel
        # bandwidth is set per-forward by the median heuristic, so no new knob.
        n = Z.shape[0]
        Zn = F.normalize(Z, dim=1)
        sim = (Zn @ Zn.t()) / self.temp_nbr
        common = incidence_c @ incidence_c.t()                 # shared-hyperedge weight
        eye = torch.eye(n, dtype=torch.bool, device=Z.device)
        pos = (common > 0) & (~eye)
        valid = pos.any(dim=1)
        if not valid.any():
            return None
        neg_inf = float('-inf')
        denom = torch.logsumexp(sim.masked_fill(eye, neg_inf), dim=1)

        if self.edge_preserving and feats is not None:
            # log-weight positives by an expression-similarity bilateral kernel:
            #   w_ij = exp(-(1 - e_ij) / h),  e_ij = cos(feat_i, feat_j)
            #   h    = median over adjacent pairs of (1 - e_ij)   [median heuristic]
            with torch.no_grad():
                Fn = F.normalize(feats, dim=1)
                dissim = 1.0 - (Fn @ Fn.t())                   # in [0, 2]
                h = dissim[pos].median().clamp_min(1e-6)
            log_w = -dissim / h                                # log positive weights (finite)
            # Mask non-positives as the *final* op (like the isotropic branch below):
            # masked_fill's backward assigns 0 grad at masked entries. Folding -inf in
            # earlier via addition would instead let NaN from all-(-inf) rows -- cells
            # with no same-type neighbor -- propagate into sim and blow up training.
            num = torch.logsumexp((sim + log_w).masked_fill(~pos, neg_inf), dim=1)
        else:
            num = torch.logsumexp(sim.masked_fill(~pos, neg_inf), dim=1)

        return (denom - num)[valid].mean()

    def forward(self,
                orig_feats,
                recon_feats,
                node_embeddings,
                incidence_matrix,
                classes,
                sample_n_cells = None,
                sample_n_hyperedges = None,
                hyperedge_embeddings = None,
                hyperedge_recon = None,
                hyperedge_feats = None):

        observed_classes = torch.unique(classes)
        n_observed_classes = observed_classes.numel()
        use_vc = self.lambda_var > 0 or self.lambda_cov > 0

        total_recon_loss = 0
        var_terms, cov_terms, nbr_terms = [], [], []
        for c in observed_classes:
            mask = classes == c
            total_recon_loss += self.reconloss(orig_feats[mask], recon_feats[mask])

            if mask.sum() < 2:
                continue
            Z_c = node_embeddings[mask]
            if use_vc:
                v, cov = self._variance_covariance(Z_c, self.var_target)
                var_terms.append(v)
                cov_terms.append(cov)
            if self.lambda_nbr > 0:
                nbr = self._neighborhood_contrastive(Z_c, incidence_matrix[mask],
                                                     feats=orig_feats[mask])
                if nbr is not None:
                    nbr_terms.append(nbr)

        # progress bar reports node-total and hyperedge-total (recon + regularizers);
        # total loss = node_total + hyperedge_total
        node_total = self.rho * (total_recon_loss / n_observed_classes)
        if var_terms:
            node_total = node_total + self.lambda_var * torch.stack(var_terms).mean()
        if cov_terms:
            node_total = node_total + self.lambda_cov * torch.stack(cov_terms).mean()
        if nbr_terms:
            node_total = node_total + self.lambda_nbr * torch.stack(nbr_terms).mean()
        hyperedge_total = node_total.new_zeros(())

        # ── opt-in hyperedge autoencoder branch ──────────────────────────────
        # Hyperedge embeddings are otherwise unsupervised (a pooled readout).
        # Reconstruct the center spot's own pseudobulk + optional var/cov/nbr on
        # the hyperedge embeddings. Restrict to hyperedges with a valid (own-cell)
        # target: a size-normalized log1p pseudobulk of a cell-less spot is an
        # all-zero row, which would be a bogus reconstruction target (see design
        # notes on the cell-driven dataloader).
        if (self.optimize_hyperedges and hyperedge_recon is not None
                and hyperedge_feats is not None):
            valid_h = hyperedge_feats.abs().sum(dim=1) > 0
            if valid_h.any():
                He = hyperedge_embeddings[valid_h]
                Ht = hyperedge_feats[valid_h]
                Hr = hyperedge_recon[valid_h]
                hyperedge_total = hyperedge_total + self.rho_h * self.reconloss(Hr, Ht)
                if (self.lambda_var_h > 0 or self.lambda_cov_h > 0) and He.shape[0] >= 2:
                    vh, covh = self._variance_covariance(He, self.var_target)
                    hyperedge_total = hyperedge_total + self.lambda_var_h * vh + self.lambda_cov_h * covh
                if self.lambda_nbr_h > 0 and incidence_matrix is not None:
                    # hyperedge adjacency = hyperedges sharing >=1 cell:
                    # inc_h @ inc_h.t(), inc_h = (valid hyperedges × cells)
                    inc_h = incidence_matrix[:, valid_h].t()
                    nbr_h = self._neighborhood_contrastive(He, inc_h, feats=Ht)
                    if nbr_h is not None:
                        hyperedge_total = hyperedge_total + self.lambda_nbr_h * nbr_h

        loss = node_total + hyperedge_total
        return loss, node_total.detach(), hyperedge_total.detach()