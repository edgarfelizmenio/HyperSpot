import gc
import torch
import torch.nn as nn

from tqdm import tqdm

class HGCTSVGLoss(nn.Module):
    def __init__(self,
                 input_dim,
                 projection_dim = 100,
                 temp_node = 0.5,
                 temp_hyperedge = 0.5,
                 alpha = 1.0,
                 beta = 1.0):
        super(HGCTSVGLoss, self).__init__()
        
        self.temp_node = temp_node
        self.temp_hyperedge = temp_hyperedge
        self.alpha = alpha
        self.beta = beta

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

    def __compute_similarity_matrices(self, projections, augmented_projections):
        projections_norm = projections / projections.norm(dim=1, keepdim=True)
        aug_projections_norm = augmented_projections / augmented_projections.norm(dim=2, keepdim=True)

        similarity = projections_norm @ projections_norm.t()
        
        # cross_view_similarity is the similarity between a projection in the original view and its augmented counterpart. 
        # Dimension: 2xN
        cross_view_similarity = torch.einsum("nf,knf->kn", projections_norm, aug_projections_norm)
        
        return similarity, cross_view_similarity

    def __compute_interclass_cl(self, similarity, same_class_mask):
        pos_pair_sim = similarity * same_class_mask
        pos_pair_loss = (pos_pair_sim / self.temp_node).exp()
        all_pair_loss = (similarity / self.temp_node).exp()
        node_pos_loss = pos_pair_loss.sum(dim=1)
        node_all_loss = all_pair_loss.sum(dim=1)

        node_interclass_loss = -(node_pos_loss.log() - node_all_loss.log())
        return node_interclass_loss

    def __compute_hyfi_cl(self, similarity, cross_view_similarity, commonality_matrix, temperature, same_class_mask=None):
        # compute positive pair score and sum for all projections
        positive_pair_loss = (cross_view_similarity / temperature).exp().sum(dim=0)  # (N,)

        if same_class_mask is not None:
            no_common = (commonality_matrix == 0) * same_class_mask
            commonality_matrix *= same_class_mask
        else:
            no_common = (commonality_matrix == 0)    
        
        diag = commonality_matrix.diag().unsqueeze(1)
        pos_pair_prob = commonality_matrix / diag
        weak_positive_weight = commonality_matrix * pos_pair_prob
        
        # pair-wise loss of a projection vs all projections in the batch
        # Dimension: NxN
        pair_wise_loss = (similarity / temperature).exp()
        
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
        node_intraclass_loss = -((positive_pair_loss + weak_positive_pair_loss).log() - (positive_pair_loss + weak_positive_pair_loss + negative_pair_loss).log())
        return node_intraclass_loss
    
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
        hyperedge_commonality = incidence_matrix.t() @ incidence_matrix

        node_similarity, node_cross_view_similarity = self.__compute_similarity_matrices(node_projections, augmented_node_projections)
        node_interclass_cl = self.__compute_interclass_cl(node_similarity, same_class_mask)
        node_intraclass_cl = self.__compute_hyfi_cl(node_similarity, node_cross_view_similarity, node_commonality, self.temp_node, same_class_mask)

        hyperedge_similarity, hyperedge_cross_view_similarity = self.__compute_similarity_matrices(hyperedge_projections, augmented_hyperedge_projections)
        hyperedge_level_cl = self.__compute_hyfi_cl(hyperedge_similarity, hyperedge_cross_view_similarity, hyperedge_commonality, self.temp_hyperedge)
        
        if sample_n_cells:
            # limit error computation to the sampled batch
            classes = classes[:sample_n_cells]
            node_interclass_cl = node_interclass_cl[:sample_n_cells]
            node_intraclass_cl = node_intraclass_cl[:sample_n_cells]
            hyperedge_level_cl = hyperedge_level_cl[:sample_n_hyperedges]

        # nodes_per_class = classes.bincount()
        # observed_classes = nodes_per_class > 0
        # total_node_loss_per_class = classes.bincount(weights=node_level_cl)
        # mean_loss_per_class = total_node_loss_per_class[observed_classes] / nodes_per_class[observed_classes]
        # intraclass_node_level_cl = mean_loss_per_class.mean()
        
        # batch_cl = intraclass_node_level_cl + self.alpha * hyperedge_level_cl.mean()
        batch_cl = node_interclass_cl.mean() + self.alpha * node_intraclass_cl.mean() + self.beta * hyperedge_level_cl.mean()
        
        return batch_cl
        
    
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