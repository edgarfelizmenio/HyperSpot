import gc
import torch
import torch.nn as nn

from tqdm import tqdm

from collections import namedtuple

class HyFiLoss(nn.Module):
    def __init__(self, 
                 input_dim,
                 projection_dim = 100,
                 temp_node = 0.5,
                 temp_hyperedge = 0.5,
                 alpha=1.0):
        super(HyFiLoss, self).__init__()
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

    def __compute_cl(self, projections, augmented_projections, commonality_matrix, temp): 
        projections_norm = projections / projections.norm(dim=1, keepdim=True)
        aug_projections_norm = augmented_projections / augmented_projections.norm(dim=2, keepdim=True)

        # cross_view_similarity is the similarity between a projection in the original view and its augmented counterpart. 
        # Dimension: 2xN
        cross_view_similarity = torch.einsum("nf,knf->kn", projections_norm, aug_projections_norm)
        
        # compute positive pair score and sum for all projections
        positive_pair_loss = (cross_view_similarity / temp).exp().sum(dim=0)  # (N,)

        diag = commonality_matrix.diag().unsqueeze(1)
        # commonality_matrix = commonality_matrix.to_sparse()
        pos_pair_prob = commonality_matrix/diag
        weak_positive_weight = commonality_matrix * pos_pair_prob
        # no_common indicates no common relationships (nodes with no common hyperedges, hyperedges with no common nodes)
        no_common = (commonality_matrix == 0)       
        
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

        # row, col = commonality_matrix.indices()
        # val = commonality_matrix.values()     # (E,)
        
        # pos-pair-prob only for edges
        # pos_pair_prob_vals = val / diag[row]  # (E,)
        # weak_positive_weight_vals = val * pos_pair_prob_vals  # (E,)
        
        # ------------------------------
        # 4. Sparse pairwise similarities only on edges
        # ------------------------------

        # sim_vals = (projections_norm[row] * projections_norm[col]).sum(dim=1)  # (E,)
        # pair_vals = torch.exp(sim_vals / temp)


        # print(hyperedge_index.nbytes /  (1024**3))
        # print(projections.nbytes /  (1024**3))
        # print(projections_norm.nbytes /  (1024**3))
        # print(augmented_projections.nbytes /  (1024**3))
        # print(aug_projections_norm.nbytes /  (1024**3))
        # print(cross_view_similarity.nbytes /  (1024**3))
        # print(hyperedge_index.nbytes /  (1024**3))
        # print(commonality_matrix.nbytes /  (1024**3))


        
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
        
            # # flatten all edges in batch into a big E_batch
            # E_batch = int(lens.sum().item())
        
            # if E_batch == 0:
            #     continue

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
                n_cells = None,
                n_hyperedges = None):

        # project (two-layer MLP with ELU activation)
        node_projections = self.project_nodes(node_embeddings)
        hyperedge_projections = self.project_hyperedges(hyperedge_embeddings)

        augmented_node_projections = self.project_nodes(augmented_node_embeddings)
        augmented_hyperedge_projections = self.project_hyperedges(augmented_hyperedge_embeddings)

        # incidence_matrix = incidence_matrix.to_sparse()        
        node_commonality = incidence_matrix @ incidence_matrix.t()
        node_level_cl = self.__compute_cl(node_projections, augmented_node_projections, node_commonality, self.temp_node)
        # node_level_cl = self.__compute_cl_batch(node_projections, augmented_node_projections, incidence_matrix, self.temp_node)

        hyperedge_commonality = incidence_matrix.t() @ incidence_matrix
        hyperedge_level_cl = self.__compute_cl(hyperedge_projections, augmented_hyperedge_projections, hyperedge_commonality, self.temp_hyperedge)

        batch_cl = node_level_cl.mean() + self.alpha * hyperedge_level_cl.mean()

        return batch_cl

class HyFi(nn.Module):
    def __init__(self, encoder, embedding_dim, num_views=2):
        super(HyFi, self).__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_views = num_views

    def __add_gaussian_noise(self, node_feats, std=0.1):
        noise = torch.normal(0, std, size=node_feats.size()).to(node_feats.device)
        aug_feats = torch.where(node_feats - noise < 0, node_feats + noise, node_feats - noise)
        return aug_feats
    
    def forward(self, batch):
        # forward on original data
        # node_embeddings, hyperedge_embeddings = self.encoder(node_feats, 
        #                                                      hyperedge_ids, 
        #                                                      incidence_matrix)
        node_embeddings, hyperedge_embeddings = self.encoder(batch)
        
        # make multiple views
        augmented_node_embeddings = torch.zeros((self.num_views, batch.batch_subgraph_cells.shape[0], self.embedding_dim)).to(batch.batch_cell_feats.device)
        augmented_hyperedge_embeddings = torch.zeros((self.num_views, batch.batch_subgraph_hyperedges.shape[0], self.embedding_dim)).to(batch.batch_cell_feats.device)
        # forward on data with noise
        for i in range(self.num_views):
            perturbed_cell_feats = self.__add_gaussian_noise(batch.batch_cell_feats)
            augmented_node_embeddings[i], augmented_hyperedge_embeddings[i] = self.encoder(batch._replace(batch_cell_feats=perturbed_cell_feats))
        
        return node_embeddings, hyperedge_embeddings, augmented_node_embeddings, augmented_hyperedge_embeddings
 
class HyFiAE(nn.Module):
    def __init__(self, 
                 encoder, 
                 embedding_dim,
                 num_views=2):
        super(HyFiAE, self).__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_views = num_views

    def __add_gaussian_noise(self, node_feats, std=0.1):
        noise = torch.normal(0, std, size=node_feats.size()).to(node_feats.device)
        aug_feats = torch.where(node_feats - noise < 0, node_feats + noise, node_feats - noise)
        return aug_feats
    
    def forward(self, batch):
        # forward on original data
        node_embeddings, hyperedge_embeddings, reconstruction = self.encoder(batch)
        
        # make multiple views
        # augmented_node_embeddings = torch.zeros((self.num_views, batch.batch_subgraph_cells.shape[0], self.embedding_dim)).to(batch.batch_cell_feats.device)
        # augmented_hyperedge_embeddings = torch.zeros((self.num_views, batch.batch_subgraph_hyperedges.shape[0], self.embedding_dim)).to(batch.batch_cell_feats.device)

        # forward on data with noise
        # for i in range(self.num_views):
        #     perturbed_cell_feats = self.__add_gaussian_noise(batch.batch_cell_feats)
        #     augmented_node_embeddings[i], augmented_hyperedge_embeddings[i], _ = self.encoder(batch._replace(batch_cell_feats=perturbed_cell_feats))

        return node_embeddings, hyperedge_embeddings, reconstruction
        # return node_embeddings, hyperedge_embeddings, augmented_node_embeddings, augmented_hyperedge_embeddings, reconstruction
    