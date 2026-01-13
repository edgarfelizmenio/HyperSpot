import torch
import torch.nn as nn

from tqdm import tqdm

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
        
        return cl        

    def __compute_cl_batch(self, projections, augmented_projections, hyperedge_index, temp, chunk_size=7000):
        N = projections.shape[0]        

        projections_norm = projections / projections.norm(dim=1, keepdim=True)
        aug_projections_norm = augmented_projections / augmented_projections.norm(dim=2, keepdim=True)
        cross_view_similarity = torch.einsum("nf,knf->kn", projections_norm, aug_projections_norm)        
        positive_pair_loss = (cross_view_similarity / temp).exp().sum(dim=0)  # (N,)
        
        diag = torch.zeros(N, device=projections.device)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            hyperedge_index_tile = hyperedge_index[start:end]
            commonality_matrix = hyperedge_index_tile @ hyperedge_index_tile.t()
            diag[start:end] += commonality_matrix.diag()

            del hyperedge_index_tile, commonality_matrix

        weak_positive_pair_loss = torch.zeros(N, device=projections.device)
        negative_pair_loss = torch.zeros(N, device=projections.device)
        for start_i in tqdm(range(0, N, chunk_size), leave=False):
            end_i = min(start_i + chunk_size, N)
            
            for start_j in range(0, N, chunk_size):
                end_j = min(start_j + chunk_size, N)
                
                commonality_ij = hyperedge_index[start_i:end_i] @ hyperedge_index[start_j:end_j].t()
                pos_pair_prob_ij = commonality_ij / diag[start_i:end_i].unsqueeze(1)
                weak_positive_weight_ij = commonality_ij * pos_pair_prob_ij

                del pos_pair_prob_ij
                
                no_common_ij = (commonality_ij == 0)

                del commonality_ij
                
                similarity_ij = projections_norm[start_i:end_i] @ projections_norm[start_j:end_j].t()

                pair_wise_loss_ij = (similarity_ij / temp).exp()

                del similarity_ij
                
                weak_positive_pair_loss[start_i:end_i] += (pair_wise_loss_ij * weak_positive_weight_ij).sum(dim=1)

                del weak_positive_weight_ij
                
                negative_pair_loss[start_i:end_i] += (pair_wise_loss_ij * no_common_ij).sum(dim=1)

                del no_common_ij, pair_wise_loss_ij

        del diag
        
        cl = -((positive_pair_loss + weak_positive_pair_loss).log() - (positive_pair_loss + weak_positive_pair_loss + negative_pair_loss).log())
        
        return cl

    def forward(self, 
                node_embeddings, 
                hyperedge_embeddings, 
                augmented_node_embeddings, 
                augmented_hyperedge_embeddings,
                incidence_matrix,
                n_cells = None,
                n_hyperedges = None):

        # project (two-layer MLP with ELU activation)
        node_projections = self.project_nodes(node_embeddings)
        hyperedge_projections = self.project_hyperedges(hyperedge_embeddings)

        augmented_node_projections = self.project_nodes(augmented_node_embeddings)
        augmented_hyperedge_projections = self.project_hyperedges(augmented_hyperedge_embeddings)

        # incidence_matrix = incidence_matrix.to_sparse()        
        # node_commonality = incidence_matrix @ incidence_matrix.t()
        # node_level_cl = self.__compute_cl(node_projections, augmented_node_projections, node_commonality, self.temp_node)
        node_level_cl = self.__compute_cl_batch(node_projections, augmented_node_projections, incidence_matrix, self.temp_node)

        hyperedge_commonality = incidence_matrix.t() @ incidence_matrix
        hyperedge_level_cl = self.__compute_cl(hyperedge_projections, augmented_hyperedge_projections, hyperedge_commonality, self.temp_hyperedge)

        if n_cells:
            batch_cl = node_level_cl.mean() + self.alpha * hyperedge_level_cl.mean()
        else:
            batch_cl = node_level_cl[:n_cells].mean() + self.alpha * hyperedge_level_cl[:n_hyperedges].mean()
        return batch_cl
        

class STHypergraphCL(nn.Module):
    def __init__(self, encoder, embedding_dim, num_views=2):
        super(STHypergraphCL, self).__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_views = num_views

    def __add_gaussian_noise(self, node_feats, std=0.1):
        noise = torch.normal(0, std, size=node_feats.size()).to(node_feats.device)
        aug_feats = torch.where(node_feats - noise < 0, node_feats + noise, node_feats - noise)
        return aug_feats
    
    def forward(self, node_feats, hyperedge_ids, incidence_matrix):
        # forward on original data
        node_embeddings, hyperedge_embeddings = self.encoder(node_feats, 
                                                             hyperedge_ids, 
                                                             incidence_matrix)
        # make multiple views
        augmented_node_embeddings = torch.zeros((self.num_views, node_feats.shape[0], node_feats.shape[1])).to(node_feats.device)
        augmented_hyperedge_embeddings = torch.zeros((self.num_views, hyperedge_ids.shape[0], node_feats.shape[1])).to(node_feats.device)
        # forward on data with noise
        for i in range(self.num_views):
            perturbed_node_feats = self.__add_gaussian_noise(node_feats)
            augmented_node_embeddings[i], augmented_hyperedge_embeddings[i] = self.encoder(perturbed_node_feats,
                                                                                           hyperedge_ids,
                                                                                           incidence_matrix)
        
        return node_embeddings, hyperedge_embeddings, augmented_node_embeddings, augmented_hyperedge_embeddings
        # total_contrastive_loss = HyFiLoss(node_embeddings, 
        #                                   hyperedge_embeddings, 
        #                                   augmented_node_embeddings, 
        #                                   augmented_hyperedge_embeddings)


        
        # return total_contrastive_loss
    
    