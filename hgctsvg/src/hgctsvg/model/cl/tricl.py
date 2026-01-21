import gc
import torch
import torch.nn as nn

from tqdm import tqdm

class TriCLLoss(nn.Module):
    def __init__(self, 
                 input_dim,
                 projection_dim = 100,
                 temp_node = 0.5,
                 temp_hyperedge = 0.5,
                 temp_membership = 0.5,
                 weight_hyperedge_cl = 1.0,
                 weight_membership_cl = 1.0):
        super(TriCLLoss, self).__init__()
        # alpha is the edge_loss weight

        self.temp_node = temp_node
        self.temp_hyperedge = temp_hyperedge
        self.temp_membership = temp_membership
        
        self.weight_hyperedge_cl = weight_hyperedge_cl
        self.weight_membership_cl = weight_membership_cl
        
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
        self.discriminator = nn.Bilinear(input_dim, input_dim, 1)

    def __membership_sim(self, node_proj, hyperedge_proj):
        return torch.sigmoid(self.discriminator(node_proj, hyperedge_proj))
    
    def __compute_cl(self, projections_0, projections_1, temp):
        # compute similarity matrix 
        sim = projections_0 @ projections_1.t()
        pair_wise_loss = (sim / self.temp_node).exp()
        diag = pair_wise_loss.diag()
        cl = diag / pair_wise_loss.sum(dim=1)
        sym_cl = diag / pair_wise_loss.sum(dim=0)
        total_cl = (cl + sym_cl) / 2
        
        return total_cl

    def __compute_one_way_mem_cl(self, node_proj, hyp_proj, pos_row, pos_col, n_nodes, n_hyperedges):
        n_pos_pairs = len(pos_row)
        # compute positive pair loss
        disc_pos = (self.__membership_sim(node_proj[pos_row], hyp_proj[pos_col]) / self.temp_membership).exp()
        
        # compute approximate negative pair loss anchored on nodes
        hyperedge_neg = torch.randint(0, n_hyperedges, (n_pos_pairs,), device=disc_pos.device)
        disc_neg_node_anchor = (self.__membership_sim(node_proj[pos_row], hyp_proj[hyperedge_neg]) / self.temp_membership).exp()
        mem_cl_node = -torch.log(disc_pos / (disc_pos + disc_neg_node_anchor.sum()))
        
        # compute approximate negative pair loss anchored on hyperedges
        node_neg = torch.randint(0, n_nodes, (n_pos_pairs,), device=disc_pos.device)
        disc_neg_hyperedge_anchor = (self.__membership_sim(node_proj[node_neg], hyp_proj[pos_col]) / self.temp_membership).exp()
        mem_cl_hyperedge = -torch.log(disc_pos / (disc_pos + disc_neg_hyperedge_anchor.sum()))

        return mem_cl_node + mem_cl_hyperedge
                                      
    def __compute_membership_cl(self, node_proj_0, node_proj_1, hyp_proj_0, hyp_proj_1, incidence_matrix):
        pos_row, pos_col = incidence_matrix.nonzero(as_tuple=True)

        n_nodes, n_hyperedges = incidence_matrix.shape
        
        mem_cl = self.__compute_one_way_mem_cl(node_proj_0, hyp_proj_1, pos_row, pos_col, n_nodes, n_hyperedges)
        sym_mem_cl = self.__compute_one_way_mem_cl(node_proj_1, hyp_proj_0, pos_row, pos_col, n_nodes, n_hyperedges)

        return mem_cl + sym_mem_cl

    def forward(self, 
                node_embeddings, 
                hyperedge_embeddings, 
                augmented_node_embeddings, 
                augmented_hyperedge_embeddings,
                incidence_matrix,
                cell_idx = None,
                hyperedge_idx = None):

        # project (two-layer MLP with ELU activation)
        node_projections_0 = self.project_nodes(augmented_node_embeddings[0])
        hyperedge_projections_0 = self.project_hyperedges(augmented_hyperedge_embeddings[0])

        node_projections_1 = self.project_nodes(augmented_node_embeddings[1])
        hyperedge_projections_1 = self.project_hyperedges(augmented_hyperedge_embeddings[1])

        # incidence_matrix = incidence_matrix.to_sparse()        
        # node_commonality = incidence_matrix @ incidence_matrix.t()
        # node_level_cl = self.__compute_cl(node_projections, augmented_node_projections, node_commonality, self.temp_node)
        node_level_cl = self.__compute_cl(node_projections_0, node_projections_1, self.temp_node)
        hyperedge_level_cl = self.__compute_cl(hyperedge_projections_0, hyperedge_projections_1, self.temp_hyperedge)
        membership_level_cl = self.__compute_membership_cl(node_projections_0, 
                                                           node_projections_1, 
                                                           hyperedge_projections_0,
                                                           hyperedge_projections_1, 
                                                           incidence_matrix)

        if cell_idx is None:
            batch_cl = node_level_cl.mean() + \
                        self.weight_hyperedge_cl * hyperedge_level_cl.mean() + \
                        self.weight_membership_cl * membership_level_cl.mean()
        else:
            # find the indexes in membership_level_cl that indicates membership of spots to given
            # cells indicated by cell_idx and given hyperegde given by hyperedge_idx
            nz = incidence_matrix.nonzero()
            row_mask = torch.zeros(incidence_matrix.size(0), dtype=torch.bool, device=incidence_matrix.device)
            col_mask = torch.zeros(incidence_matrix.size(1), dtype=torch.bool, device=incidence_matrix.device)
            
            row_mask[cell_idx] = True
            col_mask[hyperedge_idx] = True
            
            node_hyperedge_idx = (row_mask[nz[:, 0]] & col_mask[nz[:, 1]]).nonzero(as_tuple=True)[0]
            
            # compute batch cl limited to cell_idx and hyperedge_idx
            batch_cl = node_level_cl[:cell_idx.shape[0]].mean() + \
                        self.weight_hyperedge_cl * hyperedge_level_cl[:hyperedge_idx.shape[0]].mean() + \
                        self.weight_membership_cl * membership_level_cl[node_hyperedge_idx].mean()
        return batch_cl
        

class TriCL(nn.Module):
    def __init__(self, 
                 encoder, 
                 drop_feat_prob=0.2, 
                 drop_membership_prob=0.2):
        super(TriCL, self).__init__()
        self.encoder = encoder
        self.drop_feat_prob = drop_feat_prob
        self.drop_membership_prob = drop_membership_prob
    
    def __drop_node_feats(self, node_feats):
        mask = torch.bernoulli(torch.full((node_feats.shape[1],), 1 - self.drop_feat_prob, device=node_feats.device))
        aug_node_feats = node_feats * mask
        return aug_node_feats
    
    def __drop_node_membership(self, incidence_matrix):
        # Step 1: find nonzero indices
        nonzero_idx = incidence_matrix.nonzero(as_tuple=True)  # returns tuple (rows, cols)
    
        # Step 2: generate Bernoulli mask for nonzero elements
        n_members = nonzero_idx[0].numel()
        membership_mask = torch.bernoulli(torch.full((n_members,), 1 - self.drop_membership_prob, device=incidence_matrix.device))
    
        # Step 3: apply mask
        im_masked = incidence_matrix.clone()
        im_masked[nonzero_idx] *= membership_mask
    
        return im_masked    
    
    def forward(self, batch):
        # forward on original data
        node_embeddings, hyperedge_embeddings = self.encoder(batch)
        # make multiple views
        augmented_node_embeddings = torch.zeros((2, node_feats.shape[0], node_feats.shape[1])).to(node_feats.device)
        augmented_hyperedge_embeddings = torch.zeros((2, hyperedge_ids.shape[0], node_feats.shape[1])).to(node_feats.device)
        # forward on data with noise
        orig_node_feats = batch.batch_cell_feats
        orig_incidence_matrix = batch.batch_incidence_matrix
        for i in range(2):
            batch.batch_cell_feats = self.__drop_node_feats(orig_node_feats)
            batch.batch_incidence_matrix = self.__drop_node_membership(orig_incidence_matrix)
            augmented_node_embeddings[i], augmented_hyperedge_embeddings[i] = self.encoder(batch)
        
        return node_embeddings, hyperedge_embeddings, augmented_node_embeddings, augmented_hyperedge_embeddings
    
    