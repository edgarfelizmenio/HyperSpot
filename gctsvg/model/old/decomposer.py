import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm.auto import tqdm

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
    
    def forward(self, reconstruction, X):
        return self.criterion(reconstruction, X)

def cosine_similarity(x1, x2, eps = -5e-12):
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)

    # compute cosine similarity matrix
    sim = x1_norm @ x2_norm.T
    return sim

class SpatialLoss(nn.Module):
    def __init__(self, temperature = 1):
        super(SpatialLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, reconstruction, X, edge_index):        
        sim = cosine_similarity(reconstruction, X) 
        sim /= self.temperature
        
        sim_exp = torch.exp(sim)
        
        # numerator: positive pairs -> similarity of each node to its neighbors
        # -> sum similarity of nodes with incident edges from source
        src, dst = edge_index
        edge_similarities = sim_exp[src, dst]
        
        # accumulate sum of similarity of neighbors of each node in numerator
        num = torch.zeros(reconstruction.size(0), device=reconstruction.device, dtype=reconstruction.dtype)
        num = num.index_add(0, src, edge_similarities)        

        # print(num.shape, num)
        # denominator: negative pairs + positive pairs -> similarity matrix - diagonal
        den = sim_exp.sum(axis=1) - torch.diag(sim_exp, 0)
        # print(den.shape, den)
        
        quotient = torch.div(num, den)
        
        loss = -torch.log(quotient).mean()
        
        return loss

def preprocess_prediction(expression, target_sum, sc_std, sc_max, eps=1e-8):
    # expression -> n_obs x n_cell_types x n_feats
    
    # normalize_total
    # print(expression.shape)
    expr_sum = torch.sum(expression, dim=2)
    expr_sum = expr_sum.unsqueeze(2)
    # print(expr_sum.shape)
    expr_normalized = expression / (expr_sum + eps)
    expr_normalized *= target_sum

    # log1p
    expr_log1p = torch.log1p(expr_normalized)

    # scale
    expr_scaled = expr_log1p / sc_std

    # clip
    expr_clipped = torch.where(expr_scaled > sc_max, sc_max, expr_scaled)
    return expr_clipped
    
    
class PriorLoss(nn.Module):
    def __init__(self, n_cell_types, classifier, sc_target_sum, sc_std, sc_max):
        super(PriorLoss, self).__init__()
        self.n_cell_types = n_cell_types
        self.classifier = classifier
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
        self.sc_target_sum = sc_target_sum
        self.sc_std = sc_std
        self.sc_max = sc_max
        
        # self.centroids = centroids
        # self.criterion = nn.MSELoss(reduction='mean')

    
    def forward(self, expression):
        n_cells = expression.size(0)


        # map predicted expressions to classifier input
        expression = preprocess_prediction(expression, self.sc_target_sum, self.sc_std, self.sc_max)
        expression = expression.view(-1, expression.size(-1))
        
        logits = self.classifier(expression)
        labels = torch.arange(self.n_cell_types, dtype=torch.long, device=expression.device).repeat(n_cells)
        ce_loss = self.criterion(logits, labels)
        
        # centroids = self.centroids.repeat(n_cells, 1)
        # print(type(expression), type(centroids))
        # print(expression.shape, centroids.shape)
        # ce_loss = self.criterion(expression, centroids)
        return ce_loss
        
class DecompositionLoss(nn.Module):
    def __init__(self, n_cell_types, classifier,
                 sc_target_sum, sc_std, sc_max,
                 alpha, beta, gamma):
        super(DecompositionLoss, self).__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.spatial_loss = SpatialLoss()
        self.prior_loss = PriorLoss(n_cell_types, classifier,
                                   sc_target_sum, sc_std, sc_max)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, expression, reconstruction, X, edge_index):
        recon_loss = self.reconstruction_loss(reconstruction, X)
        spatial_loss = self.spatial_loss(reconstruction, X, edge_index)
        prior_loss = self.prior_loss(expression)
        
        # total_loss = recon_loss + (self.beta * prior_loss)
        total_loss = (self.alpha * recon_loss) + (self.beta * spatial_loss) + (self.gamma * prior_loss)
        # print(total_loss, recon_loss, prior_loss, spatial_loss)
        return total_loss, recon_loss, prior_loss, spatial_loss

class Decomposer(nn.Module):
    def __init__(self, n_obs, n_feats, n_cell_types, centroids):
        super(Decomposer, self).__init__()
        self.n_obs = n_obs
        self.n_cell_types = n_cell_types
        self.W = nn.Parameter(torch.ones(n_obs, n_cell_types))
        
        init_expression = centroids.unsqueeze(0).repeat(n_obs, 1, 1)
        self.expression = nn.Parameter(init_expression)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
    
    def forward(self):
        expression = torch.round(self.expression)
        expression = torch.relu(self.expression)
        proportions = F.softmax(self.W, dim=1)
        # proportions = torch.where(proportions  < 0.05, 0, proportions)
        decomposition = proportions.unsqueeze(-1) * expression
        reconstruction = decomposition.sum(dim=1)

        return expression, proportions, reconstruction
        
class DecomposerDriver(object):
    def __init__(self, n_obs, n_feats, n_cell_types, centroids, classifier,
                 sc_target_sum, sc_std, sc_max,
                 device, dtype,
                 alpha, beta, gamma,
                lr = 1e-3):
        self.n_obs = n_obs
        self.n_feats = n_feats
        self.n_cell_types = n_cell_types
        self.device = device
        self.dtype = dtype
        self.lr = lr
    
        self.decomposer = Decomposer(n_obs, n_feats, n_cell_types, centroids).to(self.device).to(self.dtype)
        self.criterion = DecompositionLoss(n_cell_types, classifier,
                                    sc_target_sum, sc_std, sc_max,
                                    alpha, beta, gamma)
    
        self.optimizer = optim.Adam(self.decomposer.parameters(), self.lr)
    
    def fit(self, dataset, num_epochs=1200):
        total_losses = []
        recon_losses = []
        prior_losses = []
        spatial_losses = []
        with tqdm(range(num_epochs)) as e_pbar:
            X = dataset.x
            edge_index = dataset.edge_index
            # print(X.dtype)
            for epoch in e_pbar:
                
                expression, proportions, reconstruction = self.decomposer()
                # print(decomposition.dtype, proportions.dtype, reconstruction.dtype)
                total_loss, recon_loss, prior_loss, spatial_loss = self.criterion(expression, reconstruction, X, edge_index)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                total_losses.append(total_loss.item())
                recon_losses.append(recon_loss.item())
                prior_losses.append(prior_loss.item())
                spatial_losses.append(spatial_loss.item())

                e_pbar.set_postfix_str(f'Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item():.4f}')

        return total_losses, recon_losses, prior_losses, spatial_losses

    def get_decomposition(self):
        with torch.no_grad():
            expression, proportions, reconstruction = self.decomposer()
        
        expression = expression.cpu().detach()
        proportions = proportions.cpu().detach()
        reconstruction = reconstruction.cpu().detach()
        
        return expression, proportions, reconstruction