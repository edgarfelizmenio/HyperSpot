import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import GCNConv, DeepGraphInfomax
from torch_geometric.nn.models import GCN, GAT

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pathlib import Path

from .utils import compute_fc_layer_sizes, create_fc

def graph_embedding(z, *args, **kwargs):
    return z.mean(dim=0)

def shuffle_attributes(x, edge_index):
    permutation = torch.randperm(x.size(0))
    return x[permutation], edge_index

def make_baseline_data(n_obs, n_vars):
    return torch.zeros(n_obs, n_vars)

class GCNDGIAELoss(nn.Module):
    def __init__(self, alpha, beta):
        super(GCNDGIAELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reconstruction_criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, reconstruction, x, dgi_loss):
        reconstruction_loss = self.reconstruction_criterion(reconstruction, x)
        total_loss = reconstruction_loss
        total_loss = (self.alpha * reconstruction_loss) + (self.beta * dgi_loss)
        return total_loss, reconstruction_loss, dgi_loss
        
# create an embedding
# class GCNDGIAE(nn.Module):
#     def __init__(self, 
#                  n_in, 
#                  n_out,
#                  Activation="prelu",
#                  summary = graph_embedding,
#                  corruption = shuffle_attributes):
#         super(GCNDGIAE, self).__init__()
#         n_hidden = (n_in + n_out)//2
#         self.encoder = GCN(n_in, n_hidden, 2, n_out, dropout=0.2, act=Activation, jk="cat", improved=True)
#         self.decoder = GCN(n_out, n_hidden, 2, n_in, act=Activation, jk="cat", improved=True)
        
#         self.DGI = DeepGraphInfomax(n_out,
#                                    self.encoder,
#                                    summary = summary,
#                                    corruption = corruption)

#         self.DGI.reset_parameters()
#         print(self.encoder)
#         print(self.decoder)
#         print(self.DGI)
#         print(self)
        
#     def forward(self, x, edge_index):
#         pos_z, neg_z, summary = self.DGI(x, edge_index) 
#         dgi_loss = self.DGI.loss(pos_z, neg_z, summary)
#         reconstruction = self.decoder(pos_z, edge_index)
#         return pos_z, reconstruction, dgi_loss

class GCNDGIAE(nn.Module):
    def __init__(self, 
                 n_in,
                 n_out,
                 Activation = "leakyrelu",
                 summary = graph_embedding,
                 corruption = shuffle_attributes,
                 alpha=1,
                 beta=0.1):
        super(GCNDGIAE, self).__init__()
        # self.save_hyperparameters()
        n_hidden = (n_in + n_out) // 2
        self.encoder = GAT(n_in, n_hidden, 3, n_out, dropout=0.2, act=Activation, jk="cat")
        self.decoder = GAT(n_out, n_hidden, 3, n_in, act=Activation, jk="cat")        

        self.DGI = DeepGraphInfomax(n_out,
                                   self.encoder,
                                   summary = summary,
                                   corruption = corruption)

        self.DGI.reset_parameters()

    def forward(self, x, edge_index):
        pos_z = self.encoder(x, edge_index)
        pos_z, neg_z, summary = self.DGI(x, edge_index)
        dgi_loss = self.DGI.loss(pos_z, neg_z, summary)
        reconstruction = self.decoder(pos_z, edge_index)
        # return pos_z, reconstruction
        return pos_z, reconstruction, dgi_loss


class FFN(nn.Module):
    def __init__(self, n_in, n_out, num_layers=5, Activation=nn.PReLU, dropout=None):
        super(FFN, self).__init__()
        self.n_in = n_in
        self.n_out = n_out

        layer_sizes = compute_fc_layer_sizes(self.n_in, self.n_out, num_layers=num_layers)

        self.backbone_fc = create_fc(layer_sizes[:-1], bias=True, Activation=Activation, dropout=dropout)
        if dropout:
            self.out_fc = nn.Sequential(nn.Dropout(dropout), 
                                        nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        else:
            self.out_fc = nn.Linear(layer_sizes[-2], layer_sizes[-1])
    
    def forward(self, x):
        h = self.backbone_fc(x)
        out = self.out_fc(h)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, n_feats, n_code, dropout=0.2):
        super(AutoEncoder, self).__init__()
        self.encoder = FFN(n_feats, n_code, dropout = dropout)
        self.decoder = FFN(n_code, n_feats)
    
    def forward(self, X):
        code = self.encoder(X)
        reconstruction = self.decoder(code)
        return code, reconstruction

# class STDenoiser(object):
#     def __init__(self,
#                  n_in,
#                  n_out,
#                  device,
#                  dtype,
#                  alpha=1,
#                  beta=0.1,
#                  lr=1e-3,
#                  model_path='st_denoiser.pth'):
        
#         self.ae = GCNDGIAE(n_in, n_out).to(device).to(dtype)
#         self.criterion = GCNDGIAELoss(alpha, beta)
#         self.lr = lr
#         self.optimizer =  optim.Adam(self.ae.parameters(), self.lr, weight_decay=1e-5)fssdp
#         reconstruction_losses = []

#         with tqdm(range(num_epochs)) as e_pbar:
#             X = dataset.x
#             edge_index = dataset.edge_index

#             self.ae.train()
#             for epoch in e_pbar:
#                 code, reconstruction, dgi_loss = self.ae(X, edge_index)        
#                 total_loss, reconstruction_loss, contrastive_loss = self.criterion(reconstruction, X, dgi_loss)

#                 self.optimizer.zero_grad()
#                 total_loss.backward()
#                 self.optimizer.step()

#                 contrastive_losses.append(contrastive_loss.item())
#                 reconstruction_losses.append(reconstruction_loss.item())
#                 total_losses.append(total_loss.item())
                
#                 e_pbar.set_postfix_str(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}')


#         return total_losses, reconstruction_losses, contrastive_losses
    
#     def denoise(self, dataset):
#         X = dataset.x
#         edge_index = dataset.edge_index
#         self.ae.eval()
#         with torch.no_grad():
#             code, reconstruction, _ = self.ae(X, edge_index)

#         code = code.cpu().detach()
#         reconstruction = reconstruction.cpu().detach()

#         return code, reconstruction

#     def fit_denoise(self, dataset, num_epochs):
#         total_losses, reconstruction_losses, contrastive_losses = self.fit(dataset, num_epochs)
#         code, reconstruction = self.denoise(dataset)
#         return code, reconstruction, total_losses, reconstruction_losses, contrastive_losses

class SCDenoiser(object):
    def __init__(self,
                 n_in,
                 n_out,
                 device,
                 dtype,
                 lr=1e-3,
                 model_path='sc_denoiser.pth'):

        self.device = device
        self.dtype = dtype
        self.lr = lr
        
        self.ae = AutoEncoder(n_in, n_out).to(self.device).to(self.dtype)
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer =  optim.Adam(self.ae.parameters(), self.lr, weight_decay=1e-5)

        print(self.ae)
        self.model_path = Path(model_path)
        
    def fit(self, dataset, num_epochs, batch_size):
        total_losses = []

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.ae.train()
        with tqdm(range(num_epochs)) as e_pbar:
            for epoch in e_pbar:
                epoch_loss = 0
                with tqdm(data_loader, leave=False) as d_pbar:
                    for batch in d_pbar:
                        batch_X = batch[0]
                        _, reconstruction = self.ae(batch_X)
                        batch_loss = self.criterion(reconstruction, batch_X)

                        self.optimizer.zero_grad()
                        batch_loss.backward()
                        self.optimizer.step()

                        epoch_loss += batch_loss
                epoch_loss /= len(data_loader)

                total_losses.append(epoch_loss.item())
                e_pbar.set_postfix_str(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss.item():.4f}')

        return total_losses
    
    def denoise(self, dataset, batch_size):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        code = []
        reconstruction = []
        self.ae.eval()
        with torch.no_grad():
            with tqdm(data_loader, leave=False) as d_pbar:
                for batch in d_pbar:
                    batch_X = batch[0]
                    code_batch, reconstruction_batch = self.ae(batch_X)
                    code.append(code_batch)
                    reconstruction.append(reconstruction_batch)

        code = torch.cat(code, dim=0)
        reconstruction = torch.cat(reconstruction, dim=0)
                    
        code = code.cpu().detach()
        reconstruction = reconstruction.cpu().detach()

        return code, reconstruction

    def fit_denoise(self, dataset, num_epochs, batch_size):
        total_losses = self.fit(dataset, num_epochs, batch_size)
        code, reconstruction = self.denoise(dataset, batch_size)
        return code, reconstruction, total_losses