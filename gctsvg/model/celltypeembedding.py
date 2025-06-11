import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.loader import DataLoader

from tqdm.auto import tqdm

def laplacian_smoothness_loss(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]
    loss = (diff ** 2).sum(dim=1).mean()
    return loss

class LaplacianSmoothnessLoss(nn.Module):
    def __init__(self):
        super(LaplacianSmoothnessLoss, self).__init__()

    def forward(self, decomposition, X, edge_index):
        total_smoothness = sum(laplacian_smoothness_loss(x_k, edge_index)
                              for x_k in decomposition)
        return total_smoothness

class GctSVGLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(GctSVGLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction=reduction)
        self.smoothness_loss = LaplacianSmoothnessLoss()
    
    def forward(self, decomposition, X, edge_index):
        reconstruction = decomposition.sum(dim=0)

        reconstruction_loss = self.reconstruction_loss(reconstruction, X)
        smoothness_loss = self.smoothness_loss(decomposition, X, edge_index)
        return reconstruction_loss + smoothness_loss

# method 1
# method 2
# preprocessing steps

class CellTypeEncoder(nn.Module):
    def __init__(self, num_feats, cell_type_num_feats):
        super(CellTypeEncoder, self).__init__()
        self.conv = GCNConv(num_feats, cell_type_num_feats, add_self_loops=True)
        self.act1 = nn.PReLU()
        self.batchnorm = GraphNorm(cell_type_num_feats)
        self.linear = nn.Linear(cell_type_num_feats, cell_type_num_feats)
        self.act2 = nn.PReLU()
    
    def forward(self, cell_feats, edge_index, cell_type_probs):
        # Inter graph convolution
        h1 = self.conv(cell_feats, edge_index)
        # for now, no weighted averaging on inter convolution
        h1 = self.act1(h1)
        h1 = self.batchnorm(h1)

        # Convolution 2
        h1 = self.linear(h1)
        prob = cell_type_probs.t()
        prob_den = prob.sum(dim=1).reshape(-1,1)
        m = prob @ h1
        m /= prob_den
        m = self.act2(m)
        return m

class ExpressionDecoder(nn.Module):
    def __init__(self, cell_type_num_feats, num_feats):
        super(ExpressionDecoder, self).__init__()
        self.cell_type_num_feats = cell_type_num_feats
        self.num_feats = num_feats
        self.linear1 = nn.Linear(cell_type_num_feats, cell_type_num_feats)
        self.act1 = nn.PReLU()
        self.conv = GCNConv(cell_type_num_feats, num_feats)
        self.act2 = nn.PReLU()
        self.batchnorm = GraphNorm(num_feats)
        self.linear3 = nn.Linear(num_feats, num_feats)

    def forward(self, encoding, edge_index, masked_probs):
        # training mode
        # probs = probs * mask
        # print("masked_probs:", masked_probs.shape, "encoding:", encoding.shape)

        # Convolution 1
        encoding = self.linear1(encoding)
        m = masked_probs @ encoding
        m = self.act1(m)

        # Convolution 2
        z1 = self.conv(m, edge_index)
        z1 = self.act2(z1)

        # Convolution 2 - Batchnorm
        # num_cell_types, batch_size = z1.shape[0], z1.shape[1]
        # z1_reshaped = z1.view(num_cell_types * batch_size, self.num_feats)
        # z1_bn = self.batchnorm(z1_reshaped)
        # z1 = z1.view(num_cell_types, batch_size, self.num_feats)
        
        # Generate Expression
        z1 = self.linear3(z1)
        return z1

class ExpressionDecomposer(nn.Module):
    def __init__(self, num_feats, cell_type_num_feats):
        super(ExpressionDecomposer, self).__init__()
        self.encoder = CellTypeEncoder(num_feats, cell_type_num_feats)
        self.decoder = ExpressionDecoder(cell_type_num_feats, num_feats)

    def forward(self, cell_feats, edge_index, cell_type_probs):
        # print("feats:", cell_feats.shape, "probs:", cell_type_probs.shape)
        encoding = self.encoder(cell_feats, edge_index, cell_type_probs)
        # print("encoding:", encoding.shape)
        rows, cols = cell_type_probs.shape
        masked_probs = torch.zeros((cols, rows, cols), dtype=torch.float, device=cell_type_probs.device)
        idx = torch.arange(cols)
        masked_probs[idx,:,idx] = cell_type_probs[:, idx].t()
        # print("masked_probs:", masked_probs.shape)
        expression = self.decoder(encoding, edge_index, masked_probs)
        return expression

class ExpressionDecomposerDriver(object):
    def __init__(self, num_feats, num_cell_types, cell_type_num_feats, 
                 lr=1e-4,
                 device="cuda:1"):
        self.num_feats = num_feats
        self.num_cell_types = num_cell_types
        self.cell_type_num_feats = cell_type_num_feats

        self.device = device
        self.decomposer = ExpressionDecomposer(self.num_feats, self.cell_type_num_feats).to(device)

        self.lr = lr
        self.criterion = GctSVGLoss()
        self.optimizer = optim.Adam(self.decomposer.parameters(), self.lr)

    def train(self, dataset, batch_size=-1, num_epochs=1000):
        self.decomposer.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        with tqdm(range(num_epochs)) as epbar:
            for epoch in epbar:
                batch_loss = 0
                with tqdm(loader, leave=False) as bpbar:
                    for batch in bpbar:
                        feats, probs = batch.x.split([self.num_feats, self.num_cell_types], dim=1)
                        edge_index = batch.edge_index
                        batch_batch = batch.batch
            
                        decomposition = self.decomposer(feats, edge_index, probs)
                        loss = self.criterion(decomposition, feats, edge_index)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        batch_loss += loss
                epoch_loss = batch_loss / len(loader)
                losses.append(epoch_loss.item())
        
        return losses

    def get_decomposition(self, dataset, batch_size=-1):
        self.decomposer.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            with tqdm(loader) as bpbar:
                for batch in bpbar:
                    feats, probs = batch.x.split([self.num_feats, self.num_cell_types], dim=1)
                    edge_index = batch.edge_index
        
                    decomposition = self.decomposer(feats, edge_index, probs)
                    reconstruction = decomposition.sum(dim=0)
                    
                    decomposition = decomposition.cpu().numpy()
                    reconstruction = reconstruction.cpu().numpy()
                    return decomposition, reconstruction