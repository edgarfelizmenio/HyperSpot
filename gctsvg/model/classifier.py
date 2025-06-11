import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import GCNConv
from torch_geometric.nn.norm import GraphNorm

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pathlib import Path

from .utils import compute_fc_layer_sizes, create_fc

class Classifier(nn.Module):
    def __init__(self, n_feats, n_classes, num_layers=5):
        super(Classifier, self).__init__()
        self.n_feats = n_feats
        self.n_classes = n_classes

        layer_sizes = compute_fc_layer_sizes(n_feats, n_classes, num_layers = num_layers)

        self.backbone_fc = create_fc(layer_sizes[:-1], bias=True)
        self.out_fc = nn.Linear(layer_sizes[-2], layer_sizes[-1])
    
    def forward(self, x):
        h = self.backbone_fc(x)
        out = self.out_fc(h)
        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False    

class ClassifierDriver(object):
    def __init__(self, 
                 n_feats, 
                 n_classes, 
                 device,
                 dtype,
                 lr=1e-4,
                 model_path='classifier.pth'):
        self.n_feats = n_feats
        self.n_classes = n_classes
        self.device = device
        self.dtype = dtype
        self.lr = lr
        self.model_path = model_path

        self.classifier = Classifier(self.n_feats, self.n_classes).to(self.device).to(self.dtype)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optim.Adam(self.classifier.parameters(), self.lr)

        self.model_path = Path(model_path)
        
        # if model already exists, load model to classifier
        if self.trained_classifier_exists():
            self.load_classifier()
            self.trained = True
        else:
            self.trained = False

    def trained_classifier_exists(self):
        return self.model_path.is_file()
    
    def train(self, dataset, num_epochs, batch_size):
        if self.trained_classifier_exists():
            print("Classifier already trained. Skipping...")
            return None
            
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        self.classifier.train()
        with tqdm(range(num_epochs)) as e_pbar:
            for epoch in e_pbar:
                batch_loss = 0
                with tqdm(data_loader, leave=False) as d_pbar:
                    for batch in d_pbar:
                        batch_X = batch[0]
                        batch_labels = batch[1]
                        batch_classification = self.classifier(batch_X)
                        loss = self.criterion(batch_classification, batch_labels)
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        
                        batch_loss += loss
                epoch_loss = batch_loss / len(data_loader)
                e_pbar.set_postfix_str(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss.item():.4f}')
                losses.append(epoch_loss.item())

        # save model after training
        self.save_classifier()
        self.trained = True
        
        return losses

    def save_classifier(self):
        torch.save(self.classifier.state_dict(), self.model_path)

    def load_classifier(self):
        weights = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.classifier.load_state_dict(weights)    
    
    def get_classifier(self):
        if self.trained_classifier_exists():
            # self.classifier.freeze()
            self.classifier.eval()
            return self.classifier
        else:
            return None