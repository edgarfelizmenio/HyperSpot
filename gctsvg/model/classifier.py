import torch
import torch.nn as nn

from .utils import compute_fc_layer_sizes, create_fc

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, num_layers=3):
        super(EmbeddingClassifier, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        layer_sizes = compute_fc_layer_sizes(input_dim, n_classes, num_layers = num_layers)
        
        self.backbone_fc = create_fc(layer_sizes[:-1], bias=True)
        self.out_fc = nn.Linear(layer_sizes[-2], layer_sizes[-1])

    def forward(self, X):
        h = self.backbone_fc(X)
        out = self.out_fc(h)
        return out